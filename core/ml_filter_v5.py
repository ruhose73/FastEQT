"""
ml_filter_v5.py — фильтрация associations.xml по локальной магнитуде ML.

Единственный путь агрегации ML — compute_event_ml_v2 (медиана → σ-фильтр →
среднее). Все константы формулы/порогов/окон — CLI-флаги (см. таблицу
"Основные параметры" в ml_filter_v5.md), кроме AMP_SCALE_NM (м → нм, ×1e9)
— это не настраиваемый порог, а фиксированный перевод единиц. Более
старые версии (с ratio-фильтром, без CLI) — см. legacy/README.md.

Формула (Дягилев et al. 2023, Терско-Каспийский прогиб, формула 5б):
    ML = lg(A_nm) + b_log·lg(R) + b_lin·R + c + S
A   = смещение грунта в нм (remove_response output='DISP' × 1e9).
R   = S-P формула: R = Vp·Vs/(Vp-Vs) · ΔT(S-P) для каждой станции.
S   = станционная поправка (0 по умолчанию).

Запуск:
    python core/ml_filter_v5.py --info
    python core/ml_filter_v5.py --ml-threshold 1.0
    python core/ml_filter_v5.py --ml-threshold 1.0 --sta-corrections sta_corrections_dyagilev2023.csv
    python core/ml_filter_v5.py --rebuild-cache
    python core/ml_filter_v5.py --ml-outlier-sigma 2.0 --n-min-sta 5
"""

import sys

# Форсируем UTF-8 на stdout/stderr — в help-строках есть не-ASCII (σ, кириллица);
# без этого argparse.print_help() падает с UnicodeEncodeError в консоли по
# умолчанию (cp1251) на Windows.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import argparse
import copy
import csv
import gc
import glob
import math
import os
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

warnings.filterwarnings('ignore', message='.*norm_resp.*')
warnings.filterwarnings('ignore', message='.*computed and reported sensitivities.*')
warnings.filterwarnings('ignore', message='.*StationXML file has version.*')

import numpy as np

# ── Пути ──────────────────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ASSOC_IN  = os.path.join(
    _ROOT, 'workspace', 'associator', 'output', 'associations.xml')
DEFAULT_ASSOC_OUT = None   # рядом с assoc-in: associations_ml<thr>.xml
DEFAULT_WAVEFORMS = os.path.join(_ROOT, 'workspace', 'data_processors', 'output', 'geofiles')
DEFAULT_CACHE_AMP = os.path.join(_ROOT, 'workspace', 'magnitude', 'output', 'amps_filter_cache.csv')
DEFAULT_METADATA  = os.path.join(_ROOT, 'workspace', 'data_processors', 'input', 'metadata')
DEFAULT_STATION_JSON = os.path.join(_ROOT, 'workspace', 'data_processors', 'output', 'json2',
                                    'station_list_2.json')
DEFAULT_HYPOCENTERS = os.path.join(_ROOT, 'workspace', 'locator', 'output', 'hypocenters.csv')
DEFAULT_THRESHOLD = 1.0

# ── Пространства имён QuakeML ──────────────────────────────────────────────────

BED_NS   = 'http://quakeml.org/xmlns/bed/1.2'
QUAKE_NS = 'http://quakeml.org/xmlns/quakeml/1.2'
ET.register_namespace('',  BED_NS)
ET.register_namespace('q', QUAKE_NS)

# ── Константы по умолчанию (все переопределяются через CLI, см. main()) ──────

ML_B_LOG      = 1.024     # коэф. геометрического расхождения
ML_B_LIN      = 0.001648  # коэф. неупругого затухания, 1/км
ML_C          = -1.889    # калибровочная константа

AMP_SCALE_NM  = 1e9       # м → нм; фиксированный перевод единиц, не CLI-параметр
A_MAX_NM      = 1e6       # физический предел ~1 мм; выше = сбой удаления ответа
A_NM_MIN      = 0.005     # мин. амплитуда нм: ниже — артефакт remove_response (не сигнал)

N_MIN_STA        = 4      # минимум станций для оценки ML события
ML_OUTLIER_SIGMA = 1.5    # множитель σ для _filter_by_sigma_v2

VP          = 6.0
VS          = 3.4883
WIN_SEC     = 5.0
R_MIN_KM    = 20.0    # формула 5б ненадёжна ближе 20 км
R_MAX_KM    = 600.0   # формула 5б калибрована до ~200-300 км

# Полоса фильтра, соответствующая рабочей полосе WA-прибора.
# Дягилев et al. 2023 калибровали формулу по амплитудам на WA-симулированных
# записях (T=0.2–0.6 с → f≈1.7–5 Гц). Без этого фильтра микросейсмический
# шум 0.5–1 Гц завышает max(|DISP|) и смещает ML на 0.5–1.0 ед. вверх.
BP_FREQMIN  = 1.0
BP_FREQMAX  = 5.0

READ_MARGIN_SEC    = 60.0    # контекст по краям для remove_response
MAX_GROUP_SPAN_SEC = 300.0   # макс. временно́й диапазон одного кластера (сек)
N_WORKERS          = 4       # потоков; каждый работает со своими станциями

COMP_PRIORITY = ('E', 'N', 'Z')   # приоритет компонент при индексации волн


# ── Парсинг времени ───────────────────────────────────────────────────────────

def parse_time(s):
    if not s:
        return None
    s = s.strip().rstrip('Z').replace('T', ' ')
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ── Парсинг XML ───────────────────────────────────────────────────────────────

def parse_xml_full(tree):
    """
    Возвращает list of (ev_elem, pub_id, origin_time, picks_p, picks_s).
    origin_time из <origin>/<time>/<value>; None если отсутствует.
    picks_p/picks_s: {station: datetime}
    """
    ns   = BED_NS
    root = tree.getroot()
    result = []
    for ev in root.findall(f'.//{{{ns}}}event'):
        pub_id = ev.get('publicID', '')

        origin_time = None
        orig = ev.find(f'{{{ns}}}origin')
        if orig is not None:
            t_el = orig.find(f'{{{ns}}}time/{{{ns}}}value')
            if t_el is not None:
                origin_time = parse_time(t_el.text)

        picks_p, picks_s = {}, {}
        for pick in ev.findall(f'{{{ns}}}pick'):
            wf = pick.find(f'{{{ns}}}waveformID')
            if wf is None:
                continue
            sta = wf.get('stationCode', '').strip()
            if not sta:
                continue
            t_el     = pick.find(f'{{{ns}}}time/{{{ns}}}value')
            phase_el = pick.find(f'{{{ns}}}phaseHint')
            if t_el is None:
                continue
            t = parse_time(t_el.text)
            if t is None:
                continue
            phase = phase_el.text.strip() if phase_el is not None else 'P'
            if phase == 'P':
                picks_p[sta] = t
            elif phase == 'S':
                picks_s[sta] = t

        result.append((ev, pub_id, origin_time, picks_p, picks_s))
    return result


# ── S-P расстояние ────────────────────────────────────────────────────────────

def sp_distance_km(p_time, s_time, vp, vs):
    dt = (s_time - p_time).total_seconds()
    if dt <= 0.0:
        return None
    return vp * vs / (vp - vs) * dt


def compute_r_dict(picks_p, picks_s, vp, vs, r_max_km):
    r_dict = {}
    for sta in picks_p:
        if sta not in picks_s:
            continue
        r = sp_distance_km(picks_p[sta], picks_s[sta], vp, vs)
        if r is not None and r > 0.0 and r <= r_max_km:
            r_dict[sta] = r
    return r_dict


# ── R из реального гипоцентра (раздел 1.7d плана, --distance-source locator) ──
# Opt-in альтернатива compute_r_dict() — не замена (Rule 13). По умолчанию
# (--distance-source sp) поведение не меняется вообще, эти функции не вызываются.
# Независимая копия той же логики, что в core/validate_associator_v2.py (см.
# докстринг _r_dict_from_locator() там) — эти два модуля не импортируют друг друга
# (Rule 14 про намеренное дублирование между независимыми потребителями).

_FDSN_NS = {'f': 'http://www.fdsn.org/xml/station/1'}


def load_station_coords(metadata_dir):
    """station -> (lat, lon) из StationXML. Лёгкий ElementTree-парсинг, независимый
    от ObsPy Inventory, который этот модуль грузит только при промахе кэша амплитуд
    (main()) — координаты нужны всегда при --distance-source locator, вне
    зависимости от того, попал ли кэш амплитуд."""
    coords = {}
    if not os.path.isdir(metadata_dir):
        return coords
    for fname in os.listdir(metadata_dir):
        if not fname.endswith('.xml'):
            continue
        parts = fname.split('_')
        if len(parts) < 2:
            continue
        sta = parts[1].strip()
        try:
            tree = ET.parse(os.path.join(metadata_dir, fname))
        except ET.ParseError:
            continue
        root   = tree.getroot()
        lat_el = root.find('.//f:Station/f:Latitude', _FDSN_NS)
        lon_el = root.find('.//f:Station/f:Longitude', _FDSN_NS)
        if lat_el is not None and lon_el is not None:
            coords[sta] = (float(lat_el.text), float(lon_el.text))
    return coords


def load_hypocenters(path):
    """event_id (publicID) -> строка core/locator.py::CSV_FIELDS, числовые поля
    приведены к float/bool. Формат идентичен load_hypocenters() в
    core/export_bul.py/core/validate_associator_v2.py."""
    hyp = {}
    if not path or not os.path.isfile(path):
        return hyp
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            event_id = row.get('event_id', '').strip()
            if not event_id:
                continue
            row['converged'] = row.get('converged', '').strip() == 'True'
            for key in ('lat', 'lon', 'depth_km', 'depth_uncertainty_km', 'rms'):
                val = (row.get(key) or '').strip()
                row[key] = float(val) if val else None
            hyp[event_id] = row
    return hyp


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def compute_r_dict_from_locator(picks_p, picks_s, station_coords, hyp_row,
                                r_max_km, metric='hypocentral'):
    """
    Аналог compute_r_dict(), но R — из реального гипоцентра, не из S-P времени.
    Вызывающий код обязан сам проверить hyp_row['converged'] перед вызовом.

    metric='hypocentral' (default): R = sqrt(эпицентральная² + depth_km²) — та же
    физическая величина, что фактически даёт compute_r_dict() (S-P время зависит от
    полной длины луча, не только от горизонтального расстояния) — так ML между
    --distance-source sp/locator остаётся сравнимым. metric='epicentral' — только
    горизонтальная дистанция, без глубины (см. подробное обоснование в
    core/validate_associator_v2.py::_r_dict_from_locator()).
    """
    r_dict = {}
    lat0, lon0, depth = hyp_row.get('lat'), hyp_row.get('lon'), hyp_row.get('depth_km')
    if lat0 is None or lon0 is None:
        return r_dict
    for sta in picks_p:
        if sta not in picks_s:
            continue
        coords = station_coords.get(sta)
        if coords is None:
            continue
        epi = _haversine_km(lat0, lon0, coords[0], coords[1])
        r = math.sqrt(epi ** 2 + depth ** 2) if (metric == 'hypocentral' and depth is not None) else epi
        if 0.0 < r <= r_max_km:
            r_dict[sta] = r
    return r_dict


# ── Кэш амплитуд (ключ = pub_id, sta) ────────────────────────────────────────

def save_amp_cache(amplitudes, path):
    """amplitudes: {(pub_id, sta): float|None}"""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['pub_id', 'sta', 'A'])
        for (pub_id, sta), A in amplitudes.items():
            w.writerow([pub_id, sta, '' if A is None else A])


def load_amp_cache(path):
    """Возвращает {(pub_id, sta): float|None}"""
    cache = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            A = float(row['A']) if row['A'] else None
            cache[(row['pub_id'], row['sta'])] = A
    return cache


# ── Индекс форм волн ──────────────────────────────────────────────────────────

def _parse_file_times(fname):
    parts = fname.split('__')
    if len(parts) != 3:
        return None, None
    try:
        return (datetime.strptime(parts[1].rstrip('Z'), '%Y%m%dT%H%M%S'),
                datetime.strptime(parts[2].rstrip('Z'), '%Y%m%dT%H%M%S'))
    except ValueError:
        return None, None


def build_waveform_index(waveform_dir, comp_priority=COMP_PRIORITY):
    """dict: station -> sorted list of (t_start, t_end, fpath). Приоритет по comp_priority."""
    index = {}
    if not os.path.isdir(waveform_dir):
        return index
    for sta in os.listdir(waveform_dir):
        sta_path = os.path.join(waveform_dir, sta)
        if not os.path.isdir(sta_path):
            continue
        by_comp = defaultdict(list)
        for fname in os.listdir(sta_path):
            parts = fname.split('__')
            if len(parts) != 3:
                continue
            dot_parts = parts[0].split('.')
            if len(dot_parts) < 4:
                continue
            comp = dot_parts[3][-1].upper()
            if comp not in comp_priority:
                continue
            t_start, t_end = _parse_file_times(fname)
            if t_start is None:
                continue
            by_comp[comp].append((t_start, t_end, os.path.join(sta_path, fname)))
        for comp in comp_priority:
            if comp in by_comp:
                index[sta] = sorted(by_comp[comp], key=lambda x: x[0])
                break
    return index


def find_waveform_file(index, sta, t):
    for t_start, t_end, fpath in index.get(sta, []):
        if t_start <= t < t_end:
            return fpath
    return None


# ── Параллельная обработка по станциям ───────────────────────────────────────

def _split_triples_by_time(triples, max_span_sec):
    """
    Разбивает [(pub_id, sta, s_time), ...] на кластеры, где первый и последний
    s_time в кластере отличаются не более чем на max_span_sec.
    """
    if not triples:
        return []
    srt = sorted(triples, key=lambda x: x[2])
    clusters, cur = [], [srt[0]]
    for item in srt[1:]:
        if (item[2] - cur[0][2]).total_seconds() > max_span_sec:
            clusters.append(cur)
            cur = []
        cur.append(item)
    clusters.append(cur)
    return clusters


def _process_file_cluster(fpath, triples, inventory, bandpass=True,
                          win_sec=WIN_SEC, read_margin_sec=READ_MARGIN_SEC,
                          bp_freqmin=BP_FREQMIN, bp_freqmax=BP_FREQMAX):
    """
    Читает временно́е окно из MSEED, снимает ответ, возвращает
    {(pub_id, sta): float|None} в метрах.
    """
    from obspy import read as obspy_read, UTCDateTime
    result = {}
    try:
        s_utcs  = [UTCDateTime(s_time) for _, _, s_time in triples]
        t_start = min(s_utcs) - read_margin_sec
        t_end   = max(s_utcs) + win_sec + read_margin_sec

        st = obspy_read(fpath, starttime=t_start, endtime=t_end)
        if not st:
            return result
        tr = st[0]
        actual_sr = tr.stats.sampling_rate
        if actual_sr not in (50.0, 100.0, 80.0, 200.0):
            actual_sr = 100.0 if actual_sr > 75 else 50.0
            tr.stats.sampling_rate = actual_sr

        if inventory is not None:
            nyq = actual_sr / 2.0
            pre_filt = (0.5, 1.0, nyq * 0.85, nyq * 0.95)
            try:
                tr.detrend('linear')
                tr.taper(max_percentage=0.05, type='cosine')
                tr.remove_response(inventory=inventory, output='DISP',
                                   pre_filt=pre_filt, water_level=60)
                if bandpass:
                    tr.filter('bandpass', freqmin=bp_freqmin, freqmax=bp_freqmax,
                              corners=4, zerophase=True)
            except Exception:
                for pub_id, sta, _ in triples:
                    result[(pub_id, sta)] = None
                return result
        else:
            for pub_id, sta, _ in triples:
                result[(pub_id, sta)] = None
            return result

        for pub_id, sta, s_time in triples:
            try:
                s_utc  = UTCDateTime(s_time)
                tr_sig = tr.slice(s_utc, s_utc + win_sec)
                if tr_sig is None or len(tr_sig.data) == 0:
                    result[(pub_id, sta)] = None
                    continue
                A = float(np.max(np.abs(tr_sig.data)))
                result[(pub_id, sta)] = A if A > 0.0 else None
            except Exception:
                result[(pub_id, sta)] = None
        del tr, st
    except Exception:
        pass
    return result


def _worker_for_stations(station_list, all_events, waveform_index,
                         inventory, bandpass=True, win_sec=WIN_SEC,
                         read_margin_sec=READ_MARGIN_SEC,
                         max_group_span_sec=MAX_GROUP_SPAN_SEC,
                         bp_freqmin=BP_FREQMIN, bp_freqmax=BP_FREQMAX):
    """
    Обрабатывает амплитуды только для станций из station_list.
    all_events: list of (pub_id, picks_s, r_dict)
    Возвращает {(pub_id, sta): float|None}.
    """
    groups = defaultdict(list)
    for pub_id, picks_s, r_dict in all_events:
        for sta in station_list:
            if sta not in r_dict:
                continue
            s_time = picks_s.get(sta)
            if s_time is None:
                continue
            fpath = find_waveform_file(waveform_index, sta, s_time)
            if fpath is None:
                continue
            groups[fpath].append((pub_id, sta, s_time))

    result = {}
    for fpath, triples in groups.items():
        for cluster in _split_triples_by_time(triples, max_group_span_sec):
            result.update(_process_file_cluster(
                fpath, cluster, inventory, bandpass,
                win_sec=win_sec, read_margin_sec=read_margin_sec,
                bp_freqmin=bp_freqmin, bp_freqmax=bp_freqmax))
    gc.collect()
    return result


def extract_amplitudes(all_events, waveform_index, inventory=None,
                       bandpass=True, n_workers=N_WORKERS, win_sec=WIN_SEC,
                       read_margin_sec=READ_MARGIN_SEC,
                       max_group_span_sec=MAX_GROUP_SPAN_SEC,
                       bp_freqmin=BP_FREQMIN, bp_freqmax=BP_FREQMAX):
    """
    all_events: list of (pub_id, picks_p, picks_s, r_dict)

    Партиционирует станции на n_workers групп. Каждый воркер работает
    только со своими станциями — нет конкуренции за MSEED-файлы.
    Оконное чтение: obspy_read(fpath, starttime=..., endtime=...).

    Возвращает {(pub_id, sta): float|None} в метрах.
    """
    all_stations = sorted({
        sta
        for _, _, _, r_dict in all_events
        for sta in r_dict
    })
    if not all_stations:
        return {}

    # Разбиваем станции по воркерам round-robin
    station_groups = [all_stations[i::n_workers] for i in range(n_workers)]
    station_groups = [g for g in station_groups if g]

    # Передаём воркерам только нужные поля
    events_slim = [(pub_id, picks_s, r_dict)
                   for pub_id, _, picks_s, r_dict in all_events]

    total_groups = len(station_groups)
    print(f"  Станций: {len(all_stations)}, воркеров: {len(station_groups)}")
    for i, g in enumerate(station_groups):
        print(f"    воркер {i}: {g}")

    amplitudes = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futs = {
            executor.submit(
                _worker_for_stations, grp, events_slim,
                waveform_index, inventory, bandpass,
                win_sec, read_margin_sec, max_group_span_sec,
                bp_freqmin, bp_freqmax,
            ): i
            for i, grp in enumerate(station_groups)
        }
        for done, fut in enumerate(as_completed(futs), 1):
            amplitudes.update(fut.result())
            print(f"  Готово воркеров: {done}/{total_groups}...",
                  end='\r', flush=True)

    gc.collect()
    print(f"  Воркеров: {total_groups}/{total_groups} — готово.          ")
    return amplitudes


# ── Вычисление ML по формуле 5б ──────────────────────────────────────────────

def _ml_entries(pub_id, r_dict, amplitudes, sta_corr,
                amp_scale, r_min, a_nm_min, a_max_nm,
                ml_b_log, ml_b_lin, ml_c, exclude_stations=None):
    excl = set(exclude_stations) if exclude_stations else set()
    entries = []
    for sta, r_km in r_dict.items():
        if sta in excl:
            continue
        if r_km < r_min:
            continue
        A = amplitudes.get((pub_id, sta))
        if A is None or A <= 0.0:
            continue
        A_nm = A * amp_scale
        if A_nm <= 0.0 or A_nm < a_nm_min or A_nm > a_max_nm:
            continue
        S = sta_corr.get(sta, 0.0)
        ml_s = (math.log10(A_nm)
                + ml_b_log * math.log10(r_km)
                + ml_b_lin * r_km
                + ml_c + S)
        entries.append((sta, r_km, A_nm, ml_s))
    return entries


def _filter_by_sigma_v2(entries, n_min, outlier_sigma):
    """
    Медиана → удалить |MLi - Mmed| > outlier_sigma · σ → оставшиеся.
    Если после фильтрации остаётся < n_min станций — возвращает исходный список.
    """
    if len(entries) < n_min:
        return entries
    ml_vals = np.array([e[3] for e in entries])
    Mmed    = float(np.median(ml_vals))
    sigma   = float(np.std(ml_vals))
    if sigma == 0.0:
        return entries
    thr     = outlier_sigma * sigma
    inliers = [e for e in entries if abs(e[3] - Mmed) <= thr]
    if len(inliers) < n_min:
        return entries
    return inliers


def compute_event_ml_v2(pub_id, r_dict, amplitudes, sta_corr,
                        amp_scale=AMP_SCALE_NM, r_min=R_MIN_KM,
                        a_nm_min=A_NM_MIN, a_max_nm=A_MAX_NM,
                        ml_b_log=ML_B_LOG, ml_b_lin=ML_B_LIN, ml_c=ML_C,
                        n_min_sta=N_MIN_STA, outlier_sigma=ML_OUTLIER_SIGMA,
                        exclude_stations=None):
    """
    Вычисляет ML события: медиана → удаление выбросов (outlier_sigma·σ) → среднее.
    Возвращает (ml_mean, n_stations).
    """
    entries = _ml_entries(pub_id, r_dict, amplitudes, sta_corr,
                          amp_scale, r_min, a_nm_min, a_max_nm,
                          ml_b_log, ml_b_lin, ml_c, exclude_stations)
    if len(entries) < n_min_sta:
        return None, 0
    entries = _filter_by_sigma_v2(entries, n_min_sta, outlier_sigma)
    if not entries:
        return None, 0
    return float(np.mean([e[3] for e in entries])), len(entries)


# ── Дальние станции и ML по согласованным станциям (2026-10-07) ──────────────
# Ассоциатор подмешивает в событие пики чужих станций: у них S-P мало, R по S-P в разы
# меньше истинного, ML станции сильно занижена и тянет ML события вниз — реальные
# события M3-4.5 по бюллетеню получали ML < 1.5 и удалялись фильтром. Разбор и проверка
# против бюллетеня ГС РАН (май 2024, 2025 Q1/Q2) — context/locsat-plan.md, раздел 8.

REF_LAT           = 43.6   # опорная точка для --max-station-dist-km: Сочи,
REF_LON           = 40.0   # как по умолчанию в core/export_bul.py
CONSISTENCY_SIGMA = 1.0    # = validate_associator_v2 --sigma по умолчанию
CONSISTENCY_R_MIN = 20.0   # = validate_associator_v2 --r-min по умолчанию


def load_station_coords_json(path):
    """station -> (lat, lon) из сводного JSON станций (data_processors, формат
    {"STA": {"network", "channels", "coords": [lat, lon, elev]}})."""
    import json
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    coords = {}
    for sta, info in data.items():
        c = info.get('coords') if isinstance(info, dict) else None
        if c and len(c) >= 2:
            coords[sta.strip()] = (float(c[0]), float(c[1]))
    return coords


def far_stations(stations, station_coords, max_km, ref_lat=REF_LAT, ref_lon=REF_LON):
    """(далёкие, без_координат): станции дальше max_km от (ref_lat, ref_lon) и станции
    без известных координат (их расстояние неизвестно — они НЕ отсекаются)."""
    far, unknown = set(), set()
    for sta in stations:
        c = station_coords.get(sta)
        if c is None:
            unknown.add(sta)
        elif _haversine_km(ref_lat, ref_lon, c[0], c[1]) > max_km:
            far.add(sta)
    return far, unknown


def consistent_stations(picks_p, picks_s, vp, vs, sigma_mult=CONSISTENCY_SIGMA,
                        r_min=CONSISTENCY_R_MIN, exclude_stations=None):
    """
    Станции события, чьё время очага T0 по S-P согласуется с остальными — σ-фильтр
    validate_associator_v2 (estimate_origin_times + sigma_filter; импорт, не копия —
    тот же отбор, что у входа LOCSAT, validate_associator_v2.py --out-xml-inliers).
    exclude_stations убираются ДО фильтра — иначе чужие станции могут оказаться
    «согласованным большинством». Пустое множество — оценки T0 нет.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import validate_associator_v2 as v2
    excl = set(exclude_stations) if exclude_stations else set()
    picks = {sta: {'p': t, 's': picks_s[sta]} for sta, t in picks_p.items()
             if sta not in excl and sta in picks_s}
    raw = v2.estimate_origin_times(picks, vp, vs, r_min, None)
    if not raw:
        return set()
    inliers, _ = v2.sigma_filter(raw, sigma_mult)
    return {e['station'] for e in inliers}


def compute_event_ml_v3(pub_id, r_dict, amplitudes, sta_corr, consistent=None, **kwargs):
    """
    Как compute_event_ml_v2(), но если передано consistent (множество станций) — ML
    считается только по ним; если так ML не получается (меньше n_min_sta станций с
    амплитудой) — откат на ML по всем станциям события.
    Возвращает (ml, n_stations, used_consistent).
    """
    if consistent:
        r_sub = {s: r for s, r in r_dict.items() if s in consistent}
        ml, n = compute_event_ml_v2(pub_id, r_sub, amplitudes, sta_corr, **kwargs)
        if ml is not None:
            return ml, n, True
    ml, n = compute_event_ml_v2(pub_id, r_dict, amplitudes, sta_corr, **kwargs)
    return ml, n, False


# ── Фильтрация XML ────────────────────────────────────────────────────────────

def filter_xml(tree, ml_by_pubid, threshold, out_path, keep_no_ml=False):
    """
    Удаляет из XML события с ML < threshold.
    keep_no_ml=False: события без ML тоже удаляются.
    """
    tree2     = copy.deepcopy(tree)
    ns        = BED_NS
    root2     = tree2.getroot()
    ev_params = root2.find(f'.//{{{ns}}}eventParameters') or root2
    events2   = ev_params.findall(f'{{{ns}}}event')

    total = len(events2)
    kept = removed = no_ml_kept = no_ml_removed = 0
    for ev_elem in events2:
        pub_id = ev_elem.get('publicID', '')
        ml = ml_by_pubid.get(pub_id)
        if ml is None:
            if keep_no_ml:
                no_ml_kept += 1
                kept += 1
            else:
                ev_params.remove(ev_elem)
                no_ml_removed += 1
                removed += 1
        elif ml >= threshold:
            kept += 1
        else:
            ev_params.remove(ev_elem)
            removed += 1

    tree2.write(out_path, encoding='unicode', xml_declaration=True)
    return total, kept, removed, no_ml_kept, no_ml_removed


# ── Диагностика ───────────────────────────────────────────────────────────────

def print_diag_event(pub_id, r_dict, amplitudes, sta_corr,
                     amp_scale=AMP_SCALE_NM, r_min=R_MIN_KM,
                     a_nm_min=A_NM_MIN, a_max_nm=A_MAX_NM,
                     ml_b_log=ML_B_LOG, ml_b_lin=ML_B_LIN, ml_c=ML_C,
                     n_min_sta=N_MIN_STA, outlier_sigma=ML_OUTLIER_SIGMA,
                     exclude_stations=None):
    raw_entries = _ml_entries(pub_id, r_dict, amplitudes, sta_corr,
                              amp_scale, r_min, a_nm_min, a_max_nm,
                              ml_b_log, ml_b_lin, ml_c, exclude_stations)
    filt_entries = _filter_by_sigma_v2(raw_entries, n_min_sta, outlier_sigma)
    ml_final = (float(np.mean([e[3] for e in filt_entries]))
                if filt_entries else None)
    method_str = f"медиана→σ-фильтр→среднее  (σ-множитель={outlier_sigma})"

    used_stas = {e[0] for e in filt_entries}
    ml_str = f"{ml_final:.2f}" if ml_final is not None else "нет ML"

    # Параметры всех сырых записей (для отображения статуса выброса)
    if raw_entries:
        ml_vals_all = np.array([e[3] for e in raw_entries])
        _Mmed_all   = float(np.median(ml_vals_all))
        _sigma_all  = float(np.std(ml_vals_all))
    else:
        _Mmed_all = _sigma_all = None

    print(f"\n  pub_id={pub_id}  ML={ml_str}  [{method_str}]")
    if not r_dict:
        print("    нет S-P пар")
        return

    print(f"  {'Станция':8s}  {'R,км':>7s}  {'A,нм':>12s}  {'ML_sta':>7s}  {'Статус'}")
    print("  " + "-" * 62)
    for sta in sorted(r_dict):
        r_km = r_dict[sta]
        A    = amplitudes.get((pub_id, sta))
        if r_km < r_min:
            A_nm_s = f"{A * amp_scale:.2f}" if (A and A > 0) else "—"
            print(f"  {sta:8s}  {r_km:7.1f}  {A_nm_s:>12s}  {'—':>7s}  R<{r_min:.0f}км")
        elif A is None:
            print(f"  {sta:8s}  {r_km:7.1f}  {'—':>12s}  {'—':>7s}  нет ответа")
        elif A <= 0.0:
            print(f"  {sta:8s}  {r_km:7.1f}  {'0':>12s}  {'—':>7s}  A=0")
        else:
            A_nm = A * amp_scale
            if A_nm > a_max_nm:
                print(f"  {sta:8s}  {r_km:7.1f}  {A_nm:12.2e}  {'—':>7s}  A>A_MAX")
            else:
                S    = sta_corr.get(sta, 0.0)
                ml_s = (math.log10(A_nm)
                        + ml_b_log * math.log10(r_km)
                        + ml_b_lin * r_km
                        + ml_c + S)
                if sta in used_stas:
                    status = "OK"
                elif _sigma_all is not None:
                    dev = abs(ml_s - _Mmed_all)
                    thr = outlier_sigma * _sigma_all
                    status = f"|δ|={dev:.2f}>{outlier_sigma}σ={thr:.2f}"
                else:
                    status = "—"
                print(f"  {sta:8s}  {r_km:7.1f}  {A_nm:12.2f}  {ml_s:7.2f}  {status}")


# ── Статистика ────────────────────────────────────────────────────────────────

def print_distribution(ml_values, ml_by_pubid=None, n_extreme=3,
                       ml_b_log=ML_B_LOG, ml_b_lin=ML_B_LIN, ml_c=ML_C):
    vals   = sorted(v for v in ml_values if v is not None)
    n_none = sum(1 for v in ml_values if v is None)
    print(f"\n  Событий с оценкой ML:  {len(vals)}")
    print(f"  Событий без оценки ML: {n_none}  (удаляются при фильтрации)")
    if not vals:
        return
    n    = len(vals)
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Распределение ML (b_log={ml_b_log}, b_lin={ml_b_lin}, c={ml_c}):")
    for p in pcts:
        print(f"    P{p:2d}: {vals[max(0, int(p / 100 * n) - 1)]:.2f}")
    print(f"    min: {vals[0]:.2f}   max: {vals[-1]:.2f}")

    if ml_by_pubid is not None:
        ranked = sorted(
            ((ml, pub_id) for pub_id, ml in ml_by_pubid.items() if ml is not None),
            key=lambda x: x[0]
        )
        print(f"\n  Наименьшие ML (top-{n_extreme}):")
        for ml, pub_id in ranked[:n_extreme]:
            print(f"    {ml:6.2f}  {pub_id}")
        print(f"  Наибольшие ML (top-{n_extreme}):")
        for ml, pub_id in ranked[-n_extreme:][::-1]:
            print(f"    {ml:6.2f}  {pub_id}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Фильтрация associations.xml по ML v5 (все события, станционные воркеры, '
                     'все константы переопределяемы через CLI)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- пути --
    parser.add_argument('--assoc-in',       default=DEFAULT_ASSOC_IN,
                        help='Входной XML ассоциатора')
    parser.add_argument('--assoc-out',      default=DEFAULT_ASSOC_OUT,
                        help='Выходной XML (по умолчанию рядом с assoc-in)')
    parser.add_argument('--waveforms',      default=DEFAULT_WAVEFORMS,
                        help='Папка с формами волн')
    parser.add_argument('--cache-amp',      default=DEFAULT_CACHE_AMP,
                        help='CSV-кэш амплитуд (ключ: pub_id, sta)')
    parser.add_argument('--metadata-dir',   default=DEFAULT_METADATA,
                        help='Папка с FDSNStationXML для remove_response')

    # -- формула ML --
    parser.add_argument('--ml-b-log',       type=float, default=ML_B_LOG,
                        help='Коэффициент геометрического расхождения (b_log в lg(R))')
    parser.add_argument('--ml-b-lin',       type=float, default=ML_B_LIN,
                        help='Коэффициент неупругого затухания, 1/км (b_lin в R)')
    parser.add_argument('--ml-c',           type=float, default=ML_C,
                        help='Калибровочная константа формулы')
    parser.add_argument('--sta-corrections', default=None,
                        help='CSV станционных поправок: station,S')

    # -- S-P расстояние --
    parser.add_argument('--vp',             type=float, default=VP,
                        help='Скорость P-волны, км/с')
    parser.add_argument('--vs',             type=float, default=VS,
                        help='Скорость S-волны, км/с')
    parser.add_argument('--r-min-km',       type=float, default=R_MIN_KM,
                        help='Ближняя граница применимости формулы, км')
    parser.add_argument('--r-max-km',       type=float, default=R_MAX_KM,
                        help='Дальняя граница применимости формулы, км')

    # -- источник R (раздел 1.7d плана) --
    parser.add_argument('--distance-source', default='sp', choices=['sp', 'locator'],
                        help='Источник R для формулы ML: sp (default) — из S-P времени, '
                             'как раньше; locator — реальная дистанция из --hypocenters. '
                             'Событие без сошедшегося решения локатора автоматически '
                             'откатывается на sp (не ошибка)')
    parser.add_argument('--hypocenters',    default=DEFAULT_HYPOCENTERS,
                        help=f'CSV — выход core/locator.py, только при --distance-source '
                             f'locator (default: {DEFAULT_HYPOCENTERS})')
    parser.add_argument('--locator-distance-metric', default='hypocentral',
                        choices=['hypocentral', 'epicentral'],
                        help='hypocentral (default) = sqrt(эпицентральная² + depth²) — та '
                             'же физическая величина, что даёт S-P-метод, сравнимая между '
                             '--distance-source sp/locator; epicentral — только '
                             'горизонтальная дистанция, без глубины')

    # -- измерение амплитуды --
    parser.add_argument('--win-sec',        type=float, default=WIN_SEC,
                        help='Длина окна измерения амплитуды S-волны, сек')
    parser.add_argument('--read-margin-sec', type=float, default=READ_MARGIN_SEC,
                        help='Запас на краях читаемого окна для remove_response/detrend, сек')
    parser.add_argument('--max-group-span-sec', type=float, default=MAX_GROUP_SPAN_SEC,
                        help='Макс. разброс S-времён в одном кластере чтения MSEED, сек')
    parser.add_argument('--no-bandpass',    action='store_true', default=False,
                        help=f'Отключить полосовой фильтр {BP_FREQMIN}–{BP_FREQMAX} Гц '
                             '(по умолчанию фильтр включён — он соответствует рабочей '
                             'полосе WA-прибора, на котором калибровалась формула)')
    parser.add_argument('--bp-freqmin',     type=float, default=BP_FREQMIN,
                        help='Нижняя частота полосового фильтра, Гц')
    parser.add_argument('--bp-freqmax',     type=float, default=BP_FREQMAX,
                        help='Верхняя частота полосового фильтра, Гц')
    parser.add_argument('--comp-priority',  default=','.join(COMP_PRIORITY),
                        help='Приоритет компонент при индексации волн (через запятую)')

    # -- физические отсечки амплитуды --
    parser.add_argument('--a-nm-min',       type=float, default=A_NM_MIN,
                        help='Минимальная физическая амплитуда, нм')
    parser.add_argument('--a-max-nm',       type=float, default=A_MAX_NM,
                        help='Максимальная физическая амплитуда, нм')

    # -- агрегация по станциям --
    parser.add_argument('--n-min-sta',      type=int, default=N_MIN_STA,
                        help='Минимум станций для оценки ML события')
    parser.add_argument('--ml-outlier-sigma', type=float, default=ML_OUTLIER_SIGMA,
                        help='Множитель σ для отбраковки станций-выбросов')
    parser.add_argument('--exclude-stations', default=None,
                        help='Исключить станции из расчёта ML (через запятую, например KRNR,LSNR)')
    parser.add_argument('--ml-stations', choices=['all', 'consistent'], default='all',
                        help='all (default) — ML по всем станциям события; consistent — только по '
                             'станциям с согласованным T0 (σ-фильтр validate_associator_v2), при '
                             'нехватке станций — откат на all для этого события')
    parser.add_argument('--consistency-sigma', type=float, default=CONSISTENCY_SIGMA,
                        help=f'С --ml-stations consistent: множитель σ T0-фильтра (default: {CONSISTENCY_SIGMA})')
    parser.add_argument('--max-station-dist-km', type=float, default=None,
                        help='Исключить станции дальше этого расстояния от --ref-lat/--ref-lon из '
                             'расчёта ML и из T0-фильтра (default: выкл.; для кавказской сети — 800)')
    parser.add_argument('--ref-lat', type=float, default=REF_LAT,
                        help=f'Опорная точка для --max-station-dist-km, широта (default: {REF_LAT}, Сочи)')
    parser.add_argument('--ref-lon', type=float, default=REF_LON,
                        help=f'Опорная точка для --max-station-dist-km, долгота (default: {REF_LON}, Сочи)')
    parser.add_argument('--station-coords-json', default=DEFAULT_STATION_JSON,
                        help='Сводный JSON станций с координатами (coords: [lat, lon, elev]) для '
                             '--max-station-dist-km; недостающие берутся из --metadata-dir '
                             f'(default: {DEFAULT_STATION_JSON})')

    # -- фильтрация каталога --
    parser.add_argument('--ml-threshold',   type=float, default=DEFAULT_THRESHOLD,
                        help='Порог ML: оставить события с ML >= X')
    parser.add_argument('--keep-no-ml',     action='store_true', default=False,
                        help='Сохранять события без ML (по умолчанию удаляются)')

    # -- режимы запуска --
    parser.add_argument('--info',           action='store_true',
                        help='Только статистика ML, без записи файла')
    parser.add_argument('--rebuild-cache',  action='store_true',
                        help='Пересобрать кэш амплитуд')
    parser.add_argument('--workers',        type=int, default=N_WORKERS,
                        help='Число потоков; каждый работает со своими станциями')
    parser.add_argument('--diag-pubid',     default=None,
                        help='pub_id события для подробной диагностики по станциям')
    args = parser.parse_args()

    comp_priority = tuple(c.strip().upper() for c in args.comp_priority.split(',') if c.strip())
    k = args.vp * args.vs / (args.vp - args.vs)

    print(f"Входной XML:   {args.assoc_in}")
    print(f"Кэш амплитуд:  {args.cache_amp}")
    print(f"S-P формула:   R = {k:.3f} · ΔT(S-P) км")
    print(f"Воркеров:      {args.workers} (каждый — своя группа станций)\n")

    # ── Парсинг XML ───────────────────────────────────────────────────────────
    print("Парсинг XML...")
    tree        = ET.parse(args.assoc_in)
    events_full = parse_xml_full(tree)
    print(f"  Событий в XML: {len(events_full)}")

    # ── Источник R (раздел 1.7d плана: opt-in, default 'sp' не меняет поведение) ──
    hypocenters, station_coords = {}, {}
    if args.distance_source == 'locator':
        hypocenters    = load_hypocenters(args.hypocenters)
        station_coords = load_station_coords(args.metadata_dir)
        print(f"\nR для ML: locator ({args.locator_distance_metric})  "
              f"гипоцентров={len(hypocenters)}  координат станций={len(station_coords)}  "
              f"[фолбэк на S-P для событий без сошедшегося решения]  ({args.hypocenters})")

    # ── Вычисление R ──────────────────────────────────────────────────────────
    print("\nВычисление R...")
    all_events = []   # (pub_id, picks_p, picks_s, r_dict)
    n_has_r = n_from_locator = n_fallback_sp = 0
    for _, pub_id, _, picks_p, picks_s in events_full:
        hyp_row = hypocenters.get(pub_id) if args.distance_source == 'locator' else None
        if hyp_row and hyp_row.get('converged'):
            r_dict = compute_r_dict_from_locator(
                picks_p, picks_s, station_coords, hyp_row, args.r_max_km,
                args.locator_distance_metric)
            n_from_locator += 1
        else:
            r_dict = compute_r_dict(picks_p, picks_s, args.vp, args.vs, args.r_max_km)
            if args.distance_source == 'locator':
                n_fallback_sp += 1
        all_events.append((pub_id, picks_p, picks_s, r_dict))
        if r_dict:
            n_has_r += 1
    print(f"  Событий с R: {n_has_r} / {len(all_events)}")
    if args.distance_source == 'locator':
        print(f"  R из локатора: {n_from_locator}  fallback на S-P: {n_fallback_sp}")

    all_r = [r for _, _, _, rd in all_events for r in rd.values()]
    if all_r:
        print(f"  R: min={min(all_r):.1f} км, медиана={float(np.median(all_r)):.1f} км, "
              f"max={max(all_r):.1f} км")

    # ── Станционные поправки ──────────────────────────────────────────────────
    sta_corr = {}
    if args.sta_corrections:
        with open(args.sta_corrections, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                sta = row.get('station', '').strip()
                val = row.get('S', row.get('correction', '')).strip()
                if sta and val:
                    try:
                        sta_corr[sta] = float(val)
                    except ValueError:
                        pass
        print(f"  Станционных поправок: {len(sta_corr)}")

    # ── Кэш амплитуд ─────────────────────────────────────────────────────────
    if not args.rebuild_cache and os.path.isfile(args.cache_amp):
        print(f"\nЗагрузка кэша амплитуд: {args.cache_amp}")
        amplitudes = load_amp_cache(args.cache_amp)
        n_amp = sum(1 for v in amplitudes.values() if v is not None)
        print(f"  Загружено пар: {len(amplitudes)}  с амплитудой: {n_amp}")
    else:
        if args.rebuild_cache:
            print("\nПересборка кэша (--rebuild-cache).")

        # Загрузка StationXML
        inventory = None
        meta_dir  = args.metadata_dir
        if meta_dir and not os.path.isdir(meta_dir):
            meta_dir = os.path.join(_ROOT, meta_dir)
        if meta_dir and os.path.isdir(meta_dir):
            from obspy import read_inventory
            print(f"\nЗагрузка StationXML из {meta_dir}...")
            for fpath in sorted(glob.glob(os.path.join(meta_dir, '*.xml'))):
                try:
                    inv = read_inventory(fpath)
                    inventory = inv if inventory is None else inventory + inv
                except Exception:
                    pass
            if inventory is None:
                print("  Предупреждение: StationXML не найдены — remove_response пропущен")
            else:
                n_sta = sum(len(net) for net in inventory.networks)
                print(f"  Загружено: {n_sta} станций")

        # Индекс форм волн
        print(f"\nПостроение индекса форм волн...")
        waveform_index = build_waveform_index(args.waveforms, comp_priority)
        n_files = sum(len(v) for v in waveform_index.values())
        print(f"  Файлов: {n_files}  станций: {len(waveform_index)}")

        # Извлечение амплитуд
        use_bp = not args.no_bandpass
        print(f"\nИзвлечение амплитуд S-волны "
              f"(окно S..S+{args.win_sec:.0f}s, горизонталь {'>'.join(comp_priority)}, "
              f"БПФ {args.bp_freqmin}–{args.bp_freqmax} Гц: {'да' if use_bp else 'нет'})...")
        amplitudes = extract_amplitudes(
            all_events, waveform_index,
            inventory=inventory,
            bandpass=use_bp,
            n_workers=args.workers,
            win_sec=args.win_sec,
            read_margin_sec=args.read_margin_sec,
            max_group_span_sec=args.max_group_span_sec,
            bp_freqmin=args.bp_freqmin,
            bp_freqmax=args.bp_freqmax,
        )
        n_amp = sum(1 for v in amplitudes.values() if v is not None)
        print(f"  Пар с амплитудой: {n_amp}")

        save_amp_cache(amplitudes, args.cache_amp)
        print(f"  Кэш сохранён: {args.cache_amp}")

    # ── Вычисление ML ─────────────────────────────────────────────────────────
    excl = [s.strip() for s in args.exclude_stations.split(',') if s.strip()] \
           if args.exclude_stations else None
    if args.max_station_dist_km:
        coords = load_station_coords(args.metadata_dir)
        coords.update(load_station_coords_json(args.station_coords_json))
        all_sta = {s for _, pp, ps, _ in all_events for s in set(pp) | set(ps)}
        far, unknown = far_stations(all_sta, coords, args.max_station_dist_km,
                                    args.ref_lat, args.ref_lon)
        print(f"  Дальше {args.max_station_dist_km:.0f} км от ({args.ref_lat}, {args.ref_lon}): "
              f"{len(far)} станций {sorted(far)}")
        if unknown:
            print(f"  ВНИМАНИЕ: нет координат у {len(unknown)} станций — не отсекаются, "
                  f"добавьте их в {args.station_coords_json}: {sorted(unknown)}")
        excl = sorted(set(excl or []) | far)
    if excl:
        print(f"  Исключены станции: {excl}")

    ml_kwargs = dict(r_min=args.r_min_km, a_nm_min=args.a_nm_min, a_max_nm=args.a_max_nm,
                     ml_b_log=args.ml_b_log, ml_b_lin=args.ml_b_lin, ml_c=args.ml_c,
                     n_min_sta=args.n_min_sta, outlier_sigma=args.ml_outlier_sigma,
                     exclude_stations=excl)
    print(f"\nВычисление ML (медиана→{args.ml_outlier_sigma}σ→среднее, станции: "
          f"{args.ml_stations})...")
    ml_by_pubid = {}
    n_consistent = 0
    for pub_id, picks_p, picks_s, r_dict in all_events:
        if args.ml_stations == 'consistent':
            cons = consistent_stations(picks_p, picks_s, args.vp, args.vs,
                                       args.consistency_sigma, CONSISTENCY_R_MIN, excl)
            ml, _, used = compute_event_ml_v3(pub_id, r_dict, amplitudes, sta_corr,
                                              consistent=cons, **ml_kwargs)
            n_consistent += used
        else:
            ml, _ = compute_event_ml_v2(pub_id, r_dict, amplitudes, sta_corr, **ml_kwargs)
        ml_by_pubid[pub_id] = ml
    if args.ml_stations == 'consistent':
        print(f"  ML по согласованным станциям: {n_consistent}, откат на все станции: "
              f"{len(all_events) - n_consistent}")

    print_distribution(list(ml_by_pubid.values()), ml_by_pubid=ml_by_pubid,
                       ml_b_log=args.ml_b_log, ml_b_lin=args.ml_b_lin, ml_c=args.ml_c)

    if args.diag_pubid:
        ev_dict = {pub_id: (picks_p, picks_s, r_dict)
                   for pub_id, picks_p, picks_s, r_dict in all_events}
        if args.diag_pubid in ev_dict:
            _, _, r_dict = ev_dict[args.diag_pubid]
            print(f"\n{'=' * 70}")
            print("ДИАГНОСТИКА по pub_id:")
            print('=' * 70)
            print_diag_event(
                args.diag_pubid, r_dict, amplitudes, sta_corr,
                r_min=args.r_min_km, a_nm_min=args.a_nm_min, a_max_nm=args.a_max_nm,
                ml_b_log=args.ml_b_log, ml_b_lin=args.ml_b_lin, ml_c=args.ml_c,
                n_min_sta=args.n_min_sta, outlier_sigma=args.ml_outlier_sigma,
                exclude_stations=excl,
            )
            print('=' * 70)
        else:
            print(f"  pub_id не найден: {args.diag_pubid}")

    if args.info:
        return

    # ── Фильтрация и запись XML ───────────────────────────────────────────────
    thr = args.ml_threshold
    if args.assoc_out:
        out_path = args.assoc_out
    else:
        tag      = f"{thr:.1f}".replace('-', 'm').replace('.', 'p')
        out_path = os.path.join(os.path.dirname(args.assoc_in),
                                f'associations_ml{tag}.xml')

    total, kept, removed, no_ml_kept, no_ml_removed = filter_xml(
        tree, ml_by_pubid, thr, out_path, keep_no_ml=args.keep_no_ml
    )

    print(f"\n{'=' * 60}")
    print(f"Порог ML >= {thr}")
    print(f"  Входных событий:  {total}")
    kept_str = f"ML >= {thr}"
    if no_ml_kept:
        kept_str += f"  +  {no_ml_kept} без оценки"
    print(f"  Сохранено:        {kept}  ({kept_str})")
    removed_str = str(removed)
    if no_ml_removed:
        removed_str += f"  (из них {no_ml_removed} без оценки ML)"
    print(f"  Удалено:          {removed_str}")
    print(f"  Выходной файл:    {out_path}")


if __name__ == '__main__':
    main()
