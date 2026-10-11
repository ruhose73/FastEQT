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


# ── Отбор станций по P-годографу (--ml-stations p-traveltime, 2026-10-11) ────
# Два случая, которые consistent не лечит (разбор май 2024, context/10.10.26.md): в событие
# попадают чужие станции в 400–900 км, а у части своих EQT ставит S на фазу сразу за P
# (S-P 1–2 с на 100–200 км) -> R по S-P в разы меньше, своя станция выпадает по R < 20 км,
# σ-фильтр T0 (сам на S-P) оставляет чужих «согласованным большинством». Здесь «свои»
# станции определяются только по P-временам (годограф P, без S), S лишь подтверждает очаг.
# Метод «вслепую»: пики, координаты станций, годограф — без бюллетеня и места события.

DEFAULT_TT_TABLE = os.path.join(_ROOT, 'core', 'data', 'tt_p_ncau2024.csv')
PTT_TT_MIN_N   = 20                       # бины годографа с меньшим числом пар не используются
PTT_FAR_VEL    = 8.0                      # км/с за последним бином годографа
PTT_P_TOL      = 2.5                      # с: допуск T0 по P для «своих» станций
PTT_MIN_IN     = 4                        # минимум своих станций
PTT_S_TOL      = (8.0, 3.0)               # с: допуск T0 по S при уточнении точки (проход 1, 2)
PTT_N_CONFIRM  = 2                        # минимум своих станций, чьё S-P согласно с точкой
PTT_R_TOL_KM   = 30.0                     # |R_sp − R_точки| <= max(PTT_R_TOL_KM,
PTT_R_TOL_FRAC = 0.3                      #                          PTT_R_TOL_FRAC · R_точки)
PTT_DEPTH_KM   = 15.0                     # фиксированная глубина для R от точки
PTT_GRID       = (39.0, 47.0, 36.0, 50.0, 0.1)   # lat_min, lat_max, lon_min, lon_max, шаг °
PTT_SELECT_S_TOL = 4.0                    # с: locate_ps — допуск S к T0 якоря при выборе узла


def load_tt_table(path, min_n=PTT_TT_MIN_N):
    """CSV core/fit_traveltime.py -> (dist_km, tp_s) по бинам с n >= min_n."""
    tt = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    tt = tt[tt[:, 2] >= min_n]
    return tt[:, 0], tt[:, 1]


class PTravelTimeLocator:
    """
    Перебор эпицентра по сетке для P-пиков одного события. Расстояния станция–узлы сетки
    кэшируются по станции. Станции без координат в локации не участвуют.
    """

    def __init__(self, station_coords, tt_dist, tt_time, grid=PTT_GRID,
                 p_tol=PTT_P_TOL, min_in=PTT_MIN_IN, s_tol=PTT_S_TOL, vpvs=VP / VS,
                 far_vel=PTT_FAR_VEL):
        lat_min, lat_max, lon_min, lon_max, step = grid
        lat = np.arange(lat_min, lat_max + step / 10, step)
        lon = np.arange(lon_min, lon_max + step / 10, step)
        self.glat, self.glon = [a.ravel() for a in np.meshgrid(lat, lon, indexing='ij')]
        self.coords = station_coords
        self.tt_dist, self.tt_time = tt_dist, tt_time
        self.p_tol, self.min_in, self.s_tol = p_tol, min_in, tuple(s_tol)
        self.vpvs, self.far_vel = vpvs, far_vel
        self._dist = {}

    def tp(self, d):
        """tP(d): интерполяция медиан; ближе первого бина — к 0, дальше последнего — far_vel."""
        t = np.interp(d, np.r_[0.0, self.tt_dist], np.r_[0.0, self.tt_time])
        far = d > self.tt_dist[-1]
        t[far] = self.tt_time[-1] + (d[far] - self.tt_dist[-1]) / self.far_vel
        return t

    def dist_grid(self, sta):
        if sta not in self._dist:
            la, lo = map(math.radians, self.coords[sta])
            p1, l1 = np.radians(self.glat), np.radians(self.glon)
            a = np.sin((p1 - la) / 2) ** 2 + np.cos(la) * np.cos(p1) * np.sin((l1 - lo) / 2) ** 2
            self._dist[sta] = 2 * 6371.0 * np.arcsin(np.sqrt(a))
        return self._dist[sta]

    def locate(self, picks_p, picks_s=None):
        """
        -> (свои станции, {sta: эпиц. расстояние от точки, км}, (lat, lon)) или (None, None, None).
        1) В каждом узле T0_i = P_i − tP(d_i); якорь — станция, у которой больше всего станций
           с T0 в ±p_tol от её T0; узел и якорь — максимум своих (при равенстве — минимум суммы
           |T0 − T0_якоря| по своим). Своих < min_in -> None.
        2) При фиксированных своих — узел с минимальной дисперсией T0 по P своих и S своих;
           S-пик берётся, только если его T0 (tS = vpvs·tP) в ±s_tol от среднего T0 по P
           в текущем узле (два прохода, допуск сужается) — S «сразу за P» отпадает сам.
        """
        stas = [s for s in picks_p if s in self.coords]
        if len(stas) < self.min_in:
            return None, None, None
        t_ref = min(picks_p[s] for s in stas)
        D = np.stack([self.dist_grid(s) for s in stas], axis=1)              # G × N
        P = np.array([(picks_p[s] - t_ref).total_seconds() for s in stas])
        T0 = P[None, :] - self.tp(D)                                          # G × N
        close = np.abs(T0[:, :, None] - T0[:, None, :]) <= self.p_tol         # G × N(якорь) × N
        cnt = close.sum(axis=1)                                               # G × N
        best = cnt.max()
        if best < self.min_in:
            return None, None, None
        g_idx, a_idx = np.nonzero(cnt == best)
        res = np.where(close[g_idx, :, a_idx],
                       np.abs(T0[g_idx] - T0[g_idx, a_idx][:, None]), 0).sum(axis=1)
        k = int(np.argmin(res))
        g, a = g_idx[k], a_idx[k]
        mask = close[g, :, a]
        return self._refine(stas, D, T0, mask, picks_s, t_ref)

    def locate_ps(self, picks_p, picks_s=None, s_tol_select=PTT_SELECT_S_TOL):
        """
        Как locate(), но узел и якорь выбираются по сумме: своих по P + своих, у которых S тоже
        согласуется с узлом (T0 по S = S − vpvs·tP(d) в ±s_tol_select от T0 якоря); узлы, где своих
        по P меньше min_in, не рассматриваются. Свои станции — по-прежнему только по P. На
        событиях, где чужая группа P-пиков побеждает по числу, выигрывает группа, подтверждённая S.
        Используется для --out-xml-ptt (вход LOCSAT); ML (compute_event_ml_ptt) — на locate(),
        потому что P+S в ML поднимает выдачу на 17% без прироста recall (май 2024, 2026-10-11).
        """
        stas = [s for s in picks_p if s in self.coords]
        if len(stas) < self.min_in:
            return None, None, None
        t_ref = min(picks_p[s] for s in stas)
        D = np.stack([self.dist_grid(s) for s in stas], axis=1)              # G × N
        P = np.array([(picks_p[s] - t_ref).total_seconds() for s in stas])
        T0 = P[None, :] - self.tp(D)                                          # G × N
        close = np.abs(T0[:, :, None] - T0[:, None, :]) <= self.p_tol         # G × N × N (симметр.)
        cnt = close.sum(axis=1)                                               # G × якорь
        if cnt.max() < self.min_in:
            return None, None, None
        S = np.array([(picks_s[s] - t_ref).total_seconds() if picks_s and s in picks_s else np.nan
                      for s in stas])
        T0S = S[None, :] - self.vpvs * self.tp(D)                             # G × N, nan без S
        s_ok = np.abs(T0S[:, None, :] - T0[:, :, None]) <= s_tol_select       # G × якорь × N
        score = cnt + (close & s_ok).sum(axis=2)
        valid = cnt >= self.min_in
        g_idx, a_idx = np.nonzero(valid & (score == np.where(valid, score, -1).max()))
        res = np.where(close[g_idx, :, a_idx],
                       np.abs(T0[g_idx] - T0[g_idx, a_idx][:, None]), 0).sum(axis=1)
        k = int(np.argmin(res))
        g, a = g_idx[k], a_idx[k]
        mask = close[g, :, a]
        return self._refine(stas, D, T0, mask, picks_s, t_ref)

    def _refine(self, stas, D, T0, mask, picks_s, t_ref):
        """Этап 2 locate()/locate_ps(): при фиксированных своих (mask) — узел с минимальной
        дисперсией T0 по P своих и согласных S своих."""
        inl = {stas[i] for i in range(len(stas)) if mask[i]}
        T0p = T0[:, mask]
        t0s = {i: (picks_s[s] - t_ref).total_seconds() - self.vpvs * self.tp(D[:, i])
               for i, s in enumerate(stas) if mask[i] and picks_s and s in picks_s}
        g = int(np.argmin(T0p.var(axis=1)))
        for s_tol in self.s_tol:
            t0 = T0p[g].mean()
            cols = [T0p] + [v[:, None] for v in t0s.values() if abs(v[g] - t0) <= s_tol]
            g = int(np.argmin(np.hstack(cols).var(axis=1)))
        return inl, {s: float(D[g, i]) for i, s in enumerate(stas)}, (self.glat[g], self.glon[g])


def compute_event_ml_ptt(pub_id, picks_p, picks_s, r_dict, amplitudes, sta_corr, locator,
                         consistent=None, exclude_stations=None, n_confirm=PTT_N_CONFIRM,
                         r_tol_km=PTT_R_TOL_KM, r_tol_frac=PTT_R_TOL_FRAC,
                         depth_km=PTT_DEPTH_KM, r_max_km=R_MAX_KM, **kwargs):
    """
    ML события по станциям, отобранным PTravelTimeLocator (--ml-stations p-traveltime).
    consistent — станции σ-фильтра T0 (consistent_stations): по ним считается ML, если
    локации нет или очаг не подтверждён S — то есть ровно поведение --ml-stations consistent.
    Подтверждение: у >= n_confirm своих станций R по S-P согласно с R от точки
    (|ΔR| <= max(r_tol_km, r_tol_frac·R_точки), R_точки — с глубиной depth_km). Тогда ML —
    только по своим, а расходящийся R_sp заменяется R от точки (если <= r_max_km).
    Возвращает (ml, n_stations, статус): 'confirmed' — ML по своим; 'consistent' — очаг не
    подтверждён S, ML как в compute_event_ml_v3; 'unlocated' — локации нет, то же; 'all' —
    подтверждён, но по своим ML не получилось (< n_min_sta с амплитудой) — ML по всем станциям.
    """
    excl = set(exclude_stations) if exclude_stations else set()
    inl, epi, _ = locator.locate({s: t for s, t in picks_p.items() if s not in excl}, picks_s)
    if inl is None:
        ml, n, _ = compute_event_ml_v3(pub_id, r_dict, amplitudes, sta_corr, consistent=consistent,
                                       exclude_stations=exclude_stations, **kwargs)
        return ml, n, 'unlocated'
    r_new, n_ok = {}, 0
    for sta, r in r_dict.items():
        if sta not in inl:
            continue
        r_pt = math.sqrt(epi[sta] ** 2 + depth_km ** 2)
        if abs(r - r_pt) <= max(r_tol_km, r_tol_frac * r_pt):
            r_new[sta] = r
            n_ok += 1
        elif r_pt <= r_max_km:
            r_new[sta] = r_pt
    if n_ok < n_confirm:
        ml, n, _ = compute_event_ml_v3(pub_id, r_dict, amplitudes, sta_corr, consistent=consistent,
                                       exclude_stations=exclude_stations, **kwargs)
        return ml, n, 'consistent'
    ml, n = compute_event_ml_v2(pub_id, r_new, amplitudes, sta_corr,
                                exclude_stations=exclude_stations, **kwargs)
    if ml is not None:
        return ml, n, 'confirmed'
    ml, n = compute_event_ml_v2(pub_id, r_dict, amplitudes, sta_corr,
                                exclude_stations=exclude_stations, **kwargs)   # откат, как в v3
    return ml, n, 'all'


# ── ML по станциям для записи в XML (--write-ml, 2026-10-11) ─────────────────
# compute_event_ml_v2/v3/ptt возвращают только ML события. Для бюллетеня нужны и ML станций, вошедших
# в среднее. Функции ниже повторяют тот же выбор станций и R (Rule 13 — старые не меняются); main()
# сверяет их ML события с основной и считает расхождения (должно быть 0).

def _ml_v2_detail(pub_id, r_dict, amplitudes, sta_corr,
                  amp_scale=AMP_SCALE_NM, r_min=R_MIN_KM,
                  a_nm_min=A_NM_MIN, a_max_nm=A_MAX_NM,
                  ml_b_log=ML_B_LOG, ml_b_lin=ML_B_LIN, ml_c=ML_C,
                  n_min_sta=N_MIN_STA, outlier_sigma=ML_OUTLIER_SIGMA,
                  exclude_stations=None):
    """Как compute_event_ml_v2(), но возвращает (ml, n, {станция: ML станции}) по станциям, вошедшим в среднее."""
    entries = _ml_entries(pub_id, r_dict, amplitudes, sta_corr,
                          amp_scale, r_min, a_nm_min, a_max_nm,
                          ml_b_log, ml_b_lin, ml_c, exclude_stations)
    if len(entries) < n_min_sta:
        return None, 0, {}
    entries = _filter_by_sigma_v2(entries, n_min_sta, outlier_sigma)
    if not entries:
        return None, 0, {}
    return float(np.mean([e[3] for e in entries])), len(entries), {e[0]: e[3] for e in entries}


def compute_event_ml_detail(pub_id, picks_p, picks_s, r_dict, amplitudes, sta_corr, mode,
                            consistent=None, locator=None, exclude_stations=None,
                            n_confirm=PTT_N_CONFIRM, r_tol_km=PTT_R_TOL_KM,
                            r_tol_frac=PTT_R_TOL_FRAC, depth_km=PTT_DEPTH_KM,
                            r_max_km=R_MAX_KM, **kwargs):
    """
    (ml, n, {станция: ML}) — тот же выбор станций и R, что compute_event_ml_v2 (mode 'all'),
    compute_event_ml_v3 ('consistent') и compute_event_ml_ptt ('p-traveltime').
    """
    kw = dict(kwargs, exclude_stations=exclude_stations)

    def v3(cons):
        if cons:
            res = _ml_v2_detail(pub_id, {s: r for s, r in r_dict.items() if s in cons},
                                amplitudes, sta_corr, **kw)
            if res[0] is not None:
                return res
        return _ml_v2_detail(pub_id, r_dict, amplitudes, sta_corr, **kw)

    if mode == 'all':
        return _ml_v2_detail(pub_id, r_dict, amplitudes, sta_corr, **kw)
    if mode == 'consistent':
        return v3(consistent)
    excl = set(exclude_stations) if exclude_stations else set()
    inl, epi, _ = locator.locate({s: t for s, t in picks_p.items() if s not in excl}, picks_s)
    if inl is None:
        return v3(consistent)
    r_new, n_ok = {}, 0
    for sta, r in r_dict.items():
        if sta not in inl:
            continue
        r_pt = math.sqrt(epi[sta] ** 2 + depth_km ** 2)
        if abs(r - r_pt) <= max(r_tol_km, r_tol_frac * r_pt):
            r_new[sta] = r
            n_ok += 1
        elif r_pt <= r_max_km:
            r_new[sta] = r_pt
    if n_ok < n_confirm:
        return v3(consistent)
    res = _ml_v2_detail(pub_id, r_new, amplitudes, sta_corr, **kw)
    if res[0] is not None:
        return res
    return _ml_v2_detail(pub_id, r_dict, amplitudes, sta_corr, **kw)


def write_ml_to_xml(out_path, ml_detail):
    """
    --write-ml: в выходной XML для каждого события из ml_detail ({pub_id: (ml, n, {sta: ML})})
    добавить <magnitude> (ML события, type ML, stationCount) и <stationMagnitude> на каждую станцию,
    вошедшую в среднее и оставшуюся в событии (после --drop-far-from-origin-km). Вставляются перед
    первым <pick>. core/export_bul.py --ml-from-xml пишет их в бюллетень вместо собственного пересчёта.
    """
    tree = ET.parse(out_path)
    ns = BED_NS
    T = lambda n: f'{{{ns}}}{n}'
    n_ev = n_sta = 0
    for ev in tree.getroot().iter(T('event')):
        pub_id = ev.get('publicID', '')
        det = ml_detail.get(pub_id)
        if not det or det[0] is None:
            continue
        ml, n, per_sta = det
        nets = {}
        for pk in ev.findall(T('pick')):
            wf = pk.find(T('waveformID'))
            if wf is not None:
                nets.setdefault(wf.get('stationCode', '').strip(), wf.get('networkCode', ''))
        children = list(ev)
        pos = next((i for i, c in enumerate(children) if c.tag == T('pick')), len(children))
        new = []
        mag = ET.Element(T('magnitude'), publicID=f'{pub_id}/ml')
        ET.SubElement(ET.SubElement(mag, T('mag')), T('value')).text = f'{ml:.3f}'
        ET.SubElement(mag, T('type')).text = 'ML'
        ET.SubElement(mag, T('stationCount')).text = str(n)
        ET.SubElement(mag, T('methodID')).text = 'smi:local/ml_filter_v5'
        new.append(mag)
        for sta in sorted(per_sta):
            if sta not in nets:
                continue
            sm = ET.Element(T('stationMagnitude'), publicID=f'{pub_id}/ml/{sta}')
            ET.SubElement(ET.SubElement(sm, T('mag')), T('value')).text = f'{per_sta[sta]:.3f}'
            ET.SubElement(sm, T('type')).text = 'ML'
            ET.SubElement(sm, T('waveformID'), networkCode=nets[sta], stationCode=sta)
            new.append(sm)
            n_sta += 1
        for i, el in enumerate(new):
            ev.insert(pos + i, el)
        n_ev += 1
    tree.write(out_path, encoding='unicode', xml_declaration=True)
    print(f"  ML записана в XML: {n_ev} событий, {n_sta} станционных магнитуд ({out_path})")


def write_quakeml_ptt(tree, events, keep_ids, locator, out_path, exclude_stations=None,
                      vp=VP, vs=VS, r_tol_km=PTT_R_TOL_KM, r_tol_frac=PTT_R_TOL_FRAC,
                      depth_km=PTT_DEPTH_KM, select='ps', s_tol_select=PTT_SELECT_S_TOL):
    """
    --out-xml-ptt (2026-10-11): вход для core/locator.py вместо validate_associator_v2.py
    --out-xml-inliers. Для событий из keep_ids (прошедших порог ML), которые локализуются
    PTravelTimeLocator: P всех своих станций + S только тех своих, чьё R по S-P согласно с
    точкой перебора (тот же критерий, что подтверждение в compute_event_ml_ptt; S «сразу за P»
    не подаётся). Начальное приближение origin — точка перебора и T0 по P своих. Тот же
    publicID события — hypocenters.csv стыкуется с export_bul.py. Не локализованные события не
    пишутся (export_bul выводит их без координат). events — parse_xml_full(tree).
    select='ps' (default) — узел выбирается PTravelTimeLocator.locate_ps() (P + согласные S),
    'p' — locate(), как в ML. Май/апрель 2024: события бюллетеня с координатами 36 → 41 из 42,
    14 → 17 из 18 (раздел --out-xml-ptt в ml_filter_v5.md).
    """
    from datetime import timedelta
    excl = set(exclude_stations) if exclude_stations else set()
    nets = {}
    for wf in tree.getroot().iter(f'{{{BED_NS}}}waveformID'):
        nets.setdefault(wf.get('stationCode', '').strip(), wf.get('networkCode', ''))
    T = lambda n: f'{{{BED_NS}}}{n}'
    fmt = '%Y-%m-%dT%H:%M:%S.%fZ'
    root = ET.Element(f'{{{QUAKE_NS}}}quakeml')
    ep = ET.SubElement(root, T('eventParameters'), publicID='smi:local/ptt-inliers')
    n_in = n_w = n_p = n_s = n_s_drop = 0
    for _, pub_id, _, picks_p, picks_s in events:
        if pub_id not in keep_ids:
            continue
        n_in += 1
        pp = {s: t for s, t in picks_p.items() if s not in excl}
        if select == 'ps':
            inl, epi, pt = locator.locate_ps(pp, picks_s, s_tol_select)
        else:
            inl, epi, pt = locator.locate(pp, picks_s)
        if inl is None:
            continue
        t_ref = min(pp[s] for s in pp if s in locator.coords)
        stas = sorted(inl)
        t0s = (np.array([(pp[s] - t_ref).total_seconds() for s in stas])
               - locator.tp(np.array([epi[s] for s in stas])))
        t0 = t_ref + timedelta(seconds=float(t0s.mean()))

        ev = ET.SubElement(ep, T('event'), publicID=pub_id)
        orig = ET.SubElement(ev, T('origin'), publicID=f'{pub_id}/ptt')
        ET.SubElement(ET.SubElement(orig, T('time')), T('value')).text = t0.strftime(fmt)
        ET.SubElement(ET.SubElement(orig, T('latitude')), T('value')).text = f'{pt[0]:.3f}'
        ET.SubElement(ET.SubElement(orig, T('longitude')), T('value')).text = f'{pt[1]:.3f}'
        for s in stas:
            phases = [('P', pp[s])]
            if s in picks_s:
                r_sp = sp_distance_km(pp[s], picks_s[s], vp, vs)
                r_pt = math.sqrt(epi[s] ** 2 + depth_km ** 2)
                if r_sp is not None and abs(r_sp - r_pt) <= max(r_tol_km, r_tol_frac * r_pt):
                    phases.append(('S', picks_s[s]))
                else:
                    n_s_drop += 1
            for ph, t in phases:
                pk = ET.SubElement(ev, T('pick'), publicID=f'{pub_id}/{s}/{ph}')
                ET.SubElement(ET.SubElement(pk, T('time')), T('value')).text = t.strftime(fmt)
                ET.SubElement(pk, T('waveformID'), networkCode=nets.get(s, ''), stationCode=s)
                ET.SubElement(pk, T('phaseHint')).text = ph
                n_p += ph == 'P'
                n_s += ph == 'S'
        n_w += 1
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(out_path, encoding='utf-8', xml_declaration=True)
    print(f"QuakeML для locator.py (свои станции по P-годографу) записан: {out_path}  "
          f"({n_w} из {n_in} событий; пиков P {n_p}, S {n_s}, S отброшено как несогласные "
          f"с точкой {n_s_drop})")


def drop_far_picks(out_path, events, locator, max_km, exclude_stations=None,
                   select='ps', s_tol_select=PTT_SELECT_S_TOL):
    """
    --drop-far-from-origin-km (2026-10-11): в уже записанном выходном XML убрать пики (P и S) станций
    дальше max_km от очага события, найденного перебором (та же точка, что в --out-xml-ptt: select
    'ps' — locate_ps, 'p' — locate). Ассоциатор добавляет к событию далёкие станции, не относящиеся к
    нему (~43% станций событий не совпадают с бюллетенем ГС РАН) — они попадают в бюллетень. События
    без очага и станции без координат не меняются. ML уже посчитана и не пересчитывается.
    Май/апрель 2024, 2025 Q1 при 400 км: recall тот же, станций из бюллетеня у найденных событий
    73/61/71% → 87/75/85%, export_bul пишет на 10–25% событий меньше (державшихся на дальних станциях).
    events — parse_xml_full() исходного дерева (пики до фильтрации).
    """
    excl = set(exclude_stations) if exclude_stations else set()
    picks_by_id = {pub_id: (pp, ps) for _, pub_id, _, pp, ps in events}
    tree = ET.parse(out_path)
    ns = BED_NS
    n_ev = n_cut = n_rm = 0
    for ev in tree.getroot().iter(f'{{{ns}}}event'):
        pub_id = ev.get('publicID', '')
        if pub_id not in picks_by_id:
            continue
        n_ev += 1
        picks_p, picks_s = picks_by_id[pub_id]
        pp = {s: t for s, t in picks_p.items() if s not in excl}
        if select == 'ps':
            inl, _, pt = locator.locate_ps(pp, picks_s, s_tol_select)
        else:
            inl, _, pt = locator.locate(pp, picks_s)
        if inl is None:
            continue
        rm_ids = set()
        for pick in ev.findall(f'{{{ns}}}pick'):
            wf = pick.find(f'{{{ns}}}waveformID')
            c = locator.coords.get(wf.get('stationCode', '').strip()) if wf is not None else None
            if c is not None and _haversine_km(pt[0], pt[1], c[0], c[1]) > max_km:
                rm_ids.add(pick.get('publicID'))
                ev.remove(pick)
        for orig in ev.findall(f'{{{ns}}}origin'):
            for arr in orig.findall(f'{{{ns}}}arrival'):
                if arr.findtext(f'{{{ns}}}pickID') in rm_ids:
                    orig.remove(arr)
        n_cut += bool(rm_ids)
        n_rm += len(rm_ids)
    tree.write(out_path, encoding='unicode', xml_declaration=True)
    print(f"  Станции дальше {max_km:.0f} км от очага перебора убраны: {n_rm} пиков в {n_cut} из "
          f"{n_ev} событий ({out_path})")


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
    parser.add_argument('--ml-stations', choices=['all', 'consistent', 'p-traveltime'], default='all',
                        help='all (default) — ML по всем станциям события; consistent — только по '
                             'станциям с согласованным T0 (σ-фильтр validate_associator_v2), при '
                             'нехватке станций — откат на all для этого события; p-traveltime — '
                             'станции события отбираются по P-годографу (--tt-table), очаг '
                             'подтверждается S-P; не подтверждён — как consistent')
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

    # -- --ml-stations p-traveltime --
    parser.add_argument('--tt-table', default=DEFAULT_TT_TABLE,
                        help='Годограф первой P: CSV core/fit_traveltime.py (dist_km,tp_s,n,...)')
    parser.add_argument('--tt-min-n', type=int, default=PTT_TT_MIN_N,
                        help='Бины годографа с меньшим числом пар не используются')
    parser.add_argument('--ptt-grid', default=','.join(f'{v:g}' for v in PTT_GRID),
                        help='Сетка перебора эпицентра: lat_min,lat_max,lon_min,lon_max,шаг (°)')
    parser.add_argument('--ptt-p-tol', type=float, default=PTT_P_TOL,
                        help='Допуск T0 по P для «своих» станций, с')
    parser.add_argument('--ptt-min-stations', type=int, default=PTT_MIN_IN,
                        help='Минимум своих станций (меньше — как consistent)')
    parser.add_argument('--ptt-s-tol', default=','.join(f'{v:g}' for v in PTT_S_TOL),
                        help='Допуски T0 по S при уточнении точки, с (проходы через запятую)')
    parser.add_argument('--ptt-n-confirm', type=int, default=PTT_N_CONFIRM,
                        help='Минимум своих станций с S-P, согласным с точкой (меньше — как consistent)')
    parser.add_argument('--ptt-r-tol-km', type=float, default=PTT_R_TOL_KM,
                        help='Согласие R_sp с R от точки: |ΔR| <= max(этого, --ptt-r-tol-frac·R)')
    parser.add_argument('--ptt-r-tol-frac', type=float, default=PTT_R_TOL_FRAC,
                        help='Относительный допуск согласия R_sp с R от точки')
    parser.add_argument('--ptt-depth-km', type=float, default=PTT_DEPTH_KM,
                        help='Фиксированная глубина для R от точки, км')
    parser.add_argument('--out-xml-ptt', default=None,
                        help='Дополнительно записать вход для core/locator.py: события, прошедшие '
                             'порог ML, только свои станции по P-годографу (P + согласные с точкой S), '
                             'начальное приближение — точка перебора. Работает при любом --ml-stations')
    parser.add_argument('--ptt-locsat-select', choices=['ps', 'p'], default='ps',
                        help='С --out-xml-ptt: выбор точки перебора для входа LOCSAT. ps (default) — '
                             'по своим P + согласным S; p — только по P, как в ML')
    parser.add_argument('--ptt-select-s-tol', type=float, default=PTT_SELECT_S_TOL,
                        help='С --ptt-locsat-select ps: допуск S к T0 якоря при выборе узла, с')
    parser.add_argument('--write-ml', action='store_true', default=False,
                        help='Записать в выходной XML ML события (<magnitude>) и ML станций, вошедших в '
                             'среднее (<stationMagnitude>) — для core/export_bul.py --ml-from-xml. '
                             'Default: выкл. (XML как раньше, без магнитуд)')
    parser.add_argument('--drop-far-from-origin-km', type=float, default=None,
                        help='Убрать из выходного XML пики станций дальше этого расстояния от очага '
                             'события, найденного перебором по P-годографу (точка как в --out-xml-ptt; '
                             'события без очага не меняются; ML не пересчитывается). Default: выкл.; '
                             'проверено значение 400 км — recall тот же, бюллетень чище')

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
    ptt_stats = defaultdict(int)
    if args.ml_stations == 'p-traveltime' or args.out_xml_ptt or args.drop_far_from_origin_km:
        ptt_coords = load_station_coords(args.metadata_dir)
        ptt_coords.update(load_station_coords_json(args.station_coords_json))
        tt_dist, tt_time = load_tt_table(args.tt_table, args.tt_min_n)
        locator = PTravelTimeLocator(
            ptt_coords, tt_dist, tt_time,
            grid=tuple(float(v) for v in args.ptt_grid.split(',')),
            p_tol=args.ptt_p_tol, min_in=args.ptt_min_stations,
            s_tol=[float(v) for v in args.ptt_s_tol.split(',')], vpvs=args.vp / args.vs)
        print(f"  Годограф P: {args.tt_table} ({len(tt_dist)} бинов, до {tt_dist[-1]:.0f} км), "
              f"узлов сетки: {len(locator.glat)}, координат станций: {len(ptt_coords)}")
    ml_detail, n_detail_mismatch = {}, 0
    for pub_id, picks_p, picks_s, r_dict in all_events:
        cons = None
        if args.ml_stations == 'p-traveltime':
            cons = consistent_stations(picks_p, picks_s, args.vp, args.vs,
                                       args.consistency_sigma, CONSISTENCY_R_MIN, excl)
            ml, _, status = compute_event_ml_ptt(
                pub_id, picks_p, picks_s, r_dict, amplitudes, sta_corr, locator,
                consistent=cons, n_confirm=args.ptt_n_confirm, r_tol_km=args.ptt_r_tol_km,
                r_tol_frac=args.ptt_r_tol_frac, depth_km=args.ptt_depth_km,
                r_max_km=args.r_max_km, **ml_kwargs)
            ptt_stats[status] += 1
        elif args.ml_stations == 'consistent':
            cons = consistent_stations(picks_p, picks_s, args.vp, args.vs,
                                       args.consistency_sigma, CONSISTENCY_R_MIN, excl)
            ml, _, used = compute_event_ml_v3(pub_id, r_dict, amplitudes, sta_corr,
                                              consistent=cons, **ml_kwargs)
            n_consistent += used
        else:
            ml, _ = compute_event_ml_v2(pub_id, r_dict, amplitudes, sta_corr, **ml_kwargs)
        ml_by_pubid[pub_id] = ml
        if args.write_ml and ml is not None and ml >= args.ml_threshold:
            det = compute_event_ml_detail(
                pub_id, picks_p, picks_s, r_dict, amplitudes, sta_corr, args.ml_stations,
                consistent=cons, locator=locator if args.ml_stations == 'p-traveltime' else None,
                n_confirm=args.ptt_n_confirm, r_tol_km=args.ptt_r_tol_km,
                r_tol_frac=args.ptt_r_tol_frac, depth_km=args.ptt_depth_km,
                r_max_km=args.r_max_km, **ml_kwargs)
            if det[0] is None or abs(det[0] - ml) > 1e-9:
                n_detail_mismatch += 1
            ml_detail[pub_id] = det
    if args.write_ml:
        print(f"  --write-ml: ML по станциям для {len(ml_detail)} событий; расхождений с основной ML: "
              f"{n_detail_mismatch} (должно быть 0)")
    if args.ml_stations == 'consistent':
        print(f"  ML по согласованным станциям: {n_consistent}, откат на все станции: "
              f"{len(all_events) - n_consistent}")
    elif args.ml_stations == 'p-traveltime':
        print(f"  P-годограф: очаг подтверждён S (ML по своим) {ptt_stats['confirmed']}, "
              f"подтверждён, но ML по всем {ptt_stats['all']}, не подтверждён {ptt_stats['consistent']}, "
              f"не локализован {ptt_stats['unlocated']} (последние два — как consistent)")

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

    if args.drop_far_from_origin_km:
        drop_far_picks(out_path, events_full, locator, args.drop_far_from_origin_km,
                       exclude_stations=excl, select=args.ptt_locsat_select,
                       s_tol_select=args.ptt_select_s_tol)

    if args.write_ml:
        write_ml_to_xml(out_path, ml_detail)

    if args.out_xml_ptt:
        keep_ids = {p for p, ml in ml_by_pubid.items()
                    if (ml is not None and ml >= thr) or (ml is None and args.keep_no_ml)}
        write_quakeml_ptt(tree, events_full, keep_ids, locator, args.out_xml_ptt,
                          exclude_stations=excl, vp=args.vp, vs=args.vs,
                          r_tol_km=args.ptt_r_tol_km, r_tol_frac=args.ptt_r_tol_frac,
                          depth_km=args.ptt_depth_km, select=args.ptt_locsat_select,
                          s_tol_select=args.ptt_select_s_tol)


if __name__ == '__main__':
    main()
