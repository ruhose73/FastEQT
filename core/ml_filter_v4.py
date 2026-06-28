"""
ml_filter_v4.py — фильтрация associations.xml по локальной магнитуде ML.

Основана на логике v3 (формула, ratio-filter, оконное чтение MSEED),
но обрабатывает ВСЕ события из XML, а не выборку.

Отличия от v3:
  - Нет выборки (SAMPLE_CATALOG_EVENTS, build_sample, --n-between, --seed)
  - Нет раздела recall — используй validate_associator.py (--validate ниже)
  - Записывает отфильтрованный XML (как v2)
  - --validate вызывает validate_associator.py с --min-mag = ml-threshold
  - N_WORKERS: каждый воркер работает со своим набором станций (нет конкуренции за файлы)
  - Кэш по ключу pub_id (не ev_idx)

Формула (Дягилев et al. 2023, Терско-Каспийский прогиб, формула 5б):
    ML = lg(A_nm) + 1.024·lg(R) + 0.001648·R − 1.889 + S
A   = смещение грунта в нм (remove_response output='DISP' × 1e9).
R   = S-P формула: R = Vp·Vs/(Vp-Vs) · ΔT(S-P) для каждой станции.
S   = станционная поправка (0 по умолчанию).

Запуск:
    python core/ml_filter_v4.py --info
    python core/ml_filter_v4.py --ml-threshold 1.0 --validate --year 2024 --month 3
    python core/ml_filter_v4.py --ml-threshold 1.0 --sta-corrections sta_corrections_dyagilev2023.csv --validate
    python core/ml_filter_v4.py --rebuild-cache
"""

import argparse
import copy
import csv
import gc
import glob
import math
import os
import subprocess
import sys
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

warnings.filterwarnings('ignore', message='.*norm_resp.*')
warnings.filterwarnings('ignore', message='.*computed and reported sensitivities.*')

import numpy as np

# ── Пути ──────────────────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ASSOC_IN  = os.path.join(
    _ROOT, 'data-in-memory', 'gpu_splimit_45_may_v2', 'assoc_output_lim_weight_v2','associations.xml')
DEFAULT_ASSOC_OUT = None   # рядом с assoc-in: associations_ml<thr>.xml
DEFAULT_WAVEFORMS = os.path.join(_ROOT, 'geofiles')
DEFAULT_CACHE_AMP = os.path.join(_ROOT, 'amps_wa_v3_weight_ml15.csv')
DEFAULT_METADATA  = os.path.join(_ROOT, 'metadata')
DEFAULT_THRESHOLD = 1.0

# ── Пространства имён QuakeML ──────────────────────────────────────────────────

BED_NS   = 'http://quakeml.org/xmlns/bed/1.2'
QUAKE_NS = 'http://quakeml.org/xmlns/quakeml/1.2'
ET.register_namespace('',  BED_NS)
ET.register_namespace('q', QUAKE_NS)

# ── Константы ML ──────────────────────────────────────────────────────────────

ML_B_LOG      = 1.024
ML_B_LIN      = 0.001648
ML_C          = -1.889
AMP_SCALE_NM  = 1e9
A_MAX_NM      = 1e6    # физический предел ~1 мм; выше = сбой удаления ответа
A_NM_MIN      = 0.005   # мин. амплитуда нм: ниже — артефакт remove_response (не сигнал)
AMP_RATIO_MIN    = 0.1    # отношение actual/expected; ниже — неверная калибровка
N_ITER_FILTER    = 1      # итераций фильтрации по amplitude ratio
N_MIN_STA        = 4      # минимум станций после фильтрации
ML_OUTLIER_SIGMA = 1.5    # множитель σ для _filter_by_sigma_v2

VP          = 6.0
VS          = 3.4883
WIN_SEC     = 5.0
R_MIN_KM    = 20.0    # формула 5б ненадёжна ближе 30 км
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

_COMP_PRIORITY = ('E', 'N', 'Z')


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

def sp_distance_km(p_time, s_time):
    dt = (s_time - p_time).total_seconds()
    if dt <= 0.0:
        return None
    return VP * VS / (VP - VS) * dt


def compute_r_dict(picks_p, picks_s):
    r_dict = {}
    for sta in picks_p:
        if sta not in picks_s:
            continue
        r = sp_distance_km(picks_p[sta], picks_s[sta])
        if r is not None and r > 0.0 and r <= R_MAX_KM:
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


def build_waveform_index(waveform_dir):
    """dict: station -> sorted list of (t_start, t_end, fpath). Приоритет E > N > Z."""
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
            if comp not in _COMP_PRIORITY:
                continue
            t_start, t_end = _parse_file_times(fname)
            if t_start is None:
                continue
            by_comp[comp].append((t_start, t_end, os.path.join(sta_path, fname)))
        for comp in _COMP_PRIORITY:
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


def _process_file_cluster(fpath, triples, inventory, bandpass=True):
    """
    Читает временно́е окно из MSEED, снимает ответ, возвращает
    {(pub_id, sta): float|None} в метрах.
    """
    from obspy import read as obspy_read, UTCDateTime
    result = {}
    try:
        s_utcs  = [UTCDateTime(s_time) for _, _, s_time in triples]
        t_start = min(s_utcs) - READ_MARGIN_SEC
        t_end   = max(s_utcs) + WIN_SEC + READ_MARGIN_SEC

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
                    tr.filter('bandpass', freqmin=BP_FREQMIN, freqmax=BP_FREQMAX,
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
                tr_sig = tr.slice(s_utc, s_utc + WIN_SEC)
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
                         inventory, bandpass=True):
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
        for cluster in _split_triples_by_time(triples, MAX_GROUP_SPAN_SEC):
            result.update(_process_file_cluster(fpath, cluster,
                                                inventory, bandpass))
    gc.collect()
    return result


def extract_amplitudes(all_events, waveform_index, inventory=None,
                       bandpass=True, n_workers=N_WORKERS):
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
                waveform_index, inventory, bandpass
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
                amp_scale, r_min, a_max_nm, exclude_stations=None):
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
        if A_nm <= 0.0 or A_nm < A_NM_MIN or A_nm > a_max_nm:
            continue
        S = sta_corr.get(sta, 0.0)
        ml_s = (math.log10(A_nm)
                + ML_B_LOG * math.log10(r_km)
                + ML_B_LIN * r_km
                + ML_C + S)
        entries.append((sta, r_km, A_nm, ml_s))
    return entries


def _filter_by_amp_ratio(entries, sta_corr, n_iter, ratio_min, n_min):
    """
    Итеративно отбрасывает станции с амплитудой < ratio_min × ожидаемой
    при текущем медианном ML.
    """
    cur = entries
    for _ in range(n_iter):
        if len(cur) < n_min:
            break
        ml_med = float(np.median([e[3] for e in cur]))
        survivors = []
        for sta, r_km, A_nm, ml_s in cur:
            S = sta_corr.get(sta, 0.0)
            A_exp = 10 ** (ml_med
                           - ML_B_LOG * math.log10(r_km)
                           - ML_B_LIN * r_km
                           - ML_C - S)
            if A_nm / A_exp >= ratio_min:
                survivors.append((sta, r_km, A_nm, ml_s))
        if len(survivors) < n_min or len(survivors) == len(cur):
            break
        cur = survivors
    return cur


def compute_event_ml(pub_id, r_dict, amplitudes, sta_corr,
                     amp_scale=AMP_SCALE_NM, r_min=R_MIN_KM,
                     a_max_nm=A_MAX_NM):
    entries = _ml_entries(pub_id, r_dict, amplitudes, sta_corr,
                          amp_scale, r_min, a_max_nm)
    entries = _filter_by_amp_ratio(entries, sta_corr,
                                   N_ITER_FILTER, AMP_RATIO_MIN, N_MIN_STA)
    if not entries:
        return None, 0
    return float(np.median([e[3] for e in entries])), len(entries)


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
                        a_max_nm=A_MAX_NM, exclude_stations=None):
    """
    Вычисляет ML события: медиана → удаление выбросов (2σ) → среднее.
    Возвращает (ml_mean, n_stations).
    """
    entries = _ml_entries(pub_id, r_dict, amplitudes, sta_corr,
                          amp_scale, r_min, a_max_nm, exclude_stations)
    if len(entries) < N_MIN_STA:
        return None, 0
    entries = _filter_by_sigma_v2(entries, N_MIN_STA, ML_OUTLIER_SIGMA)
    if not entries:
        return None, 0
    return float(np.mean([e[3] for e in entries])), len(entries)


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
                     amp_scale=AMP_SCALE_NM, use_v2=True, exclude_stations=None):
    raw_entries = _ml_entries(pub_id, r_dict, amplitudes, sta_corr,
                              amp_scale, R_MIN_KM, A_MAX_NM, exclude_stations)
    if use_v2:
        filt_entries = _filter_by_sigma_v2(raw_entries, N_MIN_STA, ML_OUTLIER_SIGMA)
        ml_final = (float(np.mean([e[3] for e in filt_entries]))
                    if filt_entries else None)
        method_str = f"медиана→2σ→среднее  (σ-множитель={ML_OUTLIER_SIGMA})"
    else:
        filt_entries = _filter_by_amp_ratio(raw_entries, sta_corr,
                                            N_ITER_FILTER, AMP_RATIO_MIN, N_MIN_STA)
        ml_final = (float(np.median([e[3] for e in filt_entries]))
                    if filt_entries else None)
        method_str = "ratio-фильтр→медиана"

    used_stas = {e[0] for e in filt_entries}
    ml_str = f"{ml_final:.2f}" if ml_final is not None else "нет ML"

    # Для v2: параметры всех сырых записей (для отображения статуса выброса)
    if use_v2 and raw_entries:
        ml_vals_all = np.array([e[3] for e in raw_entries])
        _Mmed_all   = float(np.median(ml_vals_all))
        _sigma_all  = float(np.std(ml_vals_all))
    else:
        _Mmed_all = _sigma_all = None

    print(f"\n  pub_id={pub_id}  ML={ml_str}  [{method_str}]")
    if not r_dict:
        print("    нет S-P пар")
        return

    def expected_a(r_km, sta):
        if ml_final is None:
            return None
        S = sta_corr.get(sta, 0.0)
        return 10 ** (ml_final
                      - ML_B_LOG * math.log10(r_km)
                      - ML_B_LIN * r_km
                      - ML_C - S)

    print(f"  {'Станция':8s}  {'R,км':>7s}  {'A,нм':>12s}  {'ML_sta':>7s}  {'Статус'}")
    print("  " + "-" * 62)
    for sta in sorted(r_dict):
        r_km = r_dict[sta]
        A    = amplitudes.get((pub_id, sta))
        if r_km < R_MIN_KM:
            A_nm_s = f"{A * amp_scale:.2f}" if (A and A > 0) else "—"
            print(f"  {sta:8s}  {r_km:7.1f}  {A_nm_s:>12s}  {'—':>7s}  R<{R_MIN_KM:.0f}км")
        elif A is None:
            print(f"  {sta:8s}  {r_km:7.1f}  {'—':>12s}  {'—':>7s}  нет ответа")
        elif A <= 0.0:
            print(f"  {sta:8s}  {r_km:7.1f}  {'0':>12s}  {'—':>7s}  A=0")
        else:
            A_nm = A * amp_scale
            if A_nm > A_MAX_NM:
                print(f"  {sta:8s}  {r_km:7.1f}  {A_nm:12.2e}  {'—':>7s}  A>A_MAX")
            else:
                S    = sta_corr.get(sta, 0.0)
                ml_s = (math.log10(A_nm)
                        + ML_B_LOG * math.log10(r_km)
                        + ML_B_LIN * r_km
                        + ML_C + S)
                if sta in used_stas:
                    status = "OK"
                elif use_v2 and _sigma_all is not None:
                    dev = abs(ml_s - _Mmed_all)
                    thr = ML_OUTLIER_SIGMA * _sigma_all
                    status = f"|δ|={dev:.2f}>2σ={thr:.2f}"
                else:
                    exp    = expected_a(r_km, sta)
                    ratio  = A_nm / exp if exp else 0
                    status = f"ratio={ratio:.2f}<{AMP_RATIO_MIN}"
                print(f"  {sta:8s}  {r_km:7.1f}  {A_nm:12.2f}  {ml_s:7.2f}  {status}")


# ── Статистика ────────────────────────────────────────────────────────────────

def print_distribution(ml_values, ml_by_pubid=None, n_extreme=3):
    vals   = sorted(v for v in ml_values if v is not None)
    n_none = sum(1 for v in ml_values if v is None)
    print(f"\n  Событий с оценкой ML:  {len(vals)}")
    print(f"  Событий без оценки ML: {n_none}  (удаляются при фильтрации)")
    if not vals:
        return
    n    = len(vals)
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Распределение ML (b_log={ML_B_LOG}, b_lin={ML_B_LIN}, c={ML_C}):")
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
    global VP, VS, N_WORKERS
    parser = argparse.ArgumentParser(
        description='Фильтрация associations.xml по ML v4 (все события, станционные воркеры)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    parser.add_argument('--no-bandpass',    action='store_true', default=False,
                        help=f'Отключить полосовой фильтр {BP_FREQMIN}–{BP_FREQMAX} Гц '
                             '(по умолчанию фильтр включён — он соответствует рабочей '
                             'полосе WA-прибора, на котором калибровалась формула)')
    parser.add_argument('--sta-corrections', default=None,
                        help='CSV станционных поправок: station,S')
    parser.add_argument('--ml-threshold',   type=float, default=DEFAULT_THRESHOLD,
                        help='Порог ML: оставить события с ML >= X')
    parser.add_argument('--keep-no-ml',     action='store_true', default=False,
                        help='Сохранять события без ML (по умолчанию удаляются)')
    parser.add_argument('--info',           action='store_true',
                        help='Только статистика ML, без записи файла')
    parser.add_argument('--rebuild-cache',  action='store_true',
                        help='Пересобрать кэш амплитуд')
    parser.add_argument('--workers',        type=int, default=N_WORKERS,
                        help='Число потоков; каждый работает со своими станциями')
    parser.add_argument('--vp',             type=float, default=VP)
    parser.add_argument('--vs',             type=float, default=VS)
    parser.add_argument('--validate',       action='store_true',
                        help='Запустить validate_associator.py на выходном XML '
                             '(--min-mag = ml-threshold)')
    parser.add_argument('--year',           type=int, default=None)
    parser.add_argument('--month',          type=int, default=None)
    parser.add_argument('--exclude-stations', default=None,
                        help='Исключить станции из расчёта ML (через запятую, например KRNR,LSNR)')
    parser.add_argument('--diag-pubid',     default=None,
                        help='pub_id события для подробной диагностики по станциям')
    args = parser.parse_args()

    VP       = args.vp
    VS       = args.vs
    N_WORKERS = args.workers
    k = VP * VS / (VP - VS)

    print(f"Входной XML:   {args.assoc_in}")
    print(f"Кэш амплитуд:  {args.cache_amp}")
    print(f"S-P формула:   R = {k:.3f} · ΔT(S-P) км")
    print(f"Воркеров:      {N_WORKERS} (каждый — своя группа станций)\n")

    # ── Парсинг XML ───────────────────────────────────────────────────────────
    print("Парсинг XML...")
    tree        = ET.parse(args.assoc_in)
    events_full = parse_xml_full(tree)
    print(f"  Событий в XML: {len(events_full)}")

    # ── Вычисление R по S-P формуле ──────────────────────────────────────────
    print("\nВычисление R по S-P формуле...")
    all_events = []   # (pub_id, picks_p, picks_s, r_dict)
    n_has_r = 0
    for _, pub_id, _, picks_p, picks_s in events_full:
        r_dict = compute_r_dict(picks_p, picks_s)
        all_events.append((pub_id, picks_p, picks_s, r_dict))
        if r_dict:
            n_has_r += 1
    print(f"  Событий с S-P расстоянием: {n_has_r} / {len(all_events)}")

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
        waveform_index = build_waveform_index(args.waveforms)
        n_files = sum(len(v) for v in waveform_index.values())
        print(f"  Файлов: {n_files}  станций: {len(waveform_index)}")

        # Извлечение амплитуд
        use_bp = not args.no_bandpass
        print(f"\nИзвлечение амплитуд S-волны "
              f"(окно S..S+{WIN_SEC:.0f}s, горизонталь E>N>Z, "
              f"БПФ {BP_FREQMIN}–{BP_FREQMAX} Гц: {'да' if use_bp else 'нет'})...")
        amplitudes = extract_amplitudes(
            all_events, waveform_index,
            inventory=inventory,
            bandpass=use_bp,
            n_workers=N_WORKERS,
        )
        n_amp = sum(1 for v in amplitudes.values() if v is not None)
        print(f"  Пар с амплитудой: {n_amp}")

        save_amp_cache(amplitudes, args.cache_amp)
        print(f"  Кэш сохранён: {args.cache_amp}")

    # ── Вычисление ML ─────────────────────────────────────────────────────────
    excl = [s.strip() for s in args.exclude_stations.split(',') if s.strip()] \
           if args.exclude_stations else None
    if excl:
        print(f"  Исключены станции: {excl}")

    print("\nВычисление ML (медиана→2σ→среднее)...")
    ml_by_pubid = {}
    for pub_id, _, _, r_dict in all_events:
        ml, _ = compute_event_ml_v2(pub_id, r_dict, amplitudes, sta_corr,
                                    exclude_stations=excl)
        ml_by_pubid[pub_id] = ml

    print_distribution(list(ml_by_pubid.values()), ml_by_pubid=ml_by_pubid)

    if args.diag_pubid:
        ev_dict = {pub_id: (picks_p, picks_s, r_dict)
                   for pub_id, picks_p, picks_s, r_dict in all_events}
        if args.diag_pubid in ev_dict:
            _, _, r_dict = ev_dict[args.diag_pubid]
            print(f"\n{'=' * 70}")
            print("ДИАГНОСТИКА по pub_id:")
            print('=' * 70)
            print_diag_event(args.diag_pubid, r_dict, amplitudes, sta_corr,
                             use_v2=True, exclude_stations=excl)
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

    # ── Валидация ─────────────────────────────────────────────────────────────
    if args.validate:
        print(f"\n{'─' * 60}")
        print("Запуск validate_associator.py...")
        script = os.path.join(_ROOT, 'core', 'validate_associator.py')
        cmd = [sys.executable, script, '--assoc', out_path,
               '--min-mag', str(thr)]
        if args.year is not None:
            cmd += ['--year', str(args.year)]
        if args.month is not None:
            cmd += ['--month', str(args.month)]
        subprocess.run(cmd)


if __name__ == '__main__':
    main()
