"""
sta_correction_estimator.py — вычисление станционных поправок ML.

Алгоритм:
  1. Матчинг каталог (catalog.xlsx) → associations.xml (как в validate_associator.py).
  2. Для каждого совпавшего события читает форму волны, измеряет амплитуду S-волны.
  3. Вычисляет ML_sta_raw = lg(A_нм) + 1.024·lg(R) + 0.001648·R − 1.889 (без поправки).
  4. Станционная поправка события: S_ev = Ms_каталог − ML_sta_raw.
  5. По станции: среднее S_ev по событиям; IQR-детектор выбросов.
  6. Станция включается только если имеет >= MIN_EVENTS совпадений с каталогом.

Выходные файлы:
  --out-detail   CSV с поправкой по каждому (станция, событие)
  --out-summary  CSV в формате sta_corrections_dyagilev2023.csv (станция, среднее)

Запуск:
    python core/sta_correction_estimator.py --info
    python core/sta_correction_estimator.py --year 2024 --month 1
    python core/sta_correction_estimator.py --min-events 10 --year 2024 --month 1
    python core/sta_correction_estimator.py --rebuild-cache --year 2024 --month 1
"""

import argparse
import csv
import gc
import glob
import json
import math
import os
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

warnings.filterwarnings('ignore', message='.*norm_resp.*')
warnings.filterwarnings('ignore', message='.*computed and reported sensitivities.*')

import numpy as np
import openpyxl

# ── Пути ──────────────────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ASSOC_IN   = os.path.join(_ROOT, 'data-in-memory', 'gpu_splimit_45_may', 'assoc_output_lim', 'associations.xml')
DEFAULT_CATALOG    = os.path.join(_ROOT, 'catalog.xlsx')
DEFAULT_WAVEFORMS  = os.path.join(_ROOT, 'geofiles')
DEFAULT_METADATA   = os.path.join(_ROOT, 'metadata')
DEFAULT_STA_DIR    = os.path.join(_ROOT, 'json')
DEFAULT_CACHE_AMP  = os.path.join(_ROOT, 'amps_sta__may_corr.csv')
DEFAULT_OUT_DETAIL  = os.path.join(_ROOT, 'sta_corrections_may_detail.csv')
DEFAULT_OUT_SUMMARY = os.path.join(_ROOT, 'sta_corrections_may_detail.csv')

# ── Константы формулы (Дягилев et al. 2023, формула 5б) ──────────────────────

ML_B_LOG     = 1.024
ML_B_LIN     = 0.001648
ML_C         = -1.889
AMP_SCALE_NM = 1e9
A_MAX_NM     = 1e6
A_NM_MIN     = 0.01    # мин. амплитуда нм: ниже — артефакт remove_response (не сигнал)
R_MIN_KM     = 30.0
R_MAX_KM     = 600.0   # формула 5б калибрована до ~200-300 км; 0 = без ограничения

VP          = 6.0
VS          = 3.4883
WIN_SEC     = 5.0

BP_FREQMIN  = 1.0
BP_FREQMAX  = 5.0

READ_MARGIN_SEC    = 120.0
MAX_GROUP_SPAN_SEC = 300.0
N_WORKERS          = 4

MIN_EVENTS    = 6     # минимум событий каталога на станцию для включения в результат
MATCH_WIN_SEC = 15.0   # окно матчинга каталог→XML после поправки на travel time
IQR_K         = 1.5    # множитель IQR для детектора выбросов
AMP_RATIO_MAX = 5.0    # макс. допустимый ratio A_nm / ожидаемой_A_nm (0 = выключено)
AMP_RATIO_MIN = 0.0    # мин. допустимый ratio A_nm / ожидаемой_A_nm (0 = выключено)
LG_A_NM_MIN   = 0.0   # мин. lg(A_нм): < порога = шум                (0 = выключено)
LG_A_NM_MAX   = 0.0   # макс. lg(A_нм): > порога = насыщение         (0 = выключено)

# Константы для диагностики ML (должны совпадать с ml_filter_v4.py)
_ML_RATIO_MIN  = 0.1
_ML_N_ITER     = 1
_ML_N_MIN_STA  = 3

# Параметры outlier-фильтра для compute_event_ml_detail_v2
ML_OUTLIER_ABS   = 0.3   # абсолютный порог |MLi - Mmed|
ML_OUTLIER_SIGMA = 2.0   # множитель σ: удалять если |MLi - Mmed| > k·σ

_PAZ_WA = {
    'poles': [(-6.283185307 + 4.712388980j),
              (-6.283185307 - 4.712388980j)],
    'zeros': [0j, 0j],
    'gain': 1.0,
    'sensitivity': 2800.0,
}
_COMP_PRIORITY = ('E', 'N', 'Z')

BED_NS = 'http://quakeml.org/xmlns/bed/1.2'

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


# ── Загрузка каталога ────────────────────────────────────────────────────────

def load_catalog(path, year=None, month=None, min_ms=None, max_ms=None):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    events = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue
        if len(row) < 4:
            continue
        origin_time, lat, lon, depth_km = row[0], row[1], row[2], row[3]
        ms = float(row[4]) if len(row) > 4 and row[4] is not None else None
        if origin_time is None or lat is None or lon is None:
            continue
        if hasattr(origin_time, 'tzinfo') and origin_time.tzinfo is not None:
            origin_time = origin_time.replace(tzinfo=None)
        events.append({
            'origin_time': origin_time,
            'lat': float(lat),
            'lon': float(lon),
            'depth_km': float(depth_km) if depth_km is not None else 10.0,
            'ms': ms,
        })
    wb.close()

    if year is not None:
        events = [e for e in events if e['origin_time'].year == year]
    if month is not None:
        events = [e for e in events if e['origin_time'].month == month]
    if min_ms is not None:
        events = [e for e in events
                  if e['ms'] is not None and e['ms'] >= min_ms]
    if max_ms is not None:
        events = [e for e in events
                  if e['ms'] is not None and e['ms'] <= max_ms]
    return events


# ── Загрузка координат станций ───────────────────────────────────────────────

def load_stations(sta_dir):
    stations = {}
    for fpath in glob.glob(os.path.join(sta_dir, 'station_*.json')):
        name = os.path.basename(fpath).replace('station_', '').replace('.json', '')
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)
        key    = list(data.keys())[0]
        coords = data[key]['coords']
        stations[name] = {'lat': coords[0], 'lon': coords[1]}
    return stations


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def nearest_station_tt(ev_lat, ev_lon, depth_km, stations, vp):
    best_name, best_tt = None, None
    for name, st in stations.items():
        epi  = haversine_km(ev_lat, ev_lon, st['lat'], st['lon'])
        hypo = math.sqrt(epi ** 2 + depth_km ** 2)
        tt   = hypo / vp
        if best_tt is None or tt < best_tt:
            best_name, best_tt = name, tt
    return best_name, best_tt


# ── Парсинг XML ───────────────────────────────────────────────────────────────

def parse_xml_full(tree):
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

        result.append((pub_id, origin_time, picks_p, picks_s))
    return result


# ── Матчинг каталог → XML ─────────────────────────────────────────────────────

def match_catalog_to_xml(catalog, xml_events, stations, vp, match_win_sec):
    """
    Для каждого каталожного события ищет ближайшее XML событие.
    expected_assoc_time = catalog_origin + nearest_station_travel_time.
    Совпадение если |assoc_time - expected| <= match_win_sec.

    Возвращает list of dict:
      {cat_ev, xml_pub_id, xml_origin, dt_sec, picks_p, picks_s}
    """
    # Индекс XML событий по времени (только с origin_time)
    timed_xml = [(pub_id, ot, pp, ps)
                 for pub_id, ot, pp, ps in xml_events
                 if ot is not None]
    timed_xml.sort(key=lambda x: x[1])

    matched = []
    for cat_ev in catalog:
        if cat_ev['ms'] is None:
            continue
        _, tt = nearest_station_tt(
            cat_ev['lat'], cat_ev['lon'], cat_ev['depth_km'], stations, vp
        )
        if tt is None:
            continue
        expected = cat_ev['origin_time'] + timedelta(seconds=tt)

        best, best_dt = None, None
        for pub_id, ot, pp, ps in timed_xml:
            dt = abs((ot - expected).total_seconds())
            if dt <= match_win_sec:
                if best_dt is None or dt < best_dt:
                    best, best_dt = (pub_id, ot, pp, ps), dt

        if best is not None:
            pub_id, ot, pp, ps = best
            matched.append({
                'cat_ev':     cat_ev,
                'xml_pub_id': pub_id,
                'xml_origin': ot,
                'dt_sec':     best_dt,
                'picks_p':    pp,
                'picks_s':    ps,
            })

    return matched


# ── S-P расстояние ────────────────────────────────────────────────────────────

def sp_distance_km(p_time, s_time):
    dt = (s_time - p_time).total_seconds()
    if dt <= 0.0:
        return None
    return VP * VS / (VP - VS) * dt


def compute_r_dict(picks_p, picks_s, r_min_km=R_MIN_KM, r_max_km=R_MAX_KM):
    r_dict = {}
    for sta in picks_p:
        if sta not in picks_s:
            continue
        r = sp_distance_km(picks_p[sta], picks_s[sta])
        if r is None or r < r_min_km:
            continue
        if r_max_km and r > r_max_km:
            continue
        r_dict[sta] = r
    return r_dict


def compute_r_dict_coords(cat_ev, picks_s, station_coords,
                          r_min_km=R_MIN_KM, r_max_km=R_MAX_KM):
    """R из координат каталога + станций. Требует только S-пик (для окна амплитуды)."""
    r_dict = {}
    ev_lat = cat_ev['lat']
    ev_lon = cat_ev['lon']
    ev_dep = cat_ev['depth_km']
    for sta in picks_s:
        coords = station_coords.get(sta)
        if coords is None:
            continue
        epi = haversine_km(ev_lat, ev_lon, coords['lat'], coords['lon'])
        r   = math.sqrt(epi ** 2 + ev_dep ** 2)
        if r < r_min_km:
            continue
        if r_max_km and r > r_max_km:
            continue
        r_dict[sta] = r
    return r_dict


# ── Кэш амплитуд ─────────────────────────────────────────────────────────────

def save_amp_cache(amplitudes, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['pub_id', 'sta', 'A'])
        for (pub_id, sta), A in amplitudes.items():
            w.writerow([pub_id, sta, '' if A is None else A])


def load_amp_cache(path):
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


# ── Параллельная обработка амплитуд по станциям ───────────────────────────────

def _split_triples_by_time(triples, max_span_sec):
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


def _process_file_cluster(fpath, triples, inventory, win_sec=WIN_SEC, bandpass=True):
    from obspy import read as obspy_read, UTCDateTime
    result = {}
    try:
        s_utcs  = [UTCDateTime(s_time) for _, _, s_time in triples]
        t_start = min(s_utcs) - READ_MARGIN_SEC
        t_end   = max(s_utcs) + win_sec + READ_MARGIN_SEC

        st = obspy_read(fpath, starttime=t_start, endtime=t_end)
        if not st:
            return result
        # Merge gaps (fill with zeros) so st[0] covers the full time range
        st.merge(method=0, fill_value=0)
        tr = st[0]
        actual_sr = tr.stats.sampling_rate
        if actual_sr not in (50.0, 100.0, 80.0, 200.0):
            actual_sr = 100.0 if actual_sr > 75 else 50.0
            tr.stats.sampling_rate = actual_sr

        if inventory is not None:
            nyq = actual_sr / 2.0
            pre_filt = (0.5, 1.0, nyq * 0.85, nyq * 0.95)
            try:
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

        min_samples = max(1, int(actual_sr * win_sec * 0.5))
        for pub_id, sta, s_time in triples:
            try:
                s_utc  = UTCDateTime(s_time)
                tr_sig = tr.slice(s_utc, s_utc + win_sec)
                if tr_sig is None or len(tr_sig.data) < min_samples:
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


def _worker_for_stations(station_list, events_slim, waveform_index,
                         inventory, win_sec=WIN_SEC, bandpass=True):
    """events_slim: list of (pub_id, picks_s, r_dict)"""
    groups = defaultdict(list)
    for pub_id, picks_s, r_dict in events_slim:
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
            result.update(_process_file_cluster(fpath, cluster, inventory,
                                                win_sec=win_sec, bandpass=bandpass))
    gc.collect()
    return result


def extract_amplitudes(matched_events, waveform_index, inventory=None,
                       n_workers=N_WORKERS, win_sec=WIN_SEC, bandpass=True):
    """
    matched_events: list of {xml_pub_id, picks_p, picks_s} + r_dict
    Возвращает {(pub_id, sta): float|None} в метрах.
    """
    all_stations = sorted({
        sta
        for ev in matched_events
        for sta in ev['r_dict']
    })
    if not all_stations:
        return {}

    station_groups = [all_stations[i::n_workers] for i in range(n_workers)]
    station_groups = [g for g in station_groups if g]

    events_slim = [
        (ev['xml_pub_id'], ev['picks_s'], ev['r_dict'])
        for ev in matched_events
    ]

    print(f"  Станций: {len(all_stations)}, воркеров: {len(station_groups)}")

    amplitudes = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futs = {
            executor.submit(
                _worker_for_stations, grp, events_slim, waveform_index,
                inventory, win_sec, bandpass
            ): i
            for i, grp in enumerate(station_groups)
        }
        total = len(station_groups)
        for done, fut in enumerate(as_completed(futs), 1):
            amplitudes.update(fut.result())
            print(f"  Воркеров: {done}/{total}...", end='\r', flush=True)

    gc.collect()
    print(f"  Воркеров: {total}/{total} — готово.          ")
    return amplitudes


# ── Вычисление поправок ───────────────────────────────────────────────────────

def compute_raw_ml(pub_id, sta, r_km, amplitudes):
    """
    ML_raw = lg(A_нм) + b_log·lg(R) + b_lin·R + c  (без поправки).
    Возвращает ML_raw или None.
    """
    A = amplitudes.get((pub_id, sta))
    if A is None or A <= 0.0:
        return None
    A_nm = A * AMP_SCALE_NM
    if A_nm <= 0.0 or A_nm > A_MAX_NM:
        return None
    return (math.log10(A_nm)
            + ML_B_LOG * math.log10(r_km)
            + ML_B_LIN * r_km
            + ML_C)


def compute_event_ml_detail(pub_id, r_dict, amplitudes, sta_corr, exclude_stations=None):
    """
    Вычисляет ML события (медиана по станциям) с детализацией.
    Логика идентична ml_filter_v4.compute_event_ml.

    Возвращает: (ml, used_entries, filtered_entries)
      used_entries    = [(sta, r_km, A_nm, ml_sta), ...]  — вошли в медиану
      filtered_entries= [(sta, r_km, A_nm, ml_sta), ...]  — отброшены ratio-фильтром
    """
    excl = set(exclude_stations) if exclude_stations else set()
    entries = []
    for sta, r_km in r_dict.items():
        if sta in excl:
            continue
        if r_km < R_MIN_KM:
            continue
        A = amplitudes.get((pub_id, sta))
        if A is None or A <= 0.0:
            continue
        A_nm = A * AMP_SCALE_NM
        if A_nm <= 0.0 or A_nm > A_MAX_NM:
            continue
        S = sta_corr.get(sta, 0.0)
        ml_s = (math.log10(A_nm)
                + ML_B_LOG * math.log10(r_km)
                + ML_B_LIN * r_km
                + ML_C + S)
        entries.append((sta, r_km, A_nm, ml_s))

    cur = entries[:]
    for _ in range(_ML_N_ITER):
        if len(cur) < _ML_N_MIN_STA:
            break
        ml_med = float(np.median([e[3] for e in cur]))
        survivors = []
        for sta, r_km, A_nm, ml_s in cur:
            S = sta_corr.get(sta, 0.0)
            A_exp = 10 ** (ml_med
                           - ML_B_LOG * math.log10(r_km)
                           - ML_B_LIN * r_km
                           - ML_C - S)
            if A_nm / A_exp >= _ML_RATIO_MIN:
                survivors.append((sta, r_km, A_nm, ml_s))
        if len(survivors) < _ML_N_MIN_STA or len(survivors) == len(cur):
            break
        cur = survivors

    used_stas    = {e[0] for e in cur}
    used_entries = cur
    filt_entries = [e for e in entries if e[0] not in used_stas]
    ml = float(np.median([e[3] for e in cur])) if cur else None
    return ml, used_entries, filt_entries


def compute_event_ml_detail_v2(pub_id, r_dict, amplitudes, sta_corr,
                               exclude_stations=None,
                               outlier_sigma=ML_OUTLIER_SIGMA):
    """
    Вычисляет ML события (среднее после удаления выбросов) с детализацией.

    Алгоритм:
      1. ML_sta = ML_raw + S для каждой станции.
      2. Медиана Mmed по всем станциям.
      3. Удалить выбросы: |MLi - Mmed| > outlier_abs  ИЛИ  > outlier_sigma · σ.
      4. ML = среднее по оставшимся.

    Если станций < _ML_N_MIN_STA — возвращает среднее без фильтрации.

    Возвращает: (ml, used_entries, filtered_entries)
      used_entries     = [(sta, r_km, A_nm, ml_sta), ...]
      filtered_entries = [(sta, r_km, A_nm, ml_sta), ...]
    """
    excl = set(exclude_stations) if exclude_stations else set()
    entries = []
    for sta, r_km in r_dict.items():
        if sta in excl:
            continue
        if r_km < R_MIN_KM:
            continue
        A = amplitudes.get((pub_id, sta))
        if A is None or A <= 0.0:
            continue
        A_nm = A * AMP_SCALE_NM
        if A_nm <= 0.0 or A_nm < A_NM_MIN or A_nm > A_MAX_NM:
            continue
        S = sta_corr.get(sta, 0.0)
        ml_s = (math.log10(A_nm)
                + ML_B_LOG * math.log10(r_km)
                + ML_B_LIN * r_km
                + ML_C + S)
        entries.append((sta, r_km, A_nm, ml_s))

    if not entries:
        return None, [], []

    if len(entries) < _ML_N_MIN_STA:
        return float(np.mean([e[3] for e in entries])), entries[:], []

    ml_vals = np.array([e[3] for e in entries])
    Mmed    = float(np.median(ml_vals))
    sigma   = float(np.std(ml_vals))

    inliers, outliers = [], []
    for e in entries:
        dev       = abs(e[3] - Mmed)
        sigma_thr = outlier_sigma * sigma if sigma > 0.0 else float('inf')
        if dev > sigma_thr:
            outliers.append(e)
        else:
            inliers.append(e)

    if not inliers:
        inliers, outliers = entries[:], []

    ml = float(np.mean([e[3] for e in inliers]))
    return ml, inliers, outliers


def print_diag_ml(matched, amplitudes, sta_corr, exclude_stations=None, use_v2=True):
    """
    Печатает таблицу: для каждого совпавшего события (каталог ↔ XML)
    показывает Ms, вычисленный ML, расхождение Δ=ML−Ms,
    N станций, использованных в медиане, и их список.

    use_v2=True: алгоритм медиана → удаление выбросов → среднее (v2).
    use_v2=False: ratio-фильтр амплитуд (v1, старый).
    """
    _compute = compute_event_ml_detail_v2 if use_v2 else compute_event_ml_detail
    method_label = ('медиана→outlier→среднее'
                    if use_v2 else 'ratio-фильтр амплитуд')
    filt_label   = 'отброшены выброс' if use_v2 else 'отброшены ratio'

    print(f"\n{'=' * 90}")
    print("ДИАГНОСТИКА: вычисленный ML vs каталожный Ms")
    print(f"  (поправки применены; метод: {method_label}; порядок по времени каталога)")
    print(f"{'=' * 90}")
    hdr = (f"  {'Время каталога':<22} {'Ms':>4} {'ML':>5} {'Δ':>6}  "
           f"{'N':>2}  Станции использованы  [{filt_label}]")
    print(hdr)
    print(f"  {'-' * 86}")

    deltas = []
    for ev in sorted(matched, key=lambda x: x['cat_ev']['origin_time']):
        cat_t  = ev['cat_ev']['origin_time']
        ms     = ev['cat_ev']['ms']
        pub_id = ev['xml_pub_id']
        r_dict = ev.get('r_dict', {})

        ml, used, filtered = _compute(pub_id, r_dict, amplitudes, sta_corr,
                                      exclude_stations)

        used_str = ' '.join(sorted(e[0] for e in used))
        filt_str = ' '.join(sorted(e[0] for e in filtered))

        if ml is None:
            ml_str    = '   — '
            delta_str = '     —'
        else:
            delta = ml - ms
            deltas.append(delta)
            ml_str    = f"{ml:5.2f}"
            delta_str = f"{delta:+6.2f}"

        n_used   = len(used)
        filt_note = f"  [{filt_str}]" if filt_str else ''
        print(f"  {str(cat_t):<22} {ms:>4.1f} {ml_str} {delta_str}  "
              f"{n_used:>2}  {used_str}{filt_note}")

    if deltas:
        print(f"\n  Δ = ML − Ms  →  mean={np.mean(deltas):+.3f}  "
              f"median={float(np.median(deltas)):+.3f}  "
              f"std={float(np.std(deltas)):.3f}  "
              f"N={len(deltas)}")
    print('=' * 90)


def compute_corrections(matched_events, amplitudes,
                        amp_ratio_max=AMP_RATIO_MAX, amp_ratio_min=AMP_RATIO_MIN,
                        lg_a_min=LG_A_NM_MIN, lg_a_max=LG_A_NM_MAX):
    """
    Для каждого (событие, станция) вычисляет S = Ms - ML_raw.

    amp_ratio_max/min: относительный фильтр по отношению A_nm к ожидаемой (0 = выкл.)
    lg_a_min/max:      абсолютный фильтр по log10(A_нм) — диапазон допустимых амплитуд
                       (0 = соответствующий фильтр выключен)

    Возвращает list of dict:
      {pub_id, cat_time, Ms, station, R_km, A_nm, ML_raw, S}
    """
    rows = []
    n_filtered_ratio_hi = 0
    n_filtered_ratio_lo = 0
    n_filtered_a_hi = 0
    n_filtered_a_lo = 0
    for ev in matched_events:
        ms     = ev['cat_ev']['ms']
        pub_id = ev['xml_pub_id']
        cat_t  = ev['cat_ev']['origin_time']
        r_dict = ev['r_dict']

        for sta, r_km in r_dict.items():
            ml_raw = compute_raw_ml(pub_id, sta, r_km, amplitudes)
            if ml_raw is None:
                continue
            A = amplitudes.get((pub_id, sta))
            A_nm = A * AMP_SCALE_NM if A else None
            if A_nm is None:
                continue

            lg_a = math.log10(A_nm)

            # Абсолютный фильтр по lg(A_нм)
            if lg_a_min and lg_a < lg_a_min:
                n_filtered_a_lo += 1
                continue
            if lg_a_max and lg_a > lg_a_max:
                n_filtered_a_hi += 1
                continue

            # Относительный фильтр по amp_ratio
            if amp_ratio_max or amp_ratio_min:
                exp_log_a = ms - ML_B_LOG * math.log10(r_km) - ML_B_LIN * r_km - ML_C
                exp_a_nm  = 10.0 ** exp_log_a
                if amp_ratio_max and A_nm > amp_ratio_max * exp_a_nm:
                    n_filtered_ratio_hi += 1
                    continue
                if amp_ratio_min and A_nm < amp_ratio_min * exp_a_nm:
                    n_filtered_ratio_lo += 1
                    continue

            S = ms - ml_raw
            rows.append({
                'pub_id':   pub_id,
                'cat_time': cat_t,
                'Ms':       ms,
                'station':  sta,
                'R_km':     r_km,
                'A_nm':     A_nm,
                'ML_raw':   ml_raw,
                'S':        S,
            })
    if n_filtered_a_lo:
        print(f"  lg(A) filter: отброшено {n_filtered_a_lo} измерений "
              f"(lg(A_нм) < {lg_a_min})")
    if n_filtered_a_hi:
        print(f"  lg(A) filter: отброшено {n_filtered_a_hi} измерений "
              f"(lg(A_нм) > {lg_a_max})")
    if n_filtered_ratio_hi:
        print(f"  Amp-ratio filter: отброшено {n_filtered_ratio_hi} измерений "
              f"(A > {amp_ratio_max}× ожидаемого)")
    if n_filtered_ratio_lo:
        print(f"  Amp-ratio filter: отброшено {n_filtered_ratio_lo} измерений "
              f"(A < {amp_ratio_min}× ожидаемого)")
    return rows


# ── Агрегация и детектор выбросов ─────────────────────────────────────────────

def aggregate_by_station(rows, min_events=MIN_EVENTS, iqr_k=IQR_K):
    """
    Группирует строки по станции, находит выбросы методом IQR.

    Возвращает dict:
      station -> {
        'values':    [(cat_time, S), ...],   # все значения
        'outliers':  [(cat_time, S), ...],   # выбросы
        'inliers':   [(cat_time, S), ...],   # без выбросов
        'mean_all':  float,
        'mean_clean': float,
        'n':         int,
        'n_outliers': int,
        'included':  bool,   # False если n < min_events
      }
    """
    by_sta = defaultdict(list)
    for row in rows:
        by_sta[row['station']].append((row['cat_time'], row['S']))

    result = {}
    for sta, vals in by_sta.items():
        s_vals = [s for _, s in vals]
        mean_all = float(np.mean(s_vals))

        # IQR outlier detection
        if len(s_vals) >= 4:
            q1, q3 = np.percentile(s_vals, [25, 75])
            iqr = q3 - q1
            lo  = q1 - iqr_k * iqr
            hi  = q3 + iqr_k * iqr
            inliers  = [(t, s) for t, s in vals if lo <= s <= hi]
            outliers = [(t, s) for t, s in vals if s < lo or s > hi]
        else:
            inliers  = vals[:]
            outliers = []

        mean_clean = float(np.mean([s for _, s in inliers])) if inliers else mean_all

        result[sta] = {
            'values':     vals,
            'outliers':   outliers,
            'inliers':    inliers,
            'mean_all':   mean_all,
            'mean_clean': mean_clean,
            'n':          len(vals),
            'n_outliers': len(outliers),
            'included':   len(vals) >= min_events,
        }
    return result


# ── Вывод и запись ────────────────────────────────────────────────────────────

def print_summary(agg, min_events):
    print(f"\n{'=' * 70}")
    print(f"РЕЗУЛЬТАТЫ СТАНЦИОННЫХ ПОПРАВОК  (мин. событий: {min_events})")
    print(f"{'=' * 70}")
    print(f"\n  {'Станция':8s}  {'N':>4s}  {'Выбросы':>7s}  "
          f"{'S (чистое)':>10s}  {'S (с выбросами)':>15s}  Статус")
    print(f"  {'-' * 66}")

    included = {s: v for s, v in agg.items() if v['included']}
    excluded = {s: v for s, v in agg.items() if not v['included']}

    for sta in sorted(included):
        v = included[sta]
        outlier_s = (f"  *** ВЫБРОСЫ ({v['n_outliers']}) ***"
                     if v['n_outliers'] else '')
        print(f"  {sta:8s}  {v['n']:>4d}  {v['n_outliers']:>7d}  "
              f"{v['mean_clean']:>10.3f}  {v['mean_all']:>15.3f}{outlier_s}")

    if excluded:
        print(f"\n  Исключены (менее {min_events} событий):")
        for sta in sorted(excluded):
            v = excluded[sta]
            print(f"    {sta:8s}  N={v['n']}  S_mean={v['mean_all']:.3f}")

    # Подробности по выбросам
    has_outliers = [(s, v) for s, v in included.items() if v['n_outliers'] > 0]
    if has_outliers:
        print(f"\n{'─' * 70}")
        print("ДЕТАЛИ ВЫБРОСОВ:")
        for sta, v in sorted(has_outliers):
            q1, q3 = np.percentile([s for _, s in v['values']], [25, 75])
            iqr = q3 - q1
            print(f"\n  {sta}: IQR=[{q1:.3f}, {q3:.3f}]  "
                  f"iqr={iqr:.3f}  границы=[{q1-IQR_K*iqr:.3f}, {q3+IQR_K*iqr:.3f}]")
            for t, s in sorted(v['outliers'], key=lambda x: x[1], reverse=True):
                print(f"    {t.strftime('%Y-%m-%d %H:%M:%S')}  S={s:+.3f}  "
                      f"(отклонение от чистого: {s - v['mean_clean']:+.3f})")


def save_detail_csv(rows, agg, path):
    """Детальный CSV: одна строка = (станция, событие, поправка)."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['station', 'cat_time', 'pub_id', 'Ms',
                    'R_km', 'A_nm', 'ML_raw', 'S', 'is_outlier'])
        for row in sorted(rows, key=lambda r: (r['station'], r['cat_time'])):
            sta   = row['station']
            is_ol = any(
                abs(t - row['cat_time']).total_seconds() < 1 and abs(s - row['S']) < 1e-9
                for t, s in agg.get(sta, {}).get('outliers', [])
            )
            w.writerow([
                sta,
                row['cat_time'].strftime('%Y-%m-%dT%H:%M:%S'),
                row['pub_id'],
                f"{row['Ms']:.2f}",
                f"{row['R_km']:.1f}",
                f"{row['A_nm']:.2f}" if row['A_nm'] else '',
                f"{row['ML_raw']:.3f}",
                f"{row['S']:.3f}",
                'yes' if is_ol else '',
            ])


def save_summary_csv(agg, path, use_clean=True):
    """
    Итоговый CSV в формате sta_corrections_dyagilev2023.csv.
    Пишет только станции с included=True.
    use_clean=True: использует среднее без выбросов.
    """
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['station', 'S'])
        for sta in sorted(agg):
            v = agg[sta]
            if not v['included']:
                continue
            val = v['mean_clean'] if use_clean else v['mean_all']
            w.writerow([sta, f"{val:.3f}"])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global VP, VS, N_WORKERS, MIN_EVENTS, MATCH_WIN_SEC, IQR_K

    parser = argparse.ArgumentParser(
        description='Вычисление станционных поправок ML из каталога',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--assoc-in',     default=DEFAULT_ASSOC_IN,
                        help='Входной XML ассоциатора')
    parser.add_argument('--catalog',      default=DEFAULT_CATALOG,
                        help='Каталог событий (catalog.xlsx)')
    parser.add_argument('--waveforms',    default=DEFAULT_WAVEFORMS,
                        help='Папка с формами волн')
    parser.add_argument('--metadata-dir', default=DEFAULT_METADATA,
                        help='Папка с FDSNStationXML для remove_response')
    parser.add_argument('--stations-dir', default=DEFAULT_STA_DIR,
                        help='Папка с json/station_*.json')
    parser.add_argument('--cache-amp',    default=DEFAULT_CACHE_AMP,
                        help='CSV-кэш амплитуд')
    parser.add_argument('--out-detail',   default=DEFAULT_OUT_DETAIL,
                        help='Детальный CSV: поправка на каждое (станция, событие)')
    parser.add_argument('--out-summary',  default=DEFAULT_OUT_SUMMARY,
                        help='Итоговый CSV в формате station,S')
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='Пересобрать кэш амплитуд')
    parser.add_argument('--year',         type=int, default=None,
                        help='Фильтр каталога по году')
    parser.add_argument('--month',        type=int, default=None,
                        help='Фильтр каталога по месяцу')
    parser.add_argument('--min-ms',       type=float, default=None,
                        help='Минимальная Ms каталога для расчёта поправок')
    parser.add_argument('--max-ms',       type=float, default=None,
                        help='Максимальная Ms каталога (исключить сильные события, '
                             'где формула ML насыщается)')
    parser.add_argument('--min-events',   type=int, default=MIN_EVENTS,
                        help='Минимум событий каталога на станцию')
    parser.add_argument('--match-win',    type=float, default=MATCH_WIN_SEC,
                        help='Окно матчинга каталог→XML, сек (после поправки на TT)')
    parser.add_argument('--r-min',        type=float, default=R_MIN_KM,
                        help=f'Мин. расстояние, км (по умолч. {R_MIN_KM}; '
                             'при R<50 P-кода накладывается на S-волну и завышает амплитуду)')
    parser.add_argument('--r-max',        type=float, default=R_MAX_KM,
                        help='Макс. расстояние для включения в расчёт, км '
                             '(формула 5б не откалибрована >200-300 км; 0 = без ограничения)')
    parser.add_argument('--use-coords',  action='store_true',
                        help='R из координат каталога+станций вместо S-P ΔT '
                             '(точнее для каталожных событий с известной локацией)')
    parser.add_argument('--iqr-k',        type=float, default=IQR_K,
                        help='Множитель IQR для детектора выбросов')
    parser.add_argument('--workers',      type=int, default=N_WORKERS,
                        help='Число потоков (каждый — своя группа станций)')
    parser.add_argument('--vp',           type=float, default=VP)
    parser.add_argument('--vs',           type=float, default=VS)
    parser.add_argument('--info',         action='store_true',
                        help='Только показать матчинг и статистику, без записи')
    parser.add_argument('--amp-ratio-max', type=float, default=AMP_RATIO_MAX,
                        help='Макс. допустимый ratio A_nm / ожидаемой_A_nm по каталожной Ms. '
                             'Измерения с A_nm > ratio × exp_A_nm отбрасываются (0 = выключено)')
    parser.add_argument('--amp-ratio-min', type=float, default=AMP_RATIO_MIN,
                        help='Мин. допустимый ratio A_nm / ожидаемой_A_nm по каталожной Ms. '
                             'Измерения с A_nm < ratio × exp_A_nm отбрасываются (0 = выключено)')
    parser.add_argument('--lg-a-min',     type=float, default=LG_A_NM_MIN,
                        help='Мин. lg(A_нм): строки с lg(A_нм) < порога (шум) отбрасываются '
                             f'(по умолч. {LG_A_NM_MIN}; 0 = выключено)')
    parser.add_argument('--lg-a-max',     type=float, default=LG_A_NM_MAX,
                        help='Макс. lg(A_нм): строки с lg(A_нм) > порога (насыщение) отбрасываются '
                             f'(по умолч. {LG_A_NM_MAX}; 0 = выключено)')
    parser.add_argument('--win-sec',      type=float, default=WIN_SEC,
                        help='Длина окна S-волны, сек (окно: [S_time .. S_time + win_sec])')
    parser.add_argument('--no-bandpass',  action='store_true', default=False,
                        help=f'Отключить полосовой фильтр {BP_FREQMIN}–{BP_FREQMAX} Гц '
                             '(по умолчанию включён — соответствует рабочей полосе WA-прибора)')
    parser.add_argument('--exclude-stations', default=None,
                        help='Исключить станции из расчёта ML (через запятую, '
                             'например KMKR,KRNR). Применяется только в --diag-ml.')
    parser.add_argument('--diag-ml',     action='store_true',
                        help='Диагностика: вычисленный ML vs Ms каталога для каждого совпавшего '
                             'события. Загружает кэш амплитуд и поправки, выводит таблицу, '
                             'без записи файлов. Требует готового кэша (amps_sta_corr.csv).')
    parser.add_argument('--sta-corrections', default=DEFAULT_OUT_SUMMARY,
                        help='CSV станционных поправок для режима --diag-ml (station,S). '
                             'По умолчанию: sta_corrections_computed.csv')
    args = parser.parse_args()

    VP          = args.vp
    VS          = args.vs
    N_WORKERS   = args.workers
    MIN_EVENTS  = args.min_events
    MATCH_WIN_SEC = args.match_win
    IQR_K       = args.iqr_k

    k = VP * VS / (VP - VS)
    print(f"Ассоциатор:  {args.assoc_in}")
    print(f"Каталог:     {args.catalog}")
    if args.use_coords:
        print(f"Режим R:     координаты каталога → R = sqrt(epi² + depth²)")
    else:
        print(f"Режим R:     S-P ΔT → R = {k:.3f} · ΔT(S-P) км")
    ratio_info = (f"  |  amp-ratio-max: {args.amp_ratio_max}"
                  if args.amp_ratio_max else "  |  amp-ratio-max: выкл.")
    print(f"Окно матчинга: {MATCH_WIN_SEC} с  |  мин. событий: {MIN_EVENTS}  "
          f"|  IQR k: {IQR_K}  |  воркеров: {N_WORKERS}"
          f"  |  win-sec: {args.win_sec}{ratio_info}\n")

    # ── Загрузка данных ───────────────────────────────────────────────────────
    print("Загрузка каталога...")
    catalog = load_catalog(args.catalog, year=args.year, month=args.month,
                           min_ms=args.min_ms, max_ms=args.max_ms)
    print(f"  Событий в каталоге: {len(catalog)}")

    print("Загрузка координат станций...")
    stations = load_stations(args.stations_dir)
    print(f"  Станций: {len(stations)}")

    print("Парсинг XML...")
    tree       = ET.parse(args.assoc_in)
    xml_events = parse_xml_full(tree)
    print(f"  XML событий: {len(xml_events)}")

    # ── Матчинг ───────────────────────────────────────────────────────────────
    print(f"\nМатчинг каталог → XML (окно ±{MATCH_WIN_SEC} с)...")
    matched = match_catalog_to_xml(
        catalog, xml_events, stations, VP, MATCH_WIN_SEC
    )
    print(f"  Совпадений: {len(matched)} / {len(catalog)}")
    if not matched:
        print("  Нет совпадений — проверь пути и параметры.")
        return

    # Добавляем r_dict к каждому совпавшему событию
    r_mode = 'координаты каталога' if args.use_coords else 'S-P ΔT'
    for ev in matched:
        if args.use_coords:
            ev['r_dict'] = compute_r_dict_coords(
                ev['cat_ev'], ev['picks_s'], stations,
                r_min_km=args.r_min, r_max_km=args.r_max or 0)
        else:
            ev['r_dict'] = compute_r_dict(ev['picks_p'], ev['picks_s'],
                                          r_min_km=args.r_min,
                                          r_max_km=args.r_max or 0)

    n_with_r = sum(1 for ev in matched if ev['r_dict'])
    print(f"  Режим R: {r_mode}  |  с R: {n_with_r} / {len(matched)}")

    if args.info:
        # Показать детали матчинга
        print(f"\n{'─' * 70}")
        print("Совпавшие события:")
        print(f"  {'Время каталога':<22} {'Ms':>4} {'dt,с':>6} "
              f"{'N_sta':>5} {'pub_id'}")
        for ev in sorted(matched, key=lambda x: x['cat_ev']['origin_time']):
            print(f"  {str(ev['cat_ev']['origin_time']):<22} "
                  f"{ev['cat_ev']['ms']:>4.1f} {ev['dt_sec']:>6.1f} "
                  f"{len(ev['r_dict']):>5} {ev['xml_pub_id']}")
        return

    if args.diag_ml:
        # Загрузка станционных поправок
        sta_corr = {}
        if os.path.isfile(args.sta_corrections):
            with open(args.sta_corrections, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    sta = row.get('station', '').strip()
                    val = row.get('S', '').strip()
                    if sta and val:
                        try:
                            sta_corr[sta] = float(val)
                        except ValueError:
                            pass
            print(f"  Станционных поправок: {len(sta_corr)}  ({args.sta_corrections})")
        else:
            print(f"  Поправки не найдены: {args.sta_corrections} — ML без поправок")

        if not os.path.isfile(args.cache_amp):
            print(f"\n  Ошибка: кэш амплитуд не найден: {args.cache_amp}")
            print("  Запустите скрипт без --diag-ml сначала, чтобы построить кэш.")
            return
        diag_amps = load_amp_cache(args.cache_amp)
        n_amp = sum(1 for v in diag_amps.values() if v is not None)
        print(f"  Кэш амплитуд: {len(diag_amps)} пар, с амплитудой: {n_amp}")

        excl = [s.strip() for s in args.exclude_stations.split(',') if s.strip()] \
               if args.exclude_stations else None
        if excl:
            print(f"  Исключены станции: {excl}")
        print_diag_ml(matched, diag_amps, sta_corr, exclude_stations=excl)
        return

    # ── Амплитуды ─────────────────────────────────────────────────────────────
    if not args.rebuild_cache and os.path.isfile(args.cache_amp):
        print(f"\nЗагрузка кэша амплитуд: {args.cache_amp}")
        amplitudes = load_amp_cache(args.cache_amp)
        n_amp = sum(1 for v in amplitudes.values() if v is not None)
        print(f"  Пар: {len(amplitudes)}  с амплитудой: {n_amp}")
    else:
        if args.rebuild_cache:
            print("\nПересборка кэша (--rebuild-cache).")

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

        print(f"\nПостроение индекса форм волн...")
        waveform_index = build_waveform_index(args.waveforms)
        n_files = sum(len(v) for v in waveform_index.values())
        print(f"  Файлов: {n_files}  станций: {len(waveform_index)}")

        matched_with_r = [ev for ev in matched if ev['r_dict']]
        use_bp = not args.no_bandpass
        print(f"\nИзвлечение амплитуд S-волны для {len(matched_with_r)} событий "
              f"(БПФ {BP_FREQMIN}–{BP_FREQMAX} Гц: {'да' if use_bp else 'нет'})...")
        amplitudes = extract_amplitudes(
            matched_with_r, waveform_index,
            inventory=inventory,
            n_workers=N_WORKERS,
            win_sec=args.win_sec,
            bandpass=use_bp,
        )
        n_amp = sum(1 for v in amplitudes.values() if v is not None)
        print(f"  Пар с амплитудой: {n_amp}")

        save_amp_cache(amplitudes, args.cache_amp)
        print(f"  Кэш сохранён: {args.cache_amp}")

    # ── Вычисление поправок ───────────────────────────────────────────────────
    print("\nВычисление поправок S = Ms − ML_raw...")
    rows = compute_corrections(matched, amplitudes,
                               amp_ratio_max=args.amp_ratio_max or 0,
                               amp_ratio_min=args.amp_ratio_min or 0,
                               lg_a_min=args.lg_a_min or 0,
                               lg_a_max=args.lg_a_max or 0)
    print(f"  Пар (станция, событие) с поправкой: {len(rows)}")

    agg = aggregate_by_station(rows, min_events=MIN_EVENTS, iqr_k=IQR_K)

    print_summary(agg, MIN_EVENTS)

    # ── Запись файлов ─────────────────────────────────────────────────────────
    save_detail_csv(rows, agg, args.out_detail)
    print(f"\n  Детальный CSV: {args.out_detail}")

    save_summary_csv(agg, args.out_summary, use_clean=True)
    print(f"  Итоговый CSV:  {args.out_summary}")
    print("  (итоговый CSV использует среднее БЕЗ выбросов; "
          "колонка S (с выбросами) — только в выводе терминала)")


if __name__ == '__main__':
    main()
