"""
Оценка Ml-магнитуды и фильтрация associations.xml.

  1. Локация каждого события grid search по P-временам (numpy broadcasting)
  2. Извлечение амплитуды P-волны из форм волн, группировка по (станция, файл)
     Без --metadata-dir: амплитуда в counts (highpass 1 Hz).
     С   --metadata-dir: амплитуда в м/с после remove_response() из FDSNStationXML.
  3. Прокси-магнитуда: Ml_raw_i = log10(A_i) + log10(r_km_i)
                       Ml_raw   = median(Ml_raw_i по всем станциям события)
  4. Ml_est = Ml_raw + calib_const   (C задаётся вручную через --calib-const)
  5. Фильтрация XML по Ml_est >= ml_threshold

Запуск:
    python core/magnitude_estimator.py --info
    python core/magnitude_estimator.py --ml-threshold 0.0
    python core/magnitude_estimator.py --ml-threshold -0.5 0.0 0.5 --validate
    python core/magnitude_estimator.py --calib-const 3.0 --ml-threshold 0.0
    python core/magnitude_estimator.py --cache-loc locs.csv --ml-threshold 0.0
    python core/magnitude_estimator.py --metadata-dir metadata --cache-amp amps_vel.csv --fit-regression --info
"""
import argparse
import copy
import csv
import gc
import glob
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ASSOC_IN  = os.path.join(_ROOT, 'data-in-memory', 'gpu_splimit_45_march', 'assoc_output_lim', 'associations.xml')
DEFAULT_WAVEFORMS = os.path.join(_ROOT, 'data-in-memory', 'geofiles')
DEFAULT_STA_DIR   = os.path.join(_ROOT, 'json')
DEFAULT_METADATA  = os.path.join(_ROOT, 'metadata')

BED_NS   = 'http://quakeml.org/xmlns/bed/1.2'
QUAKE_NS = 'http://quakeml.org/xmlns/quakeml/1.2'

ET.register_namespace('',  BED_NS)
ET.register_namespace('q', QUAKE_NS)

VP_DEFAULT    = 6.0
DEPTH_DEFAULT = 10.0
GRID_STEP_DEF = 0.05
SR            = 100.0   # hardcode — reading from attrs gives wrong value (known bug)
WIN_SEC       = 3.0     # P-wave amplitude window, seconds

# Wood-Anderson seismometer PAZ (Richter 1935; T0=0.8s, h=0.8, V=2800)
_PAZ_WA = {
    'poles': [(-6.283185307 + 4.712388980j),
              (-6.283185307 - 4.712388980j)],
    'zeros': [0j, 0j],
    'gain': 1.0,
    'sensitivity': 2800.0,
}


# ── Utilities ─────────────────────────────────────────────────────────────────

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


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(max(0.0, a)))


def load_station_coords(sta_dir):
    """Returns dict: station -> (lat, lon)."""
    coords = {}
    for fpath in glob.glob(os.path.join(sta_dir, 'station_*.json')):
        name = os.path.basename(fpath).replace('station_', '').replace('.json', '')
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)
        key = list(data.keys())[0]
        c = data[key]['coords']
        coords[name] = (float(c[0]), float(c[1]))
    return coords


def load_inventory(metadata_dir):
    """Load all FDSNStationXML files from metadata_dir into a single ObsPy Inventory."""
    from obspy import read_inventory
    inv = None
    loaded = 0
    for fpath in sorted(glob.glob(os.path.join(metadata_dir, '*.xml'))):
        try:
            new_inv = read_inventory(fpath)
            inv = new_inv if inv is None else inv + new_inv
            loaded += 1
        except Exception:
            pass
    if inv is None:
        print(f"  Предупреждение: StationXML не найдены в {metadata_dir}")
        return None
    n_sta = sum(len(net) for net in inv.networks)
    print(f"  Загружено StationXML: {loaded} файлов, {n_sta} станций")
    return inv


# ── Waveform file index ────────────────────────────────────────────────────────

def _parse_file_times(fname):
    """Parse (t_start, t_end) from '...__ 20240101T000000Z__20240102T000000Z'."""
    parts = fname.split('__')
    if len(parts) != 3:
        return None, None
    try:
        t_start = datetime.strptime(parts[1].rstrip('Z'), '%Y%m%dT%H%M%S')
        t_end   = datetime.strptime(parts[2].rstrip('Z'), '%Y%m%dT%H%M%S')
        return t_start, t_end
    except ValueError:
        return None, None


_COMP_PRIORITY = ('Z', 'E', 'N')   # prefer vertical for P-wave, fallback to horizontal


def build_waveform_index(waveform_dir):
    """
    Returns dict: sta -> list of (t_start, t_end, fpath) sorted by t_start.
    Selects best available component per station: Z > E > N.
    Supports BH*/EH*/SH*/HH* channel families.
    """
    index = {}
    if not os.path.isdir(waveform_dir):
        return index
    for sta in os.listdir(waveform_dir):
        sta_path = os.path.join(waveform_dir, sta)
        if not os.path.isdir(sta_path):
            continue
        # Group files by component letter (last char of channel code)
        by_comp = defaultdict(list)
        for fname in os.listdir(sta_path):
            time_parts = fname.split('__')
            if len(time_parts) != 3:
                continue
            dot_parts = time_parts[0].split('.')
            if len(dot_parts) < 4:
                continue
            cha  = dot_parts[3]          # e.g. BHE, EHZ, SHN
            comp = cha[-1].upper()       # E, Z, N
            if comp not in _COMP_PRIORITY:
                continue
            t_start, t_end = _parse_file_times(fname)
            if t_start is None:
                continue
            by_comp[comp].append((t_start, t_end, os.path.join(sta_path, fname)))
        # Pick best available component
        for comp in _COMP_PRIORITY:
            if comp in by_comp:
                files = sorted(by_comp[comp], key=lambda x: x[0])
                index[sta] = files
                break
    return index


def find_waveform_file(index, sta, p_time):
    """Find waveform file for station covering p_time. n <= 31 so linear search is fine."""
    entries = index.get(sta)
    if not entries:
        return None
    for t_start, t_end, fpath in entries:
        if t_start <= p_time < t_end:
            return fpath
    return None


# ── Grid search location ──────────────────────────────────────────────────────

def locate_event(picks_p, sta_coords, vp=VP_DEFAULT, depth=DEPTH_DEFAULT,
                 grid_step=GRID_STEP_DEF):
    """
    Locate event by grid search over P-arrival times (numpy broadcasting).

    picks_p:    {station: datetime}
    sta_coords: {station: (lat, lon)}

    Returns (lat_est, lon_est, origin_t_est, residual_sec2, {sta: r_km})
    or None if fewer than 2 stations have known coords.
    """
    stas = [s for s in picks_p if s in sta_coords]
    if len(stas) < 2:
        return None

    sta_lats = np.array([sta_coords[s][0] for s in stas])
    sta_lons = np.array([sta_coords[s][1] for s in stas])
    t_ref    = min(picks_p[s] for s in stas)
    T_obs    = np.array([(picks_p[s] - t_ref).total_seconds() for s in stas])

    lat_min = sta_lats.min() - 0.5
    lat_max = sta_lats.max() + 0.5
    lon_min = sta_lons.min() - 0.5
    lon_max = sta_lons.max() + 0.5
    lat_grid = np.arange(lat_min, lat_max + grid_step * 0.5, grid_step)
    lon_grid = np.arange(lon_min, lon_max + grid_step * 0.5, grid_step)

    # Broadcast haversine — shape (N_lat, N_lon, N_sta)
    R = 6371.0
    phi1 = np.radians(lat_grid)[:, None, None]   # (N_lat, 1, 1)
    phi2 = np.radians(sta_lats)[None, None, :]   # (1, 1, N_sta)
    lam1 = np.radians(lon_grid)[None, :, None]   # (1, N_lon, 1)
    lam2 = np.radians(sta_lons)[None, None, :]   # (1, 1, N_sta)

    a      = np.sin((phi2 - phi1) / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin((lam2 - lam1) / 2) ** 2
    epi_km = R * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    r_km   = np.sqrt(epi_km ** 2 + depth ** 2)
    tt     = r_km / vp

    # Optimal origin time for each grid point
    T_origin = np.mean(T_obs[None, None, :] - tt, axis=2, keepdims=True)  # (N_lat, N_lon, 1)

    residuals = np.sum((T_obs[None, None, :] - T_origin - tt) ** 2, axis=2)  # (N_lat, N_lon)

    i_lat, i_lon = np.unravel_index(np.argmin(residuals), residuals.shape)
    lat_est  = float(lat_grid[i_lat])
    lon_est  = float(lon_grid[i_lon])
    origin_t = t_ref + timedelta(seconds=float(T_origin[i_lat, i_lon, 0]))
    residual = float(residuals[i_lat, i_lon])

    r_best = r_km[i_lat, i_lon, :]
    r_dict = {s: float(r_best[k]) for k, s in enumerate(stas)}

    return lat_est, lon_est, origin_t, residual, r_dict


# ── XML parsing ───────────────────────────────────────────────────────────────

def parse_events_from_xml(tree):
    """
    Returns list of (event_elem, pub_id, picks_p, picks_s).
    picks_p / picks_s: {sta: datetime}  (P-picks for location, S-picks for amplitude window)
    """
    ns = BED_NS
    root = tree.getroot()
    result = []

    for ev in root.findall(f'.//{{{ns}}}event'):
        pub_id  = ev.get('publicID', '')
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
            t     = parse_time(t_el.text)
            if t is None:
                continue
            phase = phase_el.text.strip() if phase_el is not None else 'P'
            if phase == 'P':
                picks_p[sta] = t
            elif phase == 'S':
                picks_s[sta] = t

        result.append((ev, pub_id, picks_p, picks_s))

    return result


# ── Location cache ─────────────────────────────────────────────────────────────

def save_location_cache(located, cache_path):
    """Save (ev_idx, pub_id, loc_result) to CSV."""
    with open(cache_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ev_idx', 'pub_id', 'lat_est', 'lon_est', 'origin_t', 'residual', 'r_km_json'])
        for ev_idx, pub_id, _, _, loc in located:
            if loc is None:
                w.writerow([ev_idx, pub_id, '', '', '', '', ''])
            else:
                lat_est, lon_est, origin_t, residual, r_dict = loc
                w.writerow([
                    ev_idx, pub_id, lat_est, lon_est,
                    origin_t.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                    residual, json.dumps(r_dict)
                ])


def load_location_cache(cache_path):
    """Returns dict: pub_id -> loc_result or None."""
    cache = {}
    with open(cache_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pub_id = row['pub_id']
            if not row.get('lat_est'):
                cache[pub_id] = None
                continue
            origin_t = parse_time(row['origin_t'])
            r_dict   = json.loads(row['r_km_json']) if row.get('r_km_json') else {}
            cache[pub_id] = (
                float(row['lat_est']), float(row['lon_est']),
                origin_t, float(row['residual']), r_dict
            )
    return cache


# ── Amplitude cache ────────────────────────────────────────────────────────────

def save_amplitude_cache(amplitudes, cache_path):
    """Save dict: (ev_idx, sta) -> A_counts to CSV."""
    with open(cache_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ev_idx', 'sta', 'A_counts'])
        for (ev_idx, sta), A in amplitudes.items():
            w.writerow([ev_idx, sta, '' if A is None else A])


def load_amplitude_cache(cache_path):
    """Returns dict: (ev_idx, sta) -> A_counts or None."""
    cache = {}
    with open(cache_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (int(row['ev_idx']), row['sta'])
            cache[key] = float(row['A_counts']) if row['A_counts'] else None
    return cache


# ── Amplitude extraction ──────────────────────────────────────────────────────

def extract_all_amplitudes(located, waveform_index, window_sec=WIN_SEC, inventory=None,
                           wood_anderson=False):
    """
    Groups event-station pairs by waveform file, reads each file once.

    inventory: ObsPy Inventory from FDSNStationXML files (optional).
               If provided, applies remove_response → amplitudes in m/s (VEL) or
               simulated Wood-Anderson counts (if wood_anderson=True).
               If None, applies highpass filter only → amplitudes in counts.
    wood_anderson: simulate Wood-Anderson seismometer after remove_response(DISP).
                   Requires inventory. Amplitude units match paper formula calibration.

    Returns dict: (ev_idx, sta) -> amplitude  (float; None if failed).
    """
    from obspy import read as obspy_read, UTCDateTime

    # Build: fpath -> [(ev_idx, sta, p_time, s_time, r_km)]
    groups = defaultdict(list)
    for ev_idx, pub_id, picks_p, picks_s, loc in located:
        if loc is None:
            continue
        _, _, _, _, r_dict = loc
        for sta, p_time in picks_p.items():
            if sta not in r_dict:
                continue
            fpath = find_waveform_file(waveform_index, sta, p_time)
            if fpath is None:
                continue
            groups[fpath].append((ev_idx, sta, p_time, picks_s.get(sta)))

    amplitudes = {}
    n_groups = len(groups)
    for g_idx, (fpath, pairs) in enumerate(groups.items()):
        if g_idx % 20 == 0:
            print(f"  Амплитуды: {g_idx}/{n_groups} файлов...", end='\r', flush=True)
        try:
            st_all = obspy_read(fpath)
            if len(st_all) == 0:
                del st_all
                continue
            tr = st_all[0]
            # SR is read from miniSEED/SAC headers — correct for raw waveform files.
            # The SR=100 hardcode rule applies only to EQTransformer's HDF5 attrs (different bug).
            actual_sr = tr.stats.sampling_rate
            if actual_sr not in (50.0, 100.0, 80.0, 200.0):
                actual_sr = 100.0 if actual_sr > 75 else 50.0
                tr.stats.sampling_rate = actual_sr

            if inventory is not None:
                nyq = actual_sr / 2.0
                pre_filt = (0.5, 1.0, nyq * 0.85, nyq * 0.95)
                try:
                    out = 'DISP' if wood_anderson else 'VEL'
                    tr.remove_response(inventory=inventory, output=out,
                                       pre_filt=pre_filt, water_level=60)
                    if wood_anderson:
                        tr.simulate(paz_simulate=_PAZ_WA, paz_remove=None)
                except Exception:
                    tr.filter('highpass', freq=1.0)
            else:
                tr.filter('highpass', freq=1.0)

            for ev_idx, sta, p_time, s_time in pairs:
                try:
                    p_utc = UTCDateTime(p_time)
                    if s_time is not None and (s_time - p_time).total_seconds() > 3.5:
                        end_utc = UTCDateTime(s_time) - 0.5
                    else:
                        end_utc = p_utc + window_sec

                    tr_sig = tr.slice(p_utc, end_utc)
                    if tr_sig is None or len(tr_sig.data) == 0:
                        amplitudes[(ev_idx, sta)] = None
                        continue
                    A = float(np.max(np.abs(tr_sig.data)))
                    amplitudes[(ev_idx, sta)] = A if A > 0.0 else None
                except Exception:
                    amplitudes[(ev_idx, sta)] = None

            del tr, st_all
        except Exception:
            pass
        gc.collect()

    print(f"  Амплитуды: {n_groups}/{n_groups} файлов — готово.        ")
    return amplitudes


# ── Ml computation ─────────────────────────────────────────────────────────────

def compute_ml_raw(A_counts, r_km, a=1.0, b_log=1.0, b_lin=0.0, c=0.0):
    if A_counts is None or A_counts <= 0.0 or r_km <= 0.0:
        return None
    return a * math.log10(A_counts) + b_log * math.log10(r_km) + b_lin * r_km + c


def compute_event_ml(ev_idx, r_dict, amplitudes, a=1.0, b_log=1.0, b_lin=0.0, c=0.0):
    """Returns (Ml_raw, n_sta_amp) using median over all stations."""
    ml_vals = []
    for sta, r_km in r_dict.items():
        A  = amplitudes.get((ev_idx, sta))
        ml = compute_ml_raw(A, r_km, a, b_log, b_lin, c)
        if ml is not None:
            ml_vals.append(ml)
    if not ml_vals:
        return None, 0
    return float(np.median(ml_vals)), len(ml_vals)


# ── Regression-based magnitude ─────────────────────────────────────────────────

def fit_regression(station_quads):
    """
    Global fit: MPVA_est = a*log10(A) + b*log10(r) + c

    station_quads: [(A_counts, r_km, mpva_target, station_name), ...]
    NOTE: global fit fails when inter-station sensitivity varies (b→0).
    Use fit_per_station_calibration instead.

    Returns (a, b, c, sigma) or None if insufficient data.
    """
    valid = [(A, r, m) for A, r, m, _ in station_quads
             if A and A > 0 and r > 0 and m is not None]
    if len(valid) < 4:
        print(f"  Недостаточно данных для регрессии: {len(valid)} точек (нужно >= 4)")
        return None

    logA = np.array([math.log10(A) for A, _, _ in valid])
    logR = np.array([math.log10(r) for _, r, _ in valid])
    mpva = np.array([m for _, _, m in valid])

    X = np.column_stack([logA, logR, np.ones(len(valid))])
    coeffs, _, _, _ = np.linalg.lstsq(X, mpva, rcond=None)
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    residuals = mpva - (a * logA + b * logR + c)
    sigma = float(np.std(residuals))

    print(f"\n  Глобальная регрессия по {len(valid)} наблюдениям:")
    print(f"    MPVA_est = {a:.4f}·log10(A) + {b:.4f}·log10(r) + ({c:.4f})  σ={sigma:.3f}")
    if abs(b) < 0.05:
        print("    ВНИМАНИЕ: b≈0 — глобальная регрессия не работает из-за разной")
        print("    чувствительности инструментов. Используйте per-station калибровку.")
    return a, b, c, sigma


def fit_per_station_calibration(station_quads):
    """
    Per-station calibration: c_sta = mean(MPVA - log10(A) - log10(r)) per station.

    station_quads: [(A_counts, r_km, mpva_target, station_name), ...]
    Fixed a=1, b=1 (standard Richter). Only c_sta varies per station, absorbing
    the instrument sensitivity difference.

    Returns dict {station_name: c_sta}.
    """
    by_sta = defaultdict(list)
    for A, r, mpva, sta in station_quads:
        if A and A > 0 and r > 0 and mpva is not None:
            by_sta[sta].append((math.log10(A), math.log10(r), mpva))

    sta_calibs = {}
    print(f"\n  Покалибровка по станциям (a=1, b=1, c_sta варьируется):")
    print(f"  {'Станция':<8} {'N':>3}  {'c_sta':>7}  {'σ_sta':>6}")
    for sta in sorted(by_sta):
        triples = by_sta[sta]
        if len(triples) < 2:
            continue
        diffs = [mpva - logA - logR for logA, logR, mpva in triples]
        c_sta = float(np.mean(diffs))
        sigma_sta = float(np.std(diffs))
        sta_calibs[sta] = c_sta
        print(f"  {sta:<8} {len(triples):>3}  {c_sta:>+7.3f}  {sigma_sta:>6.3f}")

    if sta_calibs:
        # Overall residual after per-station correction
        res = []
        for A, r, mpva, sta in station_quads:
            if A and A > 0 and r > 0 and mpva is not None and sta in sta_calibs:
                res.append(mpva - (math.log10(A) + math.log10(r) + sta_calibs[sta]))
        sigma_tot = float(np.std(res)) if res else 0.0
        print(f"\n  σ общий после per-station коррекции = {sigma_tot:.3f} ед."
              f"  (N={len(res)} наблюдений)")
    return sta_calibs


def compute_ml_regression(A_counts, r_km, a, b, c, b_lin=0.0):
    """Parametric magnitude using global fitted coefficients."""
    if A_counts is None or A_counts <= 0.0 or r_km <= 0.0:
        return None
    return a * math.log10(A_counts) + b * math.log10(r_km) + b_lin * r_km + c


def compute_event_ml_regression(ev_idx, r_dict, amplitudes, a, b, c, b_lin=0.0):
    """Returns (Ml_reg, n_sta_amp) using global regression formula, median over stations."""
    ml_vals = []
    for sta, r_km in r_dict.items():
        A  = amplitudes.get((ev_idx, sta))
        ml = compute_ml_regression(A, r_km, a, b, c, b_lin)
        if ml is not None:
            ml_vals.append(ml)
    if not ml_vals:
        return None, 0
    return float(np.median(ml_vals)), len(ml_vals)


def compute_event_ml_per_sta(ev_idx, r_dict, amplitudes, sta_calibs, fallback_c=None,
                             b_lin=0.0):
    """
    Per-station calibrated magnitude: ml_sta = log10(A) + log10(r) + b_lin*r + c_sta.
    Stations without calibration are skipped unless fallback_c is provided.
    Returns (Ml_est, n_sta_amp).
    """
    ml_vals = []
    for sta, r_km in r_dict.items():
        c_sta = sta_calibs.get(sta, fallback_c)
        if c_sta is None:
            continue
        A = amplitudes.get((ev_idx, sta))
        if A and A > 0 and r_km > 0:
            ml_vals.append(math.log10(A) + math.log10(r_km) + b_lin * r_km + c_sta)
    if not ml_vals:
        return None, 0
    return float(np.median(ml_vals)), len(ml_vals)


# ── Distribution ──────────────────────────────────────────────────────────────

def print_ml_distribution(event_ml_list, calib_const):
    """event_ml_list: [(ev_idx, Ml_raw, n_sta_amp, n_sta_loc, residual)]."""
    vals   = sorted(ml + calib_const for _, ml, _, _, _ in event_ml_list if ml is not None)
    n_none = sum(1 for _, ml, _, _, _ in event_ml_list if ml is None)
    print(f"\n  Событий с оценкой Ml:  {len(vals)}")
    print(f"  Событий без оценки Ml: {n_none}  (будут сохранены при фильтрации)")
    if not vals:
        return
    n = len(vals)
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Распределение Ml_est (C={calib_const:+.2f}):")
    for p in pcts:
        idx = max(0, int(p / 100 * n) - 1)
        print(f"    P{p:2d}: {vals[idx]:.2f}")
    print(f"    min: {vals[0]:.2f}   max: {vals[-1]:.2f}")


# ── Filter XML ────────────────────────────────────────────────────────────────

def filter_xml(tree, event_ml_list, threshold, calib_const, out_path):
    """
    Deep-copy tree, remove events with Ml_est < threshold.
    Events without Ml estimate are kept (conservative).
    Returns (total, kept, removed, no_ml).
    """
    ml_by_idx = {ev_idx: ml for ev_idx, ml, _, _, _ in event_ml_list}

    tree2     = copy.deepcopy(tree)
    ns        = BED_NS
    root2     = tree2.getroot()
    ev_params = root2.find(f'.//{{{ns}}}eventParameters') or root2
    events2   = ev_params.findall(f'{{{ns}}}event')   # snapshot before any removes

    total = len(events2)
    kept = removed = no_ml = 0

    for i, ev_elem in enumerate(events2):
        ml_raw = ml_by_idx.get(i)
        if ml_raw is None:
            no_ml += 1
            kept  += 1
        elif ml_raw + calib_const >= threshold:
            kept += 1
        else:
            ev_params.remove(ev_elem)
            removed += 1

    tree2.write(out_path, encoding='unicode', xml_declaration=True)
    return total, kept, removed, no_ml


# ── Catalog cross-reference ───────────────────────────────────────────────────

DEFAULT_CATALOG = os.path.join(_ROOT, 'catalog.xlsx')
_VALIDATE_WINDOW = 15.0   # seconds, same default as validate_associator.py


def _load_catalog_xlsx(path, year=None, month=None):
    # Columns: 0=origin_time, 1=lat, 2=lon, 3=depth, 4=Ms, 5=I, 6=Kp, 7=MPVA_reg, 8=mb
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    events = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue
        origin_time, lat, lon, depth_km = row[0], row[1], row[2], row[3]
        if origin_time is None or lat is None or lon is None:
            continue
        if hasattr(origin_time, 'tzinfo') and origin_time.tzinfo:
            origin_time = origin_time.replace(tzinfo=None)
        if year  is not None and origin_time.year  != year:
            continue
        if month is not None and origin_time.month != month:
            continue

        def _f(v):
            try: return float(v)
            except (TypeError, ValueError): return None

        events.append({
            'origin_time': origin_time,
            'lat':         float(lat),
            'lon':         float(lon),
            'depth_km':    float(depth_km) if depth_km else 10.0,
            'ms':          _f(row[4] if len(row) > 4 else None),
            'kp':          _f(row[6] if len(row) > 6 else None),
            'mpva':        _f(row[7] if len(row) > 7 else None),
            'mb':          _f(row[8] if len(row) > 8 else None),
        })
    wb.close()
    return events


def print_catalog_ml(located, event_ml_list, sta_coords, catalog_path, year, month,
                     vp=VP_DEFAULT, window_sec=_VALIDATE_WINDOW, amplitudes=None):
    """
    Match detected events to catalog using travel-time correction (same logic as
    validate_associator.py). Print Ml for each matched event and suggest threshold.

    If amplitudes is provided, also collects per-station (A_counts, r_km, mpva) triples
    for regression fitting.

    Returns list of (A_counts, r_km, mpva) station triples for matched catalog events.
    """
    cat_events = _load_catalog_xlsx(catalog_path, year, month)
    if not cat_events:
        print("  Каталог пуст или не найден.")
        return []

    # Use min(picks_p) as origin time — same as XML origin/time used by validate_associator
    assoc_list = []
    for ev_idx, _, picks_p, _, _ in located:
        if not picks_p:
            continue
        assoc_list.append((ev_idx, min(picks_p.values())))

    ml_by_idx  = {ev_idx: ml for ev_idx, ml, _, _, _ in event_ml_list}
    loc_by_idx = {ev_idx: loc for ev_idx, _, _, _, loc in located}

    pairs_for_calib = []   # (Ml_raw, mpva) for calibration by MPVA_reg
    station_triples = []   # (A_counts, r_km, mpva) per station, for regression
    matched_mls = []

    hdr = f"  {'Origin time':<25} {'Ms':>4} {'MPVA':>5} {'mb':>4} {'Kp':>4}  {'Ml':>7}  {'dt,с':>6}"
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))

    for cat_ev in sorted(cat_events, key=lambda e: e['origin_time']):
        best_tt = None
        for _, (slat, slon) in sta_coords.items():
            epi  = haversine_km(cat_ev['lat'], cat_ev['lon'], slat, slon)
            hypo = math.sqrt(epi ** 2 + cat_ev['depth_km'] ** 2)
            tt   = hypo / vp
            if best_tt is None or tt < best_tt:
                best_tt = tt
        if best_tt is None:
            continue
        expected_t = cat_ev['origin_time'] + timedelta(seconds=best_tt)

        best_ev_idx, best_dt = None, None
        for ev_idx, origin_t in assoc_list:
            dt = abs((origin_t - expected_t).total_seconds())
            if dt <= window_sec and (best_dt is None or dt < best_dt):
                best_ev_idx, best_dt = ev_idx, dt

        def _fmt(v, w=4): return f"{v:{w}.1f}" if v is not None else ' ' * (w - 1) + '-'
        ms_s   = _fmt(cat_ev['ms'])
        mpva_s = _fmt(cat_ev['mpva'])
        mb_s   = _fmt(cat_ev['mb'])
        kp_s   = _fmt(cat_ev['kp'])

        if best_ev_idx is not None:
            ml = ml_by_idx.get(best_ev_idx)
            ml_s = f"{ml:.2f}" if ml is not None else "   N/A"
            print(f"  {str(cat_ev['origin_time']):<25} {ms_s} {mpva_s} {mb_s} {kp_s}  {ml_s:>7}  {best_dt:>6.1f}")
            if ml is not None:
                matched_mls.append(ml)
                if cat_ev['mpva'] is not None:
                    pairs_for_calib.append((ml, cat_ev['mpva']))
            # Collect per-station quads for calibration
            if amplitudes is not None and cat_ev['mpva'] is not None:
                loc = loc_by_idx.get(best_ev_idx)
                if loc is not None:
                    r_dict = loc[4]
                    for sta, r_km in r_dict.items():
                        A = amplitudes.get((best_ev_idx, sta))
                        if A and A > 0 and r_km > 0:
                            station_triples.append((A, r_km, cat_ev['mpva'], sta))
        else:
            print(f"  {str(cat_ev['origin_time']):<25} {ms_s} {mpva_s} {mb_s} {kp_s}  {'пропущено':>7}")

    if matched_mls:
        print(f"\n  Найдено с оценкой Ml: {len(matched_mls)}")
        print(f"  Ml:  min={min(matched_mls):.2f}  max={max(matched_mls):.2f}")

    if pairs_for_calib:
        n = len(pairs_for_calib)
        diffs = [mpva - ml for ml, mpva in pairs_for_calib]
        C_mpva = sum(diffs) / n
        residuals_c = [(d - C_mpva) ** 2 for d in diffs]
        sigma = (sum(residuals_c) / n) ** 0.5
        print(f"\n  Калибровка (b=1.0) по MPVA_reg ({n} пар):")
        print(f"    C = mean(MPVA_reg - Ml_raw) = {C_mpva:+.2f}  σ={sigma:.2f}")
        print(f"\n    Диапазон MPVA_reg каталога: "
              f"{min(m for _, m in pairs_for_calib):.1f} … {max(m for _, m in pairs_for_calib):.1f}")
        safe_thr_raw = min(ml for ml, _ in pairs_for_calib) - 0.1
        print(f"    Безопасный --ml-threshold (b=1.0) = {safe_thr_raw:.2f}")

    return station_triples


# ── Validate ──────────────────────────────────────────────────────────────────

def run_validate(assoc_path, year, month):
    script = os.path.join(_ROOT, 'core', 'validate_associator.py')
    subprocess.run([sys.executable, script,
                    '--assoc', assoc_path,
                    '--year', str(year), '--month', str(month)])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Оценка Ml и фильтрация associations.xml',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--assoc-in',     default=DEFAULT_ASSOC_IN)
    parser.add_argument('--assoc-out',    default=None,
                        help='Путь к выходному XML; при нескольких порогах '
                             'генерируется автоматически рядом с assoc-in')
    parser.add_argument('--waveforms',    default=DEFAULT_WAVEFORMS,
                        help='Корень data-in-memory/input/')
    parser.add_argument('--stations-dir', default=DEFAULT_STA_DIR)
    parser.add_argument('--ml-threshold', type=float, nargs='*', default=None,
                        help='Порог(и) Ml_est; без флага — только статистика')
    parser.add_argument('--calib-const',  type=float, default=0.0,
                        help='Калибровочная константа C: Ml_est = Ml_raw + C')
    parser.add_argument('--vp',           type=float, default=VP_DEFAULT,
                        help='Скорость P-волны, км/с')
    parser.add_argument('--depth',        type=float, default=DEPTH_DEFAULT,
                        help='Фиксированная глубина, км')
    parser.add_argument('--grid-step',    type=float, default=GRID_STEP_DEF,
                        help='Шаг сетки локации, градусы')
    parser.add_argument('--window-sec',   type=float, default=WIN_SEC,
                        help='P-волновое окно амплитуды, сек')
    parser.add_argument('--info',         action='store_true',
                        help='Только статистика Ml, без записи файлов')
    parser.add_argument('--catalog-ml',   action='store_true',
                        help='Показать Ml_raw для каждого каталожного события и предложить порог')
    parser.add_argument('--catalog',      default=DEFAULT_CATALOG,
                        help='Путь к catalog.xlsx')
    parser.add_argument('--cache-loc',    default=None,
                        help='CSV-кэш локаций: читается если есть, иначе создаётся')
    parser.add_argument('--cache-amp',    default=None,
                        help='CSV-кэш амплитуд: читается если есть, иначе создаётся')
    parser.add_argument('--metadata-dir', default=None,
                        help='Папка с FDSNStationXML (.xml). При указании применяется '
                             'remove_response() → амплитуды в м/с (delete old amps cache!)')
    parser.add_argument('--fit-regression', action='store_true',
                        help='Подобрать коэффициенты a,b,c регрессии MPVA=a·log10(A)+b·log10(r)+c '
                             'по каталогу и пересчитать Ml для всех событий')
    parser.add_argument('--ml-a',          type=float, default=None,
                        help='Коэффициент a (при log10(A)); задаётся вручную вместо --fit-regression')
    parser.add_argument('--ml-b',          type=float, default=None,
                        help='Коэффициент b (при log10(R)). Формула статьи: 1.024')
    parser.add_argument('--ml-b-lin',      type=float, default=0.0,
                        help='Линейный коэффициент при R (км): b_lin*R. '
                             'Формула Терско-Каспийского прогиба: 0.001648')
    parser.add_argument('--ml-c',          type=float, default=None,
                        help='Свободный член c. Формула статьи: -1.889')
    parser.add_argument('--wood-anderson', action='store_true',
                        help='Симулировать Wood-Anderson после remove_response(DISP). '
                             'Требует --metadata-dir. При смене флага удали кэш --cache-amp.')
    parser.add_argument('--validate',     action='store_true',
                        help='Запустить validate_associator на каждом выходном XML')
    parser.add_argument('--year',         type=int, default=2024)
    parser.add_argument('--month',        type=int, default=1)
    args = parser.parse_args()

    print(f"Входной XML:  {args.assoc_in}")
    print(f"Формы волн:   {args.waveforms}")
    print(f"C (калибр.):  {args.calib_const:+.2f}  b_lin={args.ml_b_lin}")
    wa_str = "ДА (требует --metadata-dir)" if args.wood_anderson else "нет"
    print(f"Wood-Anderson: {wa_str}")
    print(f"Vp={args.vp} км/с  глубина={args.depth} км  шаг={args.grid_step}°\n")

    # ── Шаг 0: парсинг XML ─────────────────────────────────────────────────────
    print("Парсинг XML...")
    tree          = ET.parse(args.assoc_in)
    events_parsed = parse_events_from_xml(tree)
    print(f"  Событий: {len(events_parsed)}")

    sta_coords = load_station_coords(args.stations_dir)
    print(f"  Станций: {len(sta_coords)}")

    inventory = None
    if args.metadata_dir:
        meta_dir = args.metadata_dir
        if not os.path.isdir(meta_dir):
            meta_dir = os.path.join(_ROOT, meta_dir)
        print(f"\nЗагрузка StationXML из {meta_dir}...")
        inventory = load_inventory(meta_dir)
        if inventory is not None:
            print("  remove_response() будет применён при извлечении амплитуд (м/с)")

    # ── Шаг 1: локация ─────────────────────────────────────────────────────────
    cache_loaded = False
    loc_cache    = {}
    if args.cache_loc and os.path.isfile(args.cache_loc):
        print(f"\nЗагрузка кэша локаций: {args.cache_loc}")
        loc_cache    = load_location_cache(args.cache_loc)
        cache_loaded = True
        print(f"  Загружено: {len(loc_cache)} записей")

    print("\nЛокация событий (grid search)...")
    located   = []   # list of (ev_idx, pub_id, picks_p, picks_s, loc_result)
    n_ok = n_fail = 0
    for i, (ev, pub_id, picks_p, picks_s) in enumerate(events_parsed):
        if cache_loaded and pub_id in loc_cache:
            loc = loc_cache[pub_id]
        else:
            loc = locate_event(picks_p, sta_coords, args.vp, args.depth, args.grid_step)
        located.append((i, pub_id, picks_p, picks_s, loc))
        if loc:
            n_ok += 1
        else:
            n_fail += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(events_parsed)}...")

    print(f"  Успешно: {n_ok}   без локации: {n_fail}")

    if args.cache_loc and not cache_loaded:
        print(f"\nСохранение кэша локаций: {args.cache_loc}")
        save_location_cache(located, args.cache_loc)

    # ── Шаг 2: амплитуды ──────────────────────────────────────────────────────
    if args.cache_amp and os.path.isfile(args.cache_amp):
        print(f"\nЗагрузка кэша амплитуд: {args.cache_amp}")
        amplitudes = load_amplitude_cache(args.cache_amp)
        n_amp = sum(1 for v in amplitudes.values() if v is not None)
        print(f"  Загружено пар: {len(amplitudes)}  с амплитудой: {n_amp}")
    else:
        print("\nПостроение индекса форм волн...")
        waveform_index = build_waveform_index(args.waveforms)
        n_files = sum(len(v) for v in waveform_index.values())
        print(f"  Файлов: {n_files}  станций: {len(waveform_index)}")

        if args.wood_anderson and inventory is None:
            print("  Предупреждение: --wood-anderson требует --metadata-dir, WA-симуляция пропущена")
        print("\nИзвлечение амплитуд P-волны...")
        amplitudes = extract_all_amplitudes(located, waveform_index, args.window_sec,
                                            inventory=inventory,
                                            wood_anderson=args.wood_anderson and inventory is not None)
        n_amp = sum(1 for v in amplitudes.values() if v is not None)
        print(f"  Пар (событие, станция) с амплитудой: {n_amp}")

        if args.cache_amp:
            print(f"\nСохранение кэша амплитуд: {args.cache_amp}")
            save_amplitude_cache(amplitudes, args.cache_amp)

    # ── Шаг 3: Ml ─────────────────────────────────────────────────────────────
    print("\nВычисление Ml...")
    event_ml_list = []   # (ev_idx, Ml_raw, n_sta_amp, n_sta_loc, residual)
    for ev_idx, pub_id, picks_p, picks_s, loc in located:
        n_sta_loc = len(loc[4]) if loc else 0
        residual  = loc[3] if loc else None
        r_dict    = loc[4] if loc else {}
        ml_raw, n_sta_amp = compute_event_ml(ev_idx, r_dict, amplitudes,
                                             b_lin=args.ml_b_lin)
        event_ml_list.append((ev_idx, ml_raw, n_sta_amp, n_sta_loc, residual))

    n_with_ml = sum(1 for _, ml, _, _, _ in event_ml_list if ml is not None)
    print(f"  Событий с оценкой Ml: {n_with_ml} / {len(event_ml_list)}")

    # ── Шаг 3b: регрессионный пересчёт Ml ────────────────────────────────────
    use_manual_coeffs = (args.ml_a is not None and args.ml_b is not None
                         and args.ml_c is not None)
    reg_coeffs = None   # (a, b, c) — заполнится если используем регрессию
    active_calib = args.calib_const

    if args.fit_regression or use_manual_coeffs:
        if use_manual_coeffs and not args.fit_regression:
            reg_coeffs = (args.ml_a, args.ml_b, args.ml_c)
            print(f"\nКоэффициенты заданы вручную: a={reg_coeffs[0]:.4f}  b={reg_coeffs[1]:.4f}  c={reg_coeffs[2]:.4f}")
        else:
            print(f"\nСопоставление с каталогом + регрессия: {args.catalog}")
            station_triples = print_catalog_ml(
                located, event_ml_list, sta_coords,
                args.catalog, args.year, args.month, args.vp,
                amplitudes=amplitudes,
            )
            if station_triples:
                # Show global regression for reference (will warn if b≈0)
                fit_regression(station_triples)
                # Per-station calibration is the primary approach
                sta_calibs = fit_per_station_calibration(station_triples)
                if sta_calibs:
                    reg_coeffs = sta_calibs   # dict, not tuple
                    if use_manual_coeffs:
                        reg_coeffs = (args.ml_a, args.ml_b, args.ml_c)
                        print(f"  Ручное переопределение: a={reg_coeffs[0]:.4f}  b={reg_coeffs[1]:.4f}  c={reg_coeffs[2]:.4f}")

        if reg_coeffs is not None:
            if isinstance(reg_coeffs, dict):
                # Per-station calibration path
                print(f"\nПересчёт Ml с per-station калибровкой ({len(reg_coeffs)} станций)...")
                event_ml_list = []
                for ev_idx, _, picks_p, picks_s, loc in located:
                    n_sta_loc = len(loc[4]) if loc else 0
                    residual  = loc[3] if loc else None
                    r_dict    = loc[4] if loc else {}
                    ml_ps, n_sta_amp = compute_event_ml_per_sta(
                        ev_idx, r_dict, amplitudes, reg_coeffs, b_lin=args.ml_b_lin
                    )
                    event_ml_list.append((ev_idx, ml_ps, n_sta_amp, n_sta_loc, residual))
            else:
                # Global regression path (manual --ml-a/b/c)
                a_r, b_r, c_r = reg_coeffs
                print(f"\nПересчёт Ml (a={a_r:.4f}, b={b_r:.4f}, b_lin={args.ml_b_lin:.6f}, c={c_r:.4f})...")
                event_ml_list = []
                for ev_idx, _, picks_p, picks_s, loc in located:
                    n_sta_loc = len(loc[4]) if loc else 0
                    residual  = loc[3] if loc else None
                    r_dict    = loc[4] if loc else {}
                    ml_reg, n_sta_amp = compute_event_ml_regression(
                        ev_idx, r_dict, amplitudes, a_r, b_r, c_r, args.ml_b_lin
                    )
                    event_ml_list.append((ev_idx, ml_reg, n_sta_amp, n_sta_loc, residual))
            active_calib = 0.0   # c_sta already embedded
            n_with_ml = sum(1 for _, ml, _, _, _ in event_ml_list if ml is not None)
            print(f"  Событий с оценкой Ml: {n_with_ml} / {len(event_ml_list)}")
            print_ml_distribution(event_ml_list, active_calib)

    # Стандартный путь без регрессии
    else:
        print_ml_distribution(event_ml_list, active_calib)

        if args.catalog_ml:
            print(f"\nСопоставление с каталогом: {args.catalog}")
            print_catalog_ml(located, event_ml_list, sta_coords,
                             args.catalog, args.year, args.month, args.vp)

    thresholds = args.ml_threshold or []
    if args.info or not thresholds:
        return

    # ── Шаг 4: фильтрация ─────────────────────────────────────────────────────
    suffix = '_reg' if reg_coeffs is not None else ''
    base_dir = os.path.dirname(args.assoc_in)
    for thr in thresholds:
        if args.assoc_out and len(thresholds) == 1:
            out_path = args.assoc_out
        else:
            tag      = f"{thr:.1f}".replace('-', 'm').replace('.', 'p')
            out_path = os.path.join(base_dir, f'associations_ml{tag}{suffix}.xml')

        total, kept, removed, no_ml = filter_xml(
            tree, event_ml_list, thr, active_calib, out_path
        )
        print(f"\n=== ml_threshold={thr:.1f} → {os.path.basename(out_path)} ===")
        print(f"  Сохранено:       {kept} / {total}  (Ml_est >= {thr:.1f})")
        print(f"  Удалено:         {removed}")
        print(f"  Без оценки Ml:   {no_ml}  (сохранены)")

        if args.validate:
            print()
            run_validate(out_path, args.year, args.month)


if __name__ == '__main__':
    main()
