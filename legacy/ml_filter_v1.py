"""
ml_filter.py — фильтрация associations.xml по локальной магнитуде ML.

Формула (Дягилев et al. 2023, Терско-Каспийский прогиб, формула 5б):
    ML = lg(A) + 1.024·lg(R) + 0.001648·R − 1.889 + S

A — максимальное смещение земной поверхности в нм (нанометрах).
    Скрипт вычисляет A через remove_response(output='DISP') → метры,
    затем умножает на 1e9 (м→нм) перед log10.
R — гипоцентральное расстояние (км),
S — станционная поправка из CSV (0 если не задана).

Запуск:
    python core/ml_filter.py --info
    python core/ml_filter.py --ml-threshold 0.5 --validate
    python core/ml_filter.py --ml-threshold 0.5 --cache-loc locs.csv --cache-amp amps.csv
    python core/ml_filter.py --ml-threshold 0.5 --wood-anderson --metadata-dir metadata
    python core/ml_filter.py --ml-threshold 0.5 --sta-corrections sta_corr.csv --validate
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

# ── Пути по умолчанию ─────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ASSOC_IN  = os.path.join(_ROOT, 'data-in-memory', 'gpu_splimit_45_march','assoc_output_lim', 'associations.xml')
DEFAULT_ASSOC_OUT = None   # None → рядом с assoc-in, имя associations_ml<thr>.xml
DEFAULT_WAVEFORMS = os.path.join(_ROOT, 'geofiles')
DEFAULT_STA_DIR   = os.path.join(_ROOT, 'json')
DEFAULT_CACHE_LOC = os.path.join(_ROOT, 'locs_new.csv')
DEFAULT_CACHE_AMP = os.path.join(_ROOT, 'amps_disp.csv')
DEFAULT_METADATA  = os.path.join(_ROOT, 'metadata')
DEFAULT_THRESHOLD = 1.0
DEFAULT_YEAR      = 2024
DEFAULT_MONTH     = 4

BED_NS   = 'http://quakeml.org/xmlns/bed/1.2'
QUAKE_NS = 'http://quakeml.org/xmlns/quakeml/1.2'

ET.register_namespace('',  BED_NS)
ET.register_namespace('q', QUAKE_NS)

# ── Константы формулы (Терско-Каспийский прогиб) ─────────────────────────────

ML_A         = 1.0       # коэффициент при lg(A)
ML_B_LOG     = 1.024     # коэффициент при lg(R), геометрическое расхождение
ML_B_LIN     = 0.001648  # коэффициент при R (км), неупругое поглощение
ML_C         = -1.889    # калибровочная константа; A ожидается в нм (Дягилев et al. 2023, 5б)
AMP_SCALE_NM = 1e9       # перевод м→нм при remove_response(output='DISP')

VP        = 6.0      # скорость P-волны, км/с
DEPTH     = 10.0     # фиксированная глубина, км
WIN_SEC   = 3.0      # длина окна амплитуды P-волны, с
GRID_STEP = 0.05     # шаг сетки локации, градусы

# PAZ Wood-Anderson (Richter 1935; T0=0.8s, h=0.8, V=2800)
_PAZ_WA = {
    'poles': [(-6.283185307 + 4.712388980j),
              (-6.283185307 - 4.712388980j)],
    'zeros': [0j, 0j],
    'gain': 1.0,
    'sensitivity': 2800.0,
}


# ── Вспомогательные функции ───────────────────────────────────────────────────

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
    coords = {}
    for fpath in glob.glob(os.path.join(sta_dir, 'station_*.json')):
        name = os.path.basename(fpath).replace('station_', '').replace('.json', '')
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)
        key = list(data.keys())[0]
        c = data[key]['coords']
        coords[name] = (float(c[0]), float(c[1]))
    return coords


def load_sta_corrections(path):
    """CSV с колонками station,S. Возвращает dict {station: float}."""
    corr = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sta = row.get('station', '').strip()
            val = row.get('S', row.get('correction', '')).strip()
            if sta and val:
                try:
                    corr[sta] = float(val)
                except ValueError:
                    pass
    return corr


# ── Парсинг XML ───────────────────────────────────────────────────────────────

def parse_xml(tree):
    """Возвращает list of (ev_elem, pub_id, picks_p, picks_s).
    picks_p/picks_s: {station: datetime}"""
    ns = BED_NS
    root = tree.getroot()
    result = []
    for ev in root.findall(f'.//{{{ns}}}event'):
        pub_id = ev.get('publicID', '')
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
        result.append((ev, pub_id, picks_p, picks_s))
    return result


# ── Локация событий (grid search по P-временам) ───────────────────────────────

def locate_event(picks_p, sta_coords):
    """Возвращает (lat, lon, origin_t, residual, r_dict) или None."""
    stas = [s for s in picks_p if s in sta_coords]
    if len(stas) < 2:
        return None

    sta_lats = np.array([sta_coords[s][0] for s in stas])
    sta_lons = np.array([sta_coords[s][1] for s in stas])
    t_ref    = min(picks_p[s] for s in stas)
    T_obs    = np.array([(picks_p[s] - t_ref).total_seconds() for s in stas])

    lat_min = sta_lats.min() - 0.5;  lat_max = sta_lats.max() + 0.5
    lon_min = sta_lons.min() - 0.5;  lon_max = sta_lons.max() + 0.5
    lat_grid = np.arange(lat_min, lat_max + GRID_STEP * 0.5, GRID_STEP)
    lon_grid = np.arange(lon_min, lon_max + GRID_STEP * 0.5, GRID_STEP)

    R    = 6371.0
    phi1 = np.radians(lat_grid)[:, None, None]
    phi2 = np.radians(sta_lats)[None, None, :]
    lam1 = np.radians(lon_grid)[None, :, None]
    lam2 = np.radians(sta_lons)[None, None, :]
    a    = (np.sin((phi2 - phi1) / 2) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin((lam2 - lam1) / 2) ** 2)
    epi_km = R * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    r_km   = np.sqrt(epi_km ** 2 + DEPTH ** 2)
    tt     = r_km / VP

    T_origin  = np.mean(T_obs[None, None, :] - tt, axis=2, keepdims=True)
    residuals = np.sum((T_obs[None, None, :] - T_origin - tt) ** 2, axis=2)

    i_lat, i_lon = np.unravel_index(np.argmin(residuals), residuals.shape)
    origin_t = t_ref + timedelta(seconds=float(T_origin[i_lat, i_lon, 0]))
    r_best   = r_km[i_lat, i_lon, :]
    r_dict   = {s: float(r_best[k]) for k, s in enumerate(stas)}

    return (float(lat_grid[i_lat]), float(lon_grid[i_lon]),
            origin_t, float(residuals[i_lat, i_lon]), r_dict)


# ── Кэш локаций ──────────────────────────────────────────────────────────────

def save_loc_cache(located, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ev_idx', 'pub_id', 'lat', 'lon', 'origin_t', 'residual', 'r_km_json'])
        for ev_idx, pub_id, _, _, loc in located:
            if loc is None:
                w.writerow([ev_idx, pub_id, '', '', '', '', ''])
            else:
                lat, lon, origin_t, res, r_dict = loc
                w.writerow([ev_idx, pub_id, lat, lon,
                            origin_t.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                            res, json.dumps(r_dict)])


def load_loc_cache(path):
    cache = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if not row.get('lat'):
                cache[row['pub_id']] = None
                continue
            cache[row['pub_id']] = (
                float(row['lat']), float(row['lon']),
                parse_time(row['origin_t']),
                float(row['residual']),
                json.loads(row['r_km_json']),
            )
    return cache


# ── Кэш амплитуд ─────────────────────────────────────────────────────────────

def save_amp_cache(amplitudes, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ev_idx', 'sta', 'A'])
        for (ev_idx, sta), A in amplitudes.items():
            w.writerow([ev_idx, sta, '' if A is None else A])


def load_amp_cache(path):
    cache = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            cache[(int(row['ev_idx']), row['sta'])] = float(row['A']) if row['A'] else None
    return cache


# ── Индекс и поиск форм волн ──────────────────────────────────────────────────

_COMP_PRIORITY = ('Z', 'E', 'N')


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
    """dict: station -> list of (t_start, t_end, fpath), выбирается компонента Z > E > N."""
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


def find_waveform_file(index, sta, p_time):
    for t_start, t_end, fpath in index.get(sta, []):
        if t_start <= p_time < t_end:
            return fpath
    return None


# ── Извлечение амплитуд P-волны ───────────────────────────────────────────────

def extract_amplitudes(located, waveform_index, inventory=None, wood_anderson=False):
    """
    Читает каждый waveform-файл однократно, группируя пары (событие, станция) по файлу.
    Возвращает dict: (ev_idx, sta) -> амплитуда (float или None).
    """
    from obspy import read as obspy_read, UTCDateTime

    groups = defaultdict(list)
    for ev_idx, _, picks_p, picks_s, loc in located:
        if loc is None:
            continue
        r_dict = loc[4]
        for sta, p_time in picks_p.items():
            if sta not in r_dict:
                continue
            fpath = find_waveform_file(waveform_index, sta, p_time)
            if fpath is None:
                continue
            groups[fpath].append((ev_idx, sta, p_time, picks_s.get(sta)))

    amplitudes = {}
    n = len(groups)
    for i, (fpath, pairs) in enumerate(groups.items()):
        if i % 20 == 0:
            print(f"  Амплитуды: {i}/{n} файлов...", end='\r', flush=True)
        try:
            st = obspy_read(fpath)
            if not st:
                continue
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
                    if wood_anderson:
                        tr.simulate(paz_simulate=_PAZ_WA, paz_remove=None)
                except Exception:
                    tr.filter('highpass', freq=1.0)
            else:
                tr.filter('highpass', freq=1.0)

            for ev_idx, sta, p_time, s_time in pairs:
                try:
                    p_utc   = UTCDateTime(p_time)
                    end_utc = (UTCDateTime(s_time) - 0.5
                               if s_time and (s_time - p_time).total_seconds() > 3.5
                               else p_utc + WIN_SEC)
                    tr_sig  = tr.slice(p_utc, end_utc)
                    if tr_sig is None or len(tr_sig.data) == 0:
                        amplitudes[(ev_idx, sta)] = None
                        continue
                    A = float(np.max(np.abs(tr_sig.data)))
                    amplitudes[(ev_idx, sta)] = A if A > 0.0 else None
                except Exception:
                    amplitudes[(ev_idx, sta)] = None
            del tr, st
        except Exception:
            pass
        gc.collect()

    print(f"  Амплитуды: {n}/{n} — готово.         ")
    return amplitudes


# ── Вычисление ML по формуле статьи ──────────────────────────────────────────

def compute_event_ml(ev_idx, r_dict, amplitudes, sta_corr, amp_scale=AMP_SCALE_NM):
    """
    ML = lg(A_nm) + 1.024·lg(R) + 0.001648·R − 1.889 + S
    A из кэша — в метрах (DISP); amp_scale=1e9 переводит в нм перед log10.
    Медиана по всем станциям. Возвращает (ML, n_sta).
    """
    ml_vals = []
    for sta, r_km in r_dict.items():
        A = amplitudes.get((ev_idx, sta))
        if A is None or A <= 0.0 or r_km <= 0.0:
            continue
        A_nm = A * amp_scale
        if A_nm <= 0.0:
            continue
        S  = sta_corr.get(sta, 0.0)
        ml = (ML_A * math.log10(A_nm)
              + ML_B_LOG * math.log10(r_km)
              + ML_B_LIN * r_km
              + ML_C
              + S)
        ml_vals.append(ml)
    if not ml_vals:
        return None, 0
    return float(np.median(ml_vals)), len(ml_vals)


# ── Фильтрация XML ────────────────────────────────────────────────────────────

def filter_xml(tree, ml_by_idx, threshold, out_path):
    """Удаляет события с ML < threshold. События без оценки ML сохраняются."""
    tree2     = copy.deepcopy(tree)
    ns        = BED_NS
    root2     = tree2.getroot()
    ev_params = root2.find(f'.//{{{ns}}}eventParameters') or root2
    events2   = ev_params.findall(f'{{{ns}}}event')

    total = len(events2)
    kept = removed = no_ml = 0
    for i, ev_elem in enumerate(events2):
        ml = ml_by_idx.get(i)
        if ml is None:
            no_ml += 1
            kept  += 1
        elif ml >= threshold:
            kept += 1
        else:
            ev_params.remove(ev_elem)
            removed += 1

    tree2.write(out_path, encoding='unicode', xml_declaration=True)
    return total, kept, removed, no_ml


# ── Статистика распределения ML ───────────────────────────────────────────────

def print_distribution(ml_values):
    vals   = sorted(v for v in ml_values if v is not None)
    n_none = sum(1 for v in ml_values if v is None)
    print(f"\n  Событий с оценкой ML:  {len(vals)}")
    print(f"  Событий без оценки ML: {n_none}  (сохраняются при фильтрации)")
    if not vals:
        return
    n    = len(vals)
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Распределение ML (a={ML_A}, b_log={ML_B_LOG}, b_lin={ML_B_LIN}, c={ML_C}):")
    for p in pcts:
        print(f"    P{p:2d}: {vals[max(0, int(p / 100 * n) - 1)]:.2f}")
    print(f"    min: {vals[0]:.2f}   max: {vals[-1]:.2f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Фильтрация associations.xml по ML (формула Терско-Каспийского прогиба)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--assoc-in',       default=DEFAULT_ASSOC_IN,
                        help='Входной XML ассоциатора')
    parser.add_argument('--assoc-out',      default=DEFAULT_ASSOC_OUT,
                        help='Выходной XML. По умолчанию: рядом с assoc-in, '
                             'имя: associations_ml<threshold>.xml')
    parser.add_argument('--waveforms',      default=DEFAULT_WAVEFORMS,
                        help='Корень папки с формами волн (data-in-memory/input/)')
    parser.add_argument('--stations-dir',   default=DEFAULT_STA_DIR,
                        help='Папка с json/station_*.json')
    parser.add_argument('--sta-corrections', default=None,
                        help='CSV с колонками station,S — станционные поправки (S-term)')
    parser.add_argument('--ml-threshold',   type=float, default=DEFAULT_THRESHOLD,
                        help='Порог ML: оставить события с ML >= X. '
                             'Без флага — только статистика')
    parser.add_argument('--info',           action='store_true',
                        help='Только показать распределение ML, без записи файлов')
    parser.add_argument('--cache-loc',      default=DEFAULT_CACHE_LOC,
                        help='CSV-кэш локаций. Создаётся при первом запуске, '
                             'потом читается — экономит ~10 мин на 20k событий')
    parser.add_argument('--cache-amp',      default=DEFAULT_CACHE_AMP,
                        help='CSV-кэш амплитуд. При смене --wood-anderson '
                             'нужен отдельный файл (другие единицы!)')
    parser.add_argument('--metadata-dir',   default=DEFAULT_METADATA,
                        help='Папка с FDSNStationXML (.xml) для remove_response')
    parser.add_argument('--wood-anderson',  action='store_true', default=False,
                        help='Симулировать Wood-Anderson после remove_response(DISP). '
                             'Не рекомендуется — формула 5б ожидает смещение грунта '
                             'в нм, не WA-амплитуду.')
    parser.add_argument('--validate',       action='store_true',
                        help='Запустить validate_associator.py на выходном XML')
    parser.add_argument('--year',           type=int, default=DEFAULT_YEAR)
    parser.add_argument('--month',          type=int, default=DEFAULT_MONTH)
    args = parser.parse_args()

    # ── Заголовок ─────────────────────────────────────────────────────────────
    print(f"Входной XML:   {args.assoc_in}")
    print(f"Формы волн:    {args.waveforms}")
    wa_s = "ДА (требует --metadata-dir)" if args.wood_anderson else "нет"
    print(f"Wood-Anderson: {wa_s}")
    print(f"Формула:       ML = lg(A) + {ML_B_LOG}·lg(R) + {ML_B_LIN}·R + ({ML_C}) + S\n")

    if args.wood_anderson and args.metadata_dir is None:
        print("ОШИБКА: --wood-anderson требует --metadata-dir")
        sys.exit(1)

    # ── Парсинг XML ───────────────────────────────────────────────────────────
    print("Парсинг XML...")
    tree          = ET.parse(args.assoc_in)
    events_parsed = parse_xml(tree)
    print(f"  Событий: {len(events_parsed)}")

    sta_coords = load_station_coords(args.stations_dir)
    print(f"  Станций: {len(sta_coords)}")

    sta_corr = {}
    if args.sta_corrections:
        sta_corr = load_sta_corrections(args.sta_corrections)
        print(f"  Станционных поправок: {len(sta_corr)}")

    # ── StationXML / inventory ─────────────────────────────────────────────────
    inventory = None
    if args.metadata_dir:
        from obspy import read_inventory
        meta_dir = args.metadata_dir
        if not os.path.isdir(meta_dir):
            meta_dir = os.path.join(_ROOT, meta_dir)
        print(f"\nЗагрузка StationXML из {meta_dir}...")
        for fpath in sorted(glob.glob(os.path.join(meta_dir, '*.xml'))):
            try:
                inv = read_inventory(fpath)
                inventory = inv if inventory is None else inventory + inv
            except Exception:
                pass
        if inventory is None:
            print("  Предупреждение: StationXML не найдены, используется highpass")
        else:
            n_sta = sum(len(net) for net in inventory.networks)
            print(f"  Загружено: {n_sta} станций")

    if args.wood_anderson and inventory is None:
        print("  Предупреждение: нет StationXML → WA пропущена, используется highpass")

    # ── Локация ───────────────────────────────────────────────────────────────
    loc_cache    = {}
    cache_loaded = False
    if args.cache_loc and os.path.isfile(args.cache_loc):
        print(f"\nЗагрузка кэша локаций: {args.cache_loc}")
        loc_cache    = load_loc_cache(args.cache_loc)
        cache_loaded = True
        print(f"  Загружено: {len(loc_cache)} записей")

    print("\nЛокация событий (grid search по P-временам)...")
    located = []
    n_ok = n_fail = 0
    for i, (_, pub_id, picks_p, picks_s) in enumerate(events_parsed):
        if cache_loaded:
            loc = loc_cache.get(pub_id)
        else:
            loc = locate_event(picks_p, sta_coords)
        located.append((i, pub_id, picks_p, picks_s, loc))
        if loc:
            n_ok += 1
        else:
            n_fail += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(events_parsed)}...")
    print(f"  Успешно: {n_ok}   без локации: {n_fail}")

    if args.cache_loc and not cache_loaded:
        save_loc_cache(located, args.cache_loc)
        print(f"  Кэш сохранён: {args.cache_loc}")

    # ── Амплитуды ─────────────────────────────────────────────────────────────
    if args.cache_amp and os.path.isfile(args.cache_amp):
        print(f"\nЗагрузка кэша амплитуд: {args.cache_amp}")
        amplitudes = load_amp_cache(args.cache_amp)
        n_amp = sum(1 for v in amplitudes.values() if v is not None)
        print(f"  Загружено пар: {len(amplitudes)}  с амплитудой: {n_amp}")
    else:
        print("\nПостроение индекса форм волн...")
        waveform_index = build_waveform_index(args.waveforms)
        n_files = sum(len(v) for v in waveform_index.values())
        print(f"  Файлов: {n_files}  станций: {len(waveform_index)}")

        print("\nИзвлечение амплитуд P-волны...")
        use_wa = args.wood_anderson and inventory is not None
        amplitudes = extract_amplitudes(located, waveform_index,
                                        inventory=inventory,
                                        wood_anderson=use_wa)
        n_amp = sum(1 for v in amplitudes.values() if v is not None)
        print(f"  Пар с амплитудой: {n_amp}")

        if args.cache_amp:
            save_amp_cache(amplitudes, args.cache_amp)
            print(f"  Кэш сохранён: {args.cache_amp}")

    # ── Вычисление ML ─────────────────────────────────────────────────────────
    print("\nВычисление ML...")
    ml_by_idx = {}
    for ev_idx, pub_id, picks_p, picks_s, loc in located:
        r_dict = loc[4] if loc else {}
        ml, _ = compute_event_ml(ev_idx, r_dict, amplitudes, sta_corr)
        ml_by_idx[ev_idx] = ml

    print_distribution(list(ml_by_idx.values()))

    if args.info or args.ml_threshold is None:
        return

    # ── Фильтрация и запись XML ───────────────────────────────────────────────
    thr = args.ml_threshold
    if args.assoc_out:
        out_path = args.assoc_out
    else:
        tag      = f"{thr:.1f}".replace('-', 'm').replace('.', 'p')
        out_path = os.path.join(os.path.dirname(args.assoc_in),
                                f'associations_ml{tag}.xml')

    total, kept, removed, no_ml = filter_xml(tree, ml_by_idx, thr, out_path)

    print(f"\n{'=' * 60}")
    print(f"Порог ML >= {thr}")
    print(f"  Входных событий:  {total}")
    print(f"  Сохранено:        {kept}  (ML >= {thr}  +  {no_ml} без оценки)")
    print(f"  Удалено:          {removed}")
    print(f"  Выходной файл:    {out_path}")

    if args.validate:
        print(f"\n{'─' * 60}")
        print("Запуск validate_associator.py...")
        script = os.path.join(_ROOT, 'core', 'validate_associator.py')
        subprocess.run([sys.executable, script,
                        '--assoc', out_path,
                        '--year',  str(args.year),
                        '--month', str(args.month)])


if __name__ == '__main__':
    main()
