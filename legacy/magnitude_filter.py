"""
Фильтрует associations.xml по прокси-магнитуде с поправкой на расстояние.

  corrected_snr_i = p_snr_i × (r_i / r_ref)

где r_i — расстояние от гипоцентра (origin lat/lon из XML) до станции,
r_ref=100 км — нормировочное расстояние.

Это убирает дистанционную зависимость p_snr: событие Ms=2.0 на 280 км
и событие Ms=2.0 на 80 км дают сопоставимый corrected_snr.

Запуск:
    python core/magnitude_filter.py --info          # только перцентили, без фильтрации
    python core/magnitude_filter.py --min-snr 5
    python core/magnitude_filter.py --min-snr 3 5 8 --validate
    python core/magnitude_filter.py --min-snr 5 --no-correct  # без поправки на расстояние
"""
import argparse
import bisect
import copy
import csv
import glob
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ASSOC_IN   = os.path.join(_ROOT, 'data-in-memory', 'association_gpu_100_150', 'associations.xml')
DEFAULT_STAGING    = os.path.join(_ROOT, 'data-in-memory', 'assoc_input_gpu_100_150')
DEFAULT_STA_DIR    = os.path.join(_ROOT, 'json')

BED_NS   = 'http://quakeml.org/xmlns/bed/1.2'
QUAKE_NS = 'http://quakeml.org/xmlns/quakeml/1.2'

ET.register_namespace('',  BED_NS)
ET.register_namespace('q', QUAKE_NS)


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
    """Возвращает dict: station -> (lat, lon)."""
    coords = {}
    for fpath in glob.glob(os.path.join(sta_dir, 'station_*.json')):
        name = os.path.basename(fpath).replace('station_', '').replace('.json', '')
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)
        key = list(data.keys())[0]
        c = data[key]['coords']
        coords[name] = (float(c[0]), float(c[1]))
    return coords


def load_staging(staging_dir):
    """Возвращает dict: station -> (sorted_timestamps_float, snr_list)."""
    result = {}
    for sta in os.listdir(staging_dir):
        csv_path = os.path.join(staging_dir, sta, 'X_prediction_results.csv')
        if not os.path.isfile(csv_path):
            continue
        picks = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                p_str = row.get('p_arrival_time', '').strip()
                if not p_str or p_str.lower() == 'none':
                    continue
                t = parse_time(p_str)
                if t is None:
                    continue
                try:
                    snr = float(row.get('p_snr', 0) or 0)
                except ValueError:
                    snr = 0.0
                picks.append((t.timestamp(), snr))
        picks.sort(key=lambda x: x[0])
        result[sta] = ([p[0] for p in picks], [p[1] for p in picks])
    return result


def best_snr(sta_picks, station, ref_time, tol_sec):
    """O(log n) поиск лучшего p_snr для станции в пределах tol_sec."""
    entry = sta_picks.get(station)
    if entry is None:
        return None
    timestamps, snrs = entry
    ts = ref_time.timestamp()
    lo = bisect.bisect_left(timestamps, ts - tol_sec)
    hi = bisect.bisect_right(timestamps, ts + tol_sec)
    if lo >= hi:
        return None
    return max(snrs[lo:hi])


# ── Вычисление corrected_snr ──────────────────────────────────────────────────

def compute_event_snrs(tree, sta_picks, sta_coords, tol_sec, r_ref, r_floor, correct_distance):
    """
    Однократный проход по всем событиям XML.
    Возвращает list of (element, ev_params_ref, max_corrected_snr_or_None).
    """
    ns = BED_NS
    root = tree.getroot()
    ev_params = root.find(f'.//{{{ns}}}eventParameters') or root
    events = ev_params.findall(f'{{{ns}}}event')

    results = []
    n_no_loc = 0

    for ev in events:
        origin = ev.find(f'{{{ns}}}origin')
        origin_t = origin_lat = origin_lon = None

        if origin is not None:
            t_el  = origin.find(f'{{{ns}}}time/{{{ns}}}value')
            la_el = origin.find(f'{{{ns}}}latitude/{{{ns}}}value')
            lo_el = origin.find(f'{{{ns}}}longitude/{{{ns}}}value')
            if t_el  is not None: origin_t   = parse_time(t_el.text)
            if la_el is not None:
                try: origin_lat = float(la_el.text)
                except (ValueError, TypeError): pass
            if lo_el is not None:
                try: origin_lon = float(lo_el.text)
                except (ValueError, TypeError): pass

        has_loc = (origin_lat is not None and origin_lon is not None)
        if not has_loc:
            n_no_loc += 1

        snr_values = []
        for pick in ev.findall(f'{{{ns}}}pick'):
            wf = pick.find(f'{{{ns}}}waveformID')
            if wf is None:
                continue
            sta = wf.get('stationCode', '').strip()
            if not sta:
                continue

            pt_el = pick.find(f'{{{ns}}}time/{{{ns}}}value')
            ref_t = parse_time(pt_el.text) if pt_el is not None else origin_t
            if ref_t is None:
                continue

            snr = best_snr(sta_picks, sta, ref_t, tol_sec)
            if snr is None:
                continue

            if correct_distance and has_loc and sta in sta_coords:
                r = haversine_km(origin_lat, origin_lon, *sta_coords[sta])
                r = max(r, r_floor)
                corrected = snr * (r / r_ref)
            else:
                corrected = snr

            snr_values.append(corrected)

        max_snr = max(snr_values) if snr_values else None
        results.append((ev, ev_params, max_snr))

    if n_no_loc > 0:
        print(f"  Предупреждение: {n_no_loc} событий без координат в XML → поправка на расстояние не применена")

    return results


def print_distribution(event_snrs):
    """Печатает перцентили corrected_snr для выбора порога."""
    vals = sorted(v for _, _, v in event_snrs if v is not None)
    if not vals:
        print("  Нет данных SNR.")
        return
    n = len(vals)
    pcts = [10, 25, 50, 75, 90, 95, 99]
    print(f"  Распределение max_corrected_snr ({n} событий с данными):")
    for p in pcts:
        idx = max(0, int(p / 100 * n) - 1)
        print(f"    P{p:2d}: {vals[idx]:.1f}")
    print(f"    max: {vals[-1]:.1f}")


# ── Применение порога и запись XML ────────────────────────────────────────────

def apply_threshold(tree, event_snrs, min_snr, out_path):
    tree2 = copy.deepcopy(tree)
    ns = BED_NS
    root2 = tree2.getroot()
    ev_params2 = root2.find(f'.//{{{ns}}}eventParameters') or root2
    events2 = ev_params2.findall(f'{{{ns}}}event')

    kept = removed = no_data = 0
    for i, (_, _, max_snr) in enumerate(event_snrs):
        if max_snr is None:
            no_data += 1
            kept += 1
        elif max_snr >= min_snr:
            kept += 1
        else:
            ev_params2.remove(events2[i])
            removed += 1

    tree2.write(out_path, encoding='unicode', xml_declaration=True)
    return len(event_snrs), kept, removed, no_data


def run_validate(assoc_path, year, month):
    script = os.path.join(_ROOT, 'core', 'validate_associator.py')
    subprocess.run([sys.executable, script,
                    '--assoc', assoc_path,
                    '--year', str(year), '--month', str(month)])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Фильтр ассоциаций по distance-corrected p_snr')
    parser.add_argument('--min-snr',      type=float, nargs='*', default=None,
                        help='Порог(и) corrected_snr; если не указан — только статистика')
    parser.add_argument('--info',         action='store_true',
                        help='Только вывести распределение SNR, без фильтрации')
    parser.add_argument('--no-correct',   action='store_true',
                        help='Отключить поправку на расстояние (использовать сырой p_snr)')
    parser.add_argument('--assoc-in',     default=DEFAULT_ASSOC_IN)
    parser.add_argument('--assoc-out',    default=None)
    parser.add_argument('--staging',      default=DEFAULT_STAGING)
    parser.add_argument('--stations-dir', default=DEFAULT_STA_DIR)
    parser.add_argument('--tol',          type=float, default=30.0,
                        help='Допуск совпадения пика по времени, сек (default 30)')
    parser.add_argument('--r-ref',        type=float, default=100.0,
                        help='Нормировочное расстояние км (default 100)')
    parser.add_argument('--r-floor',      type=float, default=5.0,
                        help='Минимальное расстояние км (default 5)')
    parser.add_argument('--validate',     action='store_true')
    parser.add_argument('--year',         type=int, default=2024)
    parser.add_argument('--month',        type=int, default=1)
    args = parser.parse_args()

    correct = not args.no_correct
    mode_str = f"distance-corrected (r_ref={args.r_ref} км)" if correct else "сырой p_snr"
    print(f"Входной XML: {args.assoc_in}")
    print(f"Стейджинг:   {args.staging}")
    print(f"Станции:     {args.stations_dir}")
    print(f"Режим:       {mode_str}")
    print(f"tol:         ±{args.tol}с\n")

    print("Загрузка стейджинг-CSV...")
    sta_picks = load_staging(args.staging)
    print(f"  Загружено станций: {len(sta_picks)}")

    sta_coords = {}
    if correct:
        sta_coords = load_station_coords(args.stations_dir)
        print(f"  Координаты станций: {len(sta_coords)}")

    print("\nВычисление corrected_snr для всех событий...")
    tree = ET.parse(args.assoc_in)
    event_snrs = compute_event_snrs(
        tree, sta_picks, sta_coords,
        args.tol, args.r_ref, args.r_floor, correct
    )
    print(f"  Событий обработано: {len(event_snrs)}\n")

    print_distribution(event_snrs)

    thresholds = args.min_snr or []
    if args.info or not thresholds:
        return

    base_dir = os.path.dirname(args.assoc_in)
    suffix = "" if correct else "_raw"

    for min_snr in thresholds:
        if args.assoc_out and len(thresholds) == 1:
            out_path = args.assoc_out
        else:
            tag = str(int(min_snr)) if min_snr == int(min_snr) else str(min_snr)
            out_path = os.path.join(base_dir, f'associations_snr{tag}{suffix}.xml')

        total, kept, removed, no_data = apply_threshold(tree, event_snrs, min_snr, out_path)

        print(f"\n=== min_snr={min_snr} → {os.path.basename(out_path)} ===")
        print(f"  Сохранено:      {kept}  / {total}  (max_corrected_snr >= {min_snr})")
        print(f"  Удалено:        {removed}")
        print(f"  Без SNR данных: {no_data} (сохранены)")

        if args.validate:
            print()
            run_validate(out_path, args.year, args.month)


if __name__ == '__main__':
    main()
