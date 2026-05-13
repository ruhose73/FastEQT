"""
Подбор порогов детекции без перезапуска модели.

Читает soc.csv (или любой другой csv детекций), применяет разные комбинации
порогов и для каждой считает recall по каталогу и количество детекций.

Использование:
    python threshold_tuner.py
    python threshold_tuner.py --detections data-in-memory/output/SOC/soc.csv
    python threshold_tuner.py --det-steps 5 --p-steps 3 --s-steps 3
"""
import argparse
import csv
import math
import os
import numpy as np
from datetime import datetime, timedelta
from itertools import product

import openpyxl

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CATALOG_PATH       = os.path.join(_ROOT, "Каталог Сочи 300 км.xlsx")
DEFAULT_DETECTIONS = os.path.join(_ROOT, "data-in-memory", "output", "SOC", "soc.csv")
DEFAULT_WINDOW_SEC = 15.0
DEFAULT_VP = 6.0


# ---------------------------------------------------------------------------
# Геометрия
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def travel_time_sec(ev_lat, ev_lon, depth_km, station_lat, station_lon, vp):
    epi_km = haversine_km(ev_lat, ev_lon, station_lat, station_lon)
    hypo_km = math.sqrt(epi_km ** 2 + depth_km ** 2)
    return hypo_km / vp


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_catalog(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    events = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue
        origin_time, lat, lon, depth_km = row[0], row[1], row[2], row[3]
        ms = row[4] if len(row) > 4 else None
        if origin_time is None or lat is None or lon is None:
            continue
        events.append({
            'origin_time': origin_time,
            'lat': float(lat),
            'lon': float(lon),
            'depth_km': float(depth_km) if depth_km is not None else 10.0,
            'ms': float(ms) if ms is not None else None,
        })
    wb.close()
    return events


def load_detections(path):
    """Загружает все детекции без фильтрации порогов."""
    detections = []
    station_lat = station_lon = None
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                p_str = row.get('p_arrival_time', '').strip()
                if not p_str or p_str.lower() == 'none':
                    continue
                try:
                    p_time = datetime.fromisoformat(p_str)
                except ValueError:
                    continue

                def safe_float(key):
                    v = row.get(key, '').strip()
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        return None

                if station_lat is None:
                    try:
                        station_lat = float(row['station_lat'])
                        station_lon = float(row['station_lon'])
                    except (KeyError, ValueError):
                        pass

                detections.append({
                    'p_arrival_time': p_time,
                    'detection_probability': safe_float('detection_probability'),
                    'p_probability': safe_float('p_probability'),
                    's_probability': safe_float('s_probability'),
                    's_arrival_time': row.get('s_arrival_time', '').strip(),
                })
    except FileNotFoundError:
        print(f"Файл не найден: {path}")
        return [], None, None
    return detections, station_lat, station_lon


# ---------------------------------------------------------------------------
# Фильтрация и recall
# ---------------------------------------------------------------------------

def apply_thresholds(detections, det_thr, p_thr, s_thr, keep_ps):
    """Возвращает только детекции, прошедшие пороги."""
    result = []
    for d in detections:
        if d['detection_probability'] is None:
            continue
        if d['detection_probability'] < det_thr:
            continue
        has_p = d['p_probability'] is not None and d['p_probability'] >= p_thr
        has_s = (d['s_arrival_time'] not in ('', 'None', 'none')
                 and d['s_probability'] is not None
                 and d['s_probability'] >= s_thr)
        if keep_ps:
            if not (has_p and has_s):
                continue
        else:
            if not (has_p or has_s):
                continue
        result.append(d)
    return result


def compute_recall(catalog, detections, station_lat, station_lon, vp, window_sec):
    """Считает сколько событий каталога найдено в детекциях."""
    matched = 0
    for ev in catalog:
        tt = travel_time_sec(ev['lat'], ev['lon'], ev['depth_km'],
                             station_lat, station_lon, vp)
        p_expected = ev['origin_time'] + timedelta(seconds=tt)
        for det in detections:
            diff = abs((det['p_arrival_time'] - p_expected).total_seconds())
            if diff <= window_sec:
                matched += 1
                break
    return matched


# ---------------------------------------------------------------------------
# Сетка порогов
# ---------------------------------------------------------------------------

def make_grid(det_steps, p_steps, s_steps):
    """Равномерная сетка порогов от 0.1 до 0.9."""
    det_vals = np.linspace(0.1, 0.9, det_steps).round(2)
    p_vals   = np.linspace(0.1, 0.9, p_steps).round(2)
    s_vals   = np.linspace(0.1, 0.9, s_steps).round(2)
    return det_vals, p_vals, s_vals


# ---------------------------------------------------------------------------
# Основной цикл
# ---------------------------------------------------------------------------

def run_tuning(catalog, detections, station_lat, station_lon,
               det_vals, p_vals, s_vals, keep_ps_options,
               vp, window_sec):
    total_catalog = len(catalog)
    results = []

    combos = list(product(det_vals, p_vals, s_vals, keep_ps_options))
    print(f"Проверяем {len(combos)} комбинаций порогов...\n")

    for det_thr, p_thr, s_thr, keep_ps in combos:
        filtered = apply_thresholds(detections, det_thr, p_thr, s_thr, keep_ps)
        matched = compute_recall(catalog, filtered, station_lat, station_lon,
                                 vp, window_sec)
        recall_pct = matched / total_catalog * 100 if total_catalog > 0 else 0
        results.append({
            'det_thr': det_thr,
            'p_thr': p_thr,
            's_thr': s_thr,
            'keep_ps': keep_ps,
            'n_detections': len(filtered),
            'recall': matched,
            'recall_pct': recall_pct,
        })

    # Сортировка: сначала максимальный recall, потом минимальное число детекций
    results.sort(key=lambda r: (-r['recall_pct'], r['n_detections']))
    return results


def print_results(results, total_catalog, top_n=30):
    print(f"{'det':>5} {'p':>5} {'s':>5} {'PS':>4} │ {'recall':>10} {'детекций':>10}")
    print("─" * 50)
    for r in results[:top_n]:
        ps_str = "оба" if r['keep_ps'] else "любая"
        print(f"{r['det_thr']:>5.2f} {r['p_thr']:>5.2f} {r['s_thr']:>5.2f} {ps_str:>6} │"
              f" {r['recall']:>4}/{total_catalog} ({r['recall_pct']:>5.1f}%)"
              f" {r['n_detections']:>10}")

    print("\n─── Лучший компромисс (максимальный recall при минимуме детекций) ───")
    best = results[0]
    ps_str = "оба" if best['keep_ps'] else "любая"
    print(f"  detection_threshold = {best['det_thr']}")
    print(f"  P_threshold         = {best['p_thr']}")
    print(f"  S_threshold         = {best['s_thr']}")
    print(f"  keepPS              = {best['keep_ps']}  (фаза: {ps_str})")
    print(f"  Recall:    {best['recall']}/{total_catalog} ({best['recall_pct']:.1f}%)")
    print(f"  Детекций:  {best['n_detections']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Подбор порогов EQT без перезапуска модели")
    parser.add_argument('--detections', default=DEFAULT_DETECTIONS)
    parser.add_argument('--catalog', default=CATALOG_PATH)
    parser.add_argument('--window', type=float, default=DEFAULT_WINDOW_SEC,
                        help="Окно совпадения, сек (по умолчанию 15)")
    parser.add_argument('--vp', type=float, default=DEFAULT_VP,
                        help="Скорость P-волны, км/с (по умолчанию 6.0)")
    parser.add_argument('--det-steps', type=int, default=9,
                        help="Шагов для detection_threshold (по умолчанию 9 → 0.1..0.9)")
    parser.add_argument('--p-steps', type=int, default=5,
                        help="Шагов для P_threshold")
    parser.add_argument('--s-steps', type=int, default=5,
                        help="Шагов для S_threshold")
    parser.add_argument('--top', type=int, default=30,
                        help="Сколько строк вывести в таблице")
    parser.add_argument('--year', type=int, default=None,
                        help="Фильтр каталога: только события этого года")
    parser.add_argument('--month', type=int, default=None,
                        help="Фильтр каталога: только события этого месяца (1-12)")
    parser.add_argument('--min-mag', type=float, default=None,
                        help="Фильтр каталога: только события с Ms >= значения (например 2.5)")
    args = parser.parse_args()

    print(f"Каталог:  {args.catalog}")
    print(f"Детекции: {args.detections}")
    print(f"Vp = {args.vp} км/с, окно = ±{args.window} сек\n")

    catalog = load_catalog(args.catalog)
    print(f"Событий в каталоге: {len(catalog)}")

    if args.year is not None or args.month is not None:
        before = len(catalog)
        catalog = [
            ev for ev in catalog
            if (args.year is None or ev['origin_time'].year == args.year)
            and (args.month is None or ev['origin_time'].month == args.month)
        ]
        print(f"После фильтра год={args.year} месяц={args.month}: {len(catalog)} событий (из {before})")

    if args.min_mag is not None:
        before = len(catalog)
        catalog = [ev for ev in catalog if ev['ms'] is not None and ev['ms'] >= args.min_mag]
        print(f"После фильтра Ms >= {args.min_mag}: {len(catalog)} событий (из {before})")

    detections, station_lat, station_lon = load_detections(args.detections)
    print(f"Детекций в CSV (без фильтра): {len(detections)}")
    if station_lat is None:
        print("Не удалось определить координаты станции из CSV.")
        return
    print(f"Координаты станции: {station_lat}°N, {station_lon}°E\n")

    det_vals, p_vals, s_vals = make_grid(args.det_steps, args.p_steps, args.s_steps)
    keep_ps_options = [False, True]

    results = run_tuning(catalog, detections, station_lat, station_lon,
                         det_vals, p_vals, s_vals, keep_ps_options,
                         args.vp, args.window)

    print_results(results, len(catalog), top_n=args.top)


if __name__ == "__main__":
    main()
