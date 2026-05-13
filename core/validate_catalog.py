"""
Валидация результатов EQTransformer по каталогу событий.

Для каждого события в каталоге проверяется, есть ли соответствующая
детекция в csv файле детекций. Совпадение определяется по времени прихода
P-волны на станцию с учётом расчётного времени пробега.

Координаты станции читаются автоматически из CSV файла детекций
(колонки station_lat, station_lon) — не нужно менять скрипт при смене станции.

Использование:
    python validate_catalog.py
    python validate_catalog.py --detections data-in-memory/output/SOC/soc.csv
    python validate_catalog.py --detections data-in-memory/output/ANN/ann.csv
    python validate_catalog.py --window 20 --vp 6.2
"""
import argparse
import math
import csv
import os
from datetime import datetime, timedelta

import openpyxl

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CATALOG_PATH       = os.path.join(_ROOT, "Каталог Сочи 300 км.xlsx")
DEFAULT_DETECTIONS = os.path.join(_ROOT, "data-in-memory", "output", "ANN", "ann.csv")

# Скорость P-волны, км/с. Типичное значение для верхней коры.
DEFAULT_VP = 6.0
# Допуск совпадения, секунды (±window). Покрывает погрешности каталога,
# модели скоростей и точность EQT.
DEFAULT_WINDOW = 15


def haversine_km(lat1, lon1, lat2, lon2):
    """Расстояние между двумя точками на сфере, км."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def travel_time_sec(epicenter_lat, epicenter_lon, depth_km, vp, station_lat, station_lon):
    """Время пробега P-волны от гипоцентра до станции, секунды."""
    epi_km = haversine_km(epicenter_lat, epicenter_lon, station_lat, station_lon)
    hypo_km = math.sqrt(epi_km ** 2 + depth_km ** 2)
    return hypo_km / vp


def load_catalog(path):
    """Загружает каталог из xlsx. Возвращает список dict."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    events = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue  # заголовок
        origin_time, lat, lon, depth_km = row[0], row[1], row[2], row[3]
        ms = row[4] if len(row) > 4 else None
        if origin_time is None or lat is None or lon is None:
            continue
        events.append({
            'origin_time': origin_time,
            'lat': lat,
            'lon': lon,
            'depth_km': depth_km if depth_km is not None else 10.0,
            'ms': float(ms) if ms is not None else None,
        })
    wb.close()
    return events


def load_detections(path):
    """
    Загружает детекции из csv. Возвращает (list[dict], station_lat, station_lon).
    Координаты станции берутся из первой строки CSV — не нужен хардкод.
    """
    detections = []
    station_lat = None
    station_lon = None
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
                if station_lat is None:
                    try:
                        station_lat = float(row['station_lat'])
                        station_lon = float(row['station_lon'])
                    except (KeyError, ValueError):
                        pass
                detections.append({
                    'p_arrival_time': p_time,
                    'file_name': row.get('file_name', ''),
                    'detection_probability': row.get('detection_probability', ''),
                    'p_probability': row.get('p_probability', ''),
                })
    except FileNotFoundError:
        print(f"Файл детекций не найден: {path}")
        return [], None, None
    return detections, station_lat, station_lon


def validate(catalog, detections, station_lat, station_lon, vp, window_sec):
    """
    Для каждого события каталога ищет детекцию в окне ±window_sec
    вокруг расчётного времени прихода P-волны на станцию.
    """
    matched = []
    missed = []

    for ev in catalog:
        tt = travel_time_sec(ev['lat'], ev['lon'], ev['depth_km'], vp,
                             station_lat, station_lon)
        p_expected = ev['origin_time'] + timedelta(seconds=tt)

        hit = None
        min_diff = None
        for det in detections:
            diff = abs((det['p_arrival_time'] - p_expected).total_seconds())
            if diff <= window_sec:
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    hit = det

        epi_km = haversine_km(ev['lat'], ev['lon'], station_lat, station_lon)
        record = {
            **ev,
            'epi_km': round(epi_km, 1),
            'tt_sec': round(tt, 1),
            'p_expected': p_expected,
        }
        if hit:
            record['detection'] = hit
            record['dt_sec'] = round(min_diff, 1)
            matched.append(record)
        else:
            missed.append(record)

    return matched, missed


def print_report(matched, missed, window_sec):
    total = len(matched) + len(missed)
    recall = len(matched) / total * 100 if total > 0 else 0

    print("=" * 70)
    print(f"Recall: {len(matched)}/{total} ({recall:.1f}%) при окне ±{window_sec} сек")
    print("=" * 70)

    if missed:
        print(f"\nПропущенные события ({len(missed)}):")
        print(f"  {'Origin time':<25} {'Ms':>5} {'Dist,км':>8} {'Depth':>6} {'P_exp':<25}")
        for ev in missed:
            ms_str = f"{ev['ms']:.1f}" if ev['ms'] is not None else "  -"
            print(f"  {str(ev['origin_time']):<25} {ms_str:>5} {ev['epi_km']:>8.1f} "
                  f"{ev['depth_km']:>6.1f} {str(ev['p_expected']):<25}")

    if matched:
        print(f"\nОбнаруженные события ({len(matched)}):")
        print(f"  {'Origin time':<25} {'Ms':>5} {'Dist,км':>8} {'dt,сек':>7} {'det_prob':>9}")
        for ev in matched:
            ms_str = f"{ev['ms']:.1f}" if ev['ms'] is not None else "  -"
            print(f"  {str(ev['origin_time']):<25} {ms_str:>5} {ev['epi_km']:>8.1f} "
                  f"{ev['dt_sec']:>7.1f} "
                  f"{ev['detection']['detection_probability']:>9}")


def main():
    parser = argparse.ArgumentParser(description="Валидация EQT по каталогу")
    parser.add_argument('--detections', default=DEFAULT_DETECTIONS,
                        help="Путь к ann.csv с результатами EQT")
    parser.add_argument('--catalog', default=CATALOG_PATH,
                        help="Путь к xlsx-каталогу")
    parser.add_argument('--vp', type=float, default=DEFAULT_VP,
                        help="Скорость P-волны, км/с (по умолчанию 6.0)")
    parser.add_argument('--window', type=float, default=DEFAULT_WINDOW,
                        help="Допуск совпадения, секунды (по умолчанию 15)")
    parser.add_argument('--year', type=int, default=None,
                        help="Фильтр: только события этого года (например 2024)")
    parser.add_argument('--month', type=int, default=None,
                        help="Фильтр: только события этого месяца (1-12)")
    parser.add_argument('--min-mag', type=float, default=None,
                        help="Фильтр: только события с Ms >= значения (например 2.5)")
    args = parser.parse_args()

    print(f"Каталог: {args.catalog}")
    print(f"Детекции: {args.detections}")
    print(f"Vp = {args.vp} км/с, окно = ±{args.window} сек\n")

    catalog = load_catalog(args.catalog)
    print(f"Загружено событий из каталога: {len(catalog)}")

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
    print(f"Загружено детекций EQT: {len(detections)}")

    if not detections:
        print("Нет детекций для сопоставления.")
        return

    print(f"Координаты станции: {station_lat}°N, {station_lon}°E\n")

    matched, missed = validate(catalog, detections, station_lat, station_lon,
                               args.vp, args.window)
    print_report(matched, missed, args.window)


if __name__ == "__main__":
    main()
