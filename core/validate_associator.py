"""
Сравнение результатов ассоциатора (associations.xml) с каталогом.

Origin time в XML = event_start_time первой обнаружившей станции, что примерно
равно catalog_origin_time + P_travel_time_to_nearest_station.
Скрипт автоматически вычисляет эту поправку для каждого события и сравнивает:

    |assoc_time - (catalog_time + tt_nearest)| <= window

Использование:
    python validate_associator.py --year 2024 --month 1
    python validate_associator.py --window 20 --min-mag 2.0
    python validate_associator.py --vp 6.2 --stations-dir json
"""
import argparse
import glob
import json
import math
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import openpyxl

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CATALOG_PATH    = os.path.join(_ROOT, "catalog.xlsx")
# DEFAULT_ASSOC   = os.path.join(_ROOT, "data-in-memory", "association_gpu", "sectors", "merged", "associations.xml")
DEFAULT_ASSOC   = os.path.join(_ROOT, "data-in-memory", "association_gpu_100_150", "associations.xml")
DEFAULT_WINDOW  = 15   # секунды — допуск ПОСЛЕ поправки на travel time
DEFAULT_VP      = 6.0  # км/с
DEFAULT_STA_DIR = os.path.join(_ROOT, "json")
BED_NS = "http://quakeml.org/xmlns/bed/1.2"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def load_stations(stations_dir):
    """Загружает координаты всех станций из json/station_*.json."""
    stations = {}
    for fpath in glob.glob(os.path.join(stations_dir, "station_*.json")):
        name = os.path.basename(fpath).replace("station_", "").replace(".json", "")
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        key = list(data.keys())[0]
        coords = data[key]["coords"]
        stations[name] = {"lat": coords[0], "lon": coords[1]}
    return stations


def nearest_station_tt(ev_lat, ev_lon, depth_km, stations, vp):
    """
    Возвращает (station_name, epi_km, tt_sec) для ближайшей станции.
    Учитывает гипоцентральное расстояние (с глубиной).
    """
    best_name, best_epi, best_tt = None, None, None
    for name, st in stations.items():
        epi = haversine_km(ev_lat, ev_lon, st["lat"], st["lon"])
        hypo = math.sqrt(epi ** 2 + depth_km ** 2)
        tt = hypo / vp
        if best_tt is None or tt < best_tt:
            best_name, best_epi, best_tt = name, epi, tt
    return best_name, round(best_epi, 1), round(best_tt, 1)


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
            'lat': lat,
            'lon': lon,
            'depth_km': depth_km if depth_km is not None else 10.0,
            'ms': float(ms) if ms is not None else None,
        })
    wb.close()
    return events


def load_associations(path):
    tree = ET.parse(path)
    root = tree.getroot()
    ns = BED_NS
    events = []
    for ev in root.findall(f'.//{{{ns}}}event'):
        orig = ev.find(f'{{{ns}}}origin')
        if orig is None:
            continue
        t_el  = orig.find(f'{{{ns}}}time/{{{ns}}}value')
        la_el = orig.find(f'{{{ns}}}latitude/{{{ns}}}value')
        lo_el = orig.find(f'{{{ns}}}longitude/{{{ns}}}value')
        if t_el is None:
            continue
        t_str = t_el.text.rstrip('Z')
        try:
            origin_time = datetime.fromisoformat(t_str).replace(tzinfo=None)
        except ValueError:
            continue
        lat = float(la_el.text) if la_el is not None else None
        lon = float(lo_el.text) if lo_el is not None else None
        stations = set()
        for pick in ev.findall(f'{{{ns}}}pick'):
            wf = pick.find(f'{{{ns}}}waveformID')
            if wf is not None:
                stations.add(wf.get('stationCode', '').strip())
        events.append({
            'origin_time': origin_time,
            'lat': lat,
            'lon': lon,
            'n_stations': len(stations),
        })
    return events


def validate(catalog, assoc_events, stations, vp, window_sec):
    """
    Для каждого события каталога:
      1. Находит ближайшую станцию и вычисляет P-travel time (tt).
      2. Ожидаемое время ассоциатора = origin_time + tt.
      3. Ищет совпадение в пределах ±window_sec от этого ожидаемого времени.
    """
    REF_LAT, REF_LON = 43.57, 39.763
    matched, missed = [], []

    for ev in catalog:
        cat_t = ev['origin_time']
        if hasattr(cat_t, 'tzinfo') and cat_t.tzinfo is not None:
            cat_t = cat_t.replace(tzinfo=None)

        nearest, nearest_epi, tt = nearest_station_tt(
            ev['lat'], ev['lon'], ev['depth_km'], stations, vp
        )
        expected_assoc_t = cat_t + timedelta(seconds=tt)

        best, min_diff = None, None
        for ae in assoc_events:
            diff = abs((ae['origin_time'] - expected_assoc_t).total_seconds())
            if diff <= window_sec:
                if min_diff is None or diff < min_diff:
                    min_diff, best = diff, ae

        dist_soc = haversine_km(ev['lat'], ev['lon'], REF_LAT, REF_LON)
        record = {**ev, 'epi_km': round(dist_soc, 1),
                  'nearest_sta': nearest, 'nearest_epi': nearest_epi, 'tt_sec': tt}
        if best:
            record['assoc'] = best
            record['dt_sec'] = round(min_diff, 1)
            matched.append(record)
        else:
            missed.append(record)

    return matched, missed


def print_report(matched, missed, window_sec):
    total = len(matched) + len(missed)
    recall = len(matched) / total * 100 if total > 0 else 0

    print("=" * 80)
    print(f"Recall: {len(matched)}/{total} ({recall:.1f}%)  окно ±{window_sec} сек (после поправки на TT)")
    print("=" * 80)

    if matched:
        print(f"\nОбнаружено ({len(matched)}):")
        print(f"  {'Origin time':<25} {'Ms':>4} {'Dist':>7} {'Nearest':>6} {'TT,с':>5} {'dt,с':>6} {'N_sta':>5}")
        for ev in sorted(matched, key=lambda x: x['origin_time']):
            ms_s = f"{ev['ms']:.1f}" if ev['ms'] is not None else "  -"
            print(f"  {str(ev['origin_time']):<25} {ms_s:>4} {ev['epi_km']:>7.1f}"
                  f" {ev['nearest_sta']:>6} {ev['tt_sec']:>5.1f}"
                  f" {ev['dt_sec']:>6.1f} {ev['assoc']['n_stations']:>5}")

    if missed:
        print(f"\nПропущено ({len(missed)}):")
        print(f"  {'Origin time':<25} {'Ms':>4} {'Dist':>7} {'Nearest':>6} {'TT,с':>5} {'Depth':>6}")
        for ev in sorted(missed, key=lambda x: x['origin_time']):
            ms_s = f"{ev['ms']:.1f}" if ev['ms'] is not None else "  -"
            print(f"  {str(ev['origin_time']):<25} {ms_s:>4} {ev['epi_km']:>7.1f}"
                  f" {ev['nearest_sta']:>6} {ev['tt_sec']:>5.1f}"
                  f" {ev['depth_km']:>6.1f}")


def main():
    parser = argparse.ArgumentParser(description="Валидация ассоциатора по каталогу")
    parser.add_argument('--assoc',        default=DEFAULT_ASSOC)
    parser.add_argument('--catalog',      default=CATALOG_PATH)
    parser.add_argument('--window',       type=float, default=DEFAULT_WINDOW,
                        help="Допуск совпадения в сек ПОСЛЕ поправки на TT (default 15)")
    parser.add_argument('--vp',           type=float, default=DEFAULT_VP)
    parser.add_argument('--stations-dir', default=DEFAULT_STA_DIR,
                        help="Папка с json/station_*.json файлами")
    parser.add_argument('--year',         type=int,   default=None)
    parser.add_argument('--month',        type=int,   default=None)
    parser.add_argument('--min-mag',      type=float, default=None)
    args = parser.parse_args()

    print(f"Каталог:     {args.catalog}")
    print(f"Ассоциатор:  {args.assoc}")
    print(f"Vp:          {args.vp} км/с")
    print(f"Окно:        ±{args.window} сек (после поправки на travel time)\n")

    stations = load_stations(args.stations_dir)
    print(f"Загружено станций: {len(stations)}")

    catalog = load_catalog(args.catalog)
    print(f"Событий в каталоге: {len(catalog)}")

    if args.year is not None or args.month is not None:
        before = len(catalog)
        catalog = [
            ev for ev in catalog
            if (args.year  is None or ev['origin_time'].year  == args.year)
            and (args.month is None or ev['origin_time'].month == args.month)
        ]
        print(f"После фильтра год={args.year} месяц={args.month}: {len(catalog)} (из {before})")

    if args.min_mag is not None:
        before = len(catalog)
        catalog = [ev for ev in catalog if ev['ms'] is not None and ev['ms'] >= args.min_mag]
        print(f"После фильтра Ms >= {args.min_mag}: {len(catalog)} (из {before})")

    assoc_events = load_associations(args.assoc)
    print(f"Ассоциированных событий: {len(assoc_events)}\n")

    matched, missed = validate(catalog, assoc_events, stations, args.vp, args.window)
    print_report(matched, missed, args.window)


if __name__ == "__main__":
    main()
