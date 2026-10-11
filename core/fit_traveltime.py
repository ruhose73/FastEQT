"""
fit_traveltime.py — региональный годограф первой P по бюллетеню ГС РАН (.BUL, IMS1.0).

Для каждой пары событие-станция бюллетеня: эпицентральное расстояние (координаты события из
Origin, координаты станции — StationXML + сводный JSON станций) и время пробега tP = P − T0
(первая P-фаза станции). Медиана tP по бинам расстояния -> CSV (dist_km, tp_s, n, p16, p84),
который читает core/ml_filter_v5.py --ml-stations p-traveltime (--tt-table).

Поставляемая таблица core/data/tt_p_ncau2024.csv собрана командой по умолчанию: бюллетень
NCAU 2024 без мая (май — проверочный месяц для ML-фильтра). Для другого региона / бюллетеня —
пересобрать со своими --bul / --exclude-months / --out.

Запуск:
    python core/fit_traveltime.py
    python core/fit_traveltime.py --bul workspace/bulletin/input/2025_jan-mar_NCAU.BUL \
        --exclude-months '' --out core/data/tt_p_ncau2025q1.csv
"""

import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import argparse
import os
import re
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_bul import parse_bul, parse_bul_origins
from ml_filter_v5 import (DEFAULT_METADATA, DEFAULT_STATION_JSON, _haversine_km,
                          load_station_coords, load_station_coords_json)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_BUL = os.path.join(_ROOT, 'workspace', 'bulletin', 'input', '2024_jan-dec_NCAU.BUL')
DEFAULT_OUT = os.path.join(_ROOT, 'core', 'data', 'tt_p_ncau2024.csv')

MAX_DIST_KM = 900.0   # дальше — пар мало, для сети Кавказа не нужно
MAX_TP_SEC  = 150.0   # отсечка явных ошибок времени
BIN_KM      = 10
MIN_N       = 5       # бин с меньшим числом пар не пишется

# строка фазы: Sta, Dist, [EvAz], Phase, Time
PHASE_RE = re.compile(r'^(\S{1,5})\s+[\d.]+\s+(?:[\d.]+\s+)?([A-Za-z]\S*)\s+(\d\d:\d\d:\d\d\.\d+)')


def bul_first_p(path):
    """EVENT id -> {sta: datetime первой P-фазы станции (Pg, Pn, P, ...)}."""
    out, cur_id, date, in_ph = {}, None, None, False
    for line in open(path, encoding='utf-8', errors='replace'):
        if line.startswith('EVENT '):
            cur_id, date, in_ph = line.split()[1], None, False
            out[cur_id] = {}
            continue
        if cur_id is None:
            continue
        if date is None and re.match(r'^\d{4}/\d\d/\d\d ', line):
            date = line[:10]
        s = line.strip()
        if s.startswith('Sta') and 'Phase' in s:
            in_ph = True
            continue
        if in_ph:
            if not s:
                in_ph = False
                continue
            m = PHASE_RE.match(line)
            if not m or date is None or m.group(2)[0].upper() != 'P':
                continue
            hh, mm, ss = m.group(3).split(':')   # секунды бывают >= 60 (переход через минуту)
            t = (datetime.strptime(date, "%Y/%m/%d")
                 + timedelta(hours=int(hh), minutes=int(mm), seconds=float(ss)))
            out[cur_id].setdefault(m.group(1), t)
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Региональный годограф первой P по бюллетеню ГС РАН -> CSV для '
                    'ml_filter_v5.py --ml-stations p-traveltime',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--bul', default=DEFAULT_BUL, help='Бюллетень .BUL (IMS1.0)')
    parser.add_argument('--exclude-months', default='5',
                        help='Не использовать события этих месяцев (через запятую; пусто — все). '
                             'По умолчанию май — проверочный месяц')
    parser.add_argument('--metadata-dir', default=DEFAULT_METADATA,
                        help='Папка FDSNStationXML (координаты станций)')
    parser.add_argument('--station-coords-json', default=DEFAULT_STATION_JSON,
                        help='Сводный JSON станций (дополняет/перекрывает StationXML)')
    parser.add_argument('--max-dist-km', type=float, default=MAX_DIST_KM)
    parser.add_argument('--max-tp-sec', type=float, default=MAX_TP_SEC)
    parser.add_argument('--bin-km', type=int, default=BIN_KM)
    parser.add_argument('--min-n', type=int, default=MIN_N, help='Минимум пар в бине')
    parser.add_argument('--out', default=DEFAULT_OUT, help='Выходной CSV')
    args = parser.parse_args()

    excl_months = {int(m) for m in args.exclude_months.split(',') if m.strip()}
    coords = load_station_coords(args.metadata_dir)
    coords.update(load_station_coords_json(args.station_coords_json))

    refs = parse_bul(args.bul)
    for r, o in zip(refs, parse_bul_origins(args.bul)):
        r.update(o)
    first_p = bul_first_p(args.bul)

    d_all, t_all = [], []
    for r in refs:
        if r['t0'] is None or r['t0'].month in excl_months or r.get('lat') is None:
            continue
        for sta, p in first_p.get(r['event_id'], {}).items():
            if sta not in coords:
                continue
            d = _haversine_km(r['lat'], r['lon'], *coords[sta])
            t = (p - r['t0']).total_seconds()
            if 0 < d <= args.max_dist_km and 0 < t < args.max_tp_sec:
                d_all.append(d)
                t_all.append(t)
    d_all, t_all = np.array(d_all), np.array(t_all)
    print(f"Бюллетень: {args.bul}  (без месяцев: {sorted(excl_months) or '—'})")
    print(f"Пар событие-станция: {len(d_all)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as fh:
        fh.write('dist_km,tp_s,n,p16,p84\n')
        for lo in range(0, int(args.max_dist_km), args.bin_km):
            m = (d_all >= lo) & (d_all < lo + args.bin_km)
            if m.sum() < args.min_n:
                continue
            q16, q50, q84 = np.percentile(t_all[m], [16, 50, 84])
            mid = lo + args.bin_km / 2
            fh.write(f"{mid:g},{q50:.2f},{m.sum()},{q16:.2f},{q84:.2f}\n")
            if lo % 50 == 0:
                print(f"  {mid:5g} км: tP {q50:6.2f} с  (16–84%: {q16:.2f}–{q84:.2f}, n={m.sum()}, "
                      f"кажущаяся V {mid / q50:.2f} км/с)")
    print(f"Записано: {args.out}")


if __name__ == '__main__':
    main()
