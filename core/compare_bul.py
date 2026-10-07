"""
compare_bul.py — сравнение двух бюллетеней в формате IMS1.0:SHORT/LONG
(workspace/bulletin/input/*.BUL от ГС РАН и workspace/bulletin/output/*.BUL
из export_bul.py) по времени и станциям.

Не трогает XML/CSV пайплайна — читает напрямую .BUL файлы, используя ту же
раскладку колонок Origin/Phase Block (IDC-3.4.1Rev1, см. export_bul.py):
    Date: 1-10, Time: 12-22 (origin block, hh:mm:ss.ss)
    Sta:  1-5   (phase block)

Каждое EVENT сводится к (origin_datetime, set(stations)). Событие reference
считается найденным, если среди candidate есть событие с |dt| <= --window
и числом общих станций >= --min-shared; при нескольких подходящих кандидатах
выбирается тот, что делит больше всего станций (при равенстве — ближайший
по времени).

Использование:
    python core/compare_bul.py --reference workspace/bulletin/input/2025_jan-mar_NCAU.BUL \
                                --candidate workspace/bulletin/output/2025_q1.BUL \
                                --window 60 --min-shared 1
    python core/compare_bul.py ... --show-missed --out-csv compare.csv
    python core/compare_bul.py ... --min-mag 1.5 --location-error   # + ошибка эпицентра/глубины
"""

import argparse
import csv
import math
import statistics
from datetime import datetime


def parse_bul(path):
    """-> список {'event_id', 'region', 't0': datetime|None, 'stations': set(str)}"""
    events = []
    cur = None
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip('\n')

        if line.startswith('EVENT '):
            if cur is not None:
                events.append(cur)
            parts = line.split(None, 2)
            cur = {
                'event_id': parts[1] if len(parts) > 1 else None,
                'region':   parts[2] if len(parts) > 2 else '',
                't0':       None,
                'stations': set(),
                'mag':      None,
            }
            i += 1
            continue

        if cur is None:
            i += 1
            continue

        stripped = line.strip()

        if stripped.startswith('Date') and 'Time' in stripped and 'Latitude' in stripped:
            i += 1
            if i < n:
                cur['t0'] = _parse_origin_time(lines[i])
            i += 1
            continue

        if stripped.startswith('Magnitude') and 'Nsta' in stripped and 'OrigID' in stripped:
            # Несколько magnitude-строк на событие (разные агентства/типы) —
            # берём первую (обычно от первичного origin), остальные пропускаем.
            i += 1
            if i < n and cur['mag'] is None:
                cur['mag'] = _parse_magnitude_value(lines[i])
            i += 1
            continue

        if stripped.startswith('Sta') and 'Phase' in stripped and 'ArrID' in stripped:
            i += 1
            while i < n and lines[i].strip() != '' and not lines[i].startswith('EVENT'):
                sta = lines[i][0:5].strip()
                if sta:
                    cur['stations'].add(sta)
                i += 1
            continue

        i += 1

    if cur is not None:
        events.append(cur)
    return events


def _parse_origin_time(line):
    date_s = line[0:10].strip()
    time_s = line[11:22].strip()
    if not date_s or not time_s:
        return None
    try:
        return datetime.strptime(f"{date_s} {time_s}", "%Y/%m/%d %H:%M:%S.%f")
    except ValueError:
        return None


def _parse_magnitude_value(line):
    """Magnitude Block data row, cols 7-10 (f4.1): значение магнитуды."""
    val_s = line[6:10].strip()
    if not val_s:
        return None
    try:
        return float(val_s)
    except ValueError:
        return None


def parse_bul_origins(path):
    """
    Координаты первого Origin каждого EVENT, в том же порядке, что события parse_bul():
    список {'lat', 'lon', 'depth_km', 'rms', 'err_depth'} (None, если поле пустое —
    например, наш .BUL без --locator-mode enrich). Колонки (1-индексация, включительно,
    сверены по реальным бюллетеням ГС РАН и ISC — см. export_bul.md): RMS 31-35,
    Latitude 37-44, Longitude 46-54, Depth 72-76, Err depth 78-82. Как и t0 в
    parse_bul(), берётся только первый Origin события.
    """
    origins = []
    cur = None
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith('EVENT '):
            cur = {'lat': None, 'lon': None, 'depth_km': None, 'rms': None,
                   'err_depth': None, '_seen': False}
            origins.append(cur)
            continue
        if cur is None or cur['_seen']:
            continue
        s = line.strip()
        if s.startswith('Date') and 'Time' in s and 'Latitude' in s and i + 1 < len(lines):
            o = lines[i + 1]
            cur['_seen'] = True
            cur['lat']      = _parse_float(o[36:44])
            cur['lon']      = _parse_float(o[45:54])
            cur['depth_km'] = _parse_float(o[71:76])
            cur['rms']       = _parse_float(o[30:35])
            cur['err_depth'] = _parse_float(o[77:82])
    for o in origins:
        del o['_seen']
    return origins


def _parse_float(s):
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def print_location_report(matched):
    """
    --location-error: ошибка эпицентра/глубины candidate относительно reference по
    совпавшим событиям. Считается только там, где координаты есть в обоих бюллетенях
    (у нашего .BUL — только у событий с решением локатора, export_bul --locator-mode
    enrich). На recall не влияет.
    """
    rows = []
    for m in matched:
        ref, cand = m['ref'], m['cand']
        if None in (ref.get('lat'), ref.get('lon'), cand.get('lat'), cand.get('lon')):
            continue
        epi = _haversine_km(ref['lat'], ref['lon'], cand['lat'], cand['lon'])
        dep = (cand['depth_km'] - ref['depth_km']
               if cand.get('depth_km') is not None and ref.get('depth_km') is not None else None)
        rows.append((m, epi, dep))

    print(f"\nОшибка локации (--location-error): координаты есть у {len(rows)} из "
          f"{len(matched)} совпавших событий")
    if not rows:
        return
    epis = sorted(epi for _, epi, _ in rows)
    deps = [dep for _, _, dep in rows if dep is not None]
    print(f"  эпицентр, км: медиана {statistics.median(epis):.1f}  "
          f"макс {max(epis):.1f}  <=10 км: {sum(e <= 10 for e in epis)}/{len(epis)}")
    if deps:
        print(f"  глубина (candidate − reference), км: медиана {statistics.median(deps):+.1f}  "
              f"|медиана| {statistics.median([abs(d) for d in deps]):.1f}")
    print(f"  {'Ref origin time':<22} {'Ref id':<10} {'Mag':>5} {'Δэпи,км':>8} {'ΔH,км':>7} "
          f"{'H_ref':>6} {'H_cand':>6}")
    for m, epi, dep in sorted(rows, key=lambda x: x[0]['ref']['t0']):
        ref, cand = m['ref'], m['cand']
        mag_s = f"{ref['mag']:.1f}" if ref['mag'] is not None else "  -"
        dep_s = f"{dep:>+7.1f}" if dep is not None else f"{'—':>7}"
        h_ref = f"{ref['depth_km']:>6.1f}" if ref.get('depth_km') is not None else f"{'—':>6}"
        h_cnd = f"{cand['depth_km']:>6.1f}" if cand.get('depth_km') is not None else f"{'—':>6}"
        print(f"  {str(ref['t0']):<22} {ref['event_id'] or '-':<10} {mag_s:>5} {epi:>8.1f} "
              f"{dep_s} {h_ref} {h_cnd}")


def _loc_good(cand, max_rms, min_err_depth):
    """Решение локатора у candidate хорошее: Err depth > min_err_depth (нет упора глубины в
    границу таблицы годографов) и RMS < max_rms — тот же критерий, что качество в export_bul."""
    return (cand.get('err_depth') is not None and cand['err_depth'] > min_err_depth
            and cand.get('rms') is not None and cand['rms'] < max_rms)


def match_events_v2(reference, candidate, window_sec, min_shared, max_dist_km, no_coords_match,
                    loc_good_only=False, loc_max_rms=2.0, loc_min_err_depth=0.001):
    """
    loc_good_only: координаты candidate используются, только если решение хорошее
    (_loc_good); candidate с плохим решением сопоставляется как событие без координат —
    плохая локация не должна делать найденное событие «не найденным».

    --match-mode location: candidate с координатами совпадает с reference, если |dt| <= окна и
    эпицентры ближе max_dist_km (общие станции не требуются — состав станций у нас и в
    бюллетене может различаться). Если координат нет у candidate или у reference —
    no_coords_match: 'time' (только окно по времени) или 'stations' (окно + >= min_shared
    общих станций, как в match_events()). Среди подходящих выбирается совпавший по месту
    (ближайший по расстоянию), иначе ближайший по времени.
    """
    matched, missed = [], []
    for ref in reference:
        if ref['t0'] is None:
            missed.append(ref)
            continue
        best, best_key = None, None
        for cand in candidate:
            if cand['t0'] is None:
                continue
            dt = (cand['t0'] - ref['t0']).total_seconds()
            if abs(dt) > window_sec:
                continue
            shared = ref['stations'] & cand['stations']
            have_xy = None not in (ref.get('lat'), ref.get('lon'), cand.get('lat'), cand.get('lon'))
            if have_xy and loc_good_only and not _loc_good(cand, loc_max_rms, loc_min_err_depth):
                have_xy = False
            if have_xy:
                dist = _haversine_km(ref['lat'], ref['lon'], cand['lat'], cand['lon'])
                if dist > max_dist_km:
                    continue
                key, by = (0, dist, abs(dt)), 'location'
            else:
                if no_coords_match == 'stations' and len(shared) < min_shared:
                    continue
                key, by = (1, 0.0, abs(dt)), no_coords_match
            if best_key is None or key < best_key:
                best, best_key = {'ref': ref, 'cand': cand, 'dt': dt, 'shared': shared,
                                  'by': by}, key
        if best is not None:
            matched.append(best)
        else:
            missed.append(ref)
    return matched, missed


def match_events(reference, candidate, window_sec, min_shared):
    matched, missed = [], []
    for ref in reference:
        if ref['t0'] is None:
            missed.append(ref)
            continue

        best, best_shared, best_dt = None, -1, None
        for cand in candidate:
            if cand['t0'] is None:
                continue
            dt = (cand['t0'] - ref['t0']).total_seconds()
            if abs(dt) > window_sec:
                continue
            shared = ref['stations'] & cand['stations']
            if len(shared) < min_shared:
                continue
            if (len(shared) > best_shared or
                    (len(shared) == best_shared and (best_dt is None or abs(dt) < abs(best_dt)))):
                best, best_shared, best_dt = cand, len(shared), dt

        if best is not None:
            matched.append({'ref': ref, 'cand': best, 'dt': best_dt,
                             'shared': ref['stations'] & best['stations']})
        else:
            missed.append(ref)

    return matched, missed


def print_report(matched, missed, reference, candidate, window_sec, min_shared, show_missed):
    total = len(reference)
    recall = len(matched) / total * 100 if total else 0

    print("=" * 78)
    print(f"Reference: {total} событий   Candidate: {len(candidate)} событий")
    print(f"Recall: {len(matched)}/{total} ({recall:.1f}%)  "
          f"окно ±{window_sec:.0f}с  min-shared-sta={min_shared}")
    print("=" * 78)

    if matched:
        dts = [abs(m['dt']) for m in matched]
        shared_n = [len(m['shared']) for m in matched]
        print(f"\nСовпадения ({len(matched)}):  "
              f"|dt| среднее={sum(dts)/len(dts):.1f}с  макс={max(dts):.1f}с  "
              f"общих станций среднее={sum(shared_n)/len(shared_n):.1f}")
        hdr = f"  {'Ref origin time':<22} {'Ref rg':<8} {'Mag':>5} {'dt,с':>7} {'shared':>6} {'ref_N':>5} {'cand_N':>6}"
        print(hdr)
        for m in sorted(matched, key=lambda x: x['ref']['t0']):
            ref, cand = m['ref'], m['cand']
            mag_s = f"{ref['mag']:.1f}" if ref['mag'] is not None else "  -"
            print(f"  {str(ref['t0']):<22} {ref['event_id'] or '-':<8} {mag_s:>5} "
                  f"{m['dt']:>+7.1f} {len(m['shared']):>6} "
                  f"{len(ref['stations']):>5} {len(cand['stations']):>6}")

    if missed:
        print(f"\nНе найдено ({len(missed)}):")
        print(f"  {'Ref origin time':<22} {'Ref id':<10} {'Mag':>5} {'region':<30} {'N sta'}")
        for ref in sorted(missed, key=lambda x: x['t0'] or datetime(2099, 1, 1)):
            mag_s = f"{ref['mag']:.1f}" if ref['mag'] is not None else "  -"
            print(f"  {str(ref['t0']):<22} {ref['event_id'] or '-':<10} {mag_s:>5} "
                  f"{ref['region'][:30]:<30} {len(ref['stations'])}")
    elif show_missed:
        print("\nВсе события reference найдены.")


def write_csv(matched, missed, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['status', 'ref_event_id', 'ref_time', 'ref_region', 'ref_mag', 'ref_n_sta',
                    'cand_event_id', 'cand_time', 'dt_sec', 'shared_sta', 'cand_n_sta'])
        for m in matched:
            ref, cand = m['ref'], m['cand']
            w.writerow(['matched', ref['event_id'], ref['t0'], ref['region'], ref['mag'],
                        len(ref['stations']), cand['event_id'], cand['t0'], round(m['dt'], 1),
                        len(m['shared']), len(cand['stations'])])
        for ref in missed:
            w.writerow(['missed', ref['event_id'], ref['t0'], ref['region'], ref['mag'],
                        len(ref['stations']), '', '', '', '', ''])
    print(f"\nРезультат записан в {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Сравнение двух бюллетеней IMS1.0 по времени и станциям")
    parser.add_argument('--reference', required=True,
                        help="Эталонный бюллетень (напр. workspace/bulletin/input/2025_jan-mar_NCAU.BUL)")
    parser.add_argument('--candidate', required=True,
                        help="Наш бюллетень (напр. workspace/bulletin/output/2025_q1.BUL)")
    parser.add_argument('--window', type=float, default=60.0,
                        help="Допуск по времени, сек (default: 60)")
    parser.add_argument('--min-shared', type=int, default=1,
                        help="Минимум общих станций для совпадения (default: 1)")
    parser.add_argument('--min-mag', type=float, default=None,
                        help="Фильтр reference по магнитуде (MPVA) >= порога, до сравнения")
    parser.add_argument('--min-cand-mag', type=float, default=None,
                        help="Фильтр candidate (наша ML) >= порога, до сравнения — "
                             "события без вычисленной ML тоже отбрасываются")
    parser.add_argument('--ref-year', type=int, default=None,
                        help="Фильтр reference по году (t0), до сравнения")
    parser.add_argument('--ref-month', type=int, default=None,
                        help="Фильтр reference по месяцу (t0), до сравнения")
    parser.add_argument('--show-missed', action='store_true')
    parser.add_argument('--out-csv', default=None)
    parser.add_argument('--location-error', action='store_true',
                        help="Дополнительно: ошибка эпицентра/глубины candidate против reference "
                             "по совпавшим событиям, где координаты есть в обоих бюллетенях "
                             "(candidate — из export_bul.py --locator-mode enrich)")
    parser.add_argument('--match-mode', choices=['stations', 'location'], default='stations',
                        help="stations (default) — время + общие станции, как раньше; location — "
                             "для событий с координатами в обоих бюллетенях: время + расстояние "
                             "между эпицентрами <= --max-dist-km, без требования общих станций")
    parser.add_argument('--max-dist-km', type=float, default=50.0,
                        help="С --match-mode location: максимальное расстояние между эпицентрами, км "
                             "(default: 50)")
    parser.add_argument('--no-coords-match', choices=['stations', 'time'], default='stations',
                        help="С --match-mode location: как сопоставлять события без координат — "
                             "stations (default): время + --min-shared общих станций; time: только время")
    parser.add_argument('--loc-good-only', action='store_true',
                        help="С --match-mode location: по месту сопоставлять только события candidate "
                             "с хорошим решением локатора (Err depth > --loc-min-err-depth и RMS < "
                             "--loc-max-rms); остальные — как события без координат (--no-coords-match)")
    parser.add_argument('--loc-max-rms', type=float, default=2.0,
                        help="С --loc-good-only: максимальный RMS решения, с (default: 2.0)")
    parser.add_argument('--loc-min-err-depth', type=float, default=0.001,
                        help="С --loc-good-only: Err depth должна быть больше этого, км (default: 0.001 — "
                             "0 означает упор глубины в границу таблицы годографов)")
    args = parser.parse_args()

    reference = parse_bul(args.reference)
    candidate = parse_bul(args.candidate)
    if args.location_error or args.match_mode == 'location':
        for events, path in ((reference, args.reference), (candidate, args.candidate)):
            for ev, org in zip(events, parse_bul_origins(path)):
                ev.update(org)
    print(f"Reference: {args.reference}  ({len(reference)} событий)")
    print(f"Candidate: {args.candidate}  ({len(candidate)} событий)")

    if args.ref_year is not None or args.ref_month is not None:
        before = len(reference)
        reference = [r for r in reference if r['t0'] is not None
                     and (args.ref_year  is None or r['t0'].year  == args.ref_year)
                     and (args.ref_month is None or r['t0'].month == args.ref_month)]
        print(f"Reference после фильтра год={args.ref_year} месяц={args.ref_month}: "
              f"{len(reference)} (из {before})")

    if args.min_mag is not None:
        before = len(reference)
        no_mag = sum(1 for r in reference if r['mag'] is None)
        reference = [r for r in reference if r['mag'] is not None and r['mag'] >= args.min_mag]
        print(f"Reference после фильтра mag >= {args.min_mag}: {len(reference)} (из {before}; "
              f"{no_mag} без магнитуды в бюллетене)")

    if args.min_cand_mag is not None:
        before = len(candidate)
        no_mag = sum(1 for c in candidate if c['mag'] is None)
        candidate = [c for c in candidate if c['mag'] is not None and c['mag'] >= args.min_cand_mag]
        print(f"Candidate после фильтра ML >= {args.min_cand_mag}: {len(candidate)} (из {before}; "
              f"{no_mag} без вычисленной ML)")

    if args.match_mode == 'location':
        matched, missed = match_events_v2(reference, candidate, args.window, args.min_shared,
                                          args.max_dist_km, args.no_coords_match,
                                          args.loc_good_only, args.loc_max_rms,
                                          args.loc_min_err_depth)
        n_loc = sum(1 for m in matched if m['by'] == 'location')
        good_s = (f" (только хорошие решения: RMS < {args.loc_max_rms}, Err depth > "
                  f"{args.loc_min_err_depth})" if args.loc_good_only else "")
        print(f"Сопоставление: время ±{args.window:.0f}с + эпицентр <= {args.max_dist_km:.0f} км"
              f"{good_s}; без координат — {args.no_coords_match}. По месту: {n_loc}, "
              f"без координат: {len(matched) - n_loc}")
    else:
        matched, missed = match_events(reference, candidate, args.window, args.min_shared)
    print_report(matched, missed, reference, candidate, args.window, args.min_shared, args.show_missed)
    if args.location_error:
        print_location_report(matched)

    if args.out_csv:
        write_csv(matched, missed, args.out_csv)


if __name__ == "__main__":
    main()
