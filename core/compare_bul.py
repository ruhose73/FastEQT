"""
compare_bul.py — сравнение двух бюллетеней в формате IMS1.0:SHORT/LONG
(bul/*.BUL от ГС РАН и bul_out/*.BUL из export_bul.py) по времени и станциям.

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
    python core/compare_bul.py --reference bul/2025_jan-mar_NCAU.BUL \
                                --candidate bul_out/2025_q1.BUL \
                                --window 60 --min-shared 1
    python core/compare_bul.py ... --show-missed --out-csv compare.csv
"""

import argparse
import csv
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
                        help="Эталонный бюллетень (напр. bul/2025_jan-mar_NCAU.BUL)")
    parser.add_argument('--candidate', required=True,
                        help="Наш бюллетень (напр. bul_out/2025_q1.BUL)")
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
    args = parser.parse_args()

    reference = parse_bul(args.reference)
    candidate = parse_bul(args.candidate)
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

    matched, missed = match_events(reference, candidate, args.window, args.min_shared)
    print_report(matched, missed, reference, candidate, args.window, args.min_shared, args.show_missed)

    if args.out_csv:
        write_csv(matched, missed, args.out_csv)


if __name__ == "__main__":
    main()
