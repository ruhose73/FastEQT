"""
ml_calibrate.py — регрессионная калибровка формулы ML.

Читает sta_corrections_detail.csv (выход sta_correction_estimator.py) и решает:
    Ms_i = lg(A_nm_i) + b_log·lg(R_i) + b_lin·R_i + c + S_sta_i + ε_i

Режимы:
  full    — подбирает b_log, b_lin, c и S_sta совместно (OLS).
  station — фиксирует b_log, b_lin (Дягилев), подбирает только c и S_sta.

Использование:
    python core/ml_calibrate.py
    python core/ml_calibrate.py --mode station
    python core/ml_calibrate.py --exclude-stations KMKR,KRNR --min-ms 1.0
    python core/ml_calibrate.py --mode full --min-events 3
"""

import argparse
import csv
import math
import os
from collections import Counter

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DETAIL = os.path.join(_ROOT, 'sta_corrections_detail.csv')
DEFAULT_OUT    = os.path.join(_ROOT, 'sta_corrections_calibrated.csv')

# Исходные коэффициенты Дягилев 2023
B_LOG_ORIG = 1.024
B_LIN_ORIG = 0.001648
C_ORIG     = -1.889


# ── Загрузка данных ───────────────────────────────────────────────────────────

def load_rows(path, min_events=3, min_ms=None, max_ms=None,
              exclude_outliers=True, exclude_stations=None):
    excl = set(exclude_stations) if exclude_stations else set()
    raw = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sta = row['station'].strip()
            if sta in excl:
                continue
            if exclude_outliers and row.get('is_outlier', '') == 'yes':
                continue
            try:
                ms   = float(row['Ms'])
                r_km = float(row['R_km'])
                a_nm = float(row['A_nm'])
            except (ValueError, KeyError):
                continue
            if min_ms is not None and ms < min_ms:
                continue
            if max_ms is not None and ms > max_ms:
                continue
            if a_nm <= 0 or r_km <= 0:
                continue
            raw.append({'sta': sta, 'Ms': ms, 'R_km': r_km, 'A_nm': a_nm})

    counts = Counter(r['sta'] for r in raw)
    rows = [r for r in raw if counts[r['sta']] >= min_events]
    return rows, counts


# ── Регрессия: полная (подбирает b_log, b_lin) ───────────────────────────────

def run_full(rows):
    """
    Модель без глобального интерсепта, станционные константы его поглощают.
    X_i = [lg(R_i), R_i, 1_{sta=s1}, ..., 1_{sta=sN}]
    theta = [b_log, b_lin, S_sta_s1, ..., S_sta_sN]
    Нормировка: c = mean(S_sta), S_sta -= c.
    """
    stations = sorted(set(r['sta'] for r in rows))
    idx = {s: i for i, s in enumerate(stations)}
    n, ns = len(rows), len(stations)

    X = np.zeros((n, 2 + ns))
    y = np.zeros(n)
    for i, r in enumerate(rows):
        X[i, 0] = math.log10(r['R_km'])
        X[i, 1] = r['R_km']
        X[i, 2 + idx[r['sta']]] = 1.0
        y[i] = r['Ms'] - math.log10(r['A_nm'])

    theta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    b_log, b_lin = float(theta[0]), float(theta[1])

    s_raw = theta[2:]
    c = float(np.mean(s_raw))
    sta_corr = {s: float(s_raw[i]) - c for i, s in enumerate(stations)}

    res = y - X @ theta
    return b_log, b_lin, c, sta_corr, res, stations, rank


# ── Регрессия: только поправки (b_log, b_lin фиксированы) ────────────────────

def run_station(rows, b_log=B_LOG_ORIG, b_lin=B_LIN_ORIG):
    """
    Фиксирует b_log и b_lin, подбирает только c и S_sta.
    Эквивалентно per-station mean, но computed jointly (OLS).
    """
    stations = sorted(set(r['sta'] for r in rows))
    idx = {s: i for i, s in enumerate(stations)}
    n, ns = len(rows), len(stations)

    X = np.zeros((n, ns))
    y = np.zeros(n)
    for i, r in enumerate(rows):
        X[i, idx[r['sta']]] = 1.0
        y[i] = (r['Ms'] - math.log10(r['A_nm'])
                - b_log * math.log10(r['R_km'])
                - b_lin * r['R_km'])

    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    c = float(np.mean(theta))
    sta_corr = {s: float(theta[i]) - c for i, s in enumerate(stations)}

    res = y - X @ theta
    return b_log, b_lin, c, sta_corr, res, stations


# ── Вывод результатов ─────────────────────────────────────────────────────────

def print_results(b_log, b_lin, c, sta_corr, res, mode_label):
    print(f"\n{'=' * 65}")
    print(f"РЕЗУЛЬТАТЫ РЕГРЕССИИ  [{mode_label}]")
    print(f"{'=' * 65}")
    print(f"  Коэффициенты формулы:")
    changed_blog = abs(b_log - B_LOG_ORIG) > 1e-6
    changed_blin = abs(b_lin - B_LIN_ORIG) > 1e-6
    b_log_note = f"  ← был {B_LOG_ORIG}" if changed_blog else "  (без изменений)"
    b_lin_note = f"  ← был {B_LIN_ORIG:.6f}" if changed_blin else "  (без изменений)"
    c_note     = f"  ← был {C_ORIG}"
    print(f"    ML_B_LOG = {b_log:.4f}{b_log_note}")
    print(f"    ML_B_LIN = {b_lin:.6f}{b_lin_note}")
    print(f"    ML_C     = {c:.4f}{c_note}")

    vals = sorted(sta_corr.values())
    print(f"\n  Невязки Ms − ML_fit  (N={len(res)} строк):")
    print(f"    mean  = {np.mean(res):+.3f}")
    print(f"    std   = {np.std(res):.3f}")
    print(f"    p25   = {float(np.percentile(res, 25)):+.3f}  "
          f"p75 = {float(np.percentile(res, 75)):+.3f}")
    print(f"    min   = {np.min(res):.3f}   max = {np.max(res):.3f}")

    print(f"\n  Поправки станций  [{len(sta_corr)} станций]  "
          f"(range [{min(vals):+.3f} … {max(vals):+.3f}]):")
    print(f"  {'Станция':8s}  {'S_new':>7s}   N событий")
    print(f"  {'-' * 35}")
    for sta in sorted(sta_corr):
        print(f"  {sta:8s}  {sta_corr[sta]:+7.3f}")


def save_corrections(sta_corr, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['station', 'S'])
        for sta in sorted(sta_corr):
            w.writerow([sta, f"{sta_corr[sta]:.3f}"])
    print(f"\n  Поправки сохранены: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Регрессионная калибровка формулы ML',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--detail',           default=DEFAULT_DETAIL)
    parser.add_argument('--out',              default=DEFAULT_OUT)
    parser.add_argument('--mode',             default='full',
                        choices=['full', 'station'],
                        help='full — подбирает b_log,b_lin,c,S; '
                             'station — только c,S (b_log,b_lin Дягилева)')
    parser.add_argument('--min-events',       type=int, default=3,
                        help='Мин. событий на станцию')
    parser.add_argument('--min-ms',           type=float, default=None)
    parser.add_argument('--max-ms',           type=float, default=None)
    parser.add_argument('--exclude-outliers', action='store_true', default=True)
    parser.add_argument('--exclude-stations', default=None,
                        help='Станции через запятую: KMKR,KRNR')
    parser.add_argument('--no-save',          action='store_true',
                        help='Не сохранять выходной файл')
    parser.add_argument('--compare-both',     action='store_true',
                        help='Показать оба режима для сравнения')
    args = parser.parse_args()

    excl = ([s.strip() for s in args.exclude_stations.split(',') if s.strip()]
            if args.exclude_stations else None)

    print(f"Входной файл: {args.detail}")
    rows, counts = load_rows(
        args.detail,
        min_events=args.min_events,
        min_ms=args.min_ms,
        max_ms=args.max_ms,
        exclude_outliers=args.exclude_outliers,
        exclude_stations=excl,
    )

    if not rows:
        print("Нет данных после фильтрации.")
        return

    stations = sorted(set(r['sta'] for r in rows))
    print(f"Строк:    {len(rows)}")
    print(f"Станций:  {len(stations)}")
    print(f"Событий на станцию (мин/макс): "
          f"{min(counts[s] for s in stations)} / "
          f"{max(counts[s] for s in stations)}")
    if excl:
        print(f"Исключены: {excl}")

    if args.compare_both or args.mode == 'full':
        b_log, b_lin, c, sta_corr, res, stas, rank = run_full(rows)
        print_results(b_log, b_lin, c, sta_corr, res, 'full — b_log,b_lin,c,S')
        if not args.no_save and args.mode == 'full':
            save_corrections(sta_corr, args.out)
            print(f"  Новые константы (вставить в ml_filter_v4.py и "
                  f"sta_correction_estimator.py):")
            print(f"    ML_B_LOG = {b_log:.4f}")
            print(f"    ML_B_LIN = {b_lin:.6f}")
            print(f"    ML_C     = {c:.4f}")

    if args.compare_both or args.mode == 'station':
        b_log2, b_lin2, c2, sc2, res2, stas2 = run_station(rows)
        print_results(b_log2, b_lin2, c2, sc2, res2,
                      f'station — b_log={B_LOG_ORIG}, b_lin={B_LIN_ORIG:.6f} fixed')
        if not args.no_save and args.mode == 'station':
            save_corrections(sc2, args.out)
            print(f"  Новые константы (вставить в ml_filter_v4.py и "
                  f"sta_correction_estimator.py):")
            print(f"    ML_C     = {c2:.4f}")


if __name__ == '__main__':
    main()
