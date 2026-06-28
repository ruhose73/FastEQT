"""
Диагностика: сравнивает амплитуды в amps_sta_corr.csv (sta_correction_estimator)
и amps_wa.csv (ml_filter_v4) для событий, присутствующих в обоих кэшах.
Также показывает какой компонент найден в geofiles для каждой станции.
"""
import csv, os, glob
from collections import defaultdict

ROOT = 'd:/codding/aspirantura/EQTransformer'

# ── 1. Какой компонент используется для каждой станции ──────────────────────
print("=== КОМПОНЕНТЫ В geofiles/ ===")
geofiles = os.path.join(ROOT, 'geofiles')
COMP_PRIORITY = ('E', 'N', 'Z')
for sta in sorted(os.listdir(geofiles)):
    sta_path = os.path.join(geofiles, sta)
    if not os.path.isdir(sta_path):
        continue
    found = {}
    for fname in os.listdir(sta_path):
        parts = fname.split('__')
        if len(parts) != 3:
            continue
        dot_parts = parts[0].split('.')
        if len(dot_parts) < 4:
            continue
        comp = dot_parts[3][-1].upper()
        channel = dot_parts[3]
        found[comp] = channel
    picked = next((c for c in COMP_PRIORITY if c in found), None)
    channels_str = ', '.join(f"{c}={found[c]}" for c in sorted(found))
    print(f"  {sta:8s}  picked={picked}  available=[{channels_str}]")

# ── 2. Сравнение амплитуд между двумя кэшами ────────────────────────────────
print("\n=== СРАВНЕНИЕ КЭШЕЙ (sta_corr vs ml_filter) ===")
sta_corr_path = os.path.join(ROOT, 'amps_sta_corr.csv')
ml_filter_path = os.path.join(ROOT, 'amps_wa.csv')

cache_sc = {}
if os.path.isfile(sta_corr_path):
    with open(sta_corr_path) as f:
        for row in csv.DictReader(f):
            v = float(row['A']) if row['A'] else None
            cache_sc[(row['pub_id'], row['sta'])] = v
    print(f"  sta_corr: {len(cache_sc)} пар")
else:
    print(f"  sta_corr: файл не найден")

cache_ml = {}
if os.path.isfile(ml_filter_path):
    with open(ml_filter_path) as f:
        for row in csv.DictReader(f):
            v = float(row['A']) if row['A'] else None
            cache_ml[(row['pub_id'], row['sta'])] = v
    print(f"  ml_filter: {len(cache_ml)} пар")
else:
    print(f"  ml_filter: файл не найден")

# Пересечение
common = set(cache_sc) & set(cache_ml)
print(f"  Общих пар: {len(common)}")
if common:
    import math
    print(f"\n  {'pub_id (конец)':20s}  {'sta':6s}  {'A_sta_corr (м)':18s}  {'A_ml_filter (м)':18s}  {'ratio':>6s}")
    count = 0
    for key in sorted(common):
        a1 = cache_sc[key]
        a2 = cache_ml[key]
        if a1 is None or a2 is None:
            continue
        ratio = a1 / a2 if a2 > 0 else float('inf')
        pub_short = key[0][-20:]
        print(f"  {pub_short:20s}  {key[1]:6s}  {a1:18.6e}  {a2:18.6e}  {ratio:>6.3f}")
        count += 1
        if count >= 20:
            print("  ... (первые 20)")
            break
