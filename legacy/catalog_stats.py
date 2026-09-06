"""
Быстрая статистика каталога: магнитуды, расстояния, оценка полноты.
python catalog_stats.py
"""
import math
import os
import openpyxl
import collections

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CATALOG_PATH = os.path.join(_ROOT, "Каталог Сочи 300 км.xlsx")
REF_LAT, REF_LON = 43.57, 39.763  # SOC


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


wb = openpyxl.load_workbook(CATALOG_PATH, read_only=True, data_only=True)
ws = wb.active

events = []
for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i == 0:
        continue
    if row[0] is None or row[1] is None:
        continue
    ms = float(row[4]) if row[4] is not None else None
    dist = haversine_km(row[1], row[2], REF_LAT, REF_LON)
    events.append({'t': row[0], 'ms': ms, 'dist': dist, 'depth': row[3] or 10.0})
wb.close()

print(f"Всего событий в каталоге: {len(events)}")
print(f"Ms диапазон: {min(e['ms'] for e in events if e['ms'] is not None):.1f} "
      f"… {max(e['ms'] for e in events if e['ms'] is not None):.1f}")

# Распределение по магнитуде (шаг 0.5)
bins = collections.Counter()
for e in events:
    if e['ms'] is not None:
        b = math.floor(e['ms'] * 2) / 2  # округляем вниз до 0.5
        bins[b] += 1

print("\nРаспределение по Ms (все месяцы):")
print(f"  {'Ms':>6}  {'N':>5}  {'cumN≥Ms':>8}")
sorted_bins = sorted(bins.keys(), reverse=True)
cum = 0
for b in sorted_bins:
    cum += bins[b]
    print(f"  {b:>6.1f}  {bins[b]:>5}  {cum:>8}")

# Только январь 2024
jan = [e for e in events if e['t'].year == 2024 and e['t'].month == 1]
print(f"\nЯнварь 2024: {len(jan)} событий")
print(f"Ms диапазон: {min(e['ms'] for e in jan if e['ms'] is not None):.1f} "
      f"… {max(e['ms'] for e in jan if e['ms'] is not None):.1f}")

# Расстояния
jan_by_dist = sorted(jan, key=lambda e: e['dist'])
print("\nСобытия января по расстоянию от SOC:")
print(f"  {'Ms':>5}  {'Dist':>6}  {'Depth':>6}  {'Time'}")
for e in jan_by_dist:
    ms_s = f"{e['ms']:.1f}" if e['ms'] is not None else "  -"
    print(f"  {ms_s:>5}  {e['dist']:>6.1f}  {e['depth']:>6.1f}  {e['t']}")

# Оценка полноты (Gutenberg-Richter)
print("\nОценка Gutenberg-Richter (все месяцы, log10):")
for b in sorted(bins.keys()):
    n_above = sum(bins[bb] for bb in bins if bb >= b)
    print(f"  Ms≥{b:.1f}: {n_above:4d}  log10={math.log10(n_above):.2f}")
