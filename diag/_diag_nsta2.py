"""N_sta distribution + анализ missing event 2024-05-06 12:25:10."""
import xml.etree.ElementTree as ET, collections, os, sys, glob
from datetime import datetime, timedelta

BED_NS = 'http://quakeml.org/xmlns/bed/1.2'
ROOT = os.path.dirname(os.path.abspath(__file__))

# Ищем файл: аргумент или по умолчанию
if len(sys.argv) > 1:
    XML = sys.argv[1]
else:
    XML = os.path.join(ROOT, 'data-in-memory', 'gpu_splimit_45_may',
                       'assoc_output_lim', 'associations_ml1p5.xml')

print(f"Файл: {XML}")
tree = ET.parse(XML)
events_raw = tree.getroot().findall(f'.//{{{BED_NS}}}event')
print(f"Событий: {len(events_raw)}\n")

counts = collections.Counter()
all_events = []
for ev in events_raw:
    stas = set()
    for pick in ev.findall(f'{{{BED_NS}}}pick'):
        wf = pick.find(f'{{{BED_NS}}}waveformID')
        if wf is not None:
            stas.add(wf.get('stationCode', '').strip())
    orig = ev.find(f'.//{{{BED_NS}}}origin/{{{BED_NS}}}time/{{{BED_NS}}}value')
    t = None
    if orig is not None and orig.text:
        try:
            t = datetime.fromisoformat(orig.text.rstrip('Z'))
        except Exception:
            pass
    counts[len(stas)] += 1
    all_events.append((t, len(stas)))

# N_sta distribution
total = sum(counts.values())
cum_top = 0
print(f'{"N_sta":>5} | {"count":>6} | {"%":>5} | {"cum_top":>10}')
print('-' * 38)
for n in sorted(counts.keys(), reverse=True):
    cum_top += counts[n]
    bar = '#' * min(40, int(counts[n] / total * 400))
    print(f'{n:5d} | {counts[n]:6d} | {counts[n]/total*100:4.1f}% | '
          f'{cum_top:5d} ({cum_top/total*100:4.1f}%)')

# Ищем missing event 2024-05-06 12:25:10
TARGET = datetime(2024, 5, 6, 12, 25, 10)
WINDOW = 60  # сек
print(f'\n--- Поиск 2024-05-06 12:25:10 (±{WINDOW}s) ---')
found = [(t, n) for t, n in all_events
         if t is not None and abs((t - TARGET).total_seconds()) <= WINDOW]
if found:
    for t, n in sorted(found):
        dt = (t - TARGET).total_seconds()
        print(f'  {t}  N_sta={n}  dt={dt:+.1f}s')
else:
    print('  НЕ НАЙДЕНО в файле')

# N_sta порог → остаток событий
print('\n--- Если применить pair_n как пост-фильтр ---')
print(f'{"N_sta>=":>8} | {"events":>7} | {"% total":>7} | {"recall safe?":>12}')
print('-' * 45)
for thr in [6, 8, 10, 12, 13, 14, 15, 17]:
    n_ev = sum(v for k, v in counts.items() if k >= thr)
    # catalog min N_sta = 13 (with new run: might be 13 still)
    safe = "OK (min=13)" if thr <= 13 else "риск"
    print(f'{thr:8d} | {n_ev:7d} | {n_ev/total*100:6.1f}% | {safe:>12}')
