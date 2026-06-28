"""Распределение N_sta в associations.xml."""
import xml.etree.ElementTree as ET, collections, os

BED_NS = 'http://quakeml.org/xmlns/bed/1.2'
ROOT = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(ROOT, 'data-in-memory', 'gpu_splimit_45_may',
                   'assoc_output_lim', 'associations.xml')

tree = ET.parse(XML)
counts = collections.Counter()
for ev in tree.getroot().findall(f'.//{{{BED_NS}}}event'):
    stas = set()
    for pick in ev.findall(f'{{{BED_NS}}}pick'):
        wf = pick.find(f'{{{BED_NS}}}waveformID')
        if wf is not None:
            stas.add(wf.get('stationCode', '').strip())
    counts[len(stas)] += 1

total = sum(counts.values())
cum_top = 0
print(f'{"N_sta":>5} | {"count":>6} | {"%total":>6} | {"cum_from_top":>12}')
print('-' * 42)
for n in sorted(counts.keys(), reverse=True):
    cum_top += counts[n]
    print(f'{n:5d} | {counts[n]:6d} | {counts[n]/total*100:5.1f}%  |'
          f' {cum_top:6d} ({cum_top/total*100:5.1f}%)')

print(f'\nTotal: {total}')
print('\n--- Каталожные события (N_sta из validate_associator): ---')
# matched catalog events N_sta: 14,17,20,16,13,17,20,16,21,17,19
cat_nsta = [14, 17, 20, 16, 13, 17, 20, 16, 21, 17, 19]
print(f'min={min(cat_nsta)}, max={max(cat_nsta)}, median={sorted(cat_nsta)[len(cat_nsta)//2]}')
