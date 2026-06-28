"""Сравнение raw associations.xml и ml-фильтрованного файла."""
import xml.etree.ElementTree as ET, collections, os
from datetime import datetime

BED_NS = 'http://quakeml.org/xmlns/bed/1.2'
ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, 'data-in-memory', 'gpu_splimit_45_may', 'assoc_output_lim')

files = {
    'raw': os.path.join(BASE, 'associations.xml'),
    'ml1p5': os.path.join(BASE, 'associations_ml1p5.xml'),
}

TARGET = datetime(2024, 5, 6, 12, 25, 10)
WINDOW = 120

for label, path in files.items():
    if not os.path.exists(path):
        print(f"{label}: файл не найден ({path})")
        continue
    tree = ET.parse(path)
    evs = tree.getroot().findall(f'.//{{{BED_NS}}}event')
    counts = collections.Counter()
    found = []
    for ev in evs:
        stas = set()
        for pick in ev.findall(f'{{{BED_NS}}}pick'):
            wf = pick.find(f'{{{BED_NS}}}waveformID')
            if wf is not None:
                stas.add(wf.get('stationCode', '').strip())
        counts[len(stas)] += 1
        orig = ev.find(f'.//{{{BED_NS}}}origin/{{{BED_NS}}}time/{{{BED_NS}}}value')
        if orig is not None and orig.text:
            try:
                t = datetime.fromisoformat(orig.text.rstrip('Z'))
                if abs((t - TARGET).total_seconds()) <= WINDOW:
                    found.append((t, len(stas)))
            except Exception:
                pass

    total = len(evs)
    print(f"\n{'='*55}")
    print(f"{label}: {total} событий")
    cum = 0
    for n in sorted(counts.keys(), reverse=True):
        cum += counts[n]
        if n >= 6:
            print(f"  N_sta>={n:2d}: {cum:6d} ({cum/total*100:5.1f}%)")
        if n < 6:
            break

    print(f"\n  2024-05-06 12:25:10 ±{WINDOW}s:")
    if found:
        for t, n in sorted(found):
            dt = (t - TARGET).total_seconds()
            print(f"    {t}  N_sta={n}  dt={dt:+.1f}s")
    else:
        print("    НЕ НАЙДЕНО")
