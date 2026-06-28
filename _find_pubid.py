"""Найти pub_id ближайшего события к заданному времени в raw XML."""
import xml.etree.ElementTree as ET, os, sys
from datetime import datetime

BED_NS = 'http://quakeml.org/xmlns/bed/1.2'
ROOT = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(ROOT, 'data-in-memory', 'gpu_splimit_45_may',
                   'assoc_output_lim', 'associations.xml')

TARGET_STR = sys.argv[1] if len(sys.argv) > 1 else '2024-05-06 12:25:25'
TARGET = datetime.fromisoformat(TARGET_STR)
WINDOW = float(sys.argv[2]) if len(sys.argv) > 2 else 30

tree = ET.parse(XML)
for ev in tree.getroot().findall(f'.//{{{BED_NS}}}event'):
    orig = ev.find(f'.//{{{BED_NS}}}origin/{{{BED_NS}}}time/{{{BED_NS}}}value')
    if orig is None or not orig.text:
        continue
    try:
        t = datetime.fromisoformat(orig.text.rstrip('Z'))
    except Exception:
        continue
    if abs((t - TARGET).total_seconds()) <= WINDOW:
        stas = set()
        for pick in ev.findall(f'{{{BED_NS}}}pick'):
            wf = pick.find(f'{{{BED_NS}}}waveformID')
            if wf is not None:
                stas.add(wf.get('stationCode', '').strip())
        dt = (t - TARGET).total_seconds()
        pub_id = ev.get('publicID', '')
        print(f"dt={dt:+.1f}s  N_sta={len(stas):2d}  {t}  {pub_id}")
