"""Диагностика причин ML=None в associations.xml."""
import csv, os, xml.etree.ElementTree as ET, collections
from datetime import datetime

BED_NS = 'http://quakeml.org/xmlns/bed/1.2'
VP, VS = 6.0, 3.4883
R_MIN, R_MAX = 30.0, 600.0
N_MIN = 4
A_NM_MIN = 0.005

ROOT = os.path.dirname(os.path.abspath(__file__))
AMP_FILE = os.path.join(ROOT, 'amps_wa.csv')
XML_FILE = os.path.join(ROOT, 'data-in-memory', 'gpu_splimit_45_may',
                        'assoc_output_lim', 'associations.xml')

def sp_to_r(p_t, s_t):
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try:
            dt = (datetime.strptime(s_t, fmt) - datetime.strptime(p_t, fmt)).total_seconds()
            if dt > 0:
                return VP * VS / (VP - VS) * dt
        except Exception:
            pass
    return None

print("Загрузка амплитуд...")
amps = {}
with open(AMP_FILE, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        try:
            amps[(row['pub_id'], row['sta'])] = float(row['A'])
        except Exception:
            pass
print(f"  {len(amps)} пар")

print("Парсинг XML...")
tree = ET.parse(XML_FILE)
events = tree.getroot().findall(f'.//{{{BED_NS}}}event')
print(f"  {len(events)} событий\n")

reasons = collections.Counter()
r_lt30_counts = []   # сколько станций было в диапазоне R<30 у таких событий

for ev in events:
    pub_id = ev.get('publicID', '')
    picks_p, picks_s = {}, {}
    for pick in ev.findall(f'{{{BED_NS}}}pick'):
        wf = pick.find(f'{{{BED_NS}}}waveformID')
        ph_el = pick.find(f'{{{BED_NS}}}phaseHint')
        t_el = pick.find(f'{{{BED_NS}}}time/{{{BED_NS}}}value')
        if wf is None or ph_el is None or t_el is None:
            continue
        sta = wf.get('stationCode', '').strip()
        ph = (ph_el.text or '').strip()
        t = (t_el.text or '').rstrip('Z').replace('T', ' ')
        if ph == 'P':
            picks_p[sta] = t
        elif ph == 'S':
            picks_s[sta] = t

    n_lt30 = n_gt30 = n_with_amp = 0
    for sta in picks_p:
        if sta not in picks_s:
            continue
        r = sp_to_r(picks_p[sta], picks_s[sta])
        if r is None or r <= 0 or r > R_MAX:
            continue
        if r < R_MIN:
            n_lt30 += 1
        else:
            n_gt30 += 1
            A = amps.get((pub_id, sta))
            if A is not None and float(A) * 1e9 >= A_NM_MIN:
                n_with_amp += 1

    if n_with_amp >= N_MIN:
        reasons['has_ml'] += 1
    elif n_gt30 == 0:
        reasons['all_r_lt_30_km'] += 1
        r_lt30_counts.append(n_lt30)
    elif n_gt30 < N_MIN and n_with_amp < N_MIN:
        reasons[f'only_{n_gt30}_sta_gt30'] += 1
    elif n_with_amp < N_MIN:
        reasons['no_amp_in_cache'] += 1
    else:
        reasons['other'] += 1

print("Причины ML=None:")
for k, v in reasons.most_common():
    print(f"  {k:35s}: {v:6d}")

if r_lt30_counts:
    import statistics
    print(f"\nДля all_r_lt_30_km: медиана станций в R<30: {statistics.median(r_lt30_counts):.0f}, "
          f"макс: {max(r_lt30_counts)}")
