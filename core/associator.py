import os
import sys
import csv
import shutil
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from EQTransformer.utils.associator import run_associator_v2

SOURCE_DIR = os.path.join(_ROOT, 'data-in-memory', 'output_gpu_100_150')
INPUT_DIR  = os.path.join(_ROOT, 'data-in-memory', 'assoc_input_gpu_100_150')
OUTPUT_DIR = os.path.join(_ROOT, 'data-in-memory', 'association_gpu_100_150')

# v5-обработанные станции
STATIONS = ['BEYR', 'DOMR', 'GLDR', 'GOYR', 'GRYR', 'GUZR', 'LABN', 'MRNR', 'PYA1', 'SHA1', 'SOC', 'SPGR', 'VSLR', 'ZEI', 'SRGR']
# 'LSNR', 'GOFR', 'NEUR', 'SRGR', 'NCK' сильно портят результат (+ много шумов без повышения рекала)
# SRGR добавил +1 событие и 5к шумов
DET_THR  = 0.7
P_THR    = 0.3
S_THR    = 0.2
KEEP_PS  = False


def _passes(row):
    try:
        det = float(row.get('detection_probability', 0) or 0)
    except ValueError:
        return False
    if det < DET_THR:
        return False
    p_time = row.get('p_arrival_time', '').strip()
    p_prob_str = row.get('p_probability', '').strip()
    try:
        p_prob = float(p_prob_str)
    except ValueError:
        p_prob = 0.0
    has_p = bool(p_time) and p_time.lower() != 'none' and p_prob >= P_THR

    s_time = row.get('s_arrival_time', '').strip()
    s_prob_str = row.get('s_probability', '').strip()
    try:
        s_prob = float(s_prob_str)
    except ValueError:
        s_prob = 0.0
    has_s = bool(s_time) and s_time.lower() != 'none' and s_prob >= S_THR

    if KEEP_PS:
        return has_p and has_s
    return has_p or has_s


def prepare_station(src, dst):
    kept = 0
    with open(src, newline='', encoding='utf-8') as fin, \
         open(dst, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if _passes(row):
                writer.writerow(row)
                kept += 1
    return kept


# Удаляем каталоги станций не из STATIONS (остатки предыдущих запусков)
if os.path.isdir(INPUT_DIR):
    for stale in os.listdir(INPUT_DIR):
        if stale not in STATIONS:
            shutil.rmtree(os.path.join(INPUT_DIR, stale))
            print(f"  Removed stale staging dir: {stale}")

for st in STATIONS:
    src = os.path.join(SOURCE_DIR, st, f"{st.lower()}.csv")
    dst_dir = os.path.join(INPUT_DIR, st)
    dst = os.path.join(dst_dir, "X_prediction_results.csv")
    if os.path.exists(src):
        os.makedirs(dst_dir, exist_ok=True)
        kept = prepare_station(src, dst)
        print(f"  {st}: {kept} детекций после фильтра")
    else:
        print(f"  WARNING: {src} not found, skipping {st}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

start_time = '2024-01-01 00:00:00.000'
end_time   = '2024-02-01 00:00:00.000'

# moving_window=60: width of association window in seconds
# step = moving_window // 2 = 30s (computed inside _dbs_associator_v2)
# pair_n=3: minimum unique stations required
print("\nЗапуск ассоциативного модуля v2 (p_arrival_time + sliding window)...")
run_associator_v2(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    start_time=start_time,
    end_time=end_time,
    moving_window=30,
    consider_combination=False,
    pair_n=3,
)
