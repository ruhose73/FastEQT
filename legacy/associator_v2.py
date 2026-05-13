import os
import csv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from EQTransformer.utils.associator import run_associator

SOURCE_DIR = 'data-in-memory/output'
INPUT_DIR  = 'data-in-memory/assoc_input'   # staging: только наши станции
OUTPUT_DIR = 'data-in-memory/association'

# Stations to associate — add/remove as needed
# Только v5-обработанные станции (правильная нормализация)
STATIONS = ['SOC', 'VSLR', 'GUZR', 'BEYR', 'SHA1', 'MRNR', 'SPGR', 'DOMR', 'ZEI', 'LABN']

# Пороги фильтрации (применяются при подготовке staging CSV)
DET_THR  = 0.7
P_THR    = 0.3
S_THR    = 0.2
KEEP_PS  = True   # True = требовать обе фазы P и S (снижает шум)


def _passes(row):
    """Возвращает True если строка детекции проходит пороги."""
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
    """Копирует src в dst, оставляя только строки прошедшие пороги."""
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


# Собираем staging-папку: INPUT_DIR/{STATION}/X_prediction_results.csv
# Ассоциатор делает listdir(INPUT_DIR) сам — поэтому здесь только наши станции
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

# moving_window=60: covers P-wave travel time differences across a ~300 km network
# pair_n=2: event confirmed if seen on at least 2 stations
print("\nЗапуск ассоциативного модуля...")
run_associator(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    start_time=start_time,
    end_time=end_time,
    moving_window=60,
    consider_combination=False,
    pair_n=3,
)
