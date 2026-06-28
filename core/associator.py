import os
import sys
import csv
import shutil
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from EQTransformer.utils.associator import run_associator_v2

SOURCE_DIR = os.path.join(_ROOT, 'data-in-memory', "output_cpu")
INPUT_DIR  = os.path.join(_ROOT, 'data-in-memory', "assoc_input")
OUTPUT_DIR = os.path.join(_ROOT, 'data-in-memory', "assoc_output")

# v5-обработанные станции
STATIONS = ['ANN', 'BEYR']
# STATIONS = ['BEYR', 'DOMR', 'GLDR', 'GOYR', 'GRYR', 'GUZR', 'LABN', 'MRNR',  'PYA1', 'SHA1', 'SOC', 'SPGR', 'VSLR', 'ZEI', 'SRGR', 'DIGR', 'NCK', 'GOFR']
# 'LSNR', 'GOFR', 'NEUR', 'NCK' сильно портят результат (+ много шумов без повышения рекала)
DET_THR  = 0.75 # / 0.75
P_THR    = 0.35 # / 0.35
S_THR    = 0.20 # / 0.20
KEEP_PS  = False

# SNR ниже порога означает пик, не различимый от шума. 0.0 = фильтр отключён.
P_SNR_THR = 0.0  # / 8.0
S_SNR_THR = 0.0 # / 1.25

# Режим учёта неопределённости:
#   'none'   — uncertainty игнорируется полностью (как до MC Dropout)
#   'weight' — как в оригинальном EQT: фильтрация по raw prob,
#              но в staging CSV записываются эффективные вероятности
#              p_prob * (1 - p_unc) — downstream ассоциатор видит взвешенные значения
#   'filter' — наш вариант: фильтрация по effective prob = p_prob * (1 - p_unc),
#              в staging CSV остаются исходные вероятности
UNCERTAINTY_MODE = 'weight'


def _eff_prob(prob, unc_str):
    """prob * (1 - unc). Если unc отсутствует или вне [0,1] — возвращает prob."""
    try:
        u = float(unc_str)
        if 0.0 <= u <= 1.0:
            return prob * (1.0 - u)
    except (ValueError, TypeError):
        pass
    return prob


def _passes(row):
    """Фильтр по сырым вероятностям — как в оригинальном EQT.
    Uncertainty не влияет на прохождение фильтра."""
    try:
        det = float(row.get('detection_probability', 0) or 0)
    except ValueError:
        return False
    if det < DET_THR:
        return False

    p_time = row.get('p_arrival_time', '').strip()
    try:
        p_prob = float(row.get('p_probability', '') or 0)
    except ValueError:
        p_prob = 0.0
    has_p = bool(p_time) and p_time.lower() != 'none' and p_prob >= P_THR

    if has_p and P_SNR_THR > 0.0:
        try:
            p_snr = float(row.get('p_snr', 0) or 0)
        except ValueError:
            p_snr = 0.0
        if p_snr < P_SNR_THR:
            has_p = False

    s_time = row.get('s_arrival_time', '').strip()
    try:
        s_prob = float(row.get('s_probability', '') or 0)
    except ValueError:
        s_prob = 0.0
    has_s = bool(s_time) and s_time.lower() != 'none' and s_prob >= S_THR

    if has_s and S_SNR_THR > 0.0:
        try:
            s_snr = float(row.get('s_snr', 0) or 0)
        except ValueError:
            s_snr = 0.0
        if s_snr < S_SNR_THR:
            has_s = False

    if KEEP_PS:
        return has_p and has_s
    return has_p or has_s


def _passes_v2(row):
    """Фильтр по эффективной вероятности p_prob * (1 - p_unc).
    Пики с высокой неопределённостью отсеиваются строже."""
    try:
        det = float(row.get('detection_probability', 0) or 0)
    except ValueError:
        return False
    if _eff_prob(det, row.get('detection_uncertainty', '')) < DET_THR:
        return False

    p_time = row.get('p_arrival_time', '').strip()
    try:
        p_prob = float(row.get('p_probability', '') or 0)
    except ValueError:
        p_prob = 0.0
    has_p = (bool(p_time) and p_time.lower() != 'none'
             and _eff_prob(p_prob, row.get('p_uncertainty', '')) >= P_THR)

    if has_p and P_SNR_THR > 0.0:
        try:
            p_snr = float(row.get('p_snr', 0) or 0)
        except ValueError:
            p_snr = 0.0
        if p_snr < P_SNR_THR:
            has_p = False

    s_time = row.get('s_arrival_time', '').strip()
    try:
        s_prob = float(row.get('s_probability', '') or 0)
    except ValueError:
        s_prob = 0.0
    has_s = (bool(s_time) and s_time.lower() != 'none'
             and _eff_prob(s_prob, row.get('s_uncertainty', '')) >= S_THR)

    if has_s and S_SNR_THR > 0.0:
        try:
            s_snr = float(row.get('s_snr', 0) or 0)
        except ValueError:
            s_snr = 0.0
        if s_snr < S_SNR_THR:
            has_s = False

    if KEEP_PS:
        return has_p and has_s
    return has_p or has_s


def _apply_weight(row):
    """Возвращает копию row с вероятностями, взвешенными по uncertainty.
    Используется в режиме 'weight' — downstream ассоциатор видит взвешенные значения."""
    row = dict(row)
    for prob_col, unc_col in [('detection_probability', 'detection_uncertainty'),
                               ('p_probability',         'p_uncertainty'),
                               ('s_probability',         's_uncertainty')]:
        try:
            prob = float(row.get(prob_col, '') or 0)
        except ValueError:
            continue
        eff = _eff_prob(prob, row.get(unc_col, ''))
        row[prob_col] = round(eff, 4)
    return row


def prepare_station(src, dst):
    if UNCERTAINTY_MODE == 'filter':
        filter_fn = _passes_v2
    else:
        filter_fn = _passes   # 'none' и 'weight' — фильтр по raw prob

    kept = 0
    with open(src, newline='', encoding='utf-8') as fin, \
         open(dst, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if not filter_fn(row):
                continue
            if UNCERTAINTY_MODE == 'weight':
                row = _apply_weight(row)
            writer.writerow(row)
            kept += 1
    print(f"    режим: {UNCERTAINTY_MODE!r}, детекций: {kept}")
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

start_time = '2024-05-01 00:00:00.000'
end_time   = '2024-06-01 00:00:00.000'

# moving_window=60: width of association window in seconds
# step = moving_window // 2 = 30s (computed inside _dbs_associator_v2)
# pair_n=3: minimum unique stations required
print("\nЗапуск ассоциативного модуля v2 (p_arrival_time + sliding window)...")
run_associator_v2(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    start_time=start_time,
    end_time=end_time,
    moving_window=50,
    consider_combination=False,
    pair_n=6,
    coherence_tolerance=3.0,   # 15.0 = старое значение; 2.0-3.0s показывает лучшие значения
)
