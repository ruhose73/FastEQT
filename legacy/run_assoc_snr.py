"""
Пересобирает staging (assoc_input) с P_SNR_THR=8.0 / S_SNR_THR=1.25
(вместо отключённых 0.0/0.0) поверх уже применённого фильтра дальних станций
(59 легитимных кавказских станций, см. bul_out/keep_stations.txt), затем
гоняет run_associator_v2 с теми же pair_n/moving_window/coherence_tolerance,
что и текущий прод.

Источник — уже посчитанные CSV детектора (memory-25/output_cpu,
output_cpu_2), mseed не нужен.
"""
import csv
import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

DET_THR   = 0.75
P_THR     = 0.35
S_THR     = 0.20
KEEP_PS   = False
P_SNR_THR = 8.0
S_SNR_THR = 1.25
UNCERTAINTY_MODE = 'weight'

with open(os.path.join(_ROOT, "bul_out", "keep_stations.txt")) as f:
    STATIONS = [s.strip() for s in f if s.strip()]


def _eff_prob(prob, unc_str):
    try:
        u = float(unc_str)
        if 0.0 <= u <= 1.0:
            return prob * (1.0 - u)
    except (ValueError, TypeError):
        pass
    return prob


def _passes(row):
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


def _apply_weight(row):
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
    kept = 0
    with open(src, newline='', encoding='utf-8') as fin, \
         open(dst, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if not _passes(row):
                continue
            if UNCERTAINTY_MODE == 'weight':
                row = _apply_weight(row)
            writer.writerow(row)
            kept += 1
    return kept


def stage(source_dir, input_dir):
    if os.path.isdir(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir, exist_ok=True)
    for st in STATIONS:
        src = os.path.join(source_dir, st, f"{st.lower()}.csv")
        dst_dir = os.path.join(input_dir, st)
        dst = os.path.join(dst_dir, "X_prediction_results.csv")
        if os.path.exists(src):
            os.makedirs(dst_dir, exist_ok=True)
            kept = prepare_station(src, dst)
            print(f"  {st}: {kept} детекций после фильтра (SNR>=8.0/1.25)")
        else:
            print(f"  WARNING: {src} not found, skipping {st}")


if __name__ == "__main__":
    from EQTransformer.utils.associator import run_associator_v2

    print("=== Q1: staging (output_cpu -> assoc_input_snr) ===")
    stage(os.path.join(_ROOT, "memory-25", "output_cpu"),
          os.path.join(_ROOT, "memory-25", "assoc_input_snr"))

    print("\n=== Q1: run_associator_v2 ===")
    if os.path.exists("phase_dataset"):
        os.remove("phase_dataset")
    os.makedirs(os.path.join(_ROOT, "memory-25", "assoc_output_snr"), exist_ok=True)
    run_associator_v2(
        input_dir=os.path.join(_ROOT, "memory-25", "assoc_input_snr"),
        output_dir=os.path.join(_ROOT, "memory-25", "assoc_output_snr"),
        start_time="2025-01-01 00:00:00.000",
        end_time="2025-04-01 00:00:00.000",
        moving_window=50,
        consider_combination=False,
        pair_n=6,
        coherence_tolerance=3.0,
    )

    print("\n=== Q2: staging (output_cpu_2 -> assoc_input_cpu_2_snr) ===")
    stage(os.path.join(_ROOT, "memory-25", "output_cpu_2"),
          os.path.join(_ROOT, "memory-25", "assoc_input_cpu_2_snr"))

    print("\n=== Q2: run_associator_v2 ===")
    if os.path.exists("phase_dataset"):
        os.remove("phase_dataset")
    os.makedirs(os.path.join(_ROOT, "memory-25", "assoc_output_cpu_2_snr"), exist_ok=True)
    run_associator_v2(
        input_dir=os.path.join(_ROOT, "memory-25", "assoc_input_cpu_2_snr"),
        output_dir=os.path.join(_ROOT, "memory-25", "assoc_output_cpu_2_snr"),
        start_time="2025-04-01 00:00:00.000",
        end_time="2025-07-01 00:00:00.000",
        moving_window=50,
        consider_combination=False,
        pair_n=6,
        coherence_tolerance=3.0,
    )

    print("\nГотово: memory-25/assoc_output_snr и memory-25/assoc_output_cpu_2_snr")
