"""
associator.py — сетевой ассоциатор (CLI).

Готовит per-станционный стейджинг (фильтр по порогам вероятности/SNR,
опционально взвешивание по MC Dropout неопределённости), затем запускает
run_associator_v2 (EQTransformer/utils/associator.py) — скользящее окно по
P-пикам + фильтр физической согласованности станций. На выходе —
associations.xml (QuakeML), Y2000.phs, traceNmae_dic.json.

Все параметры задаются флагами командной строки (см. --help) — список
станций (--stations, коды через запятую), пороги/пути/параметры окна.
Весь код обёрнут в run_associator() (можно импортировать без побочных
эффектов) с argparse-обвязкой поверх; _passes/_passes_v2/prepare_station
принимают пороги как параметры функций, а не глобальные константы.

Более старые версии (построчный скрипт без функций/CLI, группировка по
event_start_time вместо скользящего окна и т. п.) — см. legacy/README.md.
"""

import argparse
import os
import sys
import csv
import shutil
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from EQTransformer.utils.associator import run_associator_v2


# ─── Конфигурация (значения по умолчанию для CLI-флагов) ─────────────────────

SOURCE_DIR = os.path.join(_ROOT, 'workspace', 'detector', 'output')
INPUT_DIR  = os.path.join(_ROOT, 'workspace', 'associator', 'input')
OUTPUT_DIR = os.path.join(_ROOT, 'workspace', 'associator', 'output')

# v5-обработанные станции. Альтернативный, более короткий состав, который
# пробовали раньше ('BEYR,DOMR,GLDR,GOYR,GRYR,GUZR,LABN,MRNR,PYA1,SHA1,SOC,
# SPGR,VSLR,ZEI,SRGR,DIGR,NCK,GOFR') — LSNR/GOFR/NEUR/NCK сильно портили
# результат (много шума без повышения recall) при таком отборе.
STATIONS = ("AKT,ANN,ARKR,ARNR,BEYR,BTKR,BTLR,BUJR,BVTR,DBC,DIGR,DLMR,DOMR,"
            "DRN,DVE,ERBR,GLDR,GLVR,GOFR,GOYR,GROC,GRYR,GUZR,HNZR,KANR,KLMR,"
            "KMGR,KMKR,KORR,KRNR,KSMR,LABN,LACR,LSNR,MAK,MRNR,NCK,NEUR,NVPR,"
            "PXTR,PYA1,RPOR,SGKR,SHA1,SOC,SPGR,SRGR,STDR,SUKR,TLTR,TMNR,TRKR,"
            "UNCR,URKR,VLKR,VSLR,ZEI")

DET_THR  = 0.75
P_THR    = 0.35
S_THR    = 0.20
KEEP_PS  = False

# SNR ниже порога означает пик, не различимый от шума. 0.0 = фильтр отключён.
P_SNR_THR = 0.0
S_SNR_THR = 0.0

# Режим учёта неопределённости:
#   'none'   — uncertainty игнорируется полностью (как до MC Dropout)
#   'weight' — как в оригинальном EQT: фильтрация по raw prob,
#              но в staging CSV записываются эффективные вероятности
#              p_prob * (1 - p_unc) — downstream ассоциатор видит взвешенные значения
#   'filter' — наш вариант: фильтрация по effective prob = p_prob * (1 - p_unc),
#              в staging CSV остаются исходные вероятности
UNCERTAINTY_MODE = 'weight'

START_TIME = '2024-05-01 00:00:00.000'
END_TIME   = '2024-06-01 00:00:00.000'

# moving_window=60: ширина окна ассоциации в секундах
# step = moving_window // 2 (вычисляется внутри _dbs_associator_v2)
# pair_n: минимум уникальных станций для события
MOVING_WINDOW        = 50
PAIR_N               = 6
COHERENCE_TOLERANCE  = 3.0   # 15.0 = старое значение; 2.0-3.0s показывает лучшие результаты

# ─────────────────────────────────────────────────────────────────────────────


def _eff_prob(prob, unc_str):
    """prob * (1 - unc). Если unc отсутствует или вне [0,1] — возвращает prob."""
    try:
        u = float(unc_str)
        if 0.0 <= u <= 1.0:
            return prob * (1.0 - u)
    except (ValueError, TypeError):
        pass
    return prob


def _passes(row, det_thr, p_thr, s_thr, keep_ps, p_snr_thr, s_snr_thr):
    """Фильтр по сырым вероятностям — как в оригинальном EQT.
    Uncertainty не влияет на прохождение фильтра."""
    try:
        det = float(row.get('detection_probability', 0) or 0)
    except ValueError:
        return False
    if det < det_thr:
        return False

    p_time = row.get('p_arrival_time', '').strip()
    try:
        p_prob = float(row.get('p_probability', '') or 0)
    except ValueError:
        p_prob = 0.0
    has_p = bool(p_time) and p_time.lower() != 'none' and p_prob >= p_thr

    if has_p and p_snr_thr > 0.0:
        try:
            p_snr = float(row.get('p_snr', 0) or 0)
        except ValueError:
            p_snr = 0.0
        if p_snr < p_snr_thr:
            has_p = False

    s_time = row.get('s_arrival_time', '').strip()
    try:
        s_prob = float(row.get('s_probability', '') or 0)
    except ValueError:
        s_prob = 0.0
    has_s = bool(s_time) and s_time.lower() != 'none' and s_prob >= s_thr

    if has_s and s_snr_thr > 0.0:
        try:
            s_snr = float(row.get('s_snr', 0) or 0)
        except ValueError:
            s_snr = 0.0
        if s_snr < s_snr_thr:
            has_s = False

    if keep_ps:
        return has_p and has_s
    return has_p or has_s


def _passes_v2(row, det_thr, p_thr, s_thr, keep_ps, p_snr_thr, s_snr_thr):
    """Фильтр по эффективной вероятности p_prob * (1 - p_unc).
    Пики с высокой неопределённостью отсеиваются строже."""
    try:
        det = float(row.get('detection_probability', 0) or 0)
    except ValueError:
        return False
    if _eff_prob(det, row.get('detection_uncertainty', '')) < det_thr:
        return False

    p_time = row.get('p_arrival_time', '').strip()
    try:
        p_prob = float(row.get('p_probability', '') or 0)
    except ValueError:
        p_prob = 0.0
    has_p = (bool(p_time) and p_time.lower() != 'none'
             and _eff_prob(p_prob, row.get('p_uncertainty', '')) >= p_thr)

    if has_p and p_snr_thr > 0.0:
        try:
            p_snr = float(row.get('p_snr', 0) or 0)
        except ValueError:
            p_snr = 0.0
        if p_snr < p_snr_thr:
            has_p = False

    s_time = row.get('s_arrival_time', '').strip()
    try:
        s_prob = float(row.get('s_probability', '') or 0)
    except ValueError:
        s_prob = 0.0
    has_s = (bool(s_time) and s_time.lower() != 'none'
             and _eff_prob(s_prob, row.get('s_uncertainty', '')) >= s_thr)

    if has_s and s_snr_thr > 0.0:
        try:
            s_snr = float(row.get('s_snr', 0) or 0)
        except ValueError:
            s_snr = 0.0
        if s_snr < s_snr_thr:
            has_s = False

    if keep_ps:
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


def prepare_station(src, dst, uncertainty_mode, det_thr, p_thr, s_thr,
                     keep_ps, p_snr_thr, s_snr_thr):
    if uncertainty_mode == 'filter':
        def filter_fn(row):
            return _passes_v2(row, det_thr, p_thr, s_thr, keep_ps, p_snr_thr, s_snr_thr)
    else:
        def filter_fn(row):  # 'none' и 'weight' — фильтр по raw prob
            return _passes(row, det_thr, p_thr, s_thr, keep_ps, p_snr_thr, s_snr_thr)

    kept = 0
    with open(src, newline='', encoding='utf-8') as fin, \
         open(dst, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if not filter_fn(row):
                continue
            if uncertainty_mode == 'weight':
                row = _apply_weight(row)
            writer.writerow(row)
            kept += 1
    print(f"    режим: {uncertainty_mode!r}, детекций: {kept}")
    return kept


def run_associator(source_dir, input_dir, output_dir, stations,
                    det_thr=DET_THR, p_thr=P_THR, s_thr=S_THR, keep_ps=KEEP_PS,
                    p_snr_thr=P_SNR_THR, s_snr_thr=S_SNR_THR,
                    uncertainty_mode=UNCERTAINTY_MODE,
                    start_time=START_TIME, end_time=END_TIME,
                    moving_window=MOVING_WINDOW, pair_n=PAIR_N,
                    coherence_tolerance=COHERENCE_TOLERANCE):
    """
    Стейджинг (фильтрация/взвешивание CSV детекций по станциям) + запуск
    run_associator_v2. Логика не менялась при переходе на CLI — раньше
    выполнялась построчно на верхнем уровне модуля, теперь в функции.
    """
    # Удаляем каталоги станций не из stations (остатки предыдущих запусков) —
    # иначе run_associator_v2 подхватил бы данные станций, исключённых из
    # текущего прогона (он сам читает все подпапки input_dir).
    if os.path.isdir(input_dir):
        for stale in os.listdir(input_dir):
            if stale not in stations:
                shutil.rmtree(os.path.join(input_dir, stale))
                print(f"  Removed stale staging dir: {stale}")

    for st in stations:
        src = os.path.join(source_dir, st, f"{st.lower()}.csv")
        dst_dir = os.path.join(input_dir, st)
        dst = os.path.join(dst_dir, "X_prediction_results.csv")
        if os.path.exists(src):
            os.makedirs(dst_dir, exist_ok=True)
            kept = prepare_station(src, dst, uncertainty_mode, det_thr, p_thr, s_thr,
                                    keep_ps, p_snr_thr, s_snr_thr)
            print(f"  {st}: {kept} детекций после фильтра")
        else:
            print(f"  WARNING: {src} not found, skipping {st}")

    os.makedirs(output_dir, exist_ok=True)

    print("\nЗапуск ассоциативного модуля v2 (p_arrival_time + sliding window)...")
    run_associator_v2(
        input_dir=input_dir,
        output_dir=output_dir,
        start_time=start_time,
        end_time=end_time,
        moving_window=moving_window,
        # _dbs_associator_v2 не поддерживает consider_combination=True
        # (бросает ValueError) — не CLI-флаг, а фиксированное ограничение
        # реализации, см. EQTransformer/utils/associator.py.
        consider_combination=False,
        pair_n=pair_n,
        coherence_tolerance=coherence_tolerance,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Сетевой ассоциатор: стейджинг CSV детекций по станциям + "
                    "run_associator_v2 (скользящее окно по P-пикам)."
    )
    parser.add_argument('--source-dir', default=SOURCE_DIR,
                        help="Откуда брать сырые CSV детекций по станциям")
    parser.add_argument('--input-dir', default=INPUT_DIR,
                        help="Куда писать отфильтрованный стейджинг")
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                        help="Куда писать результат ассоциатора (associations.xml и т.д.)")
    parser.add_argument('--stations', default=STATIONS,
                        help="Коды станций через запятую")
    parser.add_argument('--det-thr', type=float, default=DET_THR)
    parser.add_argument('--p-thr', type=float, default=P_THR)
    parser.add_argument('--s-thr', type=float, default=S_THR)
    parser.add_argument('--keep-ps', dest='keep_ps', action='store_true', default=KEEP_PS,
                        help="Строка проходит стейджинг только если есть и P, и S (AND)")
    parser.add_argument('--no-keep-ps', dest='keep_ps', action='store_false',
                        help="Достаточно P или S (OR, по умолчанию)")
    parser.add_argument('--p-snr-thr', type=float, default=P_SNR_THR,
                        help="0.0 = проверка SNR отключена")
    parser.add_argument('--s-snr-thr', type=float, default=S_SNR_THR)
    parser.add_argument('--uncertainty-mode', choices=['none', 'weight', 'filter'],
                        default=UNCERTAINTY_MODE,
                        help="Учёт MC Dropout неопределённости при стейджинге")
    parser.add_argument('--start-time', default=START_TIME,
                        help="'YYYY-MM-DD HH:MM:SS.fff'")
    parser.add_argument('--end-time', default=END_TIME)
    parser.add_argument('--moving-window', type=float, default=MOVING_WINDOW,
                        help="Ширина окна ассоциации, секунды")
    parser.add_argument('--pair-n', type=int, default=PAIR_N,
                        help="Минимум уникальных станций для события")
    parser.add_argument('--coherence-tolerance', type=float, default=COHERENCE_TOLERANCE)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    stations = [s.strip() for s in args.stations.split(',') if s.strip()]
    run_associator(
        source_dir=args.source_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        stations=stations,
        det_thr=args.det_thr,
        p_thr=args.p_thr,
        s_thr=args.s_thr,
        keep_ps=args.keep_ps,
        p_snr_thr=args.p_snr_thr,
        s_snr_thr=args.s_snr_thr,
        uncertainty_mode=args.uncertainty_mode,
        start_time=args.start_time,
        end_time=args.end_time,
        moving_window=args.moving_window,
        pair_n=args.pair_n,
        coherence_tolerance=args.coherence_tolerance,
    )
