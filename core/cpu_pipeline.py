"""
cpu_pipeline.py — параллельная обработка станций на CPU (CLI).

Отличия от gpu_pipeline.py (GPU):
  - GPU скрыт через set_visible_devices([]) → TF использует только CPU
  - TF потоки ограничены на воркер → процессы не конкурируют за ядра
  - MAX_WORKERS подбирается под число CPU ядер, а не VRAM

Рекомендации по --max-workers для 12 ядер:
  --max-workers 4 → 3 потока TF/воркер  (баланс скорость/параллелизм)
  --max-workers 6 → 2 потока TF/воркер
  --max-workers 8 → 1-2 потока TF/воркер

Модель загружается 1 раз на воркер (initializer паттерн).

Все параметры задаются флагами командной строки (см. --help): диапазон
дат — --date-from/--date-to; список станций — --stations (коды через
запятую), путь достраивается из --input-dir/--json-dir; пороги детекции
(detection_threshold и т. д.) — CLI-флаги, пробрасываются в
process_station_v3 в detector.py.
"""

import argparse
import os
import sys
import importlib.util
import concurrent.futures
from datetime import datetime

# Форсируем UTF-8 на stdout/stderr — в help-строках/докстринге есть не-ASCII
# (стрелки →, кириллица); без этого argparse.print_help() может упасть с
# UnicodeEncodeError в консоли по умолчанию (cp1251) на Windows.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


# ─── Конфигурация (значения по умолчанию для CLI-флагов) ─────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH      = os.path.join(_ROOT, "ModelsAndSampleData", "EqT_original_model.h5")
OUTPUT_BASE_DIR = os.path.join(_ROOT, "workspace", "detector", "output")

MAX_WORKERS = 4

# Потоков TF на воркер (inter_op и intra_op параллелизм внутри predict()).
# 4 воркера × 3 потока = 12 потоков (все ядра CPU).
TF_THREADS_PER_WORKER = 3

# По умолчанию — выход data_processors/main.py; workspace/detector/input/
# существует отдельно для случая, когда волновые файлы/station_*.json
# кладутся туда напрямую, минуя data_processors.
_IN = os.path.join(_ROOT, "workspace", "data_processors", "output", "geofiles")
_JS = os.path.join(_ROOT, "workspace", "data_processors", "output")

# Пусто — в оригинале все станции были закомментированы (см. legacy/cpu_pipeline.py).
STATIONS = ""

DATE_FROM = "2024-01-01"
DATE_TO   = "2024-02-01"   # не включается

DETECTION_THRESHOLD = 0.75
P_THRESHOLD         = 0.3
S_THRESHOLD         = 0.2
KEEP_PS             = True
ALLOW_ONLY_S        = False
SP_LIMIT            = 45
BATCH_SIZE          = 32

# ─────────────────────────────────────────────────────────────────────────────

_model        = None
_detector_mod = None


def _load_detector():
    spec = importlib.util.spec_from_file_location(
        "detector",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _init_worker(model_path, tf_threads):
    """Запускается один раз при старте воркера. Грузит модель в глобальную переменную."""
    global _model, _detector_mod
    import sys
    import tensorflow as tf

    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    # Скрываем GPU — только CPU
    tf.config.set_visible_devices([], 'GPU')

    # Ограничиваем TF потоки чтобы воркеры не конкурировали за ядра
    tf.config.threading.set_inter_op_parallelism_threads(tf_threads)
    tf.config.threading.set_intra_op_parallelism_threads(tf_threads)

    _detector_mod = _load_detector()
    _model = _detector_mod.load_model_cudnn_v2(model_path)
    print(f"  [init] модель загружена в воркере PID={os.getpid()}")


def run_station(args):
    """Обрабатывает одну станцию используя уже загруженную модель."""
    (base_directory, stations_json, date_from, date_to, output_base_dir,
     thresholds, gap_mode) = args
    station_name = os.path.basename(os.path.normpath(base_directory))
    process_station = (_detector_mod.process_station_v4 if gap_mode == 'merge'
                       else _detector_mod.process_station_v3)
    try:
        process_station(
            base_directory=base_directory,
            stations_json=stations_json,
            model=_model,
            date_from=date_from,
            date_to=date_to,
            output_base_dir=output_base_dir,
            **thresholds,
        )
        return (station_name, "success", "")
    except Exception:
        import traceback
        return (station_name, "error", traceback.format_exc())


def _build_stations(codes, input_dir, json_dir):
    """Достраивает пары (входная_директория, station_*.json) по кодам станций."""
    return [
        (os.path.join(input_dir, code), os.path.join(json_dir, f"station_{code}.json"))
        for code in codes
    ]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Параллельная обработка станций EQTransformer на CPU "
                    "(process_station_v3 из detector.py, ProcessPoolExecutor)."
    )
    parser.add_argument('--model-path', default=MODEL_PATH)
    parser.add_argument('--output-base-dir', default=OUTPUT_BASE_DIR)
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--tf-threads-per-worker', type=int, default=TF_THREADS_PER_WORKER,
                        help="0 = автоматически (cpu_count() // max-workers)")
    parser.add_argument('--input-dir', default=_IN,
                        help="Корневая директория с входными данными станций ({input-dir}/{код})")
    parser.add_argument('--json-dir', default=_JS,
                        help="Директория с station_*.json ({json-dir}/station_{код}.json)")
    parser.add_argument('--stations', default=STATIONS,
                        help="Коды станций через запятую")
    parser.add_argument('--date-from', default=DATE_FROM, help="UTCDateTime-совместимая строка")
    parser.add_argument('--date-to', default=DATE_TO, help="Не включается")
    parser.add_argument('--detection-threshold', type=float, default=DETECTION_THRESHOLD)
    parser.add_argument('--p-threshold', type=float, default=P_THRESHOLD)
    parser.add_argument('--s-threshold', type=float, default=S_THRESHOLD)
    parser.add_argument('--keep-ps', dest='keep_ps', action='store_true', default=KEEP_PS)
    parser.add_argument('--no-keep-ps', dest='keep_ps', action='store_false')
    parser.add_argument('--allow-only-s', action='store_true', default=ALLOW_ONLY_S)
    parser.add_argument('--sp-limit', type=float, default=SP_LIMIT)
    parser.add_argument('--estimate-uncertainty', dest='estimate_uncertainty',
                        action='store_true', default=False,
                        help="MC Dropout неопределённость (медленнее — number-of-sampling проходов на сегмент)")
    parser.add_argument('--number-of-sampling', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--gap-mode', choices=['first-trace', 'merge'], default='first-trace',
                        help="Суточный файл с разрывами: first-trace — как раньше, только до первого "
                             "разрыва (process_station_v3); merge — склеить куски и обработать сутки "
                             "целиком (process_station_v4)")
    return parser.parse_args()


if __name__ == "__main__":
    from obspy import UTCDateTime

    args = _parse_args()
    start = datetime.now()

    date_from = UTCDateTime(args.date_from)
    date_to   = UTCDateTime(args.date_to)

    station_codes = [s.strip() for s in args.stations.split(',') if s.strip()]
    stations = _build_stations(station_codes, args.input_dir, args.json_dir)

    tf_threads = args.tf_threads_per_worker or max(1, os.cpu_count() // args.max_workers)

    thresholds = dict(
        detection_threshold=args.detection_threshold,
        P_threshold=args.p_threshold,
        S_threshold=args.s_threshold,
        keep_ps=args.keep_ps,
        allow_only_s=args.allow_only_s,
        sp_limit=args.sp_limit,
        estimate_uncertainty=args.estimate_uncertainty,
        number_of_sampling=args.number_of_sampling,
        batch_size=args.batch_size,
    )

    tasks = [
        (bd, sj, date_from, date_to, args.output_base_dir, thresholds, args.gap_mode)
        for bd, sj in stations
    ]

    print(f"Режим:          CPU")
    print(f"Станций:        {len(tasks)}")
    print(f"Воркеров:       {args.max_workers}")
    print(f"Потоков TF:     {tf_threads} на воркер  ({args.max_workers * tf_threads} из {os.cpu_count()} ядер)")
    print(f"Модель грузится 1 раз на воркер (initializer)")
    print("-" * 60)

    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=_init_worker,
        initargs=(args.model_path, tf_threads)
    ) as executor:
        future_map = {executor.submit(run_station, t): t[0] for t in tasks}
        for future in concurrent.futures.as_completed(future_map):
            station_name, status, msg = future.result()
            elapsed = (datetime.now() - start).total_seconds() / 60
            print(f"[{elapsed:5.1f} мин] [{status.upper():7}] {station_name}")
            if msg:
                print(msg[:800])
            results.append((station_name, status))

    total = (datetime.now() - start).total_seconds() / 60
    ok  = sum(1 for _, s in results if s == "success")
    err = sum(1 for _, s in results if s == "error")
    print("-" * 60)
    print(f"Воркеров: {args.max_workers}  Потоков TF/воркер: {tf_threads}")
    print(f"Успешно: {ok}  Ошибок: {err}  Время: {total:.1f} мин")
