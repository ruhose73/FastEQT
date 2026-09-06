"""
gpu_pipeline.py — параллельная обработка станций на GPU (CLI).

Несколько процессов используют один GPU (initializer-паттерн: модель
грузится один раз на процесс, memory_growth включён чтобы процессы могли
делить VRAM).

Все параметры задаются флагами командной строки (см. --help): список
станций (--stations, коды через запятую, путь достраивается из
--input-dir/--json-dir), диапазон дат (--date-from/--date-to), MC Dropout
(--estimate-uncertainty/--number-of-sampling) и пороги детекции,
пробрасываются в process_station_v3 в detector.py.
"""

import argparse
import os
import importlib.util
import concurrent.futures
from datetime import datetime


# ─── Конфигурация (значения по умолчанию для CLI-флагов) ─────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH      = os.path.join(_ROOT, "ModelsAndSampleData", "EqT_original_model.h5")
OUTPUT_BASE_DIR = os.path.join(_ROOT, "workspace", "detector", "output")

MAX_WORKERS = 6

# По умолчанию — выход data_processors/main.py; workspace/detector/input/
# существует отдельно для случая, когда волновые файлы/station_*.json
# кладутся туда напрямую, минуя data_processors.
_IN = os.path.join(_ROOT, "workspace", "data_processors", "output", "geofiles")
_JS = os.path.join(_ROOT, "workspace", "data_processors", "output")

STATIONS = ("NCK,NVPR,PXTR,PYA1,RPOR,SGKR,SHA1,SOC,SPGR,SRGR,"
            "STDR,SUKR,TLTR,TMNR,TRKR,UNCR,URKR,VLKR")

DATE_FROM = "2024-05-01"
DATE_TO   = "2024-06-01"   # не включается

ESTIMATE_UNCERTAINTY = True
NUMBER_OF_SAMPLING   = 5

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


def _init_worker(model_path):
    """Запускается один раз при старте воркера. Грузит модель в глобальную переменную."""
    global _model, _detector_mod
    import sys
    import tensorflow as tf

    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    _detector_mod = _load_detector()
    _model = _detector_mod.load_model_cudnn_v2(model_path)
    print(f"  [init] модель загружена в воркере PID={os.getpid()}")


def run_station(args):
    """Обрабатывает одну станцию используя уже загруженную модель."""
    (base_directory, stations_json, date_from, date_to, output_base_dir,
     thresholds) = args
    station_name = os.path.basename(os.path.normpath(base_directory))
    try:
        _detector_mod.process_station_v3(
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
        description="Параллельная обработка станций EQTransformer на GPU "
                    "(process_station_v3 из detector.py, ProcessPoolExecutor)."
    )
    parser.add_argument('--model-path', default=MODEL_PATH)
    parser.add_argument('--output-base-dir', default=OUTPUT_BASE_DIR)
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--input-dir', default=_IN,
                        help="Корневая директория с входными данными станций ({input-dir}/{код})")
    parser.add_argument('--json-dir', default=_JS,
                        help="Директория с station_*.json ({json-dir}/station_{код}.json)")
    parser.add_argument('--stations', default=STATIONS,
                        help="Коды станций через запятую")
    parser.add_argument('--date-from', default=DATE_FROM, help="UTCDateTime-совместимая строка")
    parser.add_argument('--date-to', default=DATE_TO, help="Не включается")
    parser.add_argument('--estimate-uncertainty', dest='estimate_uncertainty',
                        action='store_true', default=ESTIMATE_UNCERTAINTY,
                        help="MC Dropout неопределённость (по умолчанию включена)")
    parser.add_argument('--no-estimate-uncertainty', dest='estimate_uncertainty',
                        action='store_false')
    parser.add_argument('--number-of-sampling', type=int, default=NUMBER_OF_SAMPLING)
    parser.add_argument('--detection-threshold', type=float, default=DETECTION_THRESHOLD)
    parser.add_argument('--p-threshold', type=float, default=P_THRESHOLD)
    parser.add_argument('--s-threshold', type=float, default=S_THRESHOLD)
    parser.add_argument('--keep-ps', dest='keep_ps', action='store_true', default=KEEP_PS)
    parser.add_argument('--no-keep-ps', dest='keep_ps', action='store_false')
    parser.add_argument('--allow-only-s', action='store_true', default=ALLOW_ONLY_S)
    parser.add_argument('--sp-limit', type=float, default=SP_LIMIT)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    from obspy import UTCDateTime

    args = _parse_args()
    start = datetime.now()

    date_from = UTCDateTime(args.date_from)
    date_to   = UTCDateTime(args.date_to)

    station_codes = [s.strip() for s in args.stations.split(',') if s.strip()]
    stations = _build_stations(station_codes, args.input_dir, args.json_dir)

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
        (bd, sj, date_from, date_to, args.output_base_dir, thresholds)
        for bd, sj in stations
    ]

    print(f"Станций к обработке: {len(tasks)}")
    print(f"Воркеров (GPU процессов): {args.max_workers}")
    print(f"CPU потоков всего: {os.cpu_count()}")
    print(f"Модель грузится 1 раз на воркер (initializer)")
    print("-" * 60)

    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=_init_worker,
        initargs=(args.model_path,)
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
    print(f"Всего CPU потоков: {os.cpu_count()}")
    print(f"Использовалось воркеров: {args.max_workers}")
    print(f"Успешно: {ok}  Ошибок: {err}  Время: {total:.1f} мин")
