"""
cpu_pipeline.py — параллельная обработка станций на CPU.

Отличия от gpu_pipeline.py (GPU):
  - GPU скрыт через set_visible_devices([]) → TF использует только CPU
  - TF потоки ограничены на воркер → процессы не конкурируют за ядра
  - MAX_WORKERS подбирается под число CPU ядер, а не VRAM

Рекомендации по MAX_WORKERS для 12 ядер:
  MAX_WORKERS=4 → 3 потока TF/воркер  (баланс скорость/параллелизм)
  MAX_WORKERS=6 → 2 потока TF/воркер
  MAX_WORKERS=8 → 1-2 потока TF/воркер

Модель загружается 1 раз на воркер (initializer паттерн).
"""

import os
import importlib.util
import concurrent.futures
from datetime import datetime


# ─── Конфигурация ────────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH      = os.path.join(_ROOT, "ModelsAndSampleData", "EqT_original_model.h5")
OUTPUT_BASE_DIR = os.path.join(_ROOT, "data-in-memory", "output_cpu")

MAX_WORKERS = 4

# Потоков TF на воркер (inter_op и intra_op параллелизм внутри predict()).
# 4 воркера × 3 потока = 12 потоков (все ядра CPU).
TF_THREADS_PER_WORKER = 3

_IN = os.path.join(_ROOT, "data-in-memory", "input")
_JS = os.path.join(_ROOT, "json")

STATIONS = [
        # (os.path.join(_IN, "SOC"),  os.path.join(_JS, "station_SOC.json")),
        # (os.path.join(_IN, "VSLR"), os.path.join(_JS, "station_VSLR.json")),
        # (os.path.join(_IN, "GUZR"), os.path.join(_JS, "station_GUZR.json")),
        # (os.path.join(_IN, "BEYR"), os.path.join(_JS, "station_BEYR.json")),
        # (os.path.join(_IN, "SHA1"), os.path.join(_JS, "station_SHA1.json")),
        # (os.path.join(_IN, "MRNR"), os.path.join(_JS, "station_MRNR.json")),
        # (os.path.join(_IN, "SPGR"), os.path.join(_JS, "station_SPGR.json")),
        # (os.path.join(_IN, "DOMR"), os.path.join(_JS, "station_DOMR.json")),
        # (os.path.join(_IN, "ZEI"),  os.path.join(_JS, "station_ZEI.json")),
        # (os.path.join(_IN, "LABN"), os.path.join(_JS, "station_LABN.json")),
]

TARGET_MONTH = 1
TARGET_YEAR  = 2024

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
    base_directory, stations_json, target_month, target_year = args
    station_name = os.path.basename(os.path.normpath(base_directory))
    try:
        _detector_mod.process_station(
            base_directory=base_directory,
            stations_json=stations_json,
            model=_model,
            target_month=target_month,
            target_year=target_year,
            output_base_dir=OUTPUT_BASE_DIR,
        )
        return (station_name, "success", "")
    except Exception:
        import traceback
        return (station_name, "error", traceback.format_exc())


if __name__ == "__main__":
    start = datetime.now()

    tf_threads = TF_THREADS_PER_WORKER or max(1, os.cpu_count() // MAX_WORKERS)

    tasks = [
        (bd, sj, TARGET_MONTH, TARGET_YEAR)
        for bd, sj in STATIONS
    ]

    print(f"Режим:          CPU")
    print(f"Станций:        {len(tasks)}")
    print(f"Воркеров:       {MAX_WORKERS}")
    print(f"Потоков TF:     {tf_threads} на воркер  ({MAX_WORKERS * tf_threads} из {os.cpu_count()} ядер)")
    print(f"Модель грузится 1 раз на воркер (initializer)")
    print("-" * 60)

    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_init_worker,
        initargs=(MODEL_PATH, tf_threads)
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
    print(f"Воркеров: {MAX_WORKERS}  Потоков TF/воркер: {tf_threads}")
    print(f"Успешно: {ok}  Ошибок: {err}  Время: {total:.1f} мин")
