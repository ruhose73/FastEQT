import os
import math
import importlib.util
import concurrent.futures
from datetime import datetime


# ─── Конфигурация ────────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH      = os.path.join(_ROOT, "ModelsAndSampleData", "EqT_original_model.h5")
OUTPUT_BASE_DIR = os.path.join(_ROOT, "data-in-memory", "output_gpu")

MAX_WORKERS = 3

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
       # (os.path.join(_IN, "GOYR"), os.path.join(_JS, "station_GOYR.json")),
       # (os.path.join(_IN, "PYA1"), os.path.join(_JS, "station_PYA1.json")),
        (os.path.join(_IN, "NCK"),  os.path.join(_JS, "station_NCK.json")),
        (os.path.join(_IN, "SRGR"), os.path.join(_JS, "station_SRGR.json")),
        (os.path.join(_IN, "GOFR"), os.path.join(_JS, "station_GOFR.json")),
]

TARGET_MONTH = 1
TARGET_YEAR  = 2024

# ─────────────────────────────────────────────────────────────────────────────

_model      = None
_cutter_mod = None


def _load_cutter():
    spec = importlib.util.spec_from_file_location(
        "cutter_v5",
        os.path.join(_ROOT, "legacy", "cutter-v5.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _init_worker(model_path):
    """Запускается один раз при старте воркера. Грузит модель в глобальную переменную."""
    global _model, _cutter_mod
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    _cutter_mod = _load_cutter()
    _model = _cutter_mod.load_model_cudnn_v2(model_path)
    print(f"  [init] модель загружена в воркере PID={os.getpid()}")


def run_station(args):
    """Обрабатывает одну станцию используя уже загруженную модель."""
    base_directory, stations_json, target_month, target_year = args
    station_name = os.path.basename(os.path.normpath(base_directory))
    try:
        _cutter_mod.main_v10(
            base_directory=base_directory,
            stations_json=stations_json,
            model=_model,
            target_month=target_month,
            target_year=target_year,
            create_figures=False,
            output_base_dir=OUTPUT_BASE_DIR,
        )
        return (station_name, "success", "")
    except Exception:
        import traceback
        return (station_name, "error", traceback.format_exc())


def get_threads_to_use(percent):
    return math.ceil(os.cpu_count() * percent / 100)


if __name__ == "__main__":
    start = datetime.now()

    tasks = [
        (bd, sj, TARGET_MONTH, TARGET_YEAR)
        for bd, sj in STATIONS
    ]

    print(f"Станций к обработке: {len(tasks)}")
    print(f"Воркеров (GPU процессов): {MAX_WORKERS}")
    print(f"CPU потоков всего: {os.cpu_count()}")
    print(f"Модель грузится 1 раз на воркер (initializer)")
    print("-" * 60)

    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_init_worker,
        initargs=(MODEL_PATH,)
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
    print(f"Использовалось воркеров: {MAX_WORKERS}")
    print(f"Успешно: {ok}  Ошибок: {err}  Время: {total:.1f} мин")
