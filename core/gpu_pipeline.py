import os
import math
import importlib.util
import concurrent.futures
from datetime import datetime
from obspy import UTCDateTime


# ─── Конфигурация ────────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH      = os.path.join(_ROOT, "ModelsAndSampleData", "EqT_original_model.h5")
OUTPUT_BASE_DIR = os.path.join(_ROOT, "data-in-memory", "gpu_splimit_45_may_v2", "output_detector")

MAX_WORKERS = 6

_IN = os.path.join(_ROOT, "geofiles")
_JS = os.path.join(_ROOT, "json")

# AKT, ANN, ARKR, ARNR, BEYR, BTKR, BTLR, BUJR, BVTR, DBC, DIGR, DLMR, DOMR, DRN, DVE, ERBR, GLDR, GLVR, VSLR,
# GOFR, GOYR, GROC, GRYR, GUZR, HNZR, KANR, KLMR, KMGR, KMKR, KORR, KRNR, KSMR, LABN, LACR, LSNR, MAK, MRNR, ZEI
# NCK, NVPR, PXTR, PYA1, RPOR, SGKR, SHA1, SOC, SPGR, SRGR, STDR, SUKR, TLTR, TMNR, TRKR, UNCR, URKR, VLKR, 

STATIONS = [
         (os.path.join(_IN, "NCK"), os.path.join(_JS, "station_NCK.json")),
         (os.path.join(_IN, "NVPR"), os.path.join(_JS, "station_NVPR.json")),
         (os.path.join(_IN, "PXTR"), os.path.join(_JS, "station_PXTR.json")),
         (os.path.join(_IN, "PYA1"), os.path.join(_JS, "station_PYA1.json")),
         (os.path.join(_IN, "RPOR"), os.path.join(_JS, "station_RPOR.json")),
         (os.path.join(_IN, "SGKR"), os.path.join(_JS, "station_SGKR.json")),
         (os.path.join(_IN, "SHA1"), os.path.join(_JS, "station_SHA1.json")),
         (os.path.join(_IN, "SOC"), os.path.join(_JS, "station_SOC.json")),
         (os.path.join(_IN, "SPGR"), os.path.join(_JS, "station_SPGR.json")),
         (os.path.join(_IN, "SRGR"), os.path.join(_JS, "station_SRGR.json")),
         (os.path.join(_IN, "STDR"), os.path.join(_JS, "station_STDR.json")),
         (os.path.join(_IN, "SUKR"), os.path.join(_JS, "station_SUKR.json")),
         (os.path.join(_IN, "TLTR"), os.path.join(_JS, "station_TLTR.json")),
         (os.path.join(_IN, "TMNR"), os.path.join(_JS, "station_TMNR.json")),
         (os.path.join(_IN, "TRKR"), os.path.join(_JS, "station_TRKR.json")),
         (os.path.join(_IN, "UNCR"), os.path.join(_JS, "station_UNCR.json")),
         (os.path.join(_IN, "URKR"), os.path.join(_JS, "station_URKR.json")),
         (os.path.join(_IN, "VLKR"), os.path.join(_JS, "station_VLKR.json")),
         # (os.path.join(_IN, "ZEI"), os.path.join(_JS, "station_ZEI.json")),
    ]

DATE_FROM = UTCDateTime(2024, 5, 1)
DATE_TO   = UTCDateTime(2024, 6, 1)   # не включается

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
    base_directory, stations_json, date_from, date_to = args
    station_name = os.path.basename(os.path.normpath(base_directory))
    try:
        _detector_mod.process_station_v3(
            base_directory=base_directory,
            stations_json=stations_json,
            model=_model,
            date_from=date_from,
            date_to=date_to,
            output_base_dir=OUTPUT_BASE_DIR,
            estimate_uncertainty=True,
            number_of_sampling=5,
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
        (bd, sj, DATE_FROM, DATE_TO)
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
