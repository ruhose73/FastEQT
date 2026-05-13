"""
cutter-v6.py — автономная обработка сейсмических данных (EQTransformer, GPU).

Параллельный режим (по умолчанию): MAX_WORKERS процессов,
  каждый загружает модель ОДИН РАЗ через initializer.
Последовательный режим: установи MAX_WORKERS = 1.

Все активные функции обработки перенесены из cutter-v5.py.
cutter-v5.py можно перенести в legacy/.
"""

import gc
import os
import csv
import re
import concurrent.futures
from datetime import datetime
from traceback import format_exc

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from EQTransformer.core.EqT_utils import FeedForward, LayerNormalization, SeqSelfAttention, f1
from obspy import read, Stream, UTCDateTime
from EQTransformer.utils.hdf5_maker import preprocessorV6_mem
from EQTransformer.core.predictor import predictor_mem_non_hdf_load_model_v4


# ─── Конфигурация ────────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH      = os.path.join(_ROOT, "ModelsAndSampleData", "EqT_original_model.h5")
OUTPUT_BASE_DIR = os.path.join(_ROOT, "data-in-memory", "output_gpu")
LOG_FILE        = os.path.join(_ROOT, "data-in-memory", "processing_log.csv")

MAX_WORKERS  = 3   # 1 = последовательный режим (одна станция за раз)
TARGET_MONTH = 1
TARGET_YEAR  = 2024

_IN = os.path.join(_ROOT, "data-in-memory", "input")
_JS = os.path.join(_ROOT, "json")

# Оптимальный состав: recall 33/49 (без NCK/SRGR/GOFR — они снижают до 32/49)
STATIONS = [
    (os.path.join(_IN, "SOC"),  os.path.join(_JS, "station_SOC.json")),
    (os.path.join(_IN, "VSLR"), os.path.join(_JS, "station_VSLR.json")),
    (os.path.join(_IN, "GUZR"), os.path.join(_JS, "station_GUZR.json")),
    (os.path.join(_IN, "BEYR"), os.path.join(_JS, "station_BEYR.json")),
    (os.path.join(_IN, "SHA1"), os.path.join(_JS, "station_SHA1.json")),
    (os.path.join(_IN, "MRNR"), os.path.join(_JS, "station_MRNR.json")),
    (os.path.join(_IN, "SPGR"), os.path.join(_JS, "station_SPGR.json")),
    (os.path.join(_IN, "DOMR"), os.path.join(_JS, "station_DOMR.json")),
    (os.path.join(_IN, "ZEI"),  os.path.join(_JS, "station_ZEI.json")),
    (os.path.join(_IN, "LABN"), os.path.join(_JS, "station_LABN.json")),
    (os.path.join(_IN, "GOYR"), os.path.join(_JS, "station_GOYR.json")),
    (os.path.join(_IN, "PYA1"), os.path.join(_JS, "station_PYA1.json")),
    # NCK/SRGR/GOFR ухудшают recall в секторном ассоциаторе (32/49 вместо 33/49)
    # (os.path.join(_IN, "NCK"),  os.path.join(_JS, "station_NCK.json")),
    # (os.path.join(_IN, "SRGR"), os.path.join(_JS, "station_SRGR.json")),
    # (os.path.join(_IN, "GOFR"), os.path.join(_JS, "station_GOFR.json")),
]

# ─────────────────────────────────────────────────────────────────────────────

CSV_HEADER = [
    'file_name', 'network', 'station', 'instrument_type',
    'station_lat', 'station_lon', 'station_elv',
    'event_start_time', 'event_end_time',
    'detection_probability', 'detection_uncertainty',
    'p_arrival_time', 'p_probability', 'p_uncertainty', 'p_snr',
    's_arrival_time', 's_probability', 's_uncertainty', 's_snr'
]


# ─── Функции обработки ───────────────────────────────────────────────────────

def geofile_splitter_multi_chanels_v2(base_directory, target_month=1, target_year=2024):
    """
    Генератор: читает один день за раз, отдаёт сегменты по одному, затем освобождает RAM.

    v1 загружала весь месяц в список до начала обработки — все дневные потоки
    оставались в RAM одновременно (~3 ГБ для 31 дня). v2 читает один день,
    отдаёт его сегменты по очереди, затем освобождает и переходит к следующему.
    Максимальное потребление: 1 день сырых данных + 1 сегмент в обработке.
    """
    pattern = re.compile(
        r'^(?P<net>[A-Z0-9]+)\.'
        r'(?P<sta>[A-Z0-9]+)\.'
        r'(?P<loc>[A-Z0-9]{0,2})\.'
        r'(?P<cha>[A-Z0-9]+)\.[A-Z_]*__'
        r'(?P<start>\d{8}T\d{6}Z)__'
        r'(?P<end>\d{8}T\d{6}Z)$'
    )

    files = [f for f in os.listdir(base_directory)
             if os.path.isfile(os.path.join(base_directory, f))]
    station_groups = {}

    for file_name in files:
        match = pattern.match(file_name)
        if not match:
            print(f"⚠️ Пропускаем файл: {file_name} — не подходит под шаблон")
            continue
        sta = match.group("sta")
        start = match.group("start")
        key = f"{sta}_{start}"
        station_groups.setdefault(key, []).append(
            os.path.join(base_directory, file_name)
        )

    total_yielded = 0

    for key, file_list in sorted(station_groups.items()):
        try:
            st_all = Stream()
            for fpath in file_list:
                try:
                    st_all += read(fpath)
                except Exception as e:
                    print(f"Ошибка чтения {fpath}: {e}")

            if len(st_all) < 2:
                print(f"⚠️ Пропускаем {key}: найдено только {len(st_all)} канал(ов)")
                del st_all
                continue

            tr0 = st_all[0]
            start_time = tr0.stats.starttime
            end_time = tr0.stats.endtime

            if start_time.month != target_month or start_time.year != target_year:
                del st_all
                continue

            window_length = 10 * 60
            step = 5 * 60
            t = start_time

            while t + window_length <= end_time:
                segment = st_all.slice(t, t + window_length)
                if len(segment) >= 2 and all(tr.stats.npts > 0 for tr in segment):
                    seg_name = (
                        f"{tr0.stats.network}.{tr0.stats.station}__"
                        f"{t.strftime('%Y%m%dT%H%M%SZ')}__"
                        f"{(t + window_length).strftime('%Y%m%dT%H%M%SZ')}"
                    )
                    total_yielded += 1
                    yield (segment, seg_name)
                t += step

            if t < end_time:
                segment = st_all.slice(end_time - window_length, end_time)
                if len(segment) >= 2 and all(tr.stats.npts > 0 for tr in segment):
                    seg_name = (
                        f"{tr0.stats.network}.{tr0.stats.station}__"
                        f"{(end_time - window_length).strftime('%Y%m%dT%H%M%SZ')}__"
                        f"{end_time.strftime('%Y%m%dT%H%M%SZ')}"
                    )
                    total_yielded += 1
                    yield (segment, seg_name)

        except Exception as e:
            print(f"❌ Ошибка обработки группы {key}: {e}")
        finally:
            if 'st_all' in dir():
                del st_all

    print(f"✅ Всего сегментов обработано генератором: {total_yielded}")


def worker_v3(segment_item, model, save_figs=None, number_of_plots=10):
    """
    Вызывает predictor v4, который возвращает строки событий без записи в файл.
    Возвращает: (seg_name, status, message, rows)
    """
    segment, seg_name, stations_json = segment_item
    status = "success"
    message = ""
    rows = []

    try:
        print(f"▶️ Обработка сегмента {seg_name}: {len(segment)} каналов → {[tr.stats.channel for tr in segment]}")
        start_time = datetime.now()

        csv_data, hdf5_data = preprocessorV6_mem(
            stream_list=[(segment, seg_name)],
            stations_json=stations_json,
            overlap=0.3,
            n_processor=1
        )
        t_preproc = (datetime.now() - start_time).total_seconds()

        seg_csv = csv_data[seg_name][1:]
        seg_hdf = hdf5_data[seg_name]

        t_pred_start = datetime.now()
        rows = predictor_mem_non_hdf_load_model_v4(
            csv_segment=seg_csv,
            hdf_segment=seg_hdf,
            model=model,
            save_figs=save_figs,
            detection_threshold=0.5,
            P_threshold=0.2,
            S_threshold=0.1,
            number_of_plots=number_of_plots,
            plot_mode='time',
            estimate_uncertainty=False,
            gpuid=0,
            keepPS=False,
            allowonlyS=False,
            spLimit=60
        )
        t_pred = (datetime.now() - t_pred_start).total_seconds()

        total = (datetime.now() - start_time).total_seconds()
        print(f"  preproc={t_preproc:.2f}s  predict={t_pred:.2f}s  total={total:.2f}s")
        print(f"✅ Сегмент {seg_name}: {len(rows)} событий")

    except Exception:
        status = "error"
        message = format_exc()
        print(f"❌ Ошибка обработки {seg_name}: {message}")

    return (seg_name, status, message, rows)


def preproc_sequential_v4(segment_gen, stations_json, model, output_csv,
                          save_figs=None, number_of_plots=10):
    """
    Обрабатывает сегменты из генератора последовательно, пишет все события в один CSV.
    Каждый сегмент явно удаляется из памяти после обработки.
    gc.collect() каждые 50 сегментов.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    log_results = []
    total_events = 0
    processed = 0

    with open(output_csv, 'w', newline='', encoding='utf-8') as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(CSV_HEADER)
        csv_out.flush()

        for segment, seg_name in segment_gen:
            seg_name_out, status, message, rows = worker_v3(
                (segment, seg_name, stations_json),
                model,
                save_figs=save_figs,
                number_of_plots=number_of_plots
            )
            del segment

            if rows:
                writer.writerows(rows)
                csv_out.flush()
                total_events += len(rows)
            log_results.append((seg_name_out, status, message, len(rows)))

            processed += 1
            if processed % 50 == 0:
                gc.collect()
                print(f"  [gc] собрано после {processed} сегментов, событий: {total_events}")

    gc.collect()

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        log_writer = csv.writer(f)
        log_writer.writerow(["file_name", "status", "message", "events_found"])
        for r in log_results:
            log_writer.writerow(r)

    print(f"Лог: {LOG_FILE}")
    print(f"CSV: {output_csv} ({total_events} событий)")


def _patch_lstm_inplace(model):
    """
    Патчит LSTM cell напрямую, без JSON rebuild.

    recurrent_dropout и implementation — read-only @property на LSTM слое,
    они читаются из lstm.cell. Патчим cell напрямую: LSTMCell хранит их
    как обычные instance-атрибуты, поэтому присвоение работает.

    Warning при load_model (трассировка графа) неизбежен.
    После патча cell — при model.predict() cuDNN-проверка проходит,
    cuDNN kernel используется, новый warning не появляется.

    Bidirectional хранит два LSTM (forward_layer, backward_layer) — патчим оба.
    """
    patched = 0
    for layer in model.layers:
        lstm_layers = []
        if isinstance(layer, tf.keras.layers.LSTM):
            lstm_layers = [layer]
        elif isinstance(layer, tf.keras.layers.Bidirectional):
            lstm_layers = [layer.forward_layer, layer.backward_layer]

        for lstm in lstm_layers:
            if not isinstance(lstm, tf.keras.layers.LSTM):
                continue
            if hasattr(lstm, 'cell'):
                lstm.cell.recurrent_dropout = 0.0
                lstm.cell.implementation = 2
                patched += 1

    print(f"  [cuDNN patch] патчировано {patched} LSTM cell объектов")
    return model


def load_model_cudnn_v2(model_path):
    """
    Загружает модель и применяет in-place cuDNN LSTM патч.

    v1 пересоздавал модель через model_from_json — TF 2.10 игнорировал patched config
    при десериализации Bidirectional. v2 патчит объекты напрямую после load_model,
    до первого вызова predict().
    """
    custom_objects = {
        'SeqSelfAttention': SeqSelfAttention,
        'FeedForward': FeedForward,
        'LayerNormalization': LayerNormalization,
        'f1': f1
    }
    model = load_model(model_path, compile=False, custom_objects=custom_objects)
    _patch_lstm_inplace(model)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=['binary_crossentropy'] * 3,
        metrics=[f1]
    )
    return model


def process_station(base_directory, stations_json, model,
                    target_month, target_year, output_base_dir):
    """Обрабатывает одну станцию с уже загруженной моделью (initializer паттерн)."""
    station_name = os.path.basename(os.path.normpath(base_directory))
    output_dir = os.path.join(output_base_dir, station_name)
    output_csv = os.path.join(output_dir, f"{station_name.lower()}.csv")

    segment_gen = geofile_splitter_multi_chanels_v2(
        base_directory, target_month, target_year
    )
    preproc_sequential_v4(
        segment_gen, stations_json, model,
        output_csv=output_csv,
        save_figs=None,
        number_of_plots=10
    )


# ─── Параллельный runner ─────────────────────────────────────────────────────

_model = None


def _init_worker(model_path):
    """Запускается один раз при старте воркера. Грузит модель в глобальную переменную."""
    global _model
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass
    _model = load_model_cudnn_v2(model_path)
    print(f"  [init] модель загружена в воркере PID={os.getpid()}")


def _run_task(args):
    """Запускается в воркере: обрабатывает одну станцию используя глобальную модель."""
    base_directory, stations_json, target_month, target_year = args
    station_name = os.path.basename(os.path.normpath(base_directory))
    try:
        process_station(
            base_directory, stations_json, _model,
            target_month, target_year, OUTPUT_BASE_DIR
        )
        return (station_name, "success", "")
    except Exception:
        import traceback
        return (station_name, "error", traceback.format_exc())


if __name__ == "__main__":
    start = datetime.now()

    tasks = [
        (bd, sj, TARGET_MONTH, TARGET_YEAR)
        for bd, sj in STATIONS
    ]

    print(f"Станций к обработке: {len(tasks)}")
    print(f"Воркеров: {MAX_WORKERS}  (MAX_WORKERS=1 → последовательный режим)")
    print(f"Модель грузится 1 раз на воркер")
    print("-" * 60)

    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_init_worker,
        initargs=(MODEL_PATH,)
    ) as executor:
        future_map = {executor.submit(_run_task, t): t[0] for t in tasks}
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
    print(f"Успешно: {ok}  Ошибок: {err}  Время: {total:.1f} мин")
