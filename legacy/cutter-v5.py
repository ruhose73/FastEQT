import gc
import json
import os
import csv
import traceback
from EQTransformer.core.EqT_utils import FeedForward, LayerNormalization, SeqSelfAttention, f1
from obspy import read, Stream, UTCDateTime
from EQTransformer.utils.hdf5_maker import preprocessorV6_mem
# from EQTransformer.core.predictor_mem import predictor_mem_non_hdf_load_model
# from EQTransformer.core.predictor import predictor_mem_non_hdf_load_model
# from EQTransformer.core.predictor import predictor_mem_non_hdf_load_model_v2 as predictor_mem_non_hdf_load_model
from EQTransformer.core.predictor import predictor_mem_non_hdf_load_model_v3 as predictor_mem_non_hdf_load_model
from EQTransformer.core.predictor import predictor_mem_non_hdf_load_model_v4
from EQTransformer.core.predictor import predictor_mem_non_hdf_load_model_v5
from traceback import format_exc
import re
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


LOG_FILE = "./data-in-memory/processing_log.csv"
OUTPUT_DIR = "./data-in-memory/output_new"

def get_process_time(start_time, end_time): 
    elapsed_time = end_time - start_time
    elapsed_seconds = elapsed_time.total_seconds()
    print('-' * 100)
    print(f"Обработка {elapsed_seconds:.2f} секунд.")
    print('-' * 100)

def worker_v2(segment_item, model):
    """
    segment_item: tuple (Stream, seg_name, stations_json)
    Обрабатывает один сегмент последовательно.
    """
    segment, seg_name, stations_json = segment_item
    status = "success"
    message = ""
    result = None

    try:
        print(f"▶️ Обработка сегмента {seg_name}: {len(segment)} каналов → {[tr.stats.channel for tr in segment]}")
        start_time = datetime.now()

        # --- Препроцессинг сегмента ---
        csv_data, hdf5_data = preprocessorV6_mem(
            stream_list=[(segment, seg_name)],
            stations_json=stations_json,
            overlap=0.3,
            n_processor=1
        )

        seg_csv = csv_data[seg_name][1:]  # пропускаем заголовок
        seg_hdf = hdf5_data[seg_name]

        # --- Обработка сегмента через предсказатель ---
        csv_file = predictor_mem_non_hdf_load_model(
            csv_segment=seg_csv,
            hdf_segment=seg_hdf,
            model=model,   # передаем уже загруженную модель
            output_dir=os.path.join(OUTPUT_DIR, seg_name),
            detection_threshold=0.8,
            P_threshold=0.6,
            S_threshold=0.5,
            number_of_plots=10,
            plot_mode='time',
            estimate_uncertainty=False,
            gpuid=0,
            keepPS=True,
            allowonlyS=True,
            spLimit=60
        )
        
        result = csv_file
        end_time = datetime.now()
        get_process_time(start_time, end_time)
        print(f"✅ Сегмент {seg_name} обработан, CSV: {csv_file}")

    except Exception:
        status = "error"
        message = format_exc()
        print(f"❌ Ошибка обработки {seg_name}: {message}")

    return (seg_name, status, message, result)

def worker(segment_item):
    """
    segment_item: tuple (Stream, seg_name, stations_json)
    Обрабатывает один сегмент последовательно.
    """
    segment, seg_name, stations_json = segment_item
    status = "success"
    message = ""
    result = None

    try:
        print(f"▶️ Обработка сегмента {seg_name}: {len(segment)} каналов → {[tr.stats.channel for tr in segment]}")
        start_time = datetime.now()
        # Препроцессинг сегмента
        csv_data, hdf5_data = preprocessorV6_mem(
            stream_list=[(segment, seg_name)],
            stations_json=stations_json,
            overlap=0.3,
            n_processor=1
        )

        seg_csv = csv_data[seg_name]      # весь CSV, включая заголовок
        seg_hdf = hdf5_data[seg_name]

        # Обработка сегмента
        csv_file = predictor_mem_non_hdf_load_model_batch(
            csv_segment=seg_csv[1:],  # пропускаем заголовок, predictor_mem_v6 ожидает только данные
            hdf_segment=seg_hdf,
            model_path='ModelsAndSampleData/EqT_original_model.h5',
            output_dir=os.path.join(OUTPUT_DIR, seg_name),
            detection_threshold=0.8,
            P_threshold=0.6,
            S_threshold=0.5,
            number_of_plots=0,  # строим графики
            plot_mode='time',
            estimate_uncertainty=True,
            gpuid=0,
            keepPS=True,
            allowonlyS=True,
            spLimit=60
        )
        result = csv_file
        end_time = datetime.now()
        get_process_time(start_time, end_time)
        print(f"✅ Сегмент {seg_name} обработан, CSV: {csv_file}")

    except Exception as e:
        status = "error"
        message = format_exc()
        print(f"❌ Ошибка обработки {seg_name}: {message}")

    return (seg_name, status, message, result)

def geofile_splitter_multi_chanels(base_directory, target_month=1, target_year=2024):
    """
    Делит данные по станциям и формирует сегменты 10 минут (шаг 5 минут)
    из нескольких каналов (E/N/Z или аналогичных). Поддерживает как с XX, так и без него.
    """
    files = [f for f in os.listdir(base_directory) if os.path.isfile(os.path.join(base_directory, f))]
    station_groups = {}

    pattern = re.compile(
        r'^(?P<net>[A-Z0-9]+)\.'          # сеть
        r'(?P<sta>[A-Z0-9]+)\.'           # станция
        r'(?P<loc>[A-Z0-9]{0,2})\.'       # локатор (может быть пустым)
        r'(?P<cha>[A-Z0-9]+)\.[A-Z_]*__'  # канал + суффикс
        r'(?P<start>\d{8}T\d{6}Z)__'      # начало
        r'(?P<end>\d{8}T\d{6}Z)$'         # конец
    )

    for file_name in files:
        match = pattern.match(file_name)
        if not match:
            print(f"⚠️ Пропускаем файл: {file_name} — не подходит под шаблон")
            continue

        sta = match.group("sta")
        start = match.group("start")
        key = f"{sta}_{start}"
        station_groups.setdefault(key, []).append(os.path.join(base_directory, file_name))

    segments = []

    for key, file_list in station_groups.items():
        try:
            st_all = Stream()
            for fpath in file_list:
                try:
                    st_all += read(fpath)
                except Exception as e:
                    print(f"Ошибка чтения {fpath}: {e}")

            if len(st_all) < 2:
                print(f"⚠️ Пропускаем {key}: найдено только {len(st_all)} канал(ов)")
                continue

            tr0 = st_all[0]
            start_time = tr0.stats.starttime
            end_time = tr0.stats.endtime

            if start_time.month != target_month or start_time.year != target_year:
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
                    segments.append((segment, seg_name))
                t += step

            # последний кусок
            if t < end_time:
                segment = st_all.slice(end_time - window_length, end_time)
                if len(segment) >= 2 and all(tr.stats.npts > 0 for tr in segment):
                    seg_name = (
                        f"{tr0.stats.network}.{tr0.stats.station}__"
                        f"{(end_time - window_length).strftime('%Y%m%dT%H%M%SZ')}__"
                        f"{end_time.strftime('%Y%m%dT%H%M%SZ')}"
                    )
                    segments.append((segment, seg_name))

        except Exception as e:
            print(f"❌ Ошибка обработки группы {key}: {e}")
            continue

    print(f"✅ Всего сегментов для обработки: {len(segments)}")
    return segments


def geofile_splitter_multi_chanels_v2(base_directory, target_month=1, target_year=2024):
    """
    Генераторная версия v1: отдаёт сегменты по одному, не накапливая в памяти.

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
            # st_all освобождается после обработки всех сегментов этого дня
            if 'st_all' in dir():
                del st_all

    print(f"✅ Всего сегментов обработано генератором: {total_yielded}")


def preproc_sequential(segments, stations_json):
    """
    Обработка сегментов последовательно.
    """
    results = []
    for segment, seg_name in segments:
        segment_item = (segment, seg_name, stations_json)
        res = worker(segment_item)
        results.append(res)

    # Запись логов
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "status", "message", "csv_file"])
        for r in results:
            writer.writerow(r)

    print(f"Лог обработки сохранён: {LOG_FILE}")
    print("Обработка завершена.")

def preproc_sequential_v2(segments, stations_json, model):
    """
    Обработка сегментов последовательно с уже загруженной моделью.
    """
    results = []
    for segment, seg_name in segments:
        segment_item = (segment, seg_name, stations_json)
        res = worker_v2(segment_item, model)
        results.append(res)

    # Запись логов
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "status", "message", "csv_file"])
        for r in results:
            writer.writerow(r)

    print(f"Лог обработки сохранён: {LOG_FILE}")
    print("Обработка завершена.")

def main_v2(base_directory, stations_json, model_path, target_month=1, target_year=2024):
    # --- Загружаем модель один раз ---
    print("🔹 Загрузка модели...")
    model = load_model(
        model_path,
        compile=False,
        custom_objects={
            'SeqSelfAttention': SeqSelfAttention,
            'FeedForward': FeedForward,
            'LayerNormalization': LayerNormalization,
            'f1': f1
        }
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=['binary_crossentropy'] * 3,
        metrics=[f1]
    )

    print("✅ Модель загружена и готова к работе.")

    segments = geofile_splitter_multi_chanels(base_directory, target_month, target_year)

    print("\n🔍 Проверка первых 5 сегментов:")
    for i, (segment, seg_name) in enumerate(segments[:5]):
        print(f"\nSegment {i+1}: {seg_name}")
        print(f"  Каналы: {[tr.stats.channel for tr in segment]}")
        print(f"  Время: {segment[0].stats.starttime} — {segment[0].stats.endtime}")
        print(f"  Длина: {segment[0].stats.npts} точек, частота: {segment[0].stats.sampling_rate} Hz")

    if segments:
        preproc_sequential_v2(segments, stations_json, model)


def main(base_directory, stations_json, target_month=1, target_year=2024):
    segments = geofile_splitter_multi_chanels(base_directory, target_month, target_year)

    print("\n🔍 Проверка первых 5 сегментов:")
    for i, (segment, seg_name) in enumerate(segments[:5]):
        print(f"\nSegment {i+1}: {seg_name}")
        print(f"  Каналы: {[tr.stats.channel for tr in segment]}")
        print(f"  Время: {segment[0].stats.starttime} — {segment[0].stats.endtime}")
        print(f"  Длина: {segment[0].stats.npts} точек, частота: {segment[0].stats.sampling_rate} Hz")

    if segments:
        preproc_sequential(segments, stations_json)


CSV_HEADER = [
    'file_name', 'network', 'station', 'instrument_type',
    'station_lat', 'station_lon', 'station_elv',
    'event_start_time', 'event_end_time',
    'detection_probability', 'detection_uncertainty',
    'p_arrival_time', 'p_probability', 'p_uncertainty', 'p_snr',
    's_arrival_time', 's_probability', 's_uncertainty', 's_snr'
]


def worker_v3(segment_item, model, save_figs=None, number_of_plots=10):
    """
    segment_item: tuple (Stream, seg_name, stations_json)
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

        end_time = datetime.now()
        total = (end_time - start_time).total_seconds()
        print(f"  preproc={t_preproc:.2f}s  predict={t_pred:.2f}s  total={total:.2f}s")
        print(f"✅ Сегмент {seg_name}: {len(rows)} событий")

    except Exception:
        status = "error"
        message = format_exc()
        print(f"❌ Ошибка обработки {seg_name}: {message}")

    return (seg_name, status, message, rows)


def preproc_sequential_v3(segments, stations_json, model, output_csv, save_figs=None, number_of_plots=10):
    """
    Обрабатывает сегменты последовательно и записывает все события в один CSV файл.
    Нет создания директорий под каждый сегмент.

    output_csv: путь к итоговому CSV файлу
    save_figs:  путь к каталогу для графиков, None — без графиков
    number_of_plots: максимум графиков на сегмент; игнорируется если save_figs=None
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    log_results = []
    total_events = 0

    with open(output_csv, 'w', newline='', encoding='utf-8') as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(CSV_HEADER)
        csv_out.flush()

        for segment, seg_name in segments:
            seg_name_out, status, message, rows = worker_v3(
                (segment, seg_name, stations_json),
                model,
                save_figs=save_figs,
                number_of_plots=number_of_plots
            )
            if rows:
                writer.writerows(rows)
                csv_out.flush()
                total_events += len(rows)
            log_results.append((seg_name_out, status, message, len(rows)))

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        log_writer = csv.writer(f)
        log_writer.writerow(["file_name", "status", "message", "events_found"])
        for r in log_results:
            log_writer.writerow(r)

    print(f"Лог обработки сохранён: {LOG_FILE}")
    print(f"Итоговый CSV: {output_csv} ({total_events} событий)")
    print("Обработка завершена.")


def main_v3(base_directory, stations_json, model_path,
            target_month=1, target_year=2024,
            create_figures=False):
    """
    Версия main без создания директорий под каждый сегмент.
    Все события пишутся в один CSV: output/<station>/<station_lower>.csv.
    create_figures=True — графики сохраняются в output/<station>/figures/.
    """
    print("🔹 Загрузка модели...")
    model = load_model(
        model_path,
        compile=False,
        custom_objects={
            'SeqSelfAttention': SeqSelfAttention,
            'FeedForward': FeedForward,
            'LayerNormalization': LayerNormalization,
            'f1': f1
        }
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=['binary_crossentropy'] * 3,
        metrics=[f1]
    )
    print("✅ Модель загружена и готова к работе.")

    station_name = os.path.basename(os.path.normpath(base_directory))
    output_dir = os.path.join("./data-in-memory/output", station_name)
    output_csv = os.path.join(output_dir, f"{station_name.lower()}.csv")
    save_figs = os.path.join(output_dir, "figures") if create_figures else None

    segments = geofile_splitter_multi_chanels(base_directory, target_month, target_year)

    print("\n🔍 Проверка первых 5 сегментов:")
    for i, (segment, seg_name) in enumerate(segments[:5]):
        print(f"\nSegment {i+1}: {seg_name}")
        print(f"  Каналы: {[tr.stats.channel for tr in segment]}")
        print(f"  Время: {segment[0].stats.starttime} — {segment[0].stats.endtime}")
        print(f"  Длина: {segment[0].stats.npts} точек, частота: {segment[0].stats.sampling_rate} Hz")

    if segments:
        preproc_sequential_v3(
            segments, stations_json, model,
            output_csv=output_csv,
            save_figs=save_figs,
            number_of_plots=10
        )


def preproc_sequential_v4(segment_gen, stations_json, model, output_csv,
                          save_figs=None, number_of_plots=10):
    """
    v3 с генератором вместо списка: сегменты обрабатываются по одному,
    каждый явно удаляется из памяти после обработки.
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
            del segment  # освобождаем данные сегмента до следующей итерации

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

    print(f"Лог обработки сохранён: {LOG_FILE}")
    print(f"Итоговый CSV: {output_csv} ({total_events} событий)")
    print("Обработка завершена.")


def main_v4(base_directory, stations_json, model_path,
            target_month=1, target_year=2024,
            create_figures=False):
    """
    v3 с генераторным сплиттером: потребление RAM не растёт с числом сегментов.
    """
    # GPU нужно настроить до первого вызова load_model — после инициализации
    # TF-контекста set_memory_growth вызвать уже нельзя.
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"✅ GPU настроен: {gpus[0].name}")
        except RuntimeError as e:
            print(f"⚠️ GPU: {e}")

    print("🔹 Загрузка модели...")
    model = load_model(
        model_path,
        compile=False,
        custom_objects={
            'SeqSelfAttention': SeqSelfAttention,
            'FeedForward': FeedForward,
            'LayerNormalization': LayerNormalization,
            'f1': f1
        }
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=['binary_crossentropy'] * 3,
        metrics=[f1]
    )
    print("✅ Модель загружена и готова к работе.")

    station_name = os.path.basename(os.path.normpath(base_directory))
    output_dir = os.path.join("./data-in-memory/output", station_name)
    output_csv = os.path.join(output_dir, f"{station_name.lower()}.csv")
    save_figs = os.path.join(output_dir, "figures") if create_figures else None

    segment_gen = geofile_splitter_multi_chanels_v2(
        base_directory, target_month, target_year
    )

    preproc_sequential_v4(
        segment_gen, stations_json, model,
        output_csv=output_csv,
        save_figs=save_figs,
        number_of_plots=10
    )


def _patch_lstm_for_cudnn(obj):
    """
    Рекурсивно патчит JSON конфиг модели: для каждого LSTM слоя
    устанавливает recurrent_dropout=0 и implementation=2.

    Почему JSON, а не clone_model:
    - clone_model с clone_function не достигает LSTM внутри Bidirectional
    - JSON-подход находит все LSTM рекурсивно по class_name
    - Модель строится заново уже с правильными параметрами → нет warning

    Почему implementation=2:
    - TF 2.10 требует implementation=2 для активации cuDNN LSTM kernels
    - implementation=1 использует пошаговый цикл, cuDNN его не поддерживает
    - implementation=2 использует батчевые матричные операции → быстрее на GPU
    """
    if isinstance(obj, dict):
        if obj.get('class_name') == 'LSTM':
            cfg = obj.get('config', {})
            cfg['recurrent_dropout'] = 0.0
            cfg['implementation'] = 2
        for v in obj.values():
            _patch_lstm_for_cudnn(v)
    elif isinstance(obj, list):
        for item in obj:
            _patch_lstm_for_cudnn(item)


def load_model_cudnn(model_path):
    """
    Загружает модель и пересоздаёт её с cuDNN-совместимыми LSTM слоями.

    Изменения относительно оригинала (только для inference, веса не меняются):
    - recurrent_dropout = 0  (dropout активен только при обучении, не при inference)
    - implementation = 2     (требование cuDNN LSTM в TF 2.10)
    """
    custom_objects = {
        'SeqSelfAttention': SeqSelfAttention,
        'FeedForward': FeedForward,
        'LayerNormalization': LayerNormalization,
        'f1': f1
    }

    model_orig = load_model(
        model_path,
        compile=False,
        custom_objects=custom_objects
    )

    model_json = json.loads(model_orig.to_json())
    _patch_lstm_for_cudnn(model_json)

    model_cudnn = tf.keras.models.model_from_json(
        json.dumps(model_json),
        custom_objects=custom_objects
    )
    model_cudnn.set_weights(model_orig.get_weights())
    model_cudnn.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=['binary_crossentropy'] * 3,
        metrics=[f1]
    )

    del model_orig
    gc.collect()

    return model_cudnn


def main_v5(base_directory, stations_json, model_path,
            target_month=1, target_year=2024,
            create_figures=False):
    """
    v4 с cuDNN-оптимизированной моделью.
    LSTM слои пересозданы без recurrent_dropout → cuDNN kernels активированы.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"✅ GPU настроен: {gpus[0].name}")
        except RuntimeError as e:
            print(f"⚠️ GPU: {e}")

    print("🔹 Загрузка и оптимизация модели для GPU (cuDNN LSTM)...")
    model = load_model_cudnn(model_path)
    print("✅ Модель готова.")

    station_name = os.path.basename(os.path.normpath(base_directory))
    output_dir = os.path.join("./data-in-memory/output", station_name)
    output_csv = os.path.join(output_dir, f"{station_name.lower()}.csv")
    save_figs = os.path.join(output_dir, "figures") if create_figures else None

    segment_gen = geofile_splitter_multi_chanels_v2(
        base_directory, target_month, target_year
    )

    preproc_sequential_v4(
        segment_gen, stations_json, model,
        output_csv=output_csv,
        save_figs=save_figs,
        number_of_plots=10
    )


def _patch_lstm_inplace(model):
    """
    Патчит LSTM cell напрямую, без JSON rebuild.

    recurrent_dropout и implementation — read-only @property на LSTM слое,
    они читаются из lstm.cell. Патчим cell напрямую: LSTMCell хранит их
    как обычные instance-атрибуты, поэтому присвоение работает.

    Важно: warning при load_model (трассировка графа) неизбежен.
    После патча cell — при model.predict() cuDNN-проверка проходит,
    cuDNN kernel используется и новый warning не появляется.

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
    v2: in-place патч вместо JSON rebuild.

    v1 (load_model_cudnn) пересоздавал модель через model_from_json —
    TF 2.10 игнорировал patched config при десериализации Bidirectional LSTM.
    v2 патчит объекты напрямую после load_model, до первого вызова predict.
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


def main_v6(base_directory, stations_json, model_path,
            target_month=1, target_year=2024,
            create_figures=False):
    """v5 с in-place LSTM патчем (load_model_cudnn_v2)."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"✅ GPU настроен: {gpus[0].name}")
        except RuntimeError as e:
            print(f"⚠️ GPU: {e}")

    print("🔹 Загрузка модели с in-place LSTM патчем...")
    model = load_model_cudnn_v2(model_path)
    print("✅ Модель готова.")

    station_name = os.path.basename(os.path.normpath(base_directory))
    output_dir = os.path.join("./data-in-memory/output", station_name)
    output_csv = os.path.join(output_dir, f"{station_name.lower()}.csv")
    save_figs = os.path.join(output_dir, "figures") if create_figures else None

    segment_gen = geofile_splitter_multi_chanels_v2(
        base_directory, target_month, target_year
    )

    preproc_sequential_v4(
        segment_gen, stations_json, model,
        output_csv=output_csv,
        save_figs=save_figs,
        number_of_plots=10
    )


def worker_v4(segment_item, model, save_figs=None, number_of_plots=10):
    """
    v5 предиктора: нормализация исправлена — только по std, без вычитания среднего.
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
        rows = predictor_mem_non_hdf_load_model_v5(
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

        end_time = datetime.now()
        total = (end_time - start_time).total_seconds()
        print(f"  preproc={t_preproc:.2f}s  predict={t_pred:.2f}s  total={total:.2f}s")
        print(f"✅ Сегмент {seg_name}: {len(rows)} событий")

    except Exception:
        status = "error"
        message = format_exc()
        print(f"❌ Ошибка обработки {seg_name}: {message}")

    return (seg_name, status, message, rows)


def preproc_sequential_v5(segment_gen, stations_json, model, output_csv,
                           save_figs=None, number_of_plots=10):
    """v4 с worker_v4 (predictor v5, исправленная нормализация)."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    header_written = False
    total_rows = 0

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for seg_count_inner, (segment, seg_name) in enumerate(segment_gen, 1):
            segment_item = (segment, seg_name, stations_json)
            _, _, _, rows = worker_v4(
                segment_item, model, save_figs, number_of_plots
            )
            if not header_written:
                writer.writerow([
                    'file_name', 'network', 'station', 'instrument_type',
                    'station_lat', 'station_lon', 'station_elv',
                    'event_start_time', 'event_end_time',
                    'detection_probability', 'detection_uncertainty',
                    'p_arrival_time', 'p_probability', 'p_uncertainty', 'p_snr',
                    's_arrival_time', 's_probability', 's_uncertainty', 's_snr'
                ])
                header_written = True
            for row in rows:
                writer.writerow(row)
            total_rows += len(rows)
            del segment, rows
            if seg_count_inner % 50 == 0:
                import gc; gc.collect()

    print(f"\n✅ Готово: {seg_count_inner} сегментов, {total_rows} событий → {output_csv}")


def main_v7(base_directory, stations_json, model_path,
            target_month=1, target_year=2024,
            create_figures=False, output_name=None):
    """v6 + output_name param. Использует predictor_v4 (z-score нормализация).
    output_name — переопределяет имя папки/файла вывода (по умолчанию = имя станции).
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"✅ GPU настроен: {gpus[0].name}")
        except RuntimeError as e:
            print(f"⚠️ GPU: {e}")

    print("🔹 Загрузка модели с in-place LSTM патчем...")
    model = load_model_cudnn_v2(model_path)
    print("✅ Модель готова.")

    station_name = output_name or os.path.basename(os.path.normpath(base_directory))
    output_dir = os.path.join("./data-in-memory/output", station_name)
    output_csv = os.path.join(output_dir, f"{station_name.lower()}.csv")
    save_figs = os.path.join(output_dir, "figures") if create_figures else None

    segment_gen = geofile_splitter_multi_chanels_v2(
        base_directory, target_month, target_year
    )

    preproc_sequential_v4(
        segment_gen, stations_json, model,
        output_csv=output_csv,
        save_figs=save_figs,
        number_of_plots=10
    )


def main_v9(base_directory, stations_json, model,
            target_month=1, target_year=2024,
            create_figures=False, output_name=None,
            output_base_dir="./data-in-memory/output"):
    """v8 с моделью переданной снаружи (initializer паттерн).
    Модель загружается один раз в воркере и переиспользуется для всех его станций.
    GPU/TF конфигурация должна быть сделана до вызова этой функции.
    """
    station_name = output_name or os.path.basename(os.path.normpath(base_directory))
    output_dir = os.path.join(output_base_dir, station_name)
    output_csv = os.path.join(output_dir, f"{station_name.lower()}.csv")
    save_figs = os.path.join(output_dir, "figures") if create_figures else None

    segment_gen = geofile_splitter_multi_chanels_v2(
        base_directory, target_month, target_year
    )

    preproc_sequential_v5(
        segment_gen, stations_json, model,
        output_csv=output_csv,
        save_figs=save_figs,
        number_of_plots=10
    )


def main_v10(base_directory, stations_json, model,
             target_month=1, target_year=2024,
             create_figures=False, output_name=None,
             output_base_dir="./data-in-memory/output"):
    """main_v9 с predictor_v4 (z-score нормализация).
    Используется в gpu_pipeline.py (initializer паттерн, модель передаётся снаружи).
    """
    station_name = output_name or os.path.basename(os.path.normpath(base_directory))
    output_dir = os.path.join(output_base_dir, station_name)
    output_csv = os.path.join(output_dir, f"{station_name.lower()}.csv")
    save_figs = os.path.join(output_dir, "figures") if create_figures else None

    segment_gen = geofile_splitter_multi_chanels_v2(
        base_directory, target_month, target_year
    )

    preproc_sequential_v4(
        segment_gen, stations_json, model,
        output_csv=output_csv,
        save_figs=save_figs,
        number_of_plots=10
    )


def main_v8(base_directory, stations_json, model_path,
            target_month=1, target_year=2024,
            create_figures=False, output_name=None,
            output_base_dir="./data-in-memory/output"):
    """v7 с настраиваемой базовой директорией вывода (output_base_dir).
    Используется для раздельного вывода GPU/CPU результатов.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"✅ GPU настроен: {gpus[0].name}")
        except RuntimeError as e:
            print(f"⚠️ GPU: {e}")

    print("🔹 Загрузка модели с in-place LSTM патчем...")
    model = load_model_cudnn_v2(model_path)
    print("✅ Модель готова.")

    station_name = output_name or os.path.basename(os.path.normpath(base_directory))
    output_dir = os.path.join(output_base_dir, station_name)
    output_csv = os.path.join(output_dir, f"{station_name.lower()}.csv")
    save_figs = os.path.join(output_dir, "figures") if create_figures else None

    segment_gen = geofile_splitter_multi_chanels_v2(
        base_directory, target_month, target_year
    )

    preproc_sequential_v5(
        segment_gen, stations_json, model,
        output_csv=output_csv,
        save_figs=save_figs,
        number_of_plots=10
    )


if __name__ == "__main__":
    model_path = "ModelsAndSampleData/EqT_original_model.h5"

    STATIONS = [
        ("./data-in-memory/input/SOC",  "./json/station_SOC.json"),
        ("./data-in-memory/input/VSLR", "./json/station_VSLR.json"),
        # ("./data-in-memory/input/GUZR", "./json/station_GUZR.json"),
        # ("./data-in-memory/input/BEYR", "./json/station_BEYR.json"),
        # ("./data-in-memory/input/SHA1", "./json/station_SHA1.json"),
        # ("./data-in-memory/input/MRNR", "./json/station_MRNR.json"),
        # ("./data-in-memory/input/SPGR", "./json/station_SPGR.json"),
        # ("./data-in-memory/input/DOMR", "./json/station_DOMR.json"),
        # ("./data-in-memory/input/ZEI",  "./json/station_ZEI.json"),
        # ("./data-in-memory/input/LABN", "./json/station_LABN.json"),
        # ("./data-in-memory/input/PYA1", "./json/station_PYA1.json"),
    ]

    from datetime import datetime
    start = datetime.now()
    for base_directory, stations_json in STATIONS:
        name = os.path.basename(os.path.normpath(base_directory))
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        main_v7(base_directory, stations_json, model_path,
                create_figures=False)
        elapsed = (datetime.now() - start).total_seconds() / 60
        print(f"  [{elapsed:.1f} мин] {name} готово")
    total = (datetime.now() - start).total_seconds() / 60
    print(f"\nВсего: {total:.1f} мин")