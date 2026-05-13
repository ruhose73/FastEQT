import os
import csv
from obspy import read
from multiprocessing import Pool, cpu_count
from EQTransformer.utils.hdf5_maker import preprocessorV3
from traceback import format_exc

LOG_FILE = "./processing_log.csv"

# === Worker на уровне модуля ===
def worker(segment_item):
    """
    segment_item: tuple (Stream, seg_name, preproc_dir, stations_json)
    Обрабатывает один сегмент и возвращает результат: (seg_name, status, message)
    """
    segment, seg_name, preproc_dir, stations_json = segment_item
    status = "success"
    message = ""

    try:
        # Передаем Stream напрямую в preprocessorV3
        preprocessorV3(
            preproc_dir=preproc_dir,
            stream_list=[(segment, seg_name)],
            stations_json=stations_json
        )
    except Exception as e:
        status = "error"
        message = format_exc()
        print(f"❌ Ошибка обработки {seg_name}: {message}")

    return (seg_name, status, message)


# === Разделение файлов на скользящие сегменты ===
def geofile_splitter(base_directory, target_month=1, target_year=2024):
    """
    Делит файлы на сегменты длиной 10 минут с шагом 5 минут.
    """
    files = os.listdir(base_directory)
    segments = []

    for file_path in files:
        full_path = os.path.join(base_directory, file_path)
        try:
            st = read(full_path)
        except Exception as e:
            print(f"Ошибка чтения {full_path}: {e}")
            continue

        tr = st[0]
        start_time = tr.stats.starttime
        end_time = tr.stats.endtime

        if start_time.month != target_month or start_time.year != target_year:
            continue

        window_length = 10 * 60  # 10 минут
        step = 5 * 60            # 5 минут

        t = start_time
        while t + window_length <= end_time:
            segment = st.slice(t, t + window_length)
            if len(segment) > 0 and segment[0].stats.npts > 0:
                seg_name = f"{file_path}__{t.strftime('%Y%m%dT%H%M%SZ')}__{(t + window_length).strftime('%Y%m%dT%H%M%SZ')}"
                segments.append((segment, seg_name))
            t += step

        # Обработка последнего окна, если оно меньше шага
        if t < end_time:
            segment = st.slice(end_time - window_length, end_time)
            if len(segment) > 0 and segment[0].stats.npts > 0:
                seg_name = f"{file_path}__{(end_time - window_length).strftime('%Y%m%dT%H%M%SZ')}__{end_time.strftime('%Y%m%dT%H%M%SZ')}"
                segments.append((segment, seg_name))

    print(f"Всего сегментов для обработки: {len(segments)}")
    return segments


# === Параллельная обработка сегментов ===
def preproc_parallel(segments, preproc_dir, stations_json):
    os.makedirs(preproc_dir, exist_ok=True)
    num_threads = max(1, cpu_count() - 1)

    task_list = [(seg, name, preproc_dir, stations_json) for seg, name in segments]

    with Pool(num_threads) as pool:
        results = pool.map(worker, task_list)

    # Запись логов
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "status", "message"])
        writer.writerows(results)

    print(f"Лог обработки сохранён: {LOG_FILE}")
    print("Обработка завершена.")


# === Основная функция ===
def main(base_directory, preproc_dir, stations_json, target_month=1, target_year=2024):
    segments = geofile_splitter(base_directory, target_month, target_year)
    if segments:
        preproc_parallel(segments, preproc_dir, stations_json)


# === Точка входа ===
if __name__ == "__main__":
    base_directory = "./data/input/ANN"
    preproc_dir = "./data/preproc"
    stations_json = "./json/station_ANN.json"
    main(base_directory, preproc_dir, stations_json)