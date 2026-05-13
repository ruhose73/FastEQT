import os
import glob
from obspy import read
from EQTransformer.utils.hdf5_maker import preprocessor
import os


INTERVAL_MINUTES = 5
TARGET_MONTH = 1
TARGET_YEAR = 2024

def geofile_splitter(base_directory, destination_base_directory):
    os.makedirs(destination_base_directory, exist_ok=True)

    files = glob.glob(os.path.join(base_directory, "*"))
    if not files:
        print(f"Файлы в {base_directory} не найдены.")
        return

    for file_path in files:
        try:
            st = read(file_path)
        except Exception as e:
            print(f"Ошибка чтения {file_path}: {e}")
            continue

        tr = st[0]
        start_time = tr.stats.starttime
        end_time = tr.stats.endtime

        # Фильтр по месяцу и году
        if start_time.month != TARGET_MONTH or start_time.year != TARGET_YEAR:
            continue

        t = start_time
        while t < end_time:
            t_next = t + INTERVAL_MINUTES * 60
            segment = st.slice(t, t_next)

            if len(segment) == 0 or segment[0].stats.npts == 0:
                t = t_next
                continue

            start_str = t.strftime('%Y%m%dT%H%M%SZ')
            end_str = min(t_next, end_time).strftime('%Y%m%dT%H%M%SZ')

            base_name = os.path.basename(file_path)
            new_file_name = f"{base_name}__{start_str}__{end_str}.mseed"
            out_path = os.path.join(destination_base_directory, new_file_name)

            segment.write(out_path, format="MSEED")
            print(f"Сохранен: {out_path}")

            t = t_next


def preproc_spltted(mseedDir, preproc):
    json_basepath = os.path.join(os.getcwd(),"json/station_ANN.json")
    num_threads = os.cpu_count()
    print(f"Количество потоков: {num_threads}")

    preprocessor(preproc_dir=preproc,
                mseed_dir=mseedDir, 
                stations_json=json_basepath, 
                overlap=0.3, 
                n_processor=num_threads)

def main(destination_preproc_directory, destination_base_directory, base_directory):
    # Вызов функции
    geofile_splitter(
        base_directory=base_directory,
        destination_base_directory=destination_base_directory
    )

    preproc_spltted(destination_base_directory, destination_preproc_directory)
        

if __name__ == "__main__":
    destination_preproc_directory="./data/preproc"
    destination_base_directory="./data/ANN"
    base_directory="./data/input/ANN"
    main(destination_preproc_directory, destination_base_directory, base_directory)