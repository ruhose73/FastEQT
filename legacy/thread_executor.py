import os
import math
from datetime import datetime
import concurrent.futures
import json
from EQTransformer.utils.hdf5_maker import preprocessor
from EQTransformer.core.predictor import predictor
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_process_time(start_time, end_time): 
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time.total_seconds() / 60
    formatted_start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print('-' * 100)
    print("Время запуска:", formatted_start_time)
    print("Время окончания:", formatted_end_time)
    print(f"Программа работала {elapsed_minutes:.2f} минут.")
    print('-' * 100)


def get_threads_to_use(percent): 
    value = percent / 100
    num_threads = os.cpu_count()
    return math.ceil(num_threads * value)

def predictor_geo(station): 
    try:
        input_dir_name = f'preproc/preproc_{station}'
        output_dir_name= f'geo_detections_hdfs/{station}' 
        print(input_dir_name)
        print(output_dir_name)
        predictor(input_dir=input_dir_name, 
                  input_model='ModelsAndSampleData/EqT_original_model.h5', 
                  output_dir=output_dir_name, 
                  detection_threshold=0.8, 
                  P_threshold=0.6, 
                  S_threshold=0.5, 
                  number_of_plots=1000, 
                  plot_mode='time', 
                  use_multiprocessing=False,
                  gpuid=0, 
                  output_probabilities=True, 
                  estimate_uncertainty=True, 
                  batch_size=500)
    except Exception as e:
        print(f'Ошибка predictor_geo: {e}')
    
def preprocess_geo(station_json, station, percent):
    try:
        output_dir_name = f'preproc/preproc_{station}'
        mseed_dir_name = f'geo-files/{station}'
        output_dir_basepath = os.path.join(os.getcwd(), output_dir_name)  
        threads = get_threads_to_use(percent)
        preprocessor(preproc_dir=output_dir_basepath,
                    mseed_dir=mseed_dir_name, 
                    stations_json=station_json, 
                    overlap=0.3, 
                    n_processor=threads)
    except Exception as e:
        print(f'Ошибка preprocess_geo: {e}')


def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def process_folder(folder_path, json_content, json_file_path, thread_percent):
    try:
        station = folder_path.split("\\")[-1]
        new_folder_path = os.path.dirname(json_file_path)
        print(f'Обработка станции: {station}')
        
        # Создаем файл с содержимым JSON только для одной станции (которую будем обрабатывать в потоке)
        output_file_name = f'station_{station}.json'
        output_file_path = os.path.join(new_folder_path, output_file_name)
        filtered_data = {key: value for key, value in json_content.items() if key == station}
        with open(output_file_path, 'w') as output_file:
            json.dump(filtered_data, output_file, indent=4)

        # Процессим станцию в рамках пула
        # preprocess_geo(output_file_path,station, thread_percent )
        predictor_geo(station)

    except Exception as e:
        print(f'Ошибка process_folder: {e}')



def main(directory, json_file_path, thread_percent):
    # Получаем список всех папок в указанной директории
    folders = [os.path.join(directory, name) for name in os.listdir(directory) 
               if os.path.isdir(os.path.join(directory, name))]
    
    # Читаем содержимое JSON файла со всеми станциями
    json_content = read_json(json_file_path)

    processors = len(folders)
    # Создаем пул процессов для препроцессинга (пробуем по количеству станций), проводим замеры
    with concurrent.futures.ProcessPoolExecutor(processors) as executor:
        # Запускаем обработку папок с передачей содержимого JSON
        future_to_folder = {executor.submit(process_folder, folder, json_content, json_file_path, thread_percent): folder for folder in folders}
        
        for future in concurrent.futures.as_completed(future_to_folder):
            folder = future_to_folder[future]
            try:
                future.result()  # Получаем результат выполнения
            except Exception as exc:
                print(f'Обработка станции {folder} сгенерировала исключение: {exc}')
    

if __name__ == "__main__":
    get_threads_to_use
    start_time = datetime.now()

    directory_path = 'geo-files'
    json_file_path = "json/station_list.json"
    thread_percent = 65

    main(directory_path, json_file_path, thread_percent)

    end_time = datetime.now()

    print('-' * 100)
    print("Всего потоков:", os.cpu_count())
    print("Использовалось потоков на пул:", get_threads_to_use(thread_percent))
    get_process_time(start_time, end_time)
    # concurrent.futures.ProcessPoolExecutor.shutdown()