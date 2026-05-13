# metadata_processor.py

from obspy import read_inventory
import json
import os
from datetime import datetime
import warnings

# Игнорировать предупреждения
warnings.filterwarnings("ignore", category=UserWarning, module='obspy')

def read_inventory_file(file_path):
# Загрузка инвентаризации из XML файла
    inventory = read_inventory(file_path)

    # Создание структуры для JSON
    output_data = {}

    # Перебор всех сетей и станций в инвентаризации
    for network in inventory:
        network_code = network.code

        station_code = network.stations[0].code

        output_data[station_code] = {
            "network": network_code,
            "channels": [],
            "coords": []
        }
        
        for station in network:
        # Получаем координаты станции
            coords = [station.latitude, station.longitude, station.elevation]
            output_data[station_code]["coords"] = coords

            for channel in station:
                last_date = channel.start_date
            # Перебор всех каналов станции
            for channel in station:
                # Проверка на дату начала канала
                if channel.start_date >= last_date:
                    output_data[station_code]["channels"].append(channel.code)
    return output_data

def combine_inventories_to_json(xml_folder_path, json_output_path):
    """Объединяет данные из всех XML файлов в один JSON файл."""
    combined_data = {}

    # Проход по всем XML файлам в папке
    for filename in os.listdir(xml_folder_path):
        if filename.endswith('.xml'):
            file_path = os.path.join(xml_folder_path, filename)
            inventory_data = read_inventory_file(file_path)

            # Объединяем данные
            for station_code, data in inventory_data.items():
                if station_code not in combined_data:
                    combined_data[station_code] = data
                else:
                    combined_data[station_code]["channels"].extend(data["channels"])

    # Сохранение результата в JSON файл
    try:
        with open(json_output_path, 'w') as json_file:
            json.dump(combined_data, json_file, indent=4)
    except FileNotFoundError:
        print(f"Ошибка: Не удалось найти путь {json_output_path}. Проверьте, существует ли директория.")

    print("Данные успешно сохранены в", json_output_path)