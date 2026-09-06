# main.py
import os
import sys
from datetime import datetime

# Пути ниже лежат в workspace/ и резолвятся от корня репозитория через
# _ROOT, а не от текущей рабочей директории — можно запускать и
# `python data_processors/main.py` из корня, и `python main.py` из самой
# data_processors/.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metadata_processor import combine_inventories_to_json
from geofile_processor import geofile_processor

# Путь к папке с XML файлами и выходному JSON файлу
metadata_xml_folder_path = os.path.join(_ROOT, 'workspace', 'data_processors', 'input', 'metadata')
json_stationlist_output_path = os.path.join(_ROOT, 'workspace', 'data_processors', 'output', 'station_list_RU.json')
geofile_input_directory = os.path.join(_ROOT, 'workspace', 'data_processors', 'input', 'raw')
geofile_output_directory = os.path.join(_ROOT, 'workspace', 'data_processors', 'output', 'geofiles')

metadata_xml_exist = os.path.exists(metadata_xml_folder_path)
json_stationlist_exist = os.path.exists(json_stationlist_output_path)
geofile_input_directory_exist = os.path.exists(geofile_input_directory)
geofile_output_directory_exist = os.path.exists(geofile_output_directory)

def get_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

if metadata_xml_exist:
    print("Создание списка станций")
    combine_inventories_to_json(metadata_xml_folder_path, json_stationlist_output_path)
else:
    print("XML", metadata_xml_exist)
    print("JSON", json_stationlist_exist)

if(geofile_input_directory_exist & geofile_output_directory_exist):
    print("Копирование рабочих сейсмических файлов")
    subdirectories = get_directories(geofile_input_directory)
    print("Станции", subdirectories)
    geofile_processor(subdirectories, geofile_input_directory, geofile_output_directory)
else:
    print("INPUT", geofile_input_directory_exist)
    print("OUTPUT", geofile_output_directory_exist)





