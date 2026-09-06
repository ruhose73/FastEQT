# main.py
import os
from datetime import datetime
from metadata_processor import combine_inventories_to_json
from geofile_processor import geofile_processor

# Путь к папке с XML файлами и выходному JSON файлу
# (production-plan.md, Трек 1 — единая рабочая директория workspace/)
metadata_xml_folder_path = 'workspace/data_processors/input/metadata'
json_stationlist_output_path = 'workspace/data_processors/output/station_list_RU.json'
geofile_input_directory = 'workspace/data_processors/input/raw'
geofile_output_directory = 'workspace/data_processors/output/geofiles'

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





