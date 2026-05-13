# geofile_processor.py

import os
import shutil
from datetime import datetime, timedelta

# Задаем месяц и год для фильтрации
target_month = 1
target_year = 2024

def geofile_processor(subdirectories, base_directory, destination_base_directory):
        # Проходим по каждому подкаталогу
    for subdir in subdirectories:
        source_directory = os.path.join(base_directory, subdir)
        destination_directory = os.path.join(destination_base_directory, subdir)

        # Создаем целевой каталог, если он не существует
        os.makedirs(destination_directory, exist_ok=True)

        # Проверяем, существует ли подкаталог
        if os.path.exists(source_directory):
            # Проходим по всем подкаталогам в текущем подкаталоге
            for root, dirs, files in os.walk(source_directory):
                for file_name in files:
                        # Извлекаем номер дня из имени файла
                        day_part = file_name.split('.')[-1]

                        # Проверяем, является ли номер дня допустимым (001-365)
                        if day_part.isdigit() and 1 <= int(day_part) <= 365:
                            day_of_year = int(day_part)

                            # Вычисляем дату на основе номера дня в году
                            date = datetime(target_year, 1, 1) + timedelta(days=day_of_year - 1)

                            # Проверяем, попадает ли дата в целевой месяц
                            if date.month == target_month:
                                # Форматируем даты в нужный формат
                                start_date_str = date.strftime('%Y%m%dT%H%M%SZ')
                                end_date_str = (date + timedelta(days=1)).strftime('%Y%m%dT%H%M%SZ')

                                # Извлекаем имя папки
                                folder_name = os.path.basename(root)

                                # Создаем новое имя файла с учетом формата
                                base_name = file_name.rsplit('.', 2)[0]   # Извлекаем часть имени между 'RU.' и '.D'
                                print(base_name)
                                new_file_name = f'{base_name}__{start_date_str}__{end_date_str}'

                                # Полные пути к файлам
                                source_file_path = os.path.join(root, file_name)
                                destination_file_path = os.path.join(destination_directory, new_file_name)

                                # Копируем и переименовываем файл
                                shutil.copy2(source_file_path, destination_file_path)
                                print(f'Скопирован: {source_file_path} -> {destination_file_path}')
        else:
            print(f'Подкаталог не найден: {source_directory}')
