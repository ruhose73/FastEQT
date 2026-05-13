#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 21:21:31 2019

@author: mostafamousavi

last update: 01/29/2021

- downsampling using the interpolation function can cause false segmentaiton error. 
    This depend on your data and its sampling rate. If you kept getting this error when 
    using multiprocessors, try using only a single cpu. 
    
"""

from obspy import read
import os
import platform
from os import listdir
from os.path import join
import h5py
import numpy as np
import csv
import shutil
import json
import pandas as pd
from multiprocessing.pool import ThreadPool
import multiprocessing
import pickle
import faulthandler; faulthandler.enable()

import json
import numpy as np
from obspy import Stream
from multiprocessing.pool import ThreadPool

def preprocessorV6_mem(stream_list, stations_json, overlap=0.3, n_processor=None):
    """
    Препроцессор потоков (Stream) в оперативной памяти для EQTransformer.
    Все окна приводятся к размеру (6000,3) с интерполяцией.
    Возвращает:
      - csv_data: dict[str, list[list[str]]]
      - hdf5_data: dict[str, dict[str, dict]]
    """
    if n_processor is None:
        import multiprocessing
        n_processor = max(1, multiprocessing.cpu_count() - 1)

    with open(stations_json, 'r') as f:
        stations_ = json.load(f)

    csv_data = {}
    hdf5_data = {}

    def process(item):
        st, base_name = item
        output_name = base_name
        csv_rows = [['trace_name', 'start_time']]
        hdf_datasets = {}

        try:
            # --- Детеренд, фильтр, оконная косинусная аподизация ---
            st.detrend('demean')
            st.filter('bandpass', freqmin=1.0, freqmax=45, corners=2, zerophase=True)
            st.taper(max_percentage=0.001, type='cosine', max_length=2)

            # --- Выравнивание по времени ---
            start_time = max(tr.stats.starttime for tr in st)
            end_time = min(tr.stats.endtime for tr in st)
            st.trim(start_time, end_time, pad=True, fill_value=0)
        except Exception as e:
            print(f"❌ Ошибка препроцессинга {base_name}: {e}")
            return

        available_channels = [tr.stats.channel[-1] for tr in st]
        required_channels = ['E', 'N', 'Z']
        channel_map = {'Z': 2, 'E': 0, '1': 0, 'N': 1, '2': 1}

        slide = int(60 - overlap * 60)
        next_slice = start_time + 60

        while next_slice <= end_time:
            w = st.slice(start_time, next_slice)
            npz_data = np.zeros([6000, 3], dtype=np.float32)

            # Заполняем доступные каналы
            for tr in w:
                ch = tr.stats.channel[-1]
                if ch in channel_map:
                    col = channel_map[ch]
                    data = tr.data
                    # Интерполяция до 6000 точек
                    if len(data) != 6000:
                        x_old = np.linspace(0, 1, len(data))
                        x_new = np.linspace(0, 1, 6000)
                        data = np.interp(x_new, x_old, data)
                    npz_data[:, col] = data

            # Дополнение недостающих каналов нулями
            for req_ch in required_channels:
                if req_ch not in available_channels:
                    col = channel_map[req_ch]
                    npz_data[:, col] = 0.0

            tr_name = f"{st[0].stats.station}_{st[0].stats.network}_{st[0].stats.channel[:2]}_{str(start_time)}"

            # Формируем структуру HDF5 в памяти
            hdf_datasets[tr_name] = {
                'data': npz_data,
                'attrs': {
                    "trace_name": tr_name,
                    "receiver_code": st[0].stats.station,
                    "network_code": stations_[st[0].stats.station]['network'],
                    "receiver_latitude": stations_[st[0].stats.station]['coords'][0],
                    "receiver_longitude": stations_[st[0].stats.station]['coords'][1],
                    "receiver_elevation_m": stations_[st[0].stats.station]['coords'][2],
                    "trace_start_time": str(start_time).replace('T', ' ').replace('Z', ''),
                    "sampling_rate": st[0].stats.sampling_rate  # добавляем частоту
                }
            }

            csv_rows.append([tr_name, str(start_time)])
            start_time += slide
            next_slice += slide

        csv_data[output_name] = csv_rows
        hdf5_data[output_name] = hdf_datasets
        print(f"✅ {output_name}: обработано {len(hdf_datasets)} окон (каналов: {len(st)})")

    from multiprocessing.dummy import Pool as ThreadPool
    with ThreadPool(n_processor) as pool:
        pool.map(process, stream_list)

    return csv_data, hdf5_data

def preprocessorV5_mem(stream_list, stations_json, overlap=0.3, n_processor=None):
    """
    Препроцессинг сегментов в памяти с сохранением частоты и точных временных меток.
    Возвращает:
    - csv_data: dict[str, list[list[str]]]
    - hdf5_data: dict[str, dict[str, dict]]
    """
    if n_processor is None:
        import multiprocessing
        n_processor = max(1, multiprocessing.cpu_count() - 1)

    with open(stations_json, 'r') as f:
        stations_ = json.load(f)

    csv_data = {}
    hdf5_data = {}

    def process(item):
        st, base_name = item
        output_name = base_name

        csv_rows = [['trace_name', 'start_time']]
        hdf_datasets = {}

        try:
            st.detrend('demean')
            st.filter('bandpass', freqmin=1.0, freqmax=45, corners=2, zerophase=True)
            st.taper(max_percentage=0.001, type='cosine', max_length=2)

            # Сохраняем оригинальную частоту
            orig_sampling_rate = st[0].stats.sampling_rate

            # Не интерполируем! Просто используем оригинальную частоту, как в мастере
            start_time = max(tr.stats.starttime for tr in st)
            end_time = min(tr.stats.endtime for tr in st)
            st.trim(start_time, end_time, pad=True, fill_value=0)

        except Exception as e:
            print(f"❌ Ошибка препроцессинга {base_name}: {e}")
            return

        required_channels = ['E', 'N', 'Z']
        channel_map = {'Z': 2, 'E': 0, '1': 0, 'N': 1, '2': 1}

        # Слайдинг по окнам без округления
        window_length = 60.0  # сек
        step = window_length * (1 - overlap)
        curr_start = start_time
        while curr_start + window_length <= end_time:
            w = st.slice(curr_start, curr_start + window_length)
            npz_data = np.zeros([int(window_length * orig_sampling_rate), 3], dtype=np.float32)

            available_channels = [tr.stats.channel[-1] for tr in w]
            for tr in w:
                ch = tr.stats.channel[-1]
                if ch in channel_map:
                    col = channel_map[ch]
                    data = tr.data
                    if len(data) > npz_data.shape[0]:
                        data = data[:npz_data.shape[0]]
                    npz_data[:len(data), col] = data

            # Дополнение недостающих каналов нулями
            for req_ch in required_channels:
                if req_ch not in available_channels:
                    col = channel_map[req_ch]
                    npz_data[:, col] = 0.0

            tr_name = f"{st[0].stats.station}_{st[0].stats.network}_{st[0].stats.channel[:2]}_{str(curr_start)}"
            hdf_datasets[tr_name] = {
                'data': npz_data,
                'attrs': {
                    "trace_name": tr_name,
                    "receiver_code": st[0].stats.station,
                    "network_code": stations_[st[0].stats.station]['network'],
                    "receiver_latitude": stations_[st[0].stats.station]['coords'][0],
                    "receiver_longitude": stations_[st[0].stats.station]['coords'][1],
                    "receiver_elevation_m": stations_[st[0].stats.station]['coords'][2],
                    "trace_start_time": str(curr_start).replace('T', ' ').replace('Z', ''),
                    "sampling_rate": orig_sampling_rate
                }
            }

            csv_rows.append([tr_name, str(curr_start)])
            curr_start += step

        csv_data[output_name] = csv_rows
        hdf5_data[output_name] = hdf_datasets
        print(f"✅ {output_name}: обработано {len(hdf_datasets)} окон (каналов: {len(st)})")

    with ThreadPool(n_processor) as pool:
        pool.map(process, stream_list)

    return csv_data, hdf5_data

def preprocessorV4_mem(stream_list, stations_json, overlap=0.3, n_processor=None):
    """
    Улучшенный препроцессор для обработки потоков (Stream) в оперативной памяти.
    Полностью эквивалентен оригинальному preprocessorV3, но:
      - работает без записи на диск;
      - автоматически дополняет недостающие каналы нулями;
      - выполняет выравнивание, фильтрацию и нормализацию.

    Возвращает:
    - csv_data: dict[str, list[list[str]]]
    - hdf5_data: dict[str, dict[str, dict]]
    """

    if n_processor is None:
        import multiprocessing
        n_processor = max(1, multiprocessing.cpu_count() - 1)

    with open(stations_json, 'r') as f:
        stations_ = json.load(f)

    csv_data = {}
    hdf5_data = {}

    def process(item):
        st, base_name = item
        output_name = base_name

        csv_rows = [['trace_name', 'start_time']]
        hdf_datasets = {}

        try:
            # === 🧹 Подготовка сигналов ===
            st.detrend('demean')
            st.filter('bandpass', freqmin=1.0, freqmax=45, corners=2, zerophase=True)
            st.taper(max_percentage=0.001, type='cosine', max_length=2)

            # Интерполяция до 100 Hz
            if any(tr.stats.sampling_rate != 100.0 for tr in st):
                st.interpolate(100.0, method="linear")

            # Выровнять начало и конец всех каналов
            start_time = max(tr.stats.starttime for tr in st)
            end_time = min(tr.stats.endtime for tr in st)
            st.trim(start_time, end_time, pad=True, fill_value=0)

        except Exception as e:
            print(f"❌ Ошибка препроцессинга {base_name}: {e}")
            return

        # === 🧭 Подготовка 3-канального массива ===
        available_channels = [tr.stats.channel[-1] for tr in st]
        required_channels = ['E', 'N', 'Z']
        channel_map = {'Z': 2, 'E': 0, '1': 0, 'N': 1, '2': 1}

        # Создаём шаблон данных (3 канала × 6000 отсчётов)
        slide = int(60 - overlap * 60)
        next_slice = start_time + 60

        while next_slice <= end_time:
            w = st.slice(start_time, next_slice)
            npz_data = np.zeros([6000, 3], dtype=np.float32)

            # Заполняем доступные каналы
            for tr in w:
                ch = tr.stats.channel[-1]
                if ch in channel_map:
                    col = channel_map[ch]
                    data = tr.data
                    if len(data) > 6000:
                        data = data[:6000]
                    npz_data[:len(data), col] = data

            # === 🧩 Дополнение недостающих каналов нулями ===
            for req_ch in required_channels:
                if req_ch not in available_channels:
                    col = channel_map[req_ch]
                    # Канал отсутствует → заполняем нулями
                    npz_data[:, col] = 0.0

            tr_name = f"{st[0].stats.station}_{st[0].stats.network}_{st[0].stats.channel[:2]}_{str(start_time)}"

            # === 📦 Формируем структуру HDF5 в памяти ===
            hdf_datasets[tr_name] = {
                'data': npz_data,
                'attrs': {
                    "trace_name": tr_name,
                    "receiver_code": st[0].stats.station,
                    "network_code": stations_[st[0].stats.station]['network'],
                    "receiver_latitude": stations_[st[0].stats.station]['coords'][0],
                    "receiver_longitude": stations_[st[0].stats.station]['coords'][1],
                    "receiver_elevation_m": stations_[st[0].stats.station]['coords'][2],
                    "trace_start_time": str(start_time).replace('T', ' ').replace('Z', '')
                }
            }

            csv_rows.append([tr_name, str(start_time)])
            start_time += slide
            next_slice += slide

        csv_data[output_name] = csv_rows
        hdf5_data[output_name] = hdf_datasets
        print(f"✅ {output_name}: обработано {len(hdf_datasets)} окон (каналов: {len(st)})")

    # === Параллельная обработка сегментов ===
    with ThreadPool(n_processor) as pool:
        pool.map(process, stream_list)

    return csv_data, hdf5_data

def preprocessorV3_mem(stream_list, stations_json, overlap=0.3, n_processor=None):
    """
    Препроцессор для оперативной памяти. Возвращает:
    - csv_data: dict[str, list[list[str]]]
    - hdf5_data: dict[str, dict[str, dict]] (ключи: dataset_name -> {'data': np.ndarray, 'attrs': dict})
    """

    if n_processor is None:
        import multiprocessing
        n_processor = max(1, multiprocessing.cpu_count() - 1)

    with open(stations_json, 'r') as f:
        stations_ = json.load(f)

    csv_data = {}
    hdf5_data = {}

    def process(item):
        st, base_name = item
        output_name = base_name

        csv_rows = [['trace_name', 'start_time']]
        hdf_datasets = {}

        longest = max([tr.stats.npts for tr in st])
        start_time = min([tr.stats.starttime for tr in st])
        end_time = max([tr.stats.endtime for tr in st])
        slide = int(60 - overlap * 60)

        next_slice = start_time + 60
        while next_slice <= end_time:
            w = st.slice(start_time, next_slice)
            npz_data = np.zeros([6000, 3], dtype=np.float32)

            channel_map = {'Z': 2, 'E': 0, '1': 0, 'N': 1, '2': 1}
            for tr in w:
                ch = tr.stats.channel[-1]
                if ch in channel_map:
                    col = channel_map[ch]
                    npz_data[:len(tr.data), col] = tr.data[:6000]

            tr_name = f"{st[0].stats.station}_{st[0].stats.network}_{st[0].stats.channel[:2]}_{str(start_time)}"

            # Сохраняем данные в памяти
            hdf_datasets[tr_name] = {
                'data': npz_data,
                'attrs': {
                    "trace_name": tr_name,
                    "receiver_code": st[0].stats.station,
                    "network_code": stations_[st[0].stats.station]['network'],
                    "receiver_latitude": stations_[st[0].stats.station]['coords'][0],
                    "receiver_longitude": stations_[st[0].stats.station]['coords'][1],
                    "receiver_elevation_m": stations_[st[0].stats.station]['coords'][2],
                    "trace_start_time": str(start_time).replace('T', ' ').replace('Z', '')
                }
            }

            csv_rows.append([tr_name, str(start_time)])

            start_time += slide
            next_slice += slide

        csv_data[output_name] = csv_rows
        hdf5_data[output_name] = hdf_datasets
        print(f"✅ {output_name} processed in memory.")

    with ThreadPool(n_processor) as pool:
        pool.map(process, stream_list)

    return csv_data, hdf5_data

def preprocessorV3(preproc_dir, stream_list, stations_json, overlap=0.3, n_processor=None):
    if n_processor is None:
        import multiprocessing
        n_processor = max(1, multiprocessing.cpu_count() - 1)

    with open(stations_json, 'r') as f:
        stations_ = json.load(f)

    os.makedirs(preproc_dir, exist_ok=True)
    repfile = open(os.path.join(preproc_dir, "X_preprocessor_report.txt"), 'w')
    data_track = dict()

    def process(item):
        st, base_name = item
        output_name = base_name

        hdf5_path = os.path.join(preproc_dir, output_name + ".hdf5")
        csv_path = os.path.join(preproc_dir, output_name + ".csv")

        HDF = h5py.File(hdf5_path, 'w')
        HDF.create_group("data")

        csvfile = open(csv_path, 'w', newline='')
        writer = csv.writer(csvfile)
        writer.writerow(['trace_name', 'start_time'])

        longest = max([tr.stats.npts for tr in st])
        start_time = min([tr.stats.starttime for tr in st])
        end_time = max([tr.stats.endtime for tr in st])
        slide = int(60 - overlap * 60)

        next_slice = start_time + 60
        while next_slice <= end_time:
            w = st.slice(start_time, next_slice)
            npz_data = np.zeros([6000, 3], dtype=np.float32)

            channel_map = {'Z': 2, 'E': 0, '1': 0, 'N': 1, '2': 1}
            for tr in w:
                ch = tr.stats.channel[-1]
                if ch in channel_map:
                    col = channel_map[ch]
                    npz_data[:len(tr.data), col] = tr.data[:6000]

            tr_name = f"{st[0].stats.station}_{st[0].stats.network}_{st[0].stats.channel[:2]}_{str(start_time)}"
            dsF = HDF.create_dataset(f"data/{tr_name}", npz_data.shape, data=npz_data)
            dsF.attrs["trace_name"] = tr_name
            dsF.attrs["receiver_code"] = st[0].stats.station
            dsF.attrs["network_code"] = stations_[st[0].stats.station]['network']
            dsF.attrs["receiver_latitude"] = stations_[st[0].stats.station]['coords'][0]
            dsF.attrs["receiver_longitude"] = stations_[st[0].stats.station]['coords'][1]
            dsF.attrs["receiver_elevation_m"] = stations_[st[0].stats.station]['coords'][2]
            dsF.attrs["trace_start_time"] = str(start_time).replace('T', ' ').replace('Z', '')

            writer.writerow([tr_name, str(start_time)])
            csvfile.flush()
            HDF.flush()

            start_time += slide
            next_slice += slide

        HDF.close()
        csvfile.close()
        data_track[output_name] = None
        repfile.write(f"{output_name} processed, slices saved to CSV and HDF5\n")
        print(f"✅ {output_name} processed.")

    with ThreadPool(n_processor) as pool:
        pool.map(process, stream_list)

    with open(os.path.join(preproc_dir, 'time_tracks.pkl'), 'wb') as f:
        pickle.dump(data_track, f, pickle.HIGHEST_PROTOCOL)
    repfile.close()

def preprocessorV2(preproc_dir, stream_list, stations_json, overlap=0.3, n_processor=None):
    """
    Preprocesses already-read mseed streams and partitions them into 1-minute slices.

    Parameters
    ----------
    preproc_dir: str
        Directory to save HDF5, CSV, and log files.
    
    stream_list: list of tuples
        List of tuples (stream, base_name), where `stream` is an obspy.Stream object
        and `base_name` is a string identifier for that stream.

    stations_json: str
        Path to a JSON file containing station metadata.

    overlap: float, default=0.3
        Overlapping fraction for slicing.

    n_processor: int, default=None
        Number of threads for parallel processing.
    """
    import json
    if n_processor is None:
        import multiprocessing
        n_processor = max(1, multiprocessing.cpu_count() - 1)

    # Load station metadata
    with open(stations_json, 'r') as f:
        stations_ = json.load(f)

    os.makedirs(preproc_dir, exist_ok=True)
    repfile_path = os.path.join(preproc_dir, "X_preprocessor_report.txt")
    repfile = open(repfile_path, 'w')

    data_track = dict()

    def process(item):
        st, base_name = item
        output_name = base_name
        time_slots, comp_types = [], []

        # Prepare HDF5 and CSV
        hdf5_path = os.path.join(preproc_dir, output_name + ".hdf5")
        csv_path = os.path.join(preproc_dir, output_name + ".csv")

        HDF = h5py.File(hdf5_path, 'w')
        HDF.create_group("data")

        csvfile = open(csv_path, 'w', newline='')
        writer = csv.writer(csvfile)
        writer.writerow(['trace_name', 'start_time'])

        # Determine longest trace for trimming
        longest = max([tr.stats.npts for tr in st])
        start_time = min([tr.stats.starttime for tr in st])
        end_time = max([tr.stats.endtime for tr in st])
        slide = int(60 - overlap * 60)

        # Channel mapping
        chanL = [tr.stats.channel[-1] for tr in st]

        next_slice = start_time + 60
        while next_slice <= end_time:
            w = st.slice(start_time, next_slice)
            npz_data = np.zeros([6000, 3], dtype=np.float32)

            # Map channels to Z, E, N
            try:
                npz_data[:, 2] = w[chanL.index('Z')].data[:6000]
            except ValueError:
                pass
            try:
                npz_data[:, 0] = w[chanL.index('E')].data[:6000]
            except ValueError:
                try:
                    npz_data[:, 0] = w[chanL.index('1')].data[:6000]
                except ValueError:
                    pass
            try:
                npz_data[:, 1] = w[chanL.index('N')].data[:6000]
            except ValueError:
                try:
                    npz_data[:, 1] = w[chanL.index('2')].data[:6000]
                except ValueError:
                    pass

            tr_name = f"{st[0].stats.station}_{st[0].stats.network}_{st[0].stats.channel[:2]}_{str(start_time)}"
            dsF = HDF.create_dataset(f"data/{tr_name}", npz_data.shape, data=npz_data)
            dsF.attrs["trace_name"] = tr_name
            dsF.attrs["receiver_code"] = st[0].stats.station
            dsF.attrs["network_code"] = stations_[st[0].stats.station]['network']
            dsF.attrs["receiver_latitude"] = stations_[st[0].stats.station]['coords'][0]
            dsF.attrs["receiver_longitude"] = stations_[st[0].stats.station]['coords'][1]
            dsF.attrs["receiver_elevation_m"] = stations_[st[0].stats.station]['coords'][2]
            dsF.attrs["trace_start_time"] = str(start_time).replace('T', ' ').replace('Z', '')

            writer.writerow([tr_name, str(start_time)])
            csvfile.flush()
            HDF.flush()

            start_time += slide
            next_slice += slide

        HDF.close()
        csvfile.close()

        data_track[output_name] = [time_slots, comp_types]
        repfile.write(f"{output_name} processed, slices saved to CSV and HDF5\n")
        print(f"✅ {output_name} processed.")

    # Parallel processing
    with ThreadPool(n_processor) as pool:
        pool.map(process, stream_list)

    # Save time tracks
    with open(os.path.join(preproc_dir, 'time_tracks.pkl'), 'wb') as f:
        pickle.dump(data_track, f, pickle.HIGHEST_PROTOCOL)

    repfile.close()

def preprocessor(preproc_dir, mseed_dir, stations_json, overlap=0.3, n_processor=None):
    
    
    """
    
    Performs preprocessing and partitions the continuous waveforms into 1-minute slices. 

    Parameters
    ----------
    preproc_dir: str
        Path of the directory where will be located the summary files generated by preprocessor step.

    mseed_dir: str
        Path of the directory where the mseed files are located. 

    stations_json: str
        Path to a JSON file containing station information.        
        
    overlap: float, default=0.3
        If set, detection, and picking are performed in overlapping windows.
           
    n_processor: int, default=None 
        The number of CPU processors for parallel preprocessing.         

    Returns
    ----------
    mseed_dir_processed_hdfs/station.csv: Phase information for the associated events in hypoInverse format. 
    
    mseed_dir_processed_hdfs/station.hdf5: Containes all slices and preprocessed traces. 
    
    preproc_dir/X_preprocessor_report.txt: A summary of processing performance. 
    
    preproc_dir/time_tracks.pkl: Contain the time track of the continous data and its type.
       
    """  
 
    
    if not n_processor:
        n_processor = multiprocessing.cpu_count()
    
    json_file = open(stations_json)
    stations_ = json.load(json_file)
    
    # save_dir = os.path.join(os.getcwd(), str(mseed_dir)+'_processed_hdfs')
    # if os.path.isdir(save_dir):
    #     print(f' *** " {save_dir} " directory already exists!')
    #     inp = input(" * --> Do you want to creat a new empty folder? Type (Yes or y) ")
    #     if inp.lower() == "yes" or inp.lower() == "y":        
    #         shutil.rmtree(save_dir)  
    # os.makedirs(save_dir)
              
    if not os.path.exists(preproc_dir):
            os.makedirs(preproc_dir)
    repfile = open(os.path.join(preproc_dir,"X_preprocessor_report.txt"), 'w');

    save_dir = preproc_dir
    
    if platform.system() == 'Windows':
        station_list = [mseed_dir.replace("/", "\\")]
        # station_list = [join(mseed_dir, ev) for ev in listdir(mseed_dir) if ev.split("\\")[-1] != ".DS_Store"];
    else:   
        station_list = [join(mseed_dir, ev) for ev in listdir(mseed_dir) if ev.split("/")[-1] != ".DS_Store"];
    
    data_track = dict()
    # print('station_list', station_list)
    
    def process(station):
    # for station in station_list:
        if platform.system() == 'Windows':
            output_name = station.split("\\")[-1]
            output_name = output_name.split('/')[-1]
        else:
            output_name = station.split("/")[-1]
        
        try:
            os.remove(output_name+'.hdf5')
            os.remove(output_name+".csv")
        except Exception:
            pass
        
        # print('save_dir', save_dir)
        # print('output_name', output_name)

        HDF = h5py.File(os.path.join(save_dir, output_name+'.hdf5'), 'a')
        HDF.create_group("data")
    
        csvfile = open(os.path.join(save_dir, output_name+".csv"), 'w')
        output_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        output_writer.writerow(['trace_name', 'start_time'])
        csvfile.flush()   
    
        if platform.system() == 'Windows':
            file_list = [join(station, ev) for ev in listdir(station) if ev.split("\\")[-1] != ".DS_Store"];
        else:
            file_list = [join(station, ev) for ev in listdir(station) if ev.split("/")[-1] != ".DS_Store"];
            
        mon = [ev.split('__')[1]+'__'+ev.split('__')[2] for ev in file_list ];
        uni_list = list(set(mon))
        uni_list.sort()        
        tim_shift = int(60-(overlap*60))
        
        time_slots, comp_types = [], []
        
        if platform.system() == 'Windows':
            print('============ Station {} has {} chunks of data.'.format(station.split("\\")[-1], len(uni_list)), flush=True)   
        else:
            print('============ Station {} has {} chunks of data.'.format(station.split("/")[-1], len(uni_list)), flush=True)  
            
        count_chuncks=0; fln=0; c1=0; c2=0; c3=0; fl_counts=1; slide_estimates=[];
        
        for ct, month in enumerate(uni_list):
            matching = [s for s in file_list if month in s]
            
            if len(matching) == 3:  
                
                st1 = read(matching[0], debug_headers=True)
                org_samplingRate = st1[0].stats.sampling_rate
                
                for tr in st1:                   
                    time_slots.append((tr.stats.starttime, tr.stats.endtime))
                    comp_types.append(3)

                try:
                    st1.merge(fill_value=0) 
                except Exception:
                    st1=_resampling(st1)
                    st1.merge(fill_value=0)                     
                st1.detrend('demean') 
                count_chuncks += 1; c3 += 1
                if platform.system() == 'Windows':
                    print('  * '+station.split("\\")[-1]+' ('+str(count_chuncks)+') .. '+month.split('T')[0]+' --> '+month.split('__')[1].split('T')[0]+' .. 3 components .. sampling rate: '+str(org_samplingRate))  
                else:
                    print('  * '+station.split("/")[-1]+' ('+str(count_chuncks)+') .. '+month.split('T')[0]+' --> '+month.split('__')[1].split('T')[0]+' .. 3 components .. sampling rate: '+str(org_samplingRate))  
                 
                st2 = read(matching[1], debug_headers=True) 
                try:
                    st2.merge(fill_value=0)                    
                except Exception:
                    st2=_resampling(st2)
                    st2.merge(fill_value=0)                    
                st2.detrend('demean')
    
                st3 = read(matching[2], debug_headers=True) 
                try:
                    st3.merge(fill_value=0)                     
                except Exception:
                    st3=_resampling(st3)
                    st3.merge(fill_value=0) 
                st3.detrend('demean')
                
                st1.append(st2[0])
                st1.append(st3[0])
                st1.filter('bandpass',freqmin = 1.0, freqmax = 45, corners=2, zerophase=True)
                st1.taper(max_percentage=0.001, type='cosine', max_length=2)
                if len([tr for tr in st1 if tr.stats.sampling_rate != 100.0]) != 0:
                    try:
                        st1.interpolate(100, method="linear")
                    except Exception:
                        st1=_resampling(st1)
                        
                                     
                longest = st1[0].stats.npts
                start_time = st1[0].stats.starttime
                end_time = st1[0].stats.endtime
                
                for tt in st1:
                    if tt.stats.npts > longest:
                        longest = tt.stats.npts
                        start_time = tt.stats.starttime
                        end_time = tt.stats.endtime
                    
                st1.trim(start_time, end_time, pad=True, fill_value=0)

                start_time = st1[0].stats.starttime
                end_time = st1[0].stats.endtime  
                slide_estimates.append((end_time - start_time)//tim_shift)                
                fl_counts += 1 
                
                chanL = [st1[0].stats.channel[-1], st1[1].stats.channel[-1], st1[2].stats.channel[-1]]
                next_slice = start_time+60               
                while next_slice <= end_time:
                    w = st1.slice(start_time, next_slice) 
                    npz_data = np.zeros([6000,3])
                                        
                    npz_data[:,2] = w[chanL.index('Z')].data[:6000]
                    try: 
                        npz_data[:,0] = w[chanL.index('E')].data[:6000]
                    except Exception:
                        npz_data[:,0] = w[chanL.index('1')].data[:6000]
                    try: 
                        npz_data[:,1] = w[chanL.index('N')].data[:6000]
                    except Exception:
                        npz_data[:,1] = w[chanL.index('2')].data[:6000]                        
                                     
                    tr_name = st1[0].stats.station+'_'+st1[0].stats.network+'_'+st1[0].stats.channel[:2]+'_'+str(start_time)
                    HDF = h5py.File(os.path.join(save_dir,output_name+'.hdf5'), 'r')
                    dsF = HDF.create_dataset('data/'+tr_name, npz_data.shape, data = npz_data, dtype= np.float32)        
                       
                    dsF.attrs["trace_name"] = tr_name 
                    if platform.system() == 'Windows':
                        dsF.attrs["receiver_code"] = station.split("\\")[-1]
                        dsF.attrs["network_code"] = stations_[station.split("\\")[-1]]['network']
                        dsF.attrs["receiver_latitude"] = stations_[station.split("\\")[-1]]['coords'][0]
                        dsF.attrs["receiver_longitude"] = stations_[station.split("\\")[-1]]['coords'][1]
                        dsF.attrs["receiver_elevation_m"] = stations_[station.split("\\")[-1]]['coords'][2] 
                    else:
                        dsF.attrs["receiver_code"] = station.split("/")[-1]
                        dsF.attrs["network_code"] = stations_[station.split("/")[-1]]['network']
                        dsF.attrs["receiver_latitude"] = stations_[station.split("/")[-1]]['coords'][0]
                        dsF.attrs["receiver_longitude"] = stations_[station.split("/")[-1]]['coords'][1]
                        dsF.attrs["receiver_elevation_m"] = stations_[station.split("/")[-1]]['coords'][2] 
                    
                    start_time_str = str(start_time)   
                    start_time_str = start_time_str.replace('T', ' ')                 
                    start_time_str = start_time_str.replace('Z', '')          
                    dsF.attrs['trace_start_time'] = start_time_str
                    HDF.flush()
                    output_writer.writerow([str(tr_name), start_time_str])  
                    csvfile.flush()
                    fln += 1            
            
                    start_time = start_time+tim_shift
                    next_slice = next_slice+tim_shift 
  
            if len(matching) == 1:  
                 count_chuncks += 1; c1 += 1
                
                 st1 = read(matching[0], debug_headers=True)
                 org_samplingRate = st1[0].stats.sampling_rate

                 for tr in st1:                   
                     time_slots.append((tr.stats.starttime, tr.stats.endtime))
                     comp_types.append(1)
                 try:
                     st1.merge(fill_value=0) 
                 except Exception:
                     st1=_resampling(st1)
                     st1.merge(fill_value=0)                 
                 st1.detrend('demean')
                 
                 if platform.system() == 'Windows':
                     print('  * '+station.split("\\")[-1]+' ('+str(count_chuncks)+') .. '+month.split('T')[0]+' --> '+month.split('__')[1].split('T')[0]+' .. 1 components .. sampling rate: '+str(org_samplingRate)) 
                 else:
                     print('  * '+station.split("/")[-1]+' ('+str(count_chuncks)+') .. '+month.split('T')[0]+' --> '+month.split('__')[1].split('T')[0]+' .. 1 components .. sampling rate: '+str(org_samplingRate)) 
                 
                 st1.filter('bandpass',freqmin = 1.0, freqmax = 45, corners=2, zerophase=True)
                 st1.taper(max_percentage=0.001, type='cosine', max_length=2)
                 if len([tr for tr in st1 if tr.stats.sampling_rate != 100.0]) != 0:
                     try:
                         st1.interpolate(100, method="linear")
                     except Exception:
                         st1=_resampling(st1) 
                         
                 chan = st1[0].stats.channel
                 start_time = st1[0].stats.starttime
                 end_time = st1[0].stats.endtime
                 slide_estimates.append((end_time - start_time)//tim_shift)
                 fl_counts += 1    

                 next_slice = start_time+60

                 while next_slice <= end_time:
                     w = st1.slice(start_time, next_slice)                    
                     npz_data = np.zeros([6000,3])
                     if chan[-1] == 'Z':
                         npz_data[:,2] = w[0].data[:6000]
                     if chan[-1] == 'E' or  chan[-1] == '1':
                         npz_data[:,0] = w[0].data[:6000]
                     if chan[-1] == 'N' or  chan[-1] == '2':
                         npz_data[:,1] = w[0].data[:6000]
                    
                     tr_name = st1[0].stats.station+'_'+st1[0].stats.network+'_'+st1[0].stats.channel[:2]+'_'+str(start_time)
                     HDF = h5py.File(os.path.join(save_dir,output_name+'.hdf5'), 'r')
                     dsF = HDF.create_dataset('data/'+tr_name, npz_data.shape, data = npz_data, dtype= np.float32)        
                     dsF.attrs["trace_name"] = tr_name 
                     
                     if platform.system() == 'Windows':
                         dsF.attrs["receiver_code"] = station.split("\\")[-1]
                         dsF.attrs["network_code"] = stations_[station.split("\\")[-1]]['network']
                         dsF.attrs["receiver_latitude"] = stations_[station.split("\\")[-1]]['coords'][0]
                         dsF.attrs["receiver_longitude"] = stations_[station.split("\\")[-1]]['coords'][1]
                         dsF.attrs["receiver_elevation_m"] = stations_[station.split("\\")[-1]]['coords'][2]  
                     else:                         
                         dsF.attrs["receiver_code"] = station.split("/")[-1]
                         dsF.attrs["network_code"] = stations_[station.split("/")[-1]]['network']
                         dsF.attrs["receiver_latitude"] = stations_[station.split("/")[-1]]['coords'][0]
                         dsF.attrs["receiver_longitude"] = stations_[station.split("/")[-1]]['coords'][1]
                         dsF.attrs["receiver_elevation_m"] = stations_[station.split("/")[-1]]['coords'][2] 
                         
                     start_time_str = str(start_time)   
                     start_time_str = start_time_str.replace('T', ' ')                 
                     start_time_str = start_time_str.replace('Z', '')          
                     dsF.attrs['trace_start_time'] = start_time_str
                     HDF.flush()
                     output_writer.writerow([str(tr_name), start_time_str])  
                     csvfile.flush()
                     fln += 1            

                     start_time = start_time+tim_shift
                     next_slice = next_slice+tim_shift                
                
            if len(matching) == 2:  
                count_chuncks += 1; c2 += 1                
                st1 = read(matching[0], debug_headers=True)
                org_samplingRate = st1[0].stats.sampling_rate

                for tr in st1:                   
                    time_slots.append((tr.stats.starttime, tr.stats.endtime))
                    comp_types.append(2)

                try:
                    st1.merge(fill_value=0) 
                except Exception:
                    st1=_resampling(st1)
                    st1.merge(fill_value=0)  
                st1.detrend('demean')  
                
                org_samplingRate = st1[0].stats.sampling_rate
                
                if platform.system() == 'Windows':
                    print('  * '+station.split("\\")[-1]+' ('+str(count_chuncks)+') .. '+month.split('T')[0]+' --> '+month.split('__')[1].split('T')[0]+' .. 2 components .. sampling rate: '+str(org_samplingRate)) 
                else:    
                    print('  * '+station.split("/")[-1]+' ('+str(count_chuncks)+') .. '+month.split('T')[0]+' --> '+month.split('__')[1].split('T')[0]+' .. 2 components .. sampling rate: '+str(org_samplingRate)) 
                 
                st2 = read(matching[1], debug_headers=True)  
                try:
                    st2.merge(fill_value=0) 
                except Exception:
                    st2=_resampling(st1)
                    st2.merge(fill_value=0)                 
                st2.detrend('demean')
    
                st1.append(st2[0])
                st1.filter('bandpass',freqmin = 1.0, freqmax = 45, corners=2, zerophase=True)
                st1.taper(max_percentage=0.001, type='cosine', max_length=2)
                if len([tr for tr in st1 if tr.stats.sampling_rate != 100.0]) != 0:
                    try:
                        st1.interpolate(100, method="linear")
                    except Exception:
                        st1=_resampling(st1)   
                        
                longest = st1[0].stats.npts
                start_time = st1[0].stats.starttime
                end_time = st1[0].stats.endtime
                
                for tt in st1:
                    if tt.stats.npts > longest:
                        longest = tt.stats.npts
                        start_time = tt.stats.starttime
                        end_time = tt.stats.endtime               
                
                st1.trim(start_time, end_time, pad=True, fill_value=0)

                start_time = st1[0].stats.starttime
                end_time = st1[0].stats.endtime
                slide_estimates.append((end_time - start_time)//tim_shift)
                
                chan1 = st1[0].stats.channel
                chan2 = st1[1].stats.channel
                fl_counts += 1  
                
                next_slice = start_time+60

                while next_slice <= end_time:
                    w = st1.slice(start_time, next_slice)                     
                    npz_data = np.zeros([6000,3])
                    if chan1[-1] == 'Z':
                        npz_data[:,2] = w[0].data[:6000]
                    elif chan1[-1] == 'E' or  chan1[-1] == '1':
                        npz_data[:,0] = w[0].data[:6000]
                    elif chan1[-1] == 'N' or  chan1[-1] == '2':
                        npz_data[:,1] = w[0].data[:6000]

                    if chan2[-1] == 'Z':
                        npz_data[:,2] = w[1].data[:6000]
                    elif chan2[-1] == 'E' or  chan2[-1] == '1':
                        npz_data[:,0] = w[1].data[:6000]
                    elif chan2[-1] == 'N' or  chan2[-1] == '2':
                        npz_data[:,1] = w[1].data[:6000]
                    
                    tr_name = st1[0].stats.station+'_'+st1[0].stats.network+'_'+st1[0].stats.channel[:2]+'_'+str(start_time)
                    HDF = h5py.File(os.path.join(save_dir,output_name+'.hdf5'), 'r')
                    dsF = HDF.create_dataset('data/'+tr_name, npz_data.shape, data = npz_data, dtype= np.float32)        
                       
                    dsF.attrs["trace_name"] = tr_name 
                    
                    if platform.system() == 'Windows':
                        dsF.attrs["receiver_code"] = station.split("\\")[-1]
                        dsF.attrs["network_code"] = stations_[station.split("\\")[-1]]['network']
                        dsF.attrs["receiver_latitude"] = stations_[station.split("\\")[-1]]['coords'][0]
                        dsF.attrs["receiver_longitude"] = stations_[station.split("\\")[-1]]['coords'][1]
                        dsF.attrs["receiver_elevation_m"] = stations_[station.split("\\")[-1]]['coords'][2] 
                    else:    
                        dsF.attrs["receiver_code"] = station.split("/")[-1]
                        dsF.attrs["network_code"] = stations_[station.split("/")[-1]]['network']
                        dsF.attrs["receiver_latitude"] = stations_[station.split("/")[-1]]['coords'][0]
                        dsF.attrs["receiver_longitude"] = stations_[station.split("/")[-1]]['coords'][1]
                        dsF.attrs["receiver_elevation_m"] = stations_[station.split("/")[-1]]['coords'][2] 
                    
                    start_time_str = str(start_time)   
                    start_time_str = start_time_str.replace('T', ' ')                 
                    start_time_str = start_time_str.replace('Z', '')          
                    dsF.attrs['trace_start_time'] = start_time_str
                    HDF.flush()
                    output_writer.writerow([str(tr_name), start_time_str])  
                    csvfile.flush()
                    fln += 1            
            
                    start_time = start_time+tim_shift
                    next_slice = next_slice+tim_shift 
                    
            st1, st2, st3 = None, None, None
                
        HDF.close() 
        csvfile.close()
        
        dd = pd.read_csv(os.path.join(save_dir, output_name+".csv"))
                
        
        assert count_chuncks == len(uni_list)  
        assert sum(slide_estimates)-(fln/100) <= len(dd) <= sum(slide_estimates)+10
        data_track[output_name]=[time_slots, comp_types]
        print(f" Station {output_name} had {len(uni_list)} chuncks of data") 
        print(f"{len(dd)} slices were written, {sum(slide_estimates)} were expected.")
        print(f"Number of 1-components: {c1}. Number of 2-components: {c2}. Number of 3-components: {c3}.")
        try:
            print(f"Original samplieng rate: {org_samplingRate}.") 
            repfile.write(f' Station {output_name} had {len(uni_list)} chuncks of data, {len(dd)} slices were written, {int(sum(slide_estimates))} were expected. Number of 1-components: {c1}, Number of 2-components: {c2}, number of 3-components: {c3}, original samplieng rate: {org_samplingRate}\n')
        except Exception:
            pass
    with ThreadPool(n_processor) as p:
        p.map(process, station_list) 
    with open(os.path.join(preproc_dir,'time_tracks.pkl'), 'wb') as f:
        pickle.dump(data_track, f, pickle.HIGHEST_PROTOCOL)

def stationListFromMseed(mseed_directory, station_locations, dir_json='./'):
    """
    Contributed by: Tyler Newton
        
    Reads all miniseed files contained within subdirectories in the specified directory and generates a station_list.json file that describes the miniseed files in the correct format for EQTransformer.
    
    Parameters
    ----------
    mseed_directory: str
        String specifying the absolute path to the directory containing miniseed files. Directory must contain subdirectories of station names, which contain miniseed files in the EQTransformer format. 
        Each component must be a seperate miniseed file, and the naming
        convention is GS.CA06.00.HH1__20190901T000000Z__20190902T000000Z
        .mseed, or more generally 
        NETWORK.STATION.LOCATION.CHANNEL__STARTTIMESTAMP__ENDTIMESTAMP.mseed
        where LOCATION is optional.
    station_locations: dict
        Dictonary with station names as keys and lists of latitude,
        longitude, and elevation as items. For example: {"CA06": [35.59962,
        -117.49268, 796.4], "CA10": [35.56736, -117.667427, 835.9]}
    dir_json: str
        String specifying the path to the output json file.
   
    Returns
    -------
    stations_list.json: A dictionary containing information for the available stations.
    
    Example
    -------
    directory = '/Users/human/Downloads/eqt/examples/downloads_mseeds'
    locations = {"CA06": [35.59962, -117.49268, 796.4], "CA10": [35.56736, -117.667427, 835.9]}
    stationListFromMseed(directoy, locations)
    """

    station_list = {}

    # loop through subdirectories of specified directory
    for subdirectory in os.scandir(mseed_directory):
        if subdirectory.is_dir():
            channels = []
            # build channel list from miniseed files
            for mseed_file in os.scandir(subdirectory.path):
                temp_stream = read(mseed_file.path, debug_headers=True)
                channels.append(temp_stream[0].stats.channel)
            # add entry to station list for the current station
            station_list[str(temp_stream[0].stats.station)] = {"network":
                        temp_stream[0].stats.network, "channels": list(set(
                        channels)), "coords": station_locations[str(
                        temp_stream[0].stats.station)]}
    
    if not os.path.exists(dir_json):
        os.makedirs(dir_json)
    jfilename = os.path.join(dir_json, 'station_list.json')
    with open(jfilename, 'w') as fp:
        json.dump(station_list, fp)             
        
def _resampling(st):
    need_resampling = [tr for tr in st if tr.stats.sampling_rate != 100.0]
    if len(need_resampling) > 0:
       # print('resampling ...', flush=True)    
        for indx, tr in enumerate(need_resampling):
            if tr.stats.delta < 0.01:
                tr.filter('lowpass',freq=45,zerophase=True)
            tr.resample(100)
            tr.stats.sampling_rate = 100
            tr.stats.delta = 0.01
            tr.data.dtype = 'int32'
            st.remove(tr)                    
            st.append(tr)    
             
    return st