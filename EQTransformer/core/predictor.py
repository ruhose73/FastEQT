#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:44:14 2018

@author: mostafamousavi
last update: 05/27/2021

"""

from __future__ import print_function
from __future__ import division
import os
os.environ['KERAS_BACKEND']='tensorflow'
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import csv
import h5py
import time
from os import listdir
import platform
import shutil
from .EqT_utils import DataGeneratorPrediction, picker, generate_arrays_from_file
from .EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
# --- Загружаем модель с кастомными слоями ---
from tqdm import tqdm
from datetime import datetime, timedelta
import multiprocessing
import contextlib
import sys
import warnings
from scipy import signal
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    f = open('setup.py')
    for li, l in enumerate(f):
        if li == 8:
            EQT_VERSION = l.split('"')[1]
except Exception:
    EQT_VERSION = "0.1.61"

class HDFMemoryDataset:
    """Эмуляция HDF5 dataset для in-memory работы с predictor_mem."""
    def __init__(self, data, attrs):
        self.data = data
        self.attrs = attrs

def predictor_mem_non_hdf_load_model(csv_segment, hdf_segment, 
                          model_path=None,  # оставляем для совместимости
                          model=None,       # если модель уже загружена
                          output_dir=None,
                          detection_threshold=0.3,                
                          P_threshold=0.1,
                          S_threshold=0.1, 
                          number_of_plots=10,
                          plot_mode='time',
                          estimate_uncertainty=False, 
                          number_of_sampling=5,
                          gpuid=None,
                          keepPS=True,
                          allowonlyS=True,
                          spLimit=60):

    # --- GPU настройка ---
    if gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)

    # --- Создание выходных папок ---
    os.makedirs(output_dir, exist_ok=True)
    save_dir = output_dir
    save_figs = os.path.join(save_dir, 'figures')
    if number_of_plots > 0:
        os.makedirs(save_figs, exist_ok=True)

    # --- CSV-файл результатов ---
    csv_file = os.path.join(save_dir, 'X_prediction_results.csv')
    csvPr_gen = open(csv_file, 'w', newline='')
    predict_writer = csv.writer(csvPr_gen)
    predict_writer.writerow([...])
    csvPr_gen.flush()

    # --- Загружаем модель только если не передали готовую ---
    if model is None:
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

    args = {
        'input_hdf5': 'in-memory',
        'input_dimention': (6000, 3),  
        'normalization_mode': 'std',
        'use_multiprocessing': False,
        'number_of_cpus': 1,
        'estimate_uncertainty': estimate_uncertainty,
        'number_of_sampling': number_of_sampling,
        'output_probabilities': False,
        'number_of_plots': number_of_plots,
        'detection_threshold': detection_threshold,
        'P_threshold': P_threshold,
        'S_threshold': S_threshold,
        'plot_mode': plot_mode
    }

    detection_memory = []
    plt_n = 0

    for row in tqdm(csv_segment, desc="Predicting"):
        ID = row[0]
        if ID not in hdf_segment:
            print(f"⚠ Пропускаем {ID}, ключ не найден в HDF5")
            continue

        class InMemoryDataset:
            def __init__(self, data_dict):
                self.data = data_dict['data']
                self.attrs = data_dict['attrs']

        dataset_obj = InMemoryDataset(hdf_segment[ID])
        prob_dic = _gen_predictor_mem([ID], args, model, {ID: dataset_obj.data})

        plt_n, detection_memory = _gen_writer_mem_non_hdf_v8(
            [ID],
            args,
            prob_dic,
            {ID: dataset_obj},
            predict_writer=predict_writer,
            save_figs=save_figs,
            csvPr_gen=csvPr_gen,
            plt_n=plt_n,
            detection_memory=detection_memory,
            keepPS=keepPS,
            allowonlyS=allowonlyS,
            spLimit=spLimit
        )

    csvPr_gen.close()
    return csv_file

def _gen_writer_mem_non_hdf_v8(new_list, args, prob_dic, pred_set,
                               predict_writer, save_figs, csvPr_gen,
                               plt_n, detection_memory, keepPS, allowonlyS, spLimit):
    """
    Обработка предсказаний модели для in-memory данных без HDF5.
    """
    for ts in range(prob_dic['DD_mean'].shape[0]):
        evi = new_list[ts]
        dataset = pred_set[evi]
        dat = np.array(dataset.data)

        # --- Приведение dat к форме (N,3) ---
        if dat.ndim == 0:
            dat = np.zeros((6000, 3), dtype=np.float32)
        elif dat.ndim == 1:
            dat = dat[:, np.newaxis]
        if dat.shape[1] < 3:
            tmp = np.zeros((dat.shape[0], 3), dtype=dat.dtype)
            tmp[:, :dat.shape[1]] = dat
            dat = tmp

        # --- Частота дискретизации ---
        sampling_rate = dataset.attrs.get("sampling_rate", 100.0)
        print(f"▶️ Частота в attrs для {dataset.attrs.get('trace_name','')}: {sampling_rate} Hz")

        # --- Детектор P/S ---
        matches, pick_errors, yh3 = picker(
            args,
            prob_dic['DD_mean'][ts],
            prob_dic['PP_mean'][ts],
            prob_dic['SS_mean'][ts],
            prob_dic['DD_std'][ts],
            prob_dic['PP_std'][ts],
            prob_dic['SS_std'][ts]
        )

        if len(matches) == 0:
            continue

        match_idx = list(matches)[0]
        match_value = matches[match_idx]

        # --- Фильтрация allowonlyS ---
        if not allowonlyS and match_value[6] is not None and match_value[3] is None:
            continue

        # --- Фильтрация keepPS и spLimit ---
        valid_event = False
        if keepPS:
            if match_value[3] is not None and match_value[6] is not None:
                if (match_value[6] - match_value[3]) < spLimit * sampling_rate:
                    valid_event = True
        else:
            if match_value[3] is not None or match_value[6] is not None:
                valid_event = True
        if not valid_event:
            continue

        # --- SNR ---
        def safe_snr(data, idx):
            if idx is None or idx < 0 or idx >= len(data):
                return np.nan
            return _get_snr(data, idx, window=min(100, len(data)))

        snr_p = safe_snr(dat, match_value[3])
        snr_s = safe_snr(dat, match_value[6])

        # --- Конвертация времени начала трассы ---
        start_time_str = dataset.attrs["trace_start_time"]
        try:
            start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')

        # --- event_start / event_end ---
        ev_strt = start_time + timedelta(seconds=match_idx / sampling_rate)
        ev_end  = start_time + timedelta(seconds=match_value[0] / sampling_rate)

        # --- Проверка на дубликаты ---
        doublet = [st for st in detection_memory if abs((st - ev_strt).total_seconds()) < 2]
        if doublet:
            continue

        # --- Вероятности и неопределённости ---
        det_prob = round(match_value[1], 2)
        det_unc  = round(match_value[2], 2) if match_value[2] is not None else np.nan

        p_time = start_time + timedelta(seconds=match_value[3] / sampling_rate) if match_value[3] is not None else None
        p_prob = round(match_value[4], 2) if match_value[4] is not None else np.nan
        p_unc  = round(match_value[5], 2) if match_value[5] is not None else np.nan

        s_time = start_time + timedelta(seconds=match_value[6] / sampling_rate) if match_value[6] is not None else None
        s_prob = round(match_value[7], 2) if match_value[7] is not None else np.nan
        s_unc  = round(match_value[8], 2) if match_value[8] is not None else np.nan

        # --- Метаданные станции ---
        trace_name = dataset.attrs["trace_name"]
        station_name = "{:<4}".format(dataset.attrs["receiver_code"])
        network_name = "{:<2}".format(dataset.attrs["network_code"])
        instrument_type = "{:<2}".format(trace_name.split('_')[2])
        station_lat = dataset.attrs["receiver_latitude"]
        station_lon = dataset.attrs["receiver_longitude"]
        station_elv = dataset.attrs["receiver_elevation_m"]

        # --- Запись в CSV ---
        def _date_convertor(r):
            return r if r is not None else ''

        predict_writer.writerow([
            trace_name, network_name, station_name, instrument_type,
            station_lat, station_lon, station_elv,
            _date_convertor(ev_strt), _date_convertor(ev_end),
            det_prob, det_unc,
            _date_convertor(p_time), p_prob, p_unc, snr_p,
            _date_convertor(s_time), s_prob, s_unc, snr_s
        ])
        csvPr_gen.flush()
        detection_memory.append(ev_strt)

        # --- Построение графиков ---
        if plt_n < args['number_of_plots'] and dat.size > 0 and plt_n > 0:
            _plotter_prediction(
                dat, evi, args, save_figs,
                prob_dic['DD_mean'][ts],
                prob_dic['PP_mean'][ts],
                prob_dic['SS_mean'][ts],
                prob_dic['DD_std'][ts],
                prob_dic['PP_std'][ts],
                prob_dic['SS_std'][ts],
                matches
            )
            plt_n += 1

    return plt_n, detection_memory


def _gen_writer_mem_non_hdf_v9(new_list, args, prob_dic, pred_set,
                               predict_writer, save_figs, csvPr_gen,
                               plt_n, detection_memory, keepPS, allowonlyS, spLimit):
    """
    Исправленная версия v8:
    - sampling_rate всегда 100.0 (данные интерполированы до 6000 отсчётов в preprocessorV6_mem)
    - plt_n >= 0 (v8 пропускала первый ивент из-за plt_n > 0)
    """
    SAMPLING_RATE = 100.0  # EQT всегда работает с 100 Гц (6000 отсчётов / 60 с)

    for ts in range(prob_dic['DD_mean'].shape[0]):
        evi = new_list[ts]
        dataset = pred_set[evi]
        dat = np.array(dataset.data)

        if dat.ndim == 0:
            dat = np.zeros((6000, 3), dtype=np.float32)
        elif dat.ndim == 1:
            dat = dat[:, np.newaxis]
        if dat.shape[1] < 3:
            tmp = np.zeros((dat.shape[0], 3), dtype=dat.dtype)
            tmp[:, :dat.shape[1]] = dat
            dat = tmp

        matches, pick_errors, yh3 = picker(
            args,
            prob_dic['DD_mean'][ts],
            prob_dic['PP_mean'][ts],
            prob_dic['SS_mean'][ts],
            prob_dic['DD_std'][ts],
            prob_dic['PP_std'][ts],
            prob_dic['SS_std'][ts]
        )

        if len(matches) == 0:
            continue

        match_idx = list(matches)[0]
        match_value = matches[match_idx]

        if not allowonlyS and match_value[6] is not None and match_value[3] is None:
            continue

        valid_event = False
        if keepPS:
            if match_value[3] is not None and match_value[6] is not None:
                if (match_value[6] - match_value[3]) < spLimit * SAMPLING_RATE:
                    valid_event = True
        else:
            if match_value[3] is not None or match_value[6] is not None:
                valid_event = True
        if not valid_event:
            continue

        def safe_snr(data, idx):
            if idx is None or idx < 0 or idx >= len(data):
                return np.nan
            return _get_snr(data, idx, window=min(100, len(data)))

        snr_p = safe_snr(dat, match_value[3])
        snr_s = safe_snr(dat, match_value[6])

        start_time_str = dataset.attrs["trace_start_time"]
        try:
            start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')

        ev_strt = start_time + timedelta(seconds=match_idx / SAMPLING_RATE)
        ev_end  = start_time + timedelta(seconds=match_value[0] / SAMPLING_RATE)

        doublet = [st for st in detection_memory if abs((st - ev_strt).total_seconds()) < 2]
        if doublet:
            continue

        det_prob = round(match_value[1], 2)
        det_unc  = round(match_value[2], 2) if match_value[2] is not None else np.nan

        p_time = start_time + timedelta(seconds=match_value[3] / SAMPLING_RATE) if match_value[3] is not None else None
        p_prob = round(match_value[4], 2) if match_value[4] is not None else np.nan
        p_unc  = round(match_value[5], 2) if match_value[5] is not None else np.nan

        s_time = start_time + timedelta(seconds=match_value[6] / SAMPLING_RATE) if match_value[6] is not None else None
        s_prob = round(match_value[7], 2) if match_value[7] is not None else np.nan
        s_unc  = round(match_value[8], 2) if match_value[8] is not None else np.nan

        trace_name = dataset.attrs["trace_name"]
        station_name = "{:<4}".format(dataset.attrs["receiver_code"])
        network_name = "{:<2}".format(dataset.attrs["network_code"])
        instrument_type = "{:<2}".format(trace_name.split('_')[2])
        station_lat = dataset.attrs["receiver_latitude"]
        station_lon = dataset.attrs["receiver_longitude"]
        station_elv = dataset.attrs["receiver_elevation_m"]

        predict_writer.writerow([
            trace_name, network_name, station_name, instrument_type,
            station_lat, station_lon, station_elv,
            ev_strt, ev_end,
            det_prob, det_unc,
            p_time, p_prob, p_unc, snr_p,
            s_time, s_prob, s_unc, snr_s
        ])
        csvPr_gen.flush()
        detection_memory.append(ev_strt)

        if plt_n < args['number_of_plots'] and dat.size > 0:  # убрано plt_n > 0
            _plotter_prediction(
                dat, evi, args, save_figs,
                prob_dic['DD_mean'][ts],
                prob_dic['PP_mean'][ts],
                prob_dic['SS_mean'][ts],
                prob_dic['DD_std'][ts],
                prob_dic['PP_std'][ts],
                prob_dic['SS_std'][ts],
                matches
            )
            plt_n += 1

    return plt_n, detection_memory


def predictor_mem_non_hdf_load_model_v2(csv_segment, hdf_segment,
                                        model_path=None,
                                        model=None,
                                        output_dir=None,
                                        detection_threshold=0.3,
                                        P_threshold=0.1,
                                        S_threshold=0.1,
                                        number_of_plots=10,
                                        plot_mode='time',
                                        estimate_uncertainty=False,
                                        number_of_sampling=5,
                                        gpuid=None,
                                        keepPS=True,
                                        allowonlyS=True,
                                        spLimit=60):
    """
    Исправленная версия predictor_mem_non_hdf_load_model:
    - правильный заголовок CSV (v1 писала Ellipsis)
    - временны́е метки считаются по 100 Гц (v1 брала sampling_rate из attrs,
      что неверно когда данные интерполированы до 6000 отсчётов в preprocessorV6_mem)
    - первый детектированный ивент теперь тоже строит график (plt_n > 0 → убрано)
    """
    if gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)

    os.makedirs(output_dir, exist_ok=True)
    save_dir = output_dir
    save_figs = os.path.join(save_dir, 'figures')
    if number_of_plots > 0:
        os.makedirs(save_figs, exist_ok=True)

    csv_file = os.path.join(save_dir, 'X_prediction_results.csv')
    csvPr_gen = open(csv_file, 'w', newline='')
    predict_writer = csv.writer(csvPr_gen)
    predict_writer.writerow([
        'file_name', 'network', 'station', 'instrument_type',
        'station_lat', 'station_lon', 'station_elv',
        'event_start_time', 'event_end_time',
        'detection_probability', 'detection_uncertainty',
        'p_arrival_time', 'p_probability', 'p_uncertainty', 'p_snr',
        's_arrival_time', 's_probability', 's_uncertainty', 's_snr'
    ])
    csvPr_gen.flush()

    if model is None:
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

    args = {
        'input_hdf5': 'in-memory',
        'input_dimention': (6000, 3),
        'normalization_mode': 'std',
        'use_multiprocessing': False,
        'number_of_cpus': 1,
        'estimate_uncertainty': estimate_uncertainty,
        'number_of_sampling': number_of_sampling,
        'output_probabilities': False,
        'number_of_plots': number_of_plots,
        'detection_threshold': detection_threshold,
        'P_threshold': P_threshold,
        'S_threshold': S_threshold,
        'plot_mode': plot_mode
    }

    detection_memory = []
    plt_n = 0

    for row in tqdm(csv_segment, desc="Predicting"):
        ID = row[0]
        if ID not in hdf_segment:
            print(f"⚠ Пропускаем {ID}, ключ не найден в HDF5")
            continue

        class InMemoryDataset:
            def __init__(self, data_dict):
                self.data = data_dict['data']
                self.attrs = data_dict['attrs']

        dataset_obj = InMemoryDataset(hdf_segment[ID])
        prob_dic = _gen_predictor_mem([ID], args, model, {ID: dataset_obj.data})

        plt_n, detection_memory = _gen_writer_mem_non_hdf_v9(
            [ID],
            args,
            prob_dic,
            {ID: dataset_obj},
            predict_writer=predict_writer,
            save_figs=save_figs,
            csvPr_gen=csvPr_gen,
            plt_n=plt_n,
            detection_memory=detection_memory,
            keepPS=keepPS,
            allowonlyS=allowonlyS,
            spLimit=spLimit
        )

    csvPr_gen.close()
    return csv_file


def predictor_mem_non_hdf_load_model_v3(csv_segment, hdf_segment,
                                        model_path=None,
                                        model=None,
                                        output_dir=None,
                                        detection_threshold=0.3,
                                        P_threshold=0.1,
                                        S_threshold=0.1,
                                        number_of_plots=10,
                                        plot_mode='time',
                                        estimate_uncertainty=False,
                                        number_of_sampling=5,
                                        batch_size=32,
                                        gpuid=None,
                                        keepPS=True,
                                        allowonlyS=True,
                                        spLimit=60):
    """
    Батчевая версия v2: все окна сегмента идут в model.predict одним вызовом.

    v2 вызывала model.predict() N раз (по одному на каждое 60-секундное окно).
    Каждый вызов TensorFlow имеет фиксированные накладные расходы ~0.3 сек,
    что даёт 13 вызовов × 0.35 сек = ~4.5 сек на сегмент.
    v3 собирает все N окон в батч (N, 6000, 3) и вызывает model.predict один раз
    → ~10× ускорение на CPU.

    batch_size: размер мини-батча внутри model.predict (не путать с числом окон).
                32 — разумный дефолт; увеличьте до 64-128 если хватает RAM.
    """
    if gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)

    os.makedirs(output_dir, exist_ok=True)
    save_dir = output_dir
    save_figs = os.path.join(save_dir, 'figures')
    if number_of_plots > 0:
        os.makedirs(save_figs, exist_ok=True)

    csv_file = os.path.join(save_dir, 'X_prediction_results.csv')
    csvPr_gen = open(csv_file, 'w', newline='')
    predict_writer = csv.writer(csvPr_gen)
    predict_writer.writerow([
        'file_name', 'network', 'station', 'instrument_type',
        'station_lat', 'station_lon', 'station_elv',
        'event_start_time', 'event_end_time',
        'detection_probability', 'detection_uncertainty',
        'p_arrival_time', 'p_probability', 'p_uncertainty', 'p_snr',
        's_arrival_time', 's_probability', 's_uncertainty', 's_snr'
    ])
    csvPr_gen.flush()

    if model is None:
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

    args = {
        'input_hdf5': 'in-memory',
        'input_dimention': (6000, 3),
        'normalization_mode': 'std',
        'use_multiprocessing': False,
        'number_of_cpus': 1,
        'estimate_uncertainty': estimate_uncertainty,
        'number_of_sampling': number_of_sampling,
        'output_probabilities': False,
        'number_of_plots': number_of_plots,
        'detection_threshold': detection_threshold,
        'P_threshold': P_threshold,
        'S_threshold': S_threshold,
        'plot_mode': plot_mode
    }

    # --- Собираем все окна сегмента ---
    id_list = [row[0] for row in csv_segment if row[0] in hdf_segment]
    if not id_list:
        csvPr_gen.close()
        return csv_file

    X = np.stack([hdf_segment[ID]['data'] for ID in id_list], axis=0)  # (N, 6000, 3)

    # Нормализация per-window, per-channel (стандарт EQT)
    mean = X.mean(axis=1, keepdims=True)   # (N, 1, 3)
    std  = X.std(axis=1, keepdims=True)    # (N, 1, 3)
    X_norm = (X - mean) / (std + 1e-6)

    # --- Один вызов model.predict на весь сегмент ---
    if estimate_uncertainty:
        n_samples = number_of_sampling
        preds_D, preds_P, preds_S = [], [], []
        for _ in range(n_samples):
            pD, pP, pS = model.predict(X_norm, batch_size=batch_size, verbose=0)
            preds_D.append(pD)
            preds_P.append(pP)
            preds_S.append(pS)
        DD_mean = np.mean(preds_D, axis=0).squeeze(-1)   # (N, 6000)
        PP_mean = np.mean(preds_P, axis=0).squeeze(-1)
        SS_mean = np.mean(preds_S, axis=0).squeeze(-1)
        DD_std  = np.std(preds_D, axis=0).squeeze(-1)
        PP_std  = np.std(preds_P, axis=0).squeeze(-1)
        SS_std  = np.std(preds_S, axis=0).squeeze(-1)
    else:
        pD, pP, pS = model.predict(X_norm, batch_size=batch_size, verbose=0)
        DD_mean = pD.squeeze(-1)   # (N, 6000)
        PP_mean = pP.squeeze(-1)
        SS_mean = pS.squeeze(-1)
        DD_std  = np.zeros_like(DD_mean)
        PP_std  = np.zeros_like(PP_mean)
        SS_std  = np.zeros_like(SS_mean)

    # --- Обрабатываем результаты по каждому окну ---
    detection_memory = []
    plt_n = 0

    class InMemoryDataset:
        def __init__(self, data_dict):
            self.data = data_dict['data']
            self.attrs = data_dict['attrs']

    for i, ID in enumerate(tqdm(id_list, desc="Writing results")):
        dataset_obj = InMemoryDataset(hdf_segment[ID])
        prob_dic = {
            'DD_mean': DD_mean[i:i+1],   # (1, 6000) — v9 ожидает shape[0] итераций
            'PP_mean': PP_mean[i:i+1],
            'SS_mean': SS_mean[i:i+1],
            'DD_std':  DD_std[i:i+1],
            'PP_std':  PP_std[i:i+1],
            'SS_std':  SS_std[i:i+1],
        }
        plt_n, detection_memory = _gen_writer_mem_non_hdf_v9(
            [ID], args, prob_dic, {ID: dataset_obj},
            predict_writer=predict_writer,
            save_figs=save_figs,
            csvPr_gen=csvPr_gen,
            plt_n=plt_n,
            detection_memory=detection_memory,
            keepPS=keepPS,
            allowonlyS=allowonlyS,
            spLimit=spLimit
        )

    csvPr_gen.close()
    return csv_file


def _gen_writer_mem_non_hdf_v10(new_list, args, prob_dic, pred_set,
                                 save_figs, plt_n, detection_memory,
                                 keepPS, allowonlyS, spLimit):
    """
    v9 без записи в CSV: возвращает список строк событий вместо writerow().
    Используется в predictor_mem_non_hdf_load_model_v4 для сбора всех событий
    в памяти с последующей записью в общий файл.
    """
    SAMPLING_RATE = 100.0

    rows = []

    for ts in range(prob_dic['DD_mean'].shape[0]):
        evi = new_list[ts]
        dataset = pred_set[evi]
        dat = np.array(dataset.data)

        if dat.ndim == 0:
            dat = np.zeros((6000, 3), dtype=np.float32)
        elif dat.ndim == 1:
            dat = dat[:, np.newaxis]
        if dat.shape[1] < 3:
            tmp = np.zeros((dat.shape[0], 3), dtype=dat.dtype)
            tmp[:, :dat.shape[1]] = dat
            dat = tmp

        matches, pick_errors, yh3 = picker(
            args,
            prob_dic['DD_mean'][ts],
            prob_dic['PP_mean'][ts],
            prob_dic['SS_mean'][ts],
            prob_dic['DD_std'][ts],
            prob_dic['PP_std'][ts],
            prob_dic['SS_std'][ts]
        )

        if len(matches) == 0:
            continue

        match_idx = list(matches)[0]
        match_value = matches[match_idx]

        if not allowonlyS and match_value[6] is not None and match_value[3] is None:
            continue

        valid_event = False
        if keepPS:
            if match_value[3] is not None and match_value[6] is not None:
                if (match_value[6] - match_value[3]) < spLimit * SAMPLING_RATE:
                    valid_event = True
        else:
            if match_value[3] is not None or match_value[6] is not None:
                valid_event = True
        if not valid_event:
            continue

        def safe_snr(data, idx):
            if idx is None or idx < 0 or idx >= len(data):
                return np.nan
            return _get_snr(data, idx, window=min(100, len(data)))

        snr_p = safe_snr(dat, match_value[3])
        snr_s = safe_snr(dat, match_value[6])

        start_time_str = dataset.attrs["trace_start_time"]
        try:
            start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')

        ev_strt = start_time + timedelta(seconds=match_idx / SAMPLING_RATE)
        ev_end  = start_time + timedelta(seconds=match_value[0] / SAMPLING_RATE)

        doublet = [st for st in detection_memory if abs((st - ev_strt).total_seconds()) < 2]
        if doublet:
            continue

        det_prob = round(match_value[1], 2)
        det_unc  = round(match_value[2], 2) if match_value[2] is not None else np.nan

        p_time = start_time + timedelta(seconds=match_value[3] / SAMPLING_RATE) if match_value[3] is not None else None
        p_prob = round(match_value[4], 2) if match_value[4] is not None else np.nan
        p_unc  = round(match_value[5], 2) if match_value[5] is not None else np.nan

        s_time = start_time + timedelta(seconds=match_value[6] / SAMPLING_RATE) if match_value[6] is not None else None
        s_prob = round(match_value[7], 2) if match_value[7] is not None else np.nan
        s_unc  = round(match_value[8], 2) if match_value[8] is not None else np.nan

        trace_name = dataset.attrs["trace_name"]
        station_name = "{:<4}".format(dataset.attrs["receiver_code"])
        network_name = "{:<2}".format(dataset.attrs["network_code"])
        instrument_type = "{:<2}".format(trace_name.split('_')[2])
        station_lat = dataset.attrs["receiver_latitude"]
        station_lon = dataset.attrs["receiver_longitude"]
        station_elv = dataset.attrs["receiver_elevation_m"]

        rows.append([
            trace_name, network_name, station_name, instrument_type,
            station_lat, station_lon, station_elv,
            ev_strt, ev_end,
            det_prob, det_unc,
            p_time, p_prob, p_unc, snr_p,
            s_time, s_prob, s_unc, snr_s
        ])
        detection_memory.append(ev_strt)

        if save_figs is not None and plt_n < args['number_of_plots'] and dat.size > 0:
            _plotter_prediction(
                dat, evi, args, save_figs,
                prob_dic['DD_mean'][ts],
                prob_dic['PP_mean'][ts],
                prob_dic['SS_mean'][ts],
                prob_dic['DD_std'][ts],
                prob_dic['PP_std'][ts],
                prob_dic['SS_std'][ts],
                matches
            )
            plt_n += 1

    return plt_n, detection_memory, rows


def predictor_mem_non_hdf_load_model_v4(csv_segment, hdf_segment,
                                         model_path=None,
                                         model=None,
                                         save_figs=None,
                                         detection_threshold=0.3,
                                         P_threshold=0.1,
                                         S_threshold=0.1,
                                         number_of_plots=10,
                                         plot_mode='time',
                                         estimate_uncertainty=False,
                                         number_of_sampling=5,
                                         batch_size=32,
                                         gpuid=None,
                                         keepPS=True,
                                         allowonlyS=True,
                                         spLimit=60):
    """
    v3 без записи в отдельный CSV-файл и без создания директорий.

    Вместо записи в файл возвращает список строк событий (List[List]),
    которые caller (cutter) накапливает и пишет в один общий CSV.

    save_figs: путь к каталогу для графиков (создаётся автоматически).
               None — графики не строятся и каталог не создаётся.
    Возвращает: list[list] — строки событий (без заголовка).
    """
    if gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except (RuntimeError, ValueError):
                pass  # уже инициализирован или virtual device сконфигурирован

    if save_figs is not None and number_of_plots > 0:
        os.makedirs(save_figs, exist_ok=True)

    if model is None:
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

    args = {
        'input_hdf5': 'in-memory',
        'input_dimention': (6000, 3),
        'normalization_mode': 'std',
        'use_multiprocessing': False,
        'number_of_cpus': 1,
        'estimate_uncertainty': estimate_uncertainty,
        'number_of_sampling': number_of_sampling,
        'output_probabilities': False,
        'number_of_plots': number_of_plots,
        'detection_threshold': detection_threshold,
        'P_threshold': P_threshold,
        'S_threshold': S_threshold,
        'plot_mode': plot_mode
    }

    id_list = [row[0] for row in csv_segment if row[0] in hdf_segment]
    if not id_list:
        return []

    X = np.stack([hdf_segment[ID]['data'] for ID in id_list], axis=0)

    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True)
    X_norm = (X - mean) / (std + 1e-6)

    if estimate_uncertainty:
        n_samples = number_of_sampling
        preds_D, preds_P, preds_S = [], [], []
        for _ in range(n_samples):
            pD, pP, pS = model.predict(X_norm, batch_size=batch_size, verbose=0)
            preds_D.append(pD)
            preds_P.append(pP)
            preds_S.append(pS)
        DD_mean = np.mean(preds_D, axis=0).squeeze(-1)
        PP_mean = np.mean(preds_P, axis=0).squeeze(-1)
        SS_mean = np.mean(preds_S, axis=0).squeeze(-1)
        DD_std  = np.std(preds_D, axis=0).squeeze(-1)
        PP_std  = np.std(preds_P, axis=0).squeeze(-1)
        SS_std  = np.std(preds_S, axis=0).squeeze(-1)
    else:
        pD, pP, pS = model.predict(X_norm, batch_size=batch_size, verbose=0)
        DD_mean = pD.squeeze(-1)
        PP_mean = pP.squeeze(-1)
        SS_mean = pS.squeeze(-1)
        DD_std  = np.zeros_like(DD_mean)
        PP_std  = np.zeros_like(PP_mean)
        SS_std  = np.zeros_like(SS_mean)

    detection_memory = []
    plt_n = 0
    all_rows = []

    class InMemoryDataset:
        def __init__(self, data_dict):
            self.data = data_dict['data']
            self.attrs = data_dict['attrs']

    for i, ID in enumerate(tqdm(id_list, desc="Writing results")):
        dataset_obj = InMemoryDataset(hdf_segment[ID])
        prob_dic = {
            'DD_mean': DD_mean[i:i+1],
            'PP_mean': PP_mean[i:i+1],
            'SS_mean': SS_mean[i:i+1],
            'DD_std':  DD_std[i:i+1],
            'PP_std':  PP_std[i:i+1],
            'SS_std':  SS_std[i:i+1],
        }
        plt_n, detection_memory, rows = _gen_writer_mem_non_hdf_v10(
            [ID], args, prob_dic, {ID: dataset_obj},
            save_figs=save_figs,
            plt_n=plt_n,
            detection_memory=detection_memory,
            keepPS=keepPS,
            allowonlyS=allowonlyS,
            spLimit=spLimit
        )
        all_rows.extend(rows)

    # Явное освобождение больших numpy массивов до возврата из функции.
    # Без del они живут до следующего вызова GC, накапливаясь в RAM.
    del X, X_norm, DD_mean, PP_mean, SS_mean, DD_std, PP_std, SS_std
    if estimate_uncertainty:
        del preds_D, preds_P, preds_S

    return all_rows


def predictor_mem_non_hdf_load_model_v5(csv_segment, hdf_segment,
                                         model_path=None,
                                         model=None,
                                         save_figs=None,
                                         detection_threshold=0.3,
                                         P_threshold=0.1,
                                         S_threshold=0.1,
                                         number_of_plots=10,
                                         plot_mode='time',
                                         estimate_uncertainty=False,
                                         number_of_sampling=5,
                                         batch_size=32,
                                         gpuid=None,
                                         keepPS=True,
                                         allowonlyS=True,
                                         spLimit=60):
    """
    v5: исправлена нормализация относительно v4.

    v4 применял z-score (вычитал среднее + делил на std), что не соответствует
    тому как модель была обучена. Оригинальный EQT нормализует только по std:
        std_data = np.std(data, axis=0); data /= std_data
    v5 воспроизводит это точно: X_norm = X / std, среднее НЕ вычитается.
    """
    if gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except (RuntimeError, ValueError):
                pass

    if save_figs is not None and number_of_plots > 0:
        os.makedirs(save_figs, exist_ok=True)

    if model is None:
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

    args = {
        'input_hdf5': 'in-memory',
        'input_dimention': (6000, 3),
        'normalization_mode': 'std',
        'use_multiprocessing': False,
        'number_of_cpus': 1,
        'estimate_uncertainty': estimate_uncertainty,
        'number_of_sampling': number_of_sampling,
        'output_probabilities': False,
        'number_of_plots': number_of_plots,
        'detection_threshold': detection_threshold,
        'P_threshold': P_threshold,
        'S_threshold': S_threshold,
        'plot_mode': plot_mode
    }

    id_list = [row[0] for row in csv_segment if row[0] in hdf_segment]
    if not id_list:
        return []

    X = np.stack([hdf_segment[ID]['data'] for ID in id_list], axis=0)

    # Нормализация по std без вычитания среднего — как в оригинальном EQT
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1
    X_norm = X / std

    if estimate_uncertainty:
        n_samples = number_of_sampling
        preds_D, preds_P, preds_S = [], [], []
        for _ in range(n_samples):
            pD, pP, pS = model.predict(X_norm, batch_size=batch_size, verbose=0)
            preds_D.append(pD)
            preds_P.append(pP)
            preds_S.append(pS)
        DD_mean = np.mean(preds_D, axis=0).squeeze(-1)
        PP_mean = np.mean(preds_P, axis=0).squeeze(-1)
        SS_mean = np.mean(preds_S, axis=0).squeeze(-1)
        DD_std  = np.std(preds_D, axis=0).squeeze(-1)
        PP_std  = np.std(preds_P, axis=0).squeeze(-1)
        SS_std  = np.std(preds_S, axis=0).squeeze(-1)
    else:
        pD, pP, pS = model.predict(X_norm, batch_size=batch_size, verbose=0)
        DD_mean = pD.squeeze(-1)
        PP_mean = pP.squeeze(-1)
        SS_mean = pS.squeeze(-1)
        DD_std  = np.zeros_like(DD_mean)
        PP_std  = np.zeros_like(PP_mean)
        SS_std  = np.zeros_like(SS_mean)

    detection_memory = []
    plt_n = 0
    all_rows = []

    class InMemoryDataset:
        def __init__(self, data_dict):
            self.data = data_dict['data']
            self.attrs = data_dict['attrs']

    for i, ID in enumerate(tqdm(id_list, desc="Writing results")):
        dataset_obj = InMemoryDataset(hdf_segment[ID])
        prob_dic = {
            'DD_mean': DD_mean[i:i+1],
            'PP_mean': PP_mean[i:i+1],
            'SS_mean': SS_mean[i:i+1],
            'DD_std':  DD_std[i:i+1],
            'PP_std':  PP_std[i:i+1],
            'SS_std':  SS_std[i:i+1],
        }
        plt_n, detection_memory, rows = _gen_writer_mem_non_hdf_v10(
            [ID], args, prob_dic, {ID: dataset_obj},
            save_figs=save_figs,
            plt_n=plt_n,
            detection_memory=detection_memory,
            keepPS=keepPS,
            allowonlyS=allowonlyS,
            spLimit=spLimit
        )
        all_rows.extend(rows)

    del X, X_norm, DD_mean, PP_mean, SS_mean, DD_std, PP_std, SS_std
    if estimate_uncertainty:
        del preds_D, preds_P, preds_S

    return all_rows


@tf.function
def _mc_forward(model, batch):
    return model(batch, training=True)


def predictor_mem_non_hdf_load_model_v6(csv_segment, hdf_segment,
                                         model_path=None,
                                         model=None,
                                         save_figs=None,
                                         detection_threshold=0.3,
                                         P_threshold=0.1,
                                         S_threshold=0.1,
                                         number_of_plots=10,
                                         plot_mode='time',
                                         estimate_uncertainty=False,
                                         number_of_sampling=10,
                                         batch_size=32,
                                         gpuid=None,
                                         keepPS=True,
                                         allowonlyS=True,
                                         spLimit=60):
    """
    v6: реальный MC Dropout при estimate_uncertainty=True.

    v4/v5 использовали model.predict() в цикле — это training=False, dropout
    выключен, предсказания детерминированы, std всегда ~0.

    v6 при estimate_uncertainty=True использует model(X_batch, training=True):
    dropout активен на каждом forward pass → разные маски → ненулевой std.
    number_of_sampling проходов дают распределение вероятностей P и S,
    из которого вычисляются mean и std.

    При estimate_uncertainty=False поведение идентично v5 (model.predict()).
    Нормализация: X / std без вычитания среднего (как в оригинальном EQT).
    """
    if gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except (RuntimeError, ValueError):
                pass

    if save_figs is not None and number_of_plots > 0:
        os.makedirs(save_figs, exist_ok=True)

    if model is None:
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

    args = {
        'input_hdf5': 'in-memory',
        'input_dimention': (6000, 3),
        'normalization_mode': 'std',
        'use_multiprocessing': False,
        'number_of_cpus': 1,
        'estimate_uncertainty': estimate_uncertainty,
        'number_of_sampling': number_of_sampling,
        'output_probabilities': False,
        'number_of_plots': number_of_plots,
        'detection_threshold': detection_threshold,
        'P_threshold': P_threshold,
        'S_threshold': S_threshold,
        'plot_mode': plot_mode
    }

    id_list = [row[0] for row in csv_segment if row[0] in hdf_segment]
    if not id_list:
        return []

    X = np.stack([hdf_segment[ID]['data'] for ID in id_list], axis=0)  # (N, 6000, 3)

    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1
    X_norm = X / std

    if estimate_uncertainty:
        # MC Dropout батчинг: n_samples копий X складываются в один батч.
        # Dropout генерирует независимые маски для каждого сэмпла в батче,
        # поэтому n_samples копий одного окна дают n_samples разных выходов —
        # валидный MC Dropout за один forward pass вместо n_samples вызовов.
        N = len(X_norm)
        X_tiled = np.tile(X_norm, (number_of_sampling, 1, 1))  # (n*N, 6000, 3)
        X_tf = tf.constant(X_tiled, dtype=tf.float32)

        batch_D, batch_P, batch_S = [], [], []
        for i in range(0, len(X_tf), batch_size):
            out = _mc_forward(model, X_tf[i:i + batch_size])
            batch_D.append(out[0].numpy())
            batch_P.append(out[1].numpy())
            batch_S.append(out[2].numpy())

        all_D = np.concatenate(batch_D, axis=0).squeeze(-1)  # (n*N, 6000)
        all_P = np.concatenate(batch_P, axis=0).squeeze(-1)
        all_S = np.concatenate(batch_S, axis=0).squeeze(-1)

        all_D = all_D.reshape(number_of_sampling, N, 6000)
        all_P = all_P.reshape(number_of_sampling, N, 6000)
        all_S = all_S.reshape(number_of_sampling, N, 6000)

        DD_mean = all_D.mean(axis=0)   # (N, 6000)
        PP_mean = all_P.mean(axis=0)
        SS_mean = all_S.mean(axis=0)
        DD_std  = all_D.std(axis=0)
        PP_std  = all_P.std(axis=0)
        SS_std  = all_S.std(axis=0)
    else:
        pD, pP, pS = model.predict(X_norm, batch_size=batch_size, verbose=0)
        DD_mean = pD.squeeze(-1)
        PP_mean = pP.squeeze(-1)
        SS_mean = pS.squeeze(-1)
        DD_std  = np.zeros_like(DD_mean)
        PP_std  = np.zeros_like(PP_mean)
        SS_std  = np.zeros_like(SS_mean)

    detection_memory = []
    plt_n = 0
    all_rows = []

    class InMemoryDataset:
        def __init__(self, data_dict):
            self.data = data_dict['data']
            self.attrs = data_dict['attrs']

    for i, ID in enumerate(tqdm(id_list, desc="Writing results")):
        dataset_obj = InMemoryDataset(hdf_segment[ID])
        prob_dic = {
            'DD_mean': DD_mean[i:i+1],
            'PP_mean': PP_mean[i:i+1],
            'SS_mean': SS_mean[i:i+1],
            'DD_std':  DD_std[i:i+1],
            'PP_std':  PP_std[i:i+1],
            'SS_std':  SS_std[i:i+1],
        }
        plt_n, detection_memory, rows = _gen_writer_mem_non_hdf_v10(
            [ID], args, prob_dic, {ID: dataset_obj},
            save_figs=save_figs,
            plt_n=plt_n,
            detection_memory=detection_memory,
            keepPS=keepPS,
            allowonlyS=allowonlyS,
            spLimit=spLimit
        )
        all_rows.extend(rows)

    del X, X_norm, DD_mean, PP_mean, SS_mean, DD_std, PP_std, SS_std
    if estimate_uncertainty:
        del X_tiled, X_tf, all_D, all_P, all_S

    return all_rows


def predictor_mem_v7(csv_segment, hdf_segment,
                     model_path,
                     output_dir=None,
                     detection_threshold=0.3,                
                     P_threshold=0.1,
                     S_threshold=0.1, 
                     number_of_plots=10,
                     plot_mode='time',
                     estimate_uncertainty=False, 
                     number_of_sampling=5,
                     batch_size=500,
                     gpuid=None,
                     output_probabilities=False,
                     keepPS=True,
                     allowonlyS=True,
                     spLimit=60):

    # --- GPU настройка ---
    if gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)

    # --- Создание выходных папок ---
    os.makedirs(output_dir, exist_ok=True)
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)
    save_figs = os.path.join(save_dir, 'figures')
    if number_of_plots > 0:
        os.makedirs(save_figs, exist_ok=True)

    # --- Файл для сохранения вероятностей ---
    out_probs = os.path.join(save_dir, 'prediction_probabilities.hdf5')
    if output_probabilities:
        HDF_PROB = h5py.File(out_probs, 'a')
        if "probabilities" not in HDF_PROB:
            HDF_PROB.create_group("probabilities")
        if "uncertainties" not in HDF_PROB:
            HDF_PROB.create_group("uncertainties")
    else:
        HDF_PROB = None

    # --- CSV-файл результатов ---
    csv_file = os.path.join(save_dir, 'X_prediction_results.csv')
    csvPr_gen = open(csv_file, 'w', newline='')
    predict_writer = csv.writer(csvPr_gen)
    predict_writer.writerow(['file_name', 'network', 'station', 'instrument_type',
                             'station_lat', 'station_lon', 'station_elv',
                             'event_start_time', 'event_end_time',
                             'detection_probability', 'detection_uncertainty', 
                             'p_arrival_time','p_probability','p_uncertainty','p_snr',
                             's_arrival_time','s_probability','s_uncertainty','s_snr'])
    csvPr_gen.flush()

    # --- Загружаем модель ---
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

    # --- Минимальный args-словарь ---
    args = {
        'input_hdf5': 'in-memory',
        'input_dimention': (6000, 3),  
        'normalization_mode': 'std',
        'use_multiprocessing': False,
        'number_of_cpus': 1,
        'estimate_uncertainty': estimate_uncertainty,
        'number_of_sampling': number_of_sampling,
        'output_probabilities': output_probabilities,
        'number_of_plots': number_of_plots,
        'detection_threshold': detection_threshold,
        'P_threshold': P_threshold,
        'S_threshold': S_threshold,
        'plot_mode': plot_mode
    }

    detection_memory = []
    plt_n = 0

    # --- Основной цикл по сегментам ---
    for row in tqdm(csv_segment, desc="Predicting"):
        ID = row[0]  # trace_name
        if ID not in hdf_segment:
            print(f"⚠ Пропускаем {ID}, ключ не найден в HDF5")
            continue

        # Используем объект с атрибутами
        class InMemoryDataset:
            def __init__(self, data_dict):
                self.data = data_dict['data']
                self.attrs = data_dict['attrs']

        dataset_obj = InMemoryDataset(hdf_segment[ID])

        # Генерация предсказаний
        prob_dic = _gen_predictor_mem([ID], args, model, {ID: dataset_obj.data})

        # Запись результатов, построение графиков
        plt_n, detection_memory = _gen_writer_mem(
            [ID],
            args,
            prob_dic,
            {ID: dataset_obj},
            HDF_PROB,
            predict_writer,
            save_figs,
            csvPr_gen,
            plt_n,
            detection_memory,
            keepPS,
            allowonlyS,
            spLimit
        )

    if HDF_PROB:
        HDF_PROB.close()
    csvPr_gen.close()
    return csv_file

def _gen_predictor_mem(new_list, args, model, data_dict):
    """
    Предсказания для батча, одноканальный сигнал, данные из памяти.
    """

    prob_dic = {}

    # --- Формирование X ---
    X = np.array([data_dict[ID] for ID in new_list])  # (N, T)
    if X.ndim == 2:
        X = np.expand_dims(X, -1)  # (N, T, 1) для модели

    # --- Нормализация ---
    norm_mode = args.get('normalization_mode', 'std')
    if norm_mode == 'std':
        X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-6)
    elif norm_mode == 'max':
        X = X / (np.max(np.abs(X), axis=1, keepdims=True) + 1e-6)

    # --- Предсказания ---
    if args.get('estimate_uncertainty', False):
        n_samples = args.get('number_of_sampling', 5)
        pred_DD, pred_PP, pred_SS = [], [], []
        for _ in range(n_samples):
            pD, pP, pS = model.predict(X, batch_size=args.get('batch_size', 500), verbose=0)
            pred_DD.append(pD)
            pred_PP.append(pP)
            pred_SS.append(pS)

        pred_DD = np.array(pred_DD).squeeze(-1)  # (samples, N, T, 1) -> (samples, N, T)
        pred_PP = np.array(pred_PP).squeeze(-1)
        pred_SS = np.array(pred_SS).squeeze(-1)

        prob_dic['DD_mean'] = pred_DD.mean(axis=0)
        prob_dic['PP_mean'] = pred_PP.mean(axis=0)
        prob_dic['SS_mean'] = pred_SS.mean(axis=0)

        prob_dic['DD_std'] = pred_DD.std(axis=0)
        prob_dic['PP_std'] = pred_PP.std(axis=0)
        prob_dic['SS_std'] = pred_SS.std(axis=0)
    else:
        pred_DD_mean, pred_PP_mean, pred_SS_mean = model.predict(X, batch_size=args.get('batch_size', 500), verbose=0)
        prob_dic['DD_mean'] = np.squeeze(pred_DD_mean, axis=-1)  # (N, T, 1) -> (N, T)
        prob_dic['PP_mean'] = np.squeeze(pred_PP_mean, axis=-1)
        prob_dic['SS_mean'] = np.squeeze(pred_SS_mean, axis=-1)
        prob_dic['DD_std'] = np.zeros_like(prob_dic['DD_mean'])
        prob_dic['PP_std'] = np.zeros_like(prob_dic['PP_mean'])
        prob_dic['SS_std'] = np.zeros_like(prob_dic['SS_mean'])

    return prob_dic

def predictor_mem(csv_segment, hdf_segment, 
                  model_path,
                  output_dir=None,
                  detection_threshold=0.3,                
                  P_threshold=0.1,
                  S_threshold=0.1, 
                  number_of_plots=10,
                  plot_mode='time',
                  estimate_uncertainty=False, 
                  number_of_sampling=5,
                  batch_size=500,
                  gpuid=None,
                  output_probabilities=False,
                  keepPS=True,
                  allowonlyS=True,
                  spLimit=60):
    """
    Predictor working directly with preprocessed CSV and HDF5 data segments.
    """

    # --- Настройка GPU ---
    if gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)

    # --- Создание выходной папки ---
    os.makedirs(output_dir, exist_ok=True)
    save_dir = os.path.join(output_dir, 'segment_outputs')
    os.makedirs(save_dir, exist_ok=True)
    save_figs = os.path.join(save_dir, 'figures')
    if number_of_plots > 0:
        os.makedirs(save_figs, exist_ok=True)

    out_probs = os.path.join(save_dir, 'prediction_probabilities.hdf5')
    if output_probabilities:
        HDF_PROB = h5py.File(out_probs, 'a')
        HDF_PROB.create_group("probabilities")
        HDF_PROB.create_group("uncertainties")
    else:
        HDF_PROB = None

    csv_file = os.path.join(save_dir, 'X_prediction_results.csv')
    csvPr_gen = open(csv_file, 'w', newline='')
    predict_writer = csv.writer(csvPr_gen, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    predict_writer.writerow(['file_name', 'network', 'station', 'instrument_type',
                             'station_lat', 'station_lon', 'station_elv',
                             'event_start_time', 'event_end_time',
                             'detection_probability', 'detection_uncertainty', 
                             'p_arrival_time','p_probability','p_uncertainty','p_snr',
                             's_arrival_time','s_probability','s_uncertainty','s_snr'])
    csvPr_gen.flush()

    # --- Загружаем модель ---
    model = load_model(model_path, compile=False)

    # --- Предсказания по сегменту ---
    prediction_list = [row[0] for row in csv_segment]  # предполагаем, что первый столбец trace_name
    detection_memory = []
    plt_n = 0

    for ID in tqdm(prediction_list, desc="Predicting"):
        # Получаем данные для предсказания
        dataset = hdf_segment[ID]['data']
        prob_dic = _gen_predictor([ID], model=model, data_dict={ID: dataset}, detection_threshold=detection_threshold,
                                  P_threshold=P_threshold, S_threshold=S_threshold)

        # Запись результатов
        plt_n, detection_memory = _gen_writer([ID], None, prob_dic, {ID: dataset}, HDF_PROB, predict_writer,
                                              save_figs, csvPr_gen, plt_n, detection_memory,
                                              keepPS, allowonlyS, spLimit)

    if HDF_PROB:
        HDF_PROB.close()
    csvPr_gen.close()

    return csv_file

def predictor(input_dir=None,
              input_model=None,
              output_dir=None,
              output_probabilities=False,
              detection_threshold=0.3,                
              P_threshold=0.1,
              S_threshold=0.1, 
              number_of_plots=10,
              plot_mode='time',
              estimate_uncertainty=False, 
              number_of_sampling=5,
              loss_weights=[0.03, 0.40, 0.58],
              loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
              input_dimention=(6000, 3),
              normalization_mode='std',
              batch_size=500,
              gpuid=None,
              gpu_limit=None,
              number_of_cpus=5,
              use_multiprocessing=True,
              keepPS=True,
              allowonlyS=True,
              spLimit=60): 
    
    
    """
    
    Applies a trained model to a windowed waveform to perform both detection and picking at the same time. 


    Parameters
    ----------
    input_dir: str, default=None
        Directory name containing hdf5 and csv files-preprocessed data.
        
    input_model: str, default=None
        Path to a trained model.

    output_dir: str, default=None
        Output directory that will be generated. 
        
    output_probabilities: bool, default=False
        If True, it will output probabilities and estimated uncertainties for each trace into an HDF file.       
         
    detection_threshold : float, default=0.3
        A value in which the detection probabilities above it will be considered as an event.
          
    P_threshold: float, default=0.1
        A value which the P probabilities above it will be considered as P arrival.

    S_threshold: float, default=0.1
        A value which the S probabilities above it will be considered as S arrival.
               
    number_of_plots: float, default=10
        The number of plots for detected events outputed for each station data.

    plot_mode: str, default='time'
        The type of plots: 'time': only time series or 'time_frequency', time and spectrograms.
          
    estimate_uncertainty: bool, default=False
        If True uncertainties in the output probabilities will be estimated.           

    number_of_sampling: int, default=5
        Number of sampling for the uncertainty estimation. 
               
    loss_weights: list, default=[0.03, 0.40, 0.58]
        Loss weights for detection, P picking, and S picking respectively.
             
    loss_types: list, default=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'] 
        Loss types for detection, P picking, and S picking respectively.

    input_dimention: tuple, default=(6000, 3)
        Loss types for detection, P picking, and S picking respectively.      

    normalization_mode: str, default='std' 
        Mode of normalization for data preprocessing, 'max', maximum amplitude among three components, 'std', standard deviation.
           
    batch_size: int, default=500 
        Batch size. This wont affect the speed much but can affect the performance. A value beteen 200 to 1000 is recommanded.

    gpuid: int, default=None
        Id of GPU used for the prediction. If using CPU set to None.
         
    gpu_limit: int, default=None
        Set the maximum percentage of memory usage for the GPU.
          
    number_of_cpus: int, default=5
        Number of CPUs used for the parallel preprocessing and feeding of data for prediction.

    use_multiprocessing: bool, default=True
        If True, multiple CPUs will be used for the preprocessing of data even when GPU is used for the prediction.        

    keepPS: bool, default=False
        If True, detected events require both P and S picks to be written. If False, individual P or S (see allowonlyS) picks may be written.
        
    allowonlyS: bool, default=True
        If True, detected events with "only S" picks will be allowed. If False, an associated P pick is required.         
        
    spLimit: int, default=60
        S - P time in seconds. It will limit the results to those detections with events that have a specific S-P time limit. 
        
    Returns
    -------- 
    ./output_dir/STATION_OUTPUT/X_prediction_results.csv: A table containing all the detection, and picking results. Duplicated events are already removed.      
    
    ./output_dir/STATION_OUTPUT/X_report.txt: A summary of the parameters used for prediction and performance.
    
    ./output_dir/STATION_OUTPUT/figures: A folder containing plots detected events and picked arrival times. 
    
    ./time_tracks.pkl: A file containing the time track of the continous data and its type.
    

    Notes
    --------
    Estimating the uncertainties requires multiple predictions and will increase the computational time. 
    
        
    """ 
   
    
    args = {
    "input_dir": input_dir,
    "input_hdf5": None,
    "input_csv": None,
    "input_model": input_model,
    "output_dir": output_dir,
    "output_probabilities": output_probabilities,
    "detection_threshold": detection_threshold,
    "P_threshold": P_threshold,
    "S_threshold": S_threshold,
    "number_of_plots": number_of_plots,
    "plot_mode": plot_mode,
    "estimate_uncertainty": estimate_uncertainty,
    "number_of_sampling": number_of_sampling,
    "loss_weights": loss_weights,     
    "loss_types": loss_types,
    "input_dimention": input_dimention,
    "normalization_mode": normalization_mode,
    "batch_size": batch_size,
    "gpuid": gpuid,
    "gpu_limit": gpu_limit,
    "number_of_cpus": number_of_cpus,
    "use_multiprocessing": use_multiprocessing,
    "keepPS": keepPS,
    "allowonlyS": allowonlyS,
    "spLimit": spLimit   
    }
        
    availble_cpus = multiprocessing.cpu_count()
    if args['number_of_cpus'] > availble_cpus:
        args['number_of_cpus'] = availble_cpus
        
    if args['gpuid']:     
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args['gpuid'])
        tf.Session(config=tf.ConfigProto(log_device_placement=True))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = float(args['gpu_limit']) 
        K.tensorflow_backend.set_session(tf.Session(config=config))          
                                  
    class DummyFile(object):
        file = None
        def __init__(self, file):
            self.file = file
    
        def write(self, x):
            # Avoid print() second call (useless \n)
            if len(x.rstrip()) > 0:
                tqdm.write(x, file=self.file)
    
    @contextlib.contextmanager
    def nostdout():
        save_stdout = sys.stdout
        sys.stdout = DummyFile(sys.stdout)
        yield
        sys.stdout = save_stdout
    

    print('============================================================================')
    print('Running EqTransformer ', str(EQT_VERSION))
            
    print(' *** Loading the model ...', flush=True)        
    model = load_model(args['input_model'], 
                       custom_objects={'SeqSelfAttention': SeqSelfAttention, 
                                       'FeedForward': FeedForward,
                                       'LayerNormalization': LayerNormalization, 
                                       'f1': f1                                                                            
                                        })
    model.compile(loss = args['loss_types'],
                  loss_weights =  args['loss_weights'],           
                  optimizer = Adam(lr = 0.001),
                  metrics = [f1])
    print('*** Loading is complete!', flush=True)  

    if isinstance(args['output_dir'], str):
        out_dir = os.path.join(os.getcwd(), str(args['output_dir']))
        
        # if os.path.isdir(out_dir):
        #     print('============================================================================')        
        #     print(f' *** {out_dir} already exists!')
        #     inp = input(" --> Type (Yes or y) to create a new empty directory! otherwise it will overwrite!   ")
        #     if inp.lower() == "yes" or inp.lower() == "y":
        #         shutil.rmtree(out_dir)  
        #         os.makedirs(out_dir) 

        if os.path.isdir(out_dir):
            print('============================================================================')        
            print(f' *** {out_dir} already exists!')
            shutil.rmtree(out_dir)  
            os.makedirs(out_dir) 
        
        if platform.system() == 'Windows': 
            station_list = [ev.split(".")[0] for ev in listdir(args["input_dir"]) if ev.split("\\")[-1] != ".DS_Store"];
        else:
            station_list = [ev.split(".")[0] for ev in listdir(args['input_dir']) if ev.split("/")[-1] != ".DS_Store"];
        station_list = sorted(set(station_list))
        
        print(f"######### There are files for {len(station_list)} stations in {args['input_dir']} directory. #########", flush=True)
        for ct, st in enumerate(station_list):
            if platform.system() == 'Windows': 
                args["input_hdf5"] = args["input_dir"]+"\\"+st+".hdf5"
                args["input_csv"] = args["input_dir"]+"\\"+st+".csv"
                print("-" * 100)
                print(args["input_hdf5"])
                print(args["input_csv"])
                print("-" * 100)
            else:            
                args["input_hdf5"] = args["input_dir"]+"/"+st+".hdf5"
                args["input_csv"] = args["input_dir"]+"/"+st+".csv"
        
            save_dir = os.path.join(out_dir, str(st)+'_outputs')
            out_probs = os.path.join(save_dir, 'prediction_probabilities.hdf5')
            save_figs = os.path.join(save_dir, 'figures') 
            if os.path.isdir(save_dir):
                shutil.rmtree(save_dir)  
            os.makedirs(save_dir) 
            if args['number_of_plots']:
                os.makedirs(save_figs) 
            try:
                os.remove(out_probs)
            except Exception:
                 pass 
            
            if args['output_probabilities']:           
                HDF_PROB = h5py.File(out_probs, 'a')
                HDF_PROB.create_group("probabilities")
                HDF_PROB.create_group("uncertainties")  
            else:
                HDF_PROB = None   
                
            csvPr_gen = open(os.path.join(save_dir,'X_prediction_results.csv'), 'w')          
            predict_writer = csv.writer(csvPr_gen, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            predict_writer.writerow(['file_name', 
                                     'network',
                                     'station',
                                     'instrument_type',
                                     'station_lat',
                                     'station_lon',
                                     'station_elv',
                                     'event_start_time',
                                     'event_end_time',
                                     'detection_probability',
                                     'detection_uncertainty', 
                                     'p_arrival_time',
                                     'p_probability',
                                     'p_uncertainty',
                                     'p_snr',
                                     's_arrival_time',
                                     's_probability',
                                     's_uncertainty',
                                     's_snr'
                                         ])  
            csvPr_gen.flush()
            print(f'========= Started working on {st}, {ct+1} out of {len(station_list)} ...', flush=True)
    
            start_Predicting = time.time()       
            detection_memory = []
            plt_n = 0
        
            df = pd.read_csv(args['input_csv']) 
            prediction_list = df.trace_name.tolist() 
            fl = h5py.File(args['input_hdf5'], 'r')    
            list_generator=generate_arrays_from_file(prediction_list, args['batch_size']) 
        
            pbar_test = tqdm(total= int(np.ceil(len(prediction_list)/args['batch_size'])), ncols=100, file=sys.stdout)        
            for bn in range(int(np.ceil(len(prediction_list) / args['batch_size']))):  
                with nostdout():              
                    pbar_test.update()
                    
                new_list = next(list_generator)  
                prob_dic=_gen_predictor(new_list, args, model)
        
                pred_set={}
                for ID in new_list:
                    dataset = fl.get('data/'+str(ID))
                    pred_set.update( {str(ID) : dataset})  
                    
                plt_n, detection_memory= _gen_writer(new_list, args, prob_dic, pred_set, HDF_PROB, predict_writer, save_figs, csvPr_gen, plt_n, detection_memory, keepPS, allowonlyS, spLimit)    
    
            end_Predicting = time.time() 
            delta = (end_Predicting - start_Predicting) 
            hour = int(delta / 3600)
            delta -= hour * 3600
            minute = int(delta / 60)
            delta -= minute * 60
            seconds = delta     
            
            
            dd = pd.read_csv(os.path.join(save_dir,'X_prediction_results.csv'))
            print(f'\n', flush=True)
            print(' *** Finished the prediction in: {} hours and {} minutes and {} seconds.'.format(hour, minute, round(seconds, 2)), flush=True)         
            print(' *** Detected: '+str(len(dd))+' events.', flush=True)
            print(' *** Wrote the results into --> " ' + str(save_dir)+' "', flush=True)
        
            with open(os.path.join(save_dir,'X_report.txt'), 'a') as the_file:    
                the_file.write('================== Overal Info =============================='+'\n')               
                the_file.write('date of report: '+str(datetime.now())+'\n')         
                the_file.write('input_hdf5: '+str(args['input_hdf5'])+'\n')            
                the_file.write('input_csv: '+str(args['input_csv'])+'\n')
                the_file.write('input_model: '+str(args['input_model'])+'\n')
                the_file.write('output_dir: '+str(save_dir)+'\n')  
                the_file.write('================== Prediction Parameters ======================='+'\n')  
                the_file.write('finished the prediction in:  {} hours and {} minutes and {} seconds \n'.format(hour, minute, round(seconds, 2))) 
                the_file.write('detected: '+str(len(dd))+' events.'+'\n')                                       
                the_file.write('writting_probability_outputs: '+str(args['output_probabilities'])+'\n')  
                the_file.write('loss_types: '+str(args['loss_types'])+'\n')
                the_file.write('loss_weights: '+str(args['loss_weights'])+'\n')
                the_file.write('batch_size: '+str(args['batch_size'])+'\n')       
                the_file.write('================== Other Parameters ========================='+'\n')            
                the_file.write('normalization_mode: '+str(args['normalization_mode'])+'\n')
                the_file.write('estimate uncertainty: '+str(args['estimate_uncertainty'])+'\n')
                the_file.write('number of Monte Carlo sampling: '+str(args['number_of_sampling'])+'\n')             
                the_file.write('detection_threshold: '+str(args['detection_threshold'])+'\n')            
                the_file.write('P_threshold: '+str(args['P_threshold'])+'\n')
                the_file.write('S_threshold: '+str(args['S_threshold'])+'\n')
                the_file.write('number_of_plots: '+str(args['number_of_plots'])+'\n')                        
                the_file.write('use_multiprocessing: '+str(args['use_multiprocessing'])+'\n')            
                the_file.write('gpuid: '+str(args['gpuid'])+'\n')
                the_file.write('gpu_limit: '+str(args['gpu_limit'])+'\n')    
                the_file.write('keepPS: '+str(args['keepPS'])+'\n')
                the_file.write('allowonlyS: '+str(args['allowonlyS'])+'\n')  
                the_file.write('spLimit: '+str(args['spLimit'])+' seconds\n')      
    else:
        NN_in = len(args['output_dir'])
        for iidir in range(NN_in):
            output_dir_cur = args['output_dir'][iidir]
            input_dir_cur = args["input_dir"][iidir]
            
            out_dir = os.path.join(os.getcwd(), str(output_dir_cur))
            if os.path.isdir(out_dir):
                print('============================================================================')        
                print(f' *** {out_dir} already exists!')
                inp = input(" --> Type (Yes or y) to create a new empty directory! otherwise it will overwrite!   ")
                if inp.lower() == "yes" or inp.lower() == "y":
                    shutil.rmtree(out_dir)  
                    os.makedirs(out_dir) 
            if platform.system() == 'Windows': 
                station_list = [ev.split(".")[0] for ev in listdir(input_dir_cur) if ev.split("\\")[-1] != ".DS_Store"];
            else:
                station_list = [ev.split(".")[0] for ev in listdir(input_dir_cur) if ev.split("/")[-1] != ".DS_Store"];
            station_list = sorted(set(station_list))
            
            print(f"######### There are files for {len(station_list)} stations in {input_dir_cur} directory. #########", flush=True)
            for ct, st in enumerate(station_list):
                if platform.system() == 'Windows': 
                    args["input_hdf5"] = input_dir_cur+"\\"+st+".hdf5"
                    args["input_csv"] = input_dir_cur+"\\"+st+".csv"
                else:            
                    args["input_hdf5"] = input_dir_cur+"/"+st+".hdf5"
                    args["input_csv"] = input_dir_cur+"/"+st+".csv"
            
                save_dir = os.path.join(out_dir, str(st)+'_outputs')
                out_probs = os.path.join(save_dir, 'prediction_probabilities.hdf5')
                save_figs = os.path.join(save_dir, 'figures') 
                if os.path.isdir(save_dir):
                    shutil.rmtree(save_dir)  
                os.makedirs(save_dir) 
                if args['number_of_plots']:
                    os.makedirs(save_figs) 
                try:
                    os.remove(out_probs)
                except Exception:
                     pass 
                
                if args['output_probabilities']:           
                    HDF_PROB = h5py.File(out_probs, 'a')
                    HDF_PROB.create_group("probabilities")
                    HDF_PROB.create_group("uncertainties")  
                else:
                    HDF_PROB = None   
                    
                csvPr_gen = open(os.path.join(save_dir,'X_prediction_results.csv'), 'w')          
                predict_writer = csv.writer(csvPr_gen, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                predict_writer.writerow(['file_name', 
                                         'network',
                                         'station',
                                         'instrument_type',
                                         'station_lat',
                                         'station_lon',
                                         'station_elv',
                                         'event_start_time',
                                         'event_end_time',
                                         'detection_probability',
                                         'detection_uncertainty', 
                                         'p_arrival_time',
                                         'p_probability',
                                         'p_uncertainty',
                                         'p_snr',
                                         's_arrival_time',
                                         's_probability',
                                         's_uncertainty',
                                         's_snr'
                                             ])  
                csvPr_gen.flush()
                print(f'========= Started working on {st}, {ct+1} out of {len(station_list)} ...', flush=True)
        
                start_Predicting = time.time()       
                detection_memory = []
                plt_n = 0
            
                df = pd.read_csv(args['input_csv']) 
                prediction_list = df.trace_name.tolist() 
                fl = h5py.File(args['input_hdf5'], 'r')    
                list_generator=generate_arrays_from_file(prediction_list, args['batch_size']) 
            
                pbar_test = tqdm(total= int(np.ceil(len(prediction_list)/args['batch_size'])), ncols=100, file=sys.stdout)        
                for bn in range(int(np.ceil(len(prediction_list) / args['batch_size']))):  
                    with nostdout():              
                        pbar_test.update()
                        
                    new_list = next(list_generator)  
                    prob_dic=_gen_predictor(new_list, args, model)
            
                    pred_set={}
                    for ID in new_list:
                        dataset = fl.get('data/'+str(ID))
                        pred_set.update( {str(ID) : dataset})  
                        
                    plt_n, detection_memory= _gen_writer(new_list, args, prob_dic, pred_set, HDF_PROB, predict_writer, save_figs, csvPr_gen, plt_n, detection_memory, keepPS, allowonlyS, spLimit)    
        
                HDF_PROB.close()
        
                end_Predicting = time.time() 
                delta = (end_Predicting - start_Predicting) 
                hour = int(delta / 3600)
                delta -= hour * 3600
                minute = int(delta / 60)
                delta -= minute * 60
                seconds = delta     
                
                
                dd = pd.read_csv(os.path.join(save_dir,'X_prediction_results.csv'))
                print(f'\n', flush=True)
                print(' *** Finished the prediction in: {} hours and {} minutes and {} seconds.'.format(hour, minute, round(seconds, 2)), flush=True)         
                print(' *** Detected: '+str(len(dd))+' events.', flush=True)
                print(' *** Wrote the results into --> " ' + str(save_dir)+' "', flush=True)
            
                with open(os.path.join(save_dir,'X_report.txt'), 'a') as the_file:    
                    the_file.write('================== Overal Info =============================='+'\n')               
                    the_file.write('date of report: '+str(datetime.now())+'\n')         
                    the_file.write('input_hdf5: '+str(args['input_hdf5'])+'\n')            
                    the_file.write('input_csv: '+str(args['input_csv'])+'\n')
                    the_file.write('input_model: '+str(args['input_model'])+'\n')
                    the_file.write('output_dir: '+str(save_dir)+'\n')  
                    the_file.write('================== Prediction Parameters ======================='+'\n')  
                    the_file.write('finished the prediction in:  {} hours and {} minutes and {} seconds \n'.format(hour, minute, round(seconds, 2))) 
                    the_file.write('detected: '+str(len(dd))+' events.'+'\n')                                       
                    the_file.write('writting_probability_outputs: '+str(args['output_probabilities'])+'\n')  
                    the_file.write('loss_types: '+str(args['loss_types'])+'\n')
                    the_file.write('loss_weights: '+str(args['loss_weights'])+'\n')
                    the_file.write('batch_size: '+str(args['batch_size'])+'\n')       
                    the_file.write('================== Other Parameters ========================='+'\n')            
                    the_file.write('normalization_mode: '+str(args['normalization_mode'])+'\n')
                    the_file.write('estimate uncertainty: '+str(args['estimate_uncertainty'])+'\n')
                    the_file.write('number of Monte Carlo sampling: '+str(args['number_of_sampling'])+'\n')             
                    the_file.write('detection_threshold: '+str(args['detection_threshold'])+'\n')            
                    the_file.write('P_threshold: '+str(args['P_threshold'])+'\n')
                    the_file.write('S_threshold: '+str(args['S_threshold'])+'\n')
                    the_file.write('number_of_plots: '+str(args['number_of_plots'])+'\n')                        
                    the_file.write('use_multiprocessing: '+str(args['use_multiprocessing'])+'\n')            
                    the_file.write('gpuid: '+str(args['gpuid'])+'\n')
                    the_file.write('gpu_limit: '+str(args['gpu_limit'])+'\n')    
                    the_file.write('keepPS: '+str(args['keepPS'])+'\n')
                    the_file.write('allowonlyS: '+str(args['allowonlyS'])+'\n')
                    the_file.write('spLimit: '+str(args['spLimit'])+' seconds\n') 
      
def _gen_predictor(new_list, args, model): 
    
    
    """ 
    
    Performs the predictions for the current batch.

    Parameters
    ----------
    new_list: list of str
        A list of trace names in the batch.
    args: dic
        A dictionary containing all of the input parameters. 

    model: 
        The compiled model used for the prediction.

    Returns
    -------
    prob_dic: dic
        A dictionary containing output probabilities and their estimated standard deviations.
        
    """    
    
    prob_dic = dict()            
    params_prediction = {'file_name': str(args['input_hdf5']), 
                         'dim': args['input_dimention'][0],
                         'batch_size': len(new_list),
                         'n_channels': args['input_dimention'][-1],
                         'norm_mode': args['normalization_mode']}     
            
    prediction_generator = DataGeneratorPrediction(new_list, **params_prediction)
    if args['estimate_uncertainty']:
        if not args['number_of_sampling'] or args['number_of_sampling'] <= 0:
            print('please define the number of Monte Carlo sampling!')
        
        pred_DD = []
        pred_PP = []
        pred_SS = []          
        for mc in range(args['number_of_sampling']):
            predD, predP, predS = model.predict_generator(generator = prediction_generator,
                                                          use_multiprocessing = args['use_multiprocessing'],
                                                          workers = args['number_of_cpus'])
            pred_DD.append(predD)
            pred_PP.append(predP)               
            pred_SS.append(predS)
                            
        pred_DD = np.array(pred_DD).reshape(args['number_of_sampling'], len(new_list), params_prediction['dim'])
        pred_DD_mean = pred_DD.mean(axis=0)
        pred_DD_std = pred_DD.std(axis=0)  
                
        pred_PP = np.array(pred_PP).reshape(args['number_of_sampling'], len(new_list), params_prediction['dim'])
        pred_PP_mean = pred_PP.mean(axis=0)
        pred_PP_std = pred_PP.std(axis=0)      
                    
        pred_SS = np.array(pred_SS).reshape(args['number_of_sampling'], len(new_list), params_prediction['dim'])
        pred_SS_mean = pred_SS.mean(axis=0)
        pred_SS_std = pred_SS.std(axis=0)                       
    else:          
        pred_DD_mean, pred_PP_mean, pred_SS_mean = model.predict_generator(generator = prediction_generator,
                                                                           use_multiprocessing = args['use_multiprocessing'],
                                                                           workers = args['number_of_cpus'])
        pred_DD_mean = pred_DD_mean.reshape(pred_DD_mean.shape[0], pred_DD_mean.shape[1]) 
        pred_PP_mean = pred_PP_mean.reshape(pred_PP_mean.shape[0], pred_PP_mean.shape[1]) 
        pred_SS_mean = pred_SS_mean.reshape(pred_SS_mean.shape[0], pred_SS_mean.shape[1]) 
                    
        pred_DD_std = np.zeros((pred_DD_mean.shape))
        pred_PP_std = np.zeros((pred_PP_mean.shape))
        pred_SS_std = np.zeros((pred_SS_mean.shape))   
                
    prob_dic['DD_mean']=pred_DD_mean   
    prob_dic['PP_mean']=pred_PP_mean   
    prob_dic['SS_mean']=pred_SS_mean   
    prob_dic['DD_std']=pred_DD_std   
    prob_dic['PP_std']=pred_PP_std   
    prob_dic['SS_std']=pred_SS_std  
    
    return prob_dic  

def _gen_writer_mem_non_hdf(new_list, args, prob_dic, pred_set, predict_writer, save_figs, csvPr_gen, plt_n, detection_memory, keepPS, allowonlyS, spLimit):
    """
    Обработка предсказаний: запись CSV и построение графиков.
    """
    for ts in range(prob_dic['DD_mean'].shape[0]):
        evi = new_list[ts]
        dataset = pred_set[evi]
        dat = np.array(dataset)

        # --- Приведение dat к форме (N,3) ---
        if dat.ndim == 0:
            dat = np.zeros((6000, 3), dtype=np.float32)
        elif dat.ndim == 1:
            dat = dat[:, np.newaxis]
        if dat.shape[1] < 3:
            tmp = np.zeros((dat.shape[0], 3), dtype=dat.dtype)
            tmp[:, :dat.shape[1]] = dat
            dat = tmp

        # --- Детектор P/S ---
        matches, pick_errors, yh3 = picker(
            args,
            prob_dic['DD_mean'][ts],
            prob_dic['PP_mean'][ts],
            prob_dic['SS_mean'][ts],
            prob_dic['DD_std'][ts],
            prob_dic['PP_std'][ts],
            prob_dic['SS_std'][ts]
        )

        # --- Фильтрация по allowonlyS ---
        if not allowonlyS:
            if len(matches) >= 1 and matches[list(matches)[0]][6] and not matches[list(matches)[0]][3]:
                continue

        # --- Фильтрация по keepPS и spLimit ---
        valid_event = False
        if keepPS:
            if len(matches) >= 1 and matches[list(matches)[0]][3] and matches[list(matches)[0]][6]:
                if (matches[list(matches)[0]][6] - matches[list(matches)[0]][3]) < spLimit * 100:
                    valid_event = True
        else:
            if len(matches) >= 1 and (matches[list(matches)[0]][3] or matches[list(matches)[0]][6]):
                valid_event = True

        if valid_event:
            snr = [
                _get_snr(dat, matches[list(matches)[0]][3], window=100),
                _get_snr(dat, matches[list(matches)[0]][6], window=100)
            ]
            pre_write = len(detection_memory)
            detection_memory = _output_writter_prediction(dataset, predict_writer, csvPr_gen, matches, snr, detection_memory)
            post_write = len(detection_memory)

            # --- Строим графики только если есть данные ---
            if plt_n < args['number_of_plots'] and post_write > pre_write:
                if dat.size == 0:
                    print(f"⚠ Пропускаем построение графика для {evi}: пустой массив")
                else:
                    _plotter_prediction(
                        dat, evi, args, save_figs,
                        prob_dic['DD_mean'][ts],
                        prob_dic['PP_mean'][ts],
                        prob_dic['SS_mean'][ts],
                        prob_dic['DD_std'][ts],
                        prob_dic['PP_std'][ts],
                        prob_dic['SS_std'][ts],
                        matches
                    )
                    plt_n += 1

    return plt_n, detection_memory

def _gen_writer_mem(new_list, args, prob_dic, pred_set, HDF_PROB, predict_writer, save_figs, csvPr_gen, plt_n, detection_memory, keepPS, allowonlyS, spLimit):
    """
    Обработка предсказаний: запись CSV, построение графиков и сохранение вероятностей/неопределённостей.
    """
    for ts in range(prob_dic['DD_mean'].shape[0]):
        evi = new_list[ts]
        dataset = pred_set[evi]
        dat = np.array(dataset)

        # --- Приведение dat к форме (N,3) ---
        if dat.ndim == 0:
            dat = np.zeros((6000, 3), dtype=np.float32)
        elif dat.ndim == 1:
            dat = dat[:, np.newaxis]
        if dat.shape[1] < 3:
            tmp = np.zeros((dat.shape[0], 3), dtype=dat.dtype)
            tmp[:, :dat.shape[1]] = dat
            dat = tmp

        # --- Сохраняем вероятности и неопределенности в HDF5 ---
        if args['output_probabilities'] and HDF_PROB is not None:
            probs = np.zeros((prob_dic['DD_mean'].shape[1], 3), dtype=np.float32)
            probs[:, 0] = prob_dic['DD_mean'][ts]
            probs[:, 1] = prob_dic['PP_mean'][ts]
            probs[:, 2] = prob_dic['SS_mean'][ts]

            uncs = np.zeros((prob_dic['DD_mean'].shape[1], 3), dtype=np.float32)
            uncs[:, 0] = prob_dic['DD_std'][ts]
            uncs[:, 1] = prob_dic['PP_std'][ts]
            uncs[:, 2] = prob_dic['SS_std'][ts]

            # удаляем предыдущие датасеты, если они есть
            for grp_name, arr in zip(['probabilities', 'uncertainties'], [probs, uncs]):
                dset_path = f"{grp_name}/{evi}"
                if dset_path in HDF_PROB:
                    del HDF_PROB[dset_path]
                HDF_PROB.create_dataset(dset_path, arr.shape, data=arr, dtype=np.float32)
            HDF_PROB.flush()

        # --- Детектор P/S ---
        matches, pick_errors, yh3 = picker(
            args,
            prob_dic['DD_mean'][ts],
            prob_dic['PP_mean'][ts],
            prob_dic['SS_mean'][ts],
            prob_dic['DD_std'][ts],
            prob_dic['PP_std'][ts],
            prob_dic['SS_std'][ts]
        )

        # --- Фильтрация по allowonlyS ---
        if not allowonlyS:
            if len(matches) >= 1 and matches[list(matches)[0]][6] and not matches[list(matches)[0]][3]:
                continue

        # --- Фильтрация по keepPS и spLimit ---
        valid_event = False
        if keepPS:
            if len(matches) >= 1 and matches[list(matches)[0]][3] and matches[list(matches)[0]][6]:
                if (matches[list(matches)[0]][6] - matches[list(matches)[0]][3]) < spLimit * 100:
                    valid_event = True
        else:
            if len(matches) >= 1 and (matches[list(matches)[0]][3] or matches[list(matches)[0]][6]):
                valid_event = True

        if valid_event:
            snr = [
                _get_snr(dat, matches[list(matches)[0]][3], window=100),
                _get_snr(dat, matches[list(matches)[0]][6], window=100)
            ]
            pre_write = len(detection_memory)
            detection_memory = _output_writter_prediction(dataset, predict_writer, csvPr_gen, matches, snr, detection_memory)
            post_write = len(detection_memory)

            # --- Строим графики только если есть данные ---
            if plt_n < args['number_of_plots'] and post_write > pre_write:
                if dat.size == 0:
                    print(f"⚠ Пропускаем построение графика для {evi}: пустой массив")
                else:
                    _plotter_prediction(
                        dat, evi, args, save_figs,
                        prob_dic['DD_mean'][ts],
                        prob_dic['PP_mean'][ts],
                        prob_dic['SS_mean'][ts],
                        prob_dic['DD_std'][ts],
                        prob_dic['PP_std'][ts],
                        prob_dic['SS_std'][ts],
                        matches
                    )
                    plt_n += 1

    return plt_n, detection_memory
  
def _gen_writer(new_list, args, prob_dic, pred_set, HDF_PROB, predict_writer, save_figs, csvPr_gen, plt_n, detection_memory, keepPS, allowonlyS, spLimit):
    
    """ 
    
    Applies the detection and picking on the output predicted probabilities and if it finds any, write them out in the CSV file,
    makes the plots, and save the probabilities and uncertainties.

    Parameters
    ----------
    new_list: list of str
        A list of trace names in the batch.

    args: dic
        A dictionary containing all of the input parameters. 

    prob_dic: dic
        A dictionary containing output probabilities and their estimated standard deviations.
        
    pred_set: dic
        A dictionary containing HDF datasets for the current batch. 

    HDF_PROB: obj
        For writing out the probabilities and uncertainties. 

    predict_writer: obj
        For writing out the detection/picking results in the CSV file.    
    
    save_figs: str
        Path to the folder for saving the plots. 

    csvPr_gen : obj
        For writing out the detection/picking results in the CSV file.   
    
    plt_n: positive integer
        Keep the track of plotted figures.     

    detection_memory: list
        Keep the track of detected events.  

    keepPS: bool, default=False
        If True, detected events require both P and S picks to be written. If False, individual P or S (see allowonlyS) picks may be written.

    allowonlyS: bool, default=True
        If True, detected events with "only S" picks will be allowed. If False, an associated P pick is required.
        
    spLimit: int, default : 60
        S - P time in seconds. It will limit the results to those detections with events that have a specific S-P time limit.
        
    Returns
    -------
    plt_n: positive integer
        Keep the track of plotted figures. 
        
    detection_memory: list
        Keep the track of detected events.  
        
        
    """    
    
    for ts in range(prob_dic['DD_mean'].shape[0]): 
        evi =  new_list[ts] 
        dataset = pred_set[evi]  
        dat = np.array(dataset)


        if args['output_probabilities']: 
            
            probs = np.zeros((prob_dic['DD_mean'].shape[1], 3))
            probs[:, 0] = prob_dic['DD_mean'][ts]
            probs[:, 1] = prob_dic['PP_mean'][ts]
            probs[:, 2] = prob_dic['SS_mean'][ts]
             
            uncs = np.zeros((prob_dic['DD_mean'].shape[1], 3))
            uncs[:, 0] = prob_dic['DD_std'][ts]
            uncs[:, 1] = prob_dic['PP_std'][ts]
            uncs[:, 2] = prob_dic['SS_std'][ts]
            
            HDF_PROB.create_dataset('probabilities/'+str(evi), probs.shape, data=probs, dtype= np.float32) 
            HDF_PROB.create_dataset('uncertainties/'+str(evi), uncs.shape, data=uncs, dtype= np.float32) 
            HDF_PROB.flush()
                               
        matches, pick_errors, yh3 =  picker(args, prob_dic['DD_mean'][ts], prob_dic['PP_mean'][ts], prob_dic['SS_mean'][ts],
                                            prob_dic['DD_std'][ts], prob_dic['PP_std'][ts], prob_dic['SS_std'][ts])

        if not allowonlyS: #if NOT limiting to "only S" picks
            if len(matches)>=1 and matches[list(matches)[0]][6] and not matches[list(matches)[0]][3]: #if S picks exist but no P...
                continue
        
        if keepPS:
            if (len(matches) >= 1) and (matches[list(matches)[0]][3] and matches[list(matches)[0]][6]):
                if (matches[list(matches)[0]][6] - matches[list(matches)[0]][3]) < spLimit*100:
                    snr = [_get_snr(dat, matches[list(matches)[0]][3], window = 100), _get_snr(dat, matches[list(matches)[0]][6], window = 100)] 
                    pre_write = len(detection_memory)
                    detection_memory=_output_writter_prediction(dataset, predict_writer, csvPr_gen, matches, snr, detection_memory)
                    post_write = len(detection_memory)
                    if plt_n < args['number_of_plots'] and post_write > pre_write:
                        _plotter_prediction(dat, evi, args, save_figs, 
                                              prob_dic['DD_mean'][ts], 
                                              prob_dic['PP_mean'][ts],
                                              prob_dic['SS_mean'][ts],
                                              prob_dic['DD_std'][ts],
                                              prob_dic['PP_std'][ts], 
                                              prob_dic['SS_std'][ts],
                                              matches)
                        plt_n += 1 ; 
        else:
            if (len(matches) >= 1) and ((matches[list(matches)[0]][3] or matches[list(matches)[0]][6])):
                snr = [_get_snr(dat, matches[list(matches)[0]][3], window = 100), _get_snr(dat, matches[list(matches)[0]][6], window = 100)] 
                pre_write = len(detection_memory)
                detection_memory=_output_writter_prediction(dataset, predict_writer, csvPr_gen, matches, snr, detection_memory)
                post_write = len(detection_memory)
                if plt_n < args['number_of_plots'] and post_write > pre_write:
                    _plotter_prediction(dat, evi, args, save_figs, 
                                          prob_dic['DD_mean'][ts], 
                                          prob_dic['PP_mean'][ts],
                                          prob_dic['SS_mean'][ts],
                                          prob_dic['DD_std'][ts],
                                          prob_dic['PP_std'][ts], 
                                          prob_dic['SS_std'][ts],
                                          matches)
                    plt_n += 1 ; 
           
                    
    return plt_n, detection_memory

def _output_writter_prediction(dataset, predict_writer, csvPr, matches, snr, detection_memory):
    """ 
    Запись результатов обнаружения и фаз P/S в CSV с корректными значениями неопределённостей и SNR.

    Parameters
    ----------
    dataset: HDFMemoryDataset
        Объект с данными сегмента.

    predict_writer: csv.writer
        Объект для записи CSV.

    csvPr: file
        Файл CSV для flush.

    matches: dict
        Содержит информацию о детекциях и пиках.

    snr: list of floats
        SNR для фаз P и S.

    detection_memory: list
        Список предыдущих обнаруженных событий для фильтрации повторов.

    Returns
    -------
    detection_memory: list
        Обновлённый список обнаруженных событий.
    """

    trace_name = dataset.attrs["trace_name"]
    station_name = "{:<4}".format(dataset.attrs["receiver_code"])
    network_name = "{:<2}".format(dataset.attrs["network_code"])
    instrument_type = "{:<2}".format(trace_name.split('_')[2])
    station_lat = dataset.attrs["receiver_latitude"]
    station_lon = dataset.attrs["receiver_longitude"]
    station_elv = dataset.attrs["receiver_elevation_m"]
    start_time = dataset.attrs["trace_start_time"]

    try:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    except Exception:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')

    def _date_convertor(r):
        if r is None:
            return ''
        if isinstance(r, str):
            try:
                return datetime.strptime(r, '%Y-%m-%d %H:%M:%S.%f')
            except Exception:
                return datetime.strptime(r, '%Y-%m-%d %H:%M:%S')
        return r

    for match_idx, match_value in matches.items():
        ev_strt = start_time + timedelta(seconds=match_idx / 100)
        ev_end = start_time + timedelta(seconds=match_value[0] / 100)

        # Проверка на повторное событие
        doublet = [st for st in detection_memory if abs((st - ev_strt).total_seconds()) < 2]
        if doublet:
            continue

        # --- Основные вероятности и неопределённости ---
        det_prob = round(match_value[1], 2)
        det_unc  = round(match_value[2], 2) if match_value[2] is not None else np.nan

        # --- Фаза P ---
        p_time = start_time + timedelta(seconds=match_value[3] / 100) if match_value[3] is not None else None
        p_prob = round(match_value[4], 2) if match_value[4] is not None else np.nan
        p_unc  = round(match_value[5], 2) if match_value[5] is not None else np.nan
        p_snr  = snr[0] if snr[0] is not None else np.nan

        # --- Фаза S ---
        s_time = start_time + timedelta(seconds=match_value[6] / 100) if match_value[6] is not None else None
        s_prob = round(match_value[7], 2) if match_value[7] is not None else np.nan
        s_unc  = round(match_value[8], 2) if match_value[8] is not None else np.nan
        s_snr  = snr[1] if snr[1] is not None else np.nan

        # --- Запись в CSV ---
        predict_writer.writerow([
            trace_name,
            network_name,
            station_name,
            instrument_type,
            station_lat,
            station_lon,
            station_elv,
            _date_convertor(ev_strt),
            _date_convertor(ev_end),
            det_prob,
            det_unc,
            _date_convertor(p_time),
            p_prob,
            p_unc,
            p_snr,
            _date_convertor(s_time),
            s_prob,
            s_unc,
            s_snr
        ])
        csvPr.flush()
        detection_memory.append(ev_strt)

    return detection_memory
  
def _plotter_prediction(data, evi, args, save_figs, yh1, yh2, yh3, yh1_std, yh2_std, yh3_std, matches):

    """ 
    
    Generates plots of detected events waveforms, output predictions, and picked arrival times.

    Parameters
    ----------
    data: NumPy array
        3 component raw waveform.

    evi : str
        Trace name.  

    args: dic
        A dictionary containing all of the input parameters. 

    save_figs: str
        Path to the folder for saving the plots. 

    yh1: 1D array
        Detection probabilities. 

    yh2: 1D array
        P arrival probabilities.    
     
    yh3: 1D array
        S arrival probabilities. 
 
    yh1_std: 1D array
        Detection standard deviations. 

    yh2_std: 1D array
        P arrival standard deviations.  
       
    yh3_std: 1D array
        S arrival standard deviations. 

    matches: dic
        Contains the information for the detected and picked event.   
         
        
    """  

    font0 = {'family': 'serif',
            'color': 'white',
            'stretch': 'condensed',
            'weight': 'normal',
            'size': 12,
            } 
   
    spt, sst, detected_events = [], [], []
    for match, match_value in matches.items():
        detected_events.append([match, match_value[0]])
        if match_value[3]: 
            spt.append(match_value[3])
        else:
            spt.append(None)
            
        if match_value[6]:
            sst.append(match_value[6])
        else:
            sst.append(None)    
            
    if args['plot_mode'] == 'time_frequency':
    
        fig = plt.figure(constrained_layout=False)
        widths = [6, 1]
        heights = [1, 1, 1, 1, 1, 1, 1.8]
        spec5 = fig.add_gridspec(ncols=2, nrows=7, width_ratios=widths,
                              height_ratios=heights, left=0.1, right=0.9, hspace=0.1)
        
        
        ax = fig.add_subplot(spec5[0, 0])         
        plt.plot(data[:, 0], 'k')
        plt.xlim(0, 6000)
        x = np.arange(6000)
     #   for ev in detected_events:
     #       l, = plt.gca().plot(x[ev[0]:ev[1]], data[ev[0]:ev[1], 0], 'mediumblue')                        
        ax.set_xticks([])
        plt.rcParams["figure.figsize"] = (10, 10)
        legend_properties = {'weight':'bold'} 
        plt.title('Trace Name: '+str(evi))
        
        pl = None
        sl = None            
        
        if len(spt) > 0 and np.count_nonzero(data[:, 0]) > 10:
            ymin, ymax = ax.get_ylim()
            for ipt, pt in enumerate(spt):
                if pt and ipt == 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
                elif pt and ipt > 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2)
                    
        if len(sst) > 0 and np.count_nonzero(data[:, 0]) > 10: 
            for ist, st in enumerate(sst): 
                if st and ist == 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
                elif st and ist > 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)
        
    
        ax = fig.add_subplot(spec5[0, 1])                 
        if pl or sl: 
            custom_lines = [Line2D([0], [0], color='k', lw=0),
                            Line2D([0], [0], color='c', lw=2),
                            Line2D([0], [0], color='m', lw=2)]
            plt.legend(custom_lines, ['E', 'Picked P', 'Picked S'], fancybox=True, shadow=True)
            plt.axis('off')
    
    
        ax = fig.add_subplot(spec5[1, 0])         
        f, t, Pxx = signal.stft(data[:, 0], fs=100, nperseg=80)
        Pxx = np.abs(Pxx)                       
        plt.pcolormesh(t, f, Pxx, alpha=None, cmap='hot', shading='flat', antialiased=True)
        plt.ylim(0, 40)
        plt.text(1, 1, 'STFT', fontdict=font0)
        plt.ylabel('Hz', fontsize=12)
        ax.set_xticks([])
        
        
        ax = fig.add_subplot(spec5[2, 0])   
        plt.plot(data[:, 1] , 'k')
        plt.xlim(0, 6000)  
    #    for ev in detected_events:
    #        l, = plt.gca().plot(x[ev[0]:ev[1]], data[ev[0]:ev[1], 0], 'mediumblue')             
        ax.set_xticks([])
        if len(spt) > 0 and np.count_nonzero(data[:, 1]) > 10:
            ymin, ymax = ax.get_ylim()
            for ipt, pt in enumerate(spt):
                if pt and ipt == 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
                elif pt and ipt > 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2) 
                    
        if len(sst) > 0 and np.count_nonzero(data[:, 1]) > 10: 
            for ist, st in enumerate(sst): 
                if st and ist == 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
                elif st and ist > 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)
                    
        ax = fig.add_subplot(spec5[2, 1])         
        if pl or sl:
            custom_lines = [Line2D([0], [0], color='k', lw=0),
                            Line2D([0], [0], color='c', lw=2),
                            Line2D([0], [0], color='m', lw=2)]
            plt.legend(custom_lines, ['N', 'Picked P', 'Picked S'], fancybox=True, shadow=True)
            plt.axis('off')
    
    
        ax = fig.add_subplot(spec5[3, 0]) 
        f, t, Pxx = signal.stft(data[:, 1], fs=100, nperseg=80)
        Pxx = np.abs(Pxx)                       
        plt.pcolormesh(t, f, Pxx, alpha=None, cmap='hot', shading='flat', antialiased=True)
        plt.ylim(0, 40)
        plt.text(1, 1, 'STFT', fontdict=font0)
        plt.ylabel('Hz', fontsize=12)
        ax.set_xticks([])        
                       
        
        ax = fig.add_subplot(spec5[4, 0]) 
        plt.plot(data[:, 2], 'k') 
        plt.xlim(0, 6000)   
    #    for ev in detected_events:
     #       l, = plt.gca().plot(x[ev[0]:ev[1]], data[ev[0]:ev[1], 0], 'mediumblue')             
        ax.set_xticks([])               
        if len(spt) > 0 and np.count_nonzero(data[:, 2]) > 10:
            ymin, ymax = ax.get_ylim()
            for ipt, pt in enumerate(spt):
                if pt and ipt == 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
                elif pt and ipt > 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2) 
                    
        if len(sst) > 0 and np.count_nonzero(data[:, 2]) > 10:
            for ist, st in enumerate(sst): 
                if st and ist == 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
                elif st and ist > 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)  
                    
        ax = fig.add_subplot(spec5[4, 1])                         
        if pl or sl:    
            custom_lines = [Line2D([0], [0], color='k', lw=0),
                            Line2D([0], [0], color='c', lw=2),
                            Line2D([0], [0], color='m', lw=2)]
            plt.legend(custom_lines, ['Z', 'Picked P', 'Picked S'], fancybox=True, shadow=True)
            plt.axis('off')        
    
        ax = fig.add_subplot(spec5[5, 0])         
        f, t, Pxx = signal.stft(data[:, 2], fs=100, nperseg=80)
        Pxx = np.abs(Pxx)                       
        plt.pcolormesh(t, f, Pxx, alpha=None, cmap='hot', shading='flat', antialiased=True)
        plt.ylim(0, 40)
        plt.text(1, 1, 'STFT', fontdict=font0)
        plt.ylabel('Hz', fontsize=12)
        ax.set_xticks([])                   
            
        ax = fig.add_subplot(spec5[6, 0])
        x = np.linspace(0, data.shape[0], data.shape[0], endpoint=True)
        if args['estimate_uncertainty']:                               
            plt.plot(x, yh1, '--', color='g', alpha = 0.5, linewidth=2, label='Earthquake')
            lowerD = yh1-yh1_std
            upperD = yh1+yh1_std
            plt.fill_between(x, lowerD, upperD, alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99')            
                                
            plt.plot(x, yh2, '--', color='b', alpha = 0.5, linewidth=2, label='P_arrival')
            lowerP = yh2-yh2_std
            upperP = yh2+yh2_std
            plt.fill_between(x, lowerP, upperP, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')  
                                         
            plt.plot(x, yh3, '--', color='r', alpha = 0.5, linewidth=2, label='S_arrival')
            lowerS = yh3-yh3_std
            upperS = yh3+yh3_std
            plt.fill_between(x, lowerS, upperS, edgecolor='#CC4F1B', facecolor='#FF9848')
            
            plt.tight_layout()                   
            plt.ylim((-0.1, 1.1))
            plt.xlim(0, 6000)            
            plt.ylabel('Probability', fontsize=12)
            plt.xlabel('Sample', fontsize=12)                    
            plt.yticks(np.arange(0, 1.1, step=0.2))
            axes = plt.gca()
            axes.yaxis.grid(color='lightgray')  
            
            font = {'family': 'serif',
                    'color': 'dimgrey',
                    'style': 'italic',
                    'stretch': 'condensed',
                    'weight': 'normal',
                    'size': 12,
                    }

                            
        else:
            plt.plot(x, yh1, '--', color='g', alpha = 0.5, linewidth=2, label='Earthquake')
            plt.plot(x, yh2, '--', color='b', alpha = 0.5, linewidth=2, label='P_arrival')
            plt.plot(x, yh3, '--', color='r', alpha = 0.5, linewidth=2, label='S_arrival')
            plt.tight_layout()       
            plt.ylim((-0.1, 1.1)) 
            plt.xlim(0, 6000)
            plt.ylabel('Probability', fontsize=12) 
            plt.xlabel('Sample', fontsize=12) 
            plt.yticks(np.arange(0, 1.1, step=0.2))
            axes = plt.gca()
            axes.yaxis.grid(color='lightgray')        
    
        ax = fig.add_subplot(spec5[6, 1])  
        custom_lines = [Line2D([0], [0], linestyle='--', color='g', lw=2),
                        Line2D([0], [0], linestyle='--', color='b', lw=2),
                        Line2D([0], [0], linestyle='--', color='r', lw=2)]
        plt.legend(custom_lines, ['Earthquake', 'P_arrival', 'S_arrival'], fancybox=True, shadow=True)
        plt.axis('off')
            
        font = {'family': 'serif',
                    'color': 'dimgrey',
                    'style': 'italic',
                    'stretch': 'condensed',
                    'weight': 'normal',
                    'size': 12,
                    }
        
        plt.text(1, 0.2, 'EQTransformer', fontdict=font)
        if EQT_VERSION:
            plt.text(2000, 0.05, str(EQT_VERSION), fontdict=font)
            
        plt.xlim(0, 6000)
        fig.tight_layout()
        fig.savefig(os.path.join(save_figs, str(evi).replace(':', '-')+'.png'), dpi=200) 
        plt.close(fig)
        plt.clf()
    

    else:        
        
        ########################################## ploting only in time domain
        fig = plt.figure(constrained_layout=True)
        widths = [1]
        heights = [1.6, 1.6, 1.6, 2.5]
        spec5 = fig.add_gridspec(ncols=1, nrows=4, width_ratios=widths,
                              height_ratios=heights)
        
        ax = fig.add_subplot(spec5[0, 0])         
        plt.plot(data[:, 0], 'k')
        x = np.arange(6000)
        plt.xlim(0, 6000)            
        
        plt.ylabel('Amplitude\nCounts')

    #    for ev in detected_events:
    #        l, = plt.gca().plot(x[ev[0]:ev[1]], data[ev[0]:ev[1], 0], 'mediumblue')                        
                    
        plt.rcParams["figure.figsize"] = (8,6)
        legend_properties = {'weight':'bold'}  
        plt.title('Trace Name: '+str(evi))
        
        pl = sl = None        
        if len(spt) > 0 and np.count_nonzero(data[:, 0]) > 10:
            ymin, ymax = ax.get_ylim()
            for ipt, pt in enumerate(spt):
                if pt and ipt == 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
                elif pt and ipt > 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2)
                    
        if len(sst) > 0 and np.count_nonzero(data[:, 0]) > 10: 
            for ist, st in enumerate(sst): 
                if st and ist == 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
                elif st and ist > 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)
                    
        if pl or sl:    
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            custom_lines = [Line2D([0], [0], color='k', lw=0),
                            Line2D([0], [0], color='c', lw=2),
                            Line2D([0], [0], color='m', lw=2)]
            plt.legend(custom_lines, ['E', 'Picked P', 'Picked S'], 
                       loc='center left', bbox_to_anchor=(1, 0.5), 
                       fancybox=True, shadow=True)
                                           
        ax = fig.add_subplot(spec5[1, 0])   
        plt.plot(data[:, 1] , 'k')
        plt.xlim(0, 6000)            
        plt.ylabel('Amplitude\nCounts')
        
     #   for ev in detected_events:
     #       l, = plt.gca().plot(x[ev[0]:ev[1]], data[ev[0]:ev[1], 0], 'mediumblue')             
                  
        if len(spt) > 0 and np.count_nonzero(data[:, 1]) > 10:
            ymin, ymax = ax.get_ylim()
            for ipt, pt in enumerate(spt):
                if pt and ipt == 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
                elif pt and ipt > 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2)
                    
        if len(sst) > 0 and np.count_nonzero(data[:, 1]) > 10: 
            for ist, st in enumerate(sst): 
                if st and ist == 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
                elif st and ist > 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)
    
        if pl or sl:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            custom_lines = [Line2D([0], [0], color='k', lw=0),
                            Line2D([0], [0], color='c', lw=2),
                            Line2D([0], [0], color='m', lw=2)]
            plt.legend(custom_lines, ['N', 'Picked P', 'Picked S'], 
                       loc='center left', bbox_to_anchor=(1, 0.5), 
                       fancybox=True, shadow=True)
                         
        ax = fig.add_subplot(spec5[2, 0]) 
        plt.plot(data[:, 2], 'k') 
        plt.xlim(0, 6000)                    
        plt.ylabel('Amplitude\nCounts')

   #     for ev in detected_events:
   #         l, = plt.gca().plot(x[ev[0]:ev[1]], data[ev[0]:ev[1], 0], 'mediumblue')             
        ax.set_xticks([])
                   
        if len(spt) > 0 and np.count_nonzero(data[:, 2]) > 10:
            ymin, ymax = ax.get_ylim()
            for ipt, pt in enumerate(spt):
                if pt and ipt == 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
                elif pt and ipt > 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2)
                    
        if len(sst) > 0 and np.count_nonzero(data[:, 2]) > 10:
            for ist, st in enumerate(sst): 
                if st and ist == 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
                elif st and ist > 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)
                    
        if pl or sl:    
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            custom_lines = [Line2D([0], [0], color='k', lw=0),
                            Line2D([0], [0], color='c', lw=2),
                            Line2D([0], [0], color='m', lw=2)]
            plt.legend(custom_lines, ['Z', 'Picked P', 'Picked S'], 
                       loc='center left', bbox_to_anchor=(1, 0.5), 
                       fancybox=True, shadow=True)       
                   
        ax = fig.add_subplot(spec5[3, 0])
        x = np.linspace(0, data.shape[0], data.shape[0], endpoint=True)
        
        if args['estimate_uncertainty']:                               
            plt.plot(x, yh1, '--', color='g', alpha = 0.5, linewidth=1.5, label='Earthquake')
            lowerD = yh1-yh1_std
            upperD = yh1+yh1_std
            plt.fill_between(x, lowerD, upperD, alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99')            
                                
            plt.plot(x, yh2, '--', color='b', alpha = 0.5, linewidth=1.5, label='P_arrival')
            lowerP = yh2-yh2_std
            upperP = yh2+yh2_std
            plt.fill_between(x, lowerP, upperP, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')  
                                         
            plt.plot(x, yh3, '--', color='r', alpha = 0.5, linewidth=1.5, label='S_arrival')
            lowerS = yh3-yh3_std
            upperS = yh3+yh3_std
            plt.fill_between(x, lowerS, upperS, edgecolor='#CC4F1B', facecolor='#FF9848')
            
            plt.tight_layout()                   
            plt.ylim((-0.1, 1.1))
            plt.xlim(0, 6000)                                
            plt.ylabel('Probability')
            plt.xlabel('Sample')                    
            plt.legend(loc='lower center', bbox_to_anchor=(0., 1.17, 1., .102), ncol=3, mode="expand",
                       prop=legend_properties,  borderaxespad=0., fancybox=True, shadow=True)
            plt.yticks(np.arange(0, 1.1, step=0.2))
            axes = plt.gca()
            axes.yaxis.grid(color='lightgray')

            font = {'family': 'serif',
                    'color': 'dimgrey',
                    'style': 'italic',
                    'stretch': 'condensed',
                    'weight': 'normal',
                    'size': 12,
                    }
    
            plt.text(6500, 0.5, 'EQTransformer', fontdict=font)
            if EQT_VERSION:
                plt.text(7000, 0.1, str(EQT_VERSION), fontdict=font)
                            
        else:
            plt.plot(x, yh1, '--', color='g', alpha = 0.5, linewidth=1.5, label='Earthquake')
            plt.plot(x, yh2, '--', color='b', alpha = 0.5, linewidth=1.5, label='P_arrival')
            plt.plot(x, yh3, '--', color='r', alpha = 0.5, linewidth=1.5, label='S_arrival')
            
            plt.tight_layout()       
            plt.ylim((-0.1, 1.1)) 
            plt.xlim(0, 6000)                                            
            plt.ylabel('Probability') 
            plt.xlabel('Sample')  
            plt.legend(loc='lower center', bbox_to_anchor=(0., 1.17, 1., .102), ncol=3, mode="expand",
                       prop=legend_properties,  borderaxespad=0., fancybox=True, shadow=True)
            plt.yticks(np.arange(0, 1.1, step=0.2))
            axes = plt.gca()
            axes.yaxis.grid(color='lightgray')
            
            font = {'family': 'serif',
                    'color': 'dimgrey',
                    'style': 'italic',
                    'stretch': 'condensed',
                    'weight': 'normal',
                    'size': 12,
                    }
    
            plt.text(6500, 0.5, 'EQTransformer', fontdict=font)
            if EQT_VERSION:
                plt.text(7000, 0.1, str(EQT_VERSION), fontdict=font)
            
        fig.tight_layout()
        fig.savefig(os.path.join(save_figs, str(evi).replace(':', '-')+'.png')) 
        plt.close(fig)
        plt.clf()
        
def _get_snr(data, pat, window = 200):
    
    """ 
    
    Estimates SNR.
    
    Parameters
    ----------
    data: NumPy array
        3 component data.     

    pat: positive integer
        Sample point where a specific phase arrives.  

    window: positive integer
        The length of the window for calculating the SNR (in the sample).         
        
    Returns
    -------   
    snr : {float, None}
       Estimated SNR in db.   
        
    """      
       
    snr = None
    if pat:
        try:
            if int(pat) >= window and (int(pat)+window) < len(data):
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)           
            elif int(pat) < window and (int(pat)+window) < len(data):
                window = int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)
            elif (int(pat)+window) > len(data):
                window = len(data)-int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)    
        except Exception:
            pass
    return snr 



