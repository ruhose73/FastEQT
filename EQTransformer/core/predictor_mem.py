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

def segment_to_windows(data, window_size=6000, step=3000):
    """
    Разбивает 2D массив (N, 3) на перекрывающиеся окна с нормализацией.
    """
    n = data.shape[0]
    windows = []
    for start in range(0, n - window_size + 1, step):
        w = data[start:start + window_size].copy()
        # --- стандартная нормализация EQTransformer ---
        std = np.std(w, axis=0)
        std[std == 0] = 1e-10
        w = w / std
        windows.append(w)
    return np.stack(windows, axis=0) if windows else np.empty((0, window_size, 3))

def merge_windows(pred_array, window_size=6000, step=3000):
    """
    Сливает вероятности окон обратно в единый временной ряд, усредняя перекрытия.
    """
    if pred_array.ndim == 3:
        pred_array = np.squeeze(pred_array, axis=-1)  # (num_windows, 6000)

    num_windows = pred_array.shape[0]
    total_len = window_size + (num_windows - 1) * step
    merged = np.zeros(total_len)
    counts = np.zeros(total_len)

    for i, win in enumerate(pred_array):
        start = i * step
        merged[start:start + window_size] += win
        counts[start:start + window_size] += 1

    counts[counts == 0] = 1
    return merged / counts

def _gen_predictor_mem_batch(id_list, args, model, data_dict,
                             window_size=6000, step=3000, batch_size=32):
    """
    Батчевый инференс для нескольких ID, с нормализацией и слиянием окон.
    """
    all_windows = []
    index_slices = []

    # --- Подготовка всех окон сразу ---
    for ID in id_list:
        data = data_dict[ID]
        if isinstance(data, dict) and 'data' in data:
            data = data['data']
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(f"{ID}: ожидается массив формы (N,3), получено {data.shape}")

        windows = segment_to_windows(data, window_size, step)
        if len(windows) == 0:
            continue

        start_idx = len(all_windows)
        all_windows.append(windows)
        end_idx = start_idx + len(windows)
        index_slices.append((ID, slice(start_idx, end_idx)))

    if not all_windows:
        return {}

    # --- Конкатенация и предсказание ---
    all_windows = np.concatenate(all_windows, axis=0)  # (total_windows, 6000, 3)
    preds = model.predict(all_windows, batch_size=batch_size, verbose=0)

    # --- Проверка выходов ---
    if not isinstance(preds, (list, tuple)) or len(preds) != 3:
        raise ValueError("Ожидалось 3 выхода (D, P, S), но модель вернула: "
                         f"{[p.shape for p in preds]}")

    DD_mean, PP_mean, SS_mean = [np.squeeze(p, axis=-1) for p in preds]  # (Nwin, 6000)

    # --- Отладочная статистика ---
    print(f"[DEBUG] DD_mean: min={DD_mean.min():.4f}, max={DD_mean.max():.4f}, mean={DD_mean.mean():.4f}")
    print(f"[DEBUG] PP_mean: min={PP_mean.min():.4f}, max={PP_mean.max():.4f}, mean={PP_mean.mean():.4f}")
    print(f"[DEBUG] SS_mean: min={SS_mean.min():.4f}, max={SS_mean.max():.4f}, mean={SS_mean.mean():.4f}")

    # --- Формирование словаря результатов ---
    prob_dic_total = {}
    for ID, sl in index_slices:
        DD_m = merge_windows(DD_mean[sl], window_size, step)
        PP_m = merge_windows(PP_mean[sl], window_size, step)
        SS_m = merge_windows(SS_mean[sl], window_size, step)

        prob_dic_total[ID] = {
            'DD_mean': np.expand_dims(DD_m, axis=0),
            'PP_mean': np.expand_dims(PP_m, axis=0),
            'SS_mean': np.expand_dims(SS_m, axis=0),
            'DD_std': np.zeros((1, DD_m.shape[0]), dtype=np.float32),
            'PP_std': np.zeros((1, PP_m.shape[0]), dtype=np.float32),
            'SS_std': np.zeros((1, SS_m.shape[0]), dtype=np.float32),
        }

    return prob_dic_total

def predictor_mem_non_hdf_load_model(csv_segment, hdf_segment,
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
                                     spLimit=60,
                                     batch_ids=8,
                                     batch_size=32):
    """
    Основная функция предсказания с пакетной обработкой окон.
    """

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
    predict_writer.writerow([
    'file_name', 'network', 'station', 'instrument_type', 'station_lat', 'station_lon', 'station_elv',
    'event_start_time', 'event_end_time', 'detection_probability', 'detection_uncertainty',
    'p_arrival_time', 'p_probability', 'p_uncertainty', 'p_snr',
    's_arrival_time', 's_probability', 's_uncertainty', 's_snr'
    ])
    csvPr_gen.flush()

    # --- Загружаем модель, если не передали готовую ---
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

    # --- Пакетная обработка ID ---
    for i in tqdm(range(0, len(csv_segment), batch_ids), desc="Predicting"):
        batch_rows = csv_segment[i:i + batch_ids]
        id_list = [r[0] for r in batch_rows if r[0] in hdf_segment]
        if not id_list:
            continue

        # Загружаем данные для батча
        data_dict = {ID: hdf_segment[ID]['data'] for ID in id_list}

        # Батчевый инференс
        prob_dic_batch = _gen_predictor_mem_batch(id_list, args, model, data_dict,
                                                  window_size=6000, step=3000,
                                                  batch_size=batch_size)

        # --- Обработка результатов ---
        for ID in id_list:
            if ID not in prob_dic_batch:
                continue

            class InMemoryDataset:
                def __init__(self, data_dict):
                    self.data = data_dict['data']
                    self.attrs = data_dict.get('attrs', {})

            dataset_obj = InMemoryDataset(hdf_segment[ID])
            prob_dic = prob_dic_batch[ID]

            plt_n, detection_memory = _gen_writer_mem_non_hdf(
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

def _output_writter_prediction(dataset, predict_writer, csvPr, matches, snr, detection_memory):
    
    """ 
    
    Writes the detection & picking results into a CSV file.

    Parameters
    ----------
    dataset: hdf5 obj
        Dataset object of the trace.

    predict_writer: obj
        For writing out the detection/picking results in the CSV file.
        
    csvPr: obj
        For writing out the detection/picking results in the CSV file.  

    matches: dic
        It contains the information for the detected and picked event.   
 
    snr: list of two floats
        Estimated signal to noise ratios for picked P and S phases.      
 
    detection_memory : list
        Keep the track of detected events.          
        
    Returns
    -------   
    detection_memory : list
        Keep the track of detected events.  
        
        
    """      

    trace_name = dataset.attrs["trace_name"]
    station_name = dataset.attrs["receiver_code"]
    station_lat = dataset.attrs["receiver_latitude"]
    station_lon = dataset.attrs["receiver_longitude"]
    station_elv = dataset.attrs["receiver_elevation_m"]
    start_time = dataset.attrs["trace_start_time"]
    station_name = "{:<4}".format(station_name)
    network_name = dataset.attrs["network_code"]
    network_name = "{:<2}".format(network_name)
    instrument_type = trace_name.split('_')[2]
    instrument_type = "{:<2}".format(instrument_type)  

    try:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    except Exception:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        
    def _date_convertor(r):  
        if isinstance(r, str):
            mls = r.split('.')
            if len(mls) == 1:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S')
            else:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S.%f')
        else:
            new_t = r
            
        return new_t
            
    for match, match_value in matches.items():
        ev_strt = start_time+timedelta(seconds= match/100)
        ev_end = start_time+timedelta(seconds= match_value[0]/100)
        
        doublet = [ st for st in detection_memory if abs((st-ev_strt).total_seconds()) < 2]
        
        if len(doublet) == 0: 
            det_prob = round(match_value[1], 2)
            if match_value[2]:
                det_unc = round(match_value[2], 2) 
            else:
                det_unc = match_value[2]
                       
            if match_value[3]: 
                p_time = start_time+timedelta(seconds= match_value[3]/100)
            else:
                p_time = None
            p_prob = match_value[4]
            p_unc = match_value[5]
            
            if p_unc:
                p_unc = round(p_unc, 2)
            if p_prob:
                p_prob = round(p_prob, 2)
                
            if match_value[6]:
                s_time = start_time+timedelta(seconds= match_value[6]/100)
            else:
                s_time = None
            s_prob = match_value[7]
            s_unc = match_value[8]

            if s_unc:
                s_unc = round(s_unc, 2)                
            if s_prob:
                s_prob = round(s_prob, 2)
                
            predict_writer.writerow([trace_name, 
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
                                         snr[0],
                                         _date_convertor(s_time), 
                                         s_prob,
                                         s_unc, 
                                         snr[1]
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
   