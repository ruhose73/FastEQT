#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 18:52:42 2019

@author: mostafamousavi

last update: 01/29/2021
"""

from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
import json
import os
import platform
import sqlite3 
import pandas as pd
import csv
from os import listdir
import h5py
#import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.signal.trigger import ar_pick
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from itertools import combinations
from obspy.core.event import Catalog, Event, Origin, Arrival, Pick, WaveformStreamID


def run_associator(input_dir,
                   start_time, 
                   end_time, 
                   moving_window=15, 
                   pair_n=3,
                   output_dir='.',
                   consider_combination=False):
    
    """
    
    It performs a very simple association based on detection times on multiple stations. It works fine when you have a small and local network of seismic stations. 

    Parameters
    ----------
    input_dir: str, default=None
        Directory name containing hdf5 and csv files-preprocessed data.
        
    start_time: str, default=None
        Start of a time period of interest in 'YYYY-MM-DD hh:mm:ss.f' format.
        
    end_time: str, default=None
        End of a timeperiod of interest in 'YYYY-MM-DD hh:mm:ss.f' format. 
        
    moving_window: int, default=15
        The length of time window used for association in second. 
        
    pair_n: int, default=2
        The minimum number of stations used for the association. 
        
    output_dir: str, default='.'
        Path to the directory to write the output file.
        
    consider_combination: bool, default=False
        If True, it will write down all possible combinations of picked arrival times for each event. This will generate multiple events with the same ID, and you will need to remove those with poor solutions after location. This helps to remove the false positives from the associated event. 


    Returns
    ----------        
    output_dir/Y2000.phs: Phase information for the associated events in hypoInverse format.     

    output_dir/associations.xml: quakeml output (containing origin and pick objects - using ObsPy functions). QuakeML is useful so that the user can then easily use ObsPy to generate input for other relocator methods (e.g. NonLinLoc). Contributed by Stephen Hicks  

    output_dir/traceNmae_dic.json: A dictionary where the trace name for all the detections associated to an event are listed. This can be used later to access the traces for calculating the cross-correlations during the relocation process. 
        
        
    Warning
    ----------        
    Unlike the other modules, this function does not create the ouput directory. So if the given path does not exist will give an error. 
        
    """   
   
    
    if os.path.exists("phase_dataset"):
        os.remove("phase_dataset")
    conn = sqlite3.connect("phase_dataset")
    cur = conn.cursor()
    
    cur.execute('''
        CREATE TABLE phase_dataset (traceID TEXT, 
                                    network TEXT,
                                    station TEXT,
                                    instrument_type TEXT,
                                    stlat NUMERIC, 
                                    stlon NUMERIC, 
                                    stelv NUMERIC,                        
                                    event_start_time DateTime, 
                                    event_end_time DateTime, 
                                    detection_prob NUMERIC, 
                                    detection_unc NUMERIC, 
                                    p_arrival_time DateTime, 
                                    p_prob NUMERIC, 
                                    p_unc NUMERIC, 
                                    p_snr NUMERIC,
                                    s_arrival_time DateTime, 
                                    s_prob NUMERIC, 
                                    s_unc NUMERIC, 
                                    s_snr NUMERIC,
                                    amp NUMERIC
                                    )''')
    if platform.system() == 'Windows':
        station_list = [ev for ev in listdir(input_dir) if ev.split("\\")[-1] != ".DS_Store"];
    else:
        station_list = [ev for ev in listdir(input_dir) if ev.split("/")[-1] != ".DS_Store"];
        
    station_list = sorted(set(station_list))

    for st in station_list:       
        print(f'reading {st} ...')
        if platform.system() == 'Windows':
            file_path = os.path.join(input_dir, st, "X_prediction_results.csv")
            _pick_database_maker(conn, cur, file_path)
        else:
            file_path = os.path.join(input_dir, st, "X_prediction_results.csv")
            _pick_database_maker(conn, cur, file_path)

    #  read the database as dataframe 
    conn = sqlite3.connect("phase_dataset")
    tbl = pd.read_sql_query("SELECT * FROM phase_dataset", conn); 
    #tbl = tbl[tbl.p_prob > 0.3]
    #tbl = tbl[tbl.s_prob > 0.3]

    tbl['event_start_time'] = tbl['event_start_time'].apply(lambda row : _date_convertor(row)) 
    tbl['event_end_time'] = tbl['event_end_time'].apply(lambda row : _date_convertor(row)) 
    tbl['p_arrival_time'] = tbl['p_arrival_time'].apply(lambda row : _date_convertor(row)) 
    tbl['s_arrival_time'] = tbl['s_arrival_time'].apply(lambda row : _date_convertor(row)) 

    _dbs_associator(start_time,
                    end_time,
                    moving_window,
                    tbl, 
                    pair_n,
                    output_dir,
                    station_list,
                    consider_combination)
    
    # os.remove("phase_dataset")

def _pick_database_maker(conn, cur, input_file):
    csv_file = open(input_file)
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if not row:
            continue
        if line_count == 0:
            #  print(f'Column names are {", ".join(row)}')
            line_count += 1
        else: 
            line_count += 1
            
            traceID = row[0]
                
            network = row[1]
            station = row[2]
            instrument_type = row[3]
            stlat = float(row[4])
            stlon = float(row[5]) 
            stelv = float(row[6])                        
            
            mls = row[7].split('.')
            if len(mls) == 1:
                event_start_time = datetime.strptime(row[7], '%Y-%m-%d %H:%M:%S')
            else:
                event_start_time = datetime.strptime(row[7], '%Y-%m-%d %H:%M:%S.%f')

            mls = row[8].split('.')
            if len(mls) == 1:
                event_end_time = datetime.strptime(row[8], '%Y-%m-%d %H:%M:%S')
            else:
                event_end_time = datetime.strptime(row[8], '%Y-%m-%d %H:%M:%S.%f')
                            
            detection_prob = float(row[9]) 
            try:
                detection_unc = float(row[10]) 
            except Exception:
                detection_unc = None          

            if len(row[11]) > 10:
           #     p_arrival_time = UTCDateTime(row[11].replace(' ', 'T')+'Z')
                mls = row[11].split('.')
                if len(mls) == 1:
                    p_arrival_time = datetime.strptime(row[11], '%Y-%m-%d %H:%M:%S')
                else:
                    p_arrival_time = datetime.strptime(row[11], '%Y-%m-%d %H:%M:%S.%f')
                                    
                p_prob = float(row[12])
                try:
                    p_unc = float(row[13]) 
                except Exception:
                    p_unc = None
            else:
                p_arrival_time = None
                p_prob = None
                p_unc = None 
                
            try:
                p_snr = float(row[14]) 
            except Exception:
                p_snr = None 
                                
            if len(row[15]) > 10:
                mls = row[15].split('.')
                if len(mls) == 1:
                    s_arrival_time = datetime.strptime(row[15], '%Y-%m-%d %H:%M:%S')
                else:
                    s_arrival_time = datetime.strptime(row[15], '%Y-%m-%d %H:%M:%S.%f')                
                                
                s_prob = float(row[16])
                try:
                    s_unc = float(row[17]) 
                except Exception:
                    s_unc = None
            else:
                s_arrival_time = None
                s_prob = None
                s_unc = None   

            try:
                s_snr = float(row[18]) 
            except Exception:
                s_snr = None
                
            amp = None

            cur.execute('''INSERT INTO phase_dataset VALUES 
                        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, ?)''', 
                        (traceID, network, station, instrument_type, stlat, stlon, stelv,
                         event_start_time, event_end_time, detection_prob, detection_unc, 
                         p_arrival_time, p_prob, p_unc, p_snr, s_arrival_time, s_prob, s_unc, s_snr,
                         amp))
            
            conn.commit()




def _decimalDegrees2DMS(value,type):
    
    'Converts a Decimal Degree Value into Degrees Minute Seconds Notation. Pass value as double type = {Latitude or Longitude} as string returns a string as D:M:S:Direction created by: anothergisblog.blogspot.com' 

    degrees = int(value)
    submin = abs( (value - int(value) ) * 60)
    direction = ""
    if type == "Longitude":
        if degrees < 0:
            direction = "W"
        elif degrees > 0:
            direction = " "
        else:
            direction = ""
        notation = ["{:>3}".format(str(abs(degrees))), direction, "{:>5}".format(str(round(submin, 2)))] 

    elif type == "Latitude":
        if degrees < 0:
            direction = "S"
        elif degrees > 0:
            direction = " "
        else:
            direction = "" 
        notation =["{:>2}".format(str(abs(degrees))), direction, "{:>5}".format(str(round(submin, 2)))] 
        
    return notation



def _weighcalculator_prob(pr):
    'calculate the picks weights'
    weight = 4
    if pr > 0.6:
        weight = 0
    elif pr <= 0.6 and pr > 0.5:
        weight = 1
    elif pr <= 0.5 and pr > 0.2:
        weight = 2
    elif pr <= 0.2 and pr > 0.1:
        weight = 3  
    elif pr <= 0.1:
        weight = 4 
         
    return weight



def _date_convertor(r): 
    'convert datatime form string'
    if r and len(r)>5:
        mls = r.split('.')
        if len(mls) == 1:
            new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S')
        else:
            new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S.%f')
        return new_t
            


def _doubleChecking(station_list, detections, preprocessed_dir, moving_window, thr_on=3.7, thr_of=0.5):
    'this function perform traditional detection (STA/LTA) and picker (AIC) to double check for events on the remaining stations when an event has been detected on more than two stations'
    for stt in station_list:
        sttt = stt.split('_')[0]
      #  print(sttt)
        if sttt not in detections['station'].to_list():
            new_picks = {}                    
            if platform.system() == 'Windows':
                file_name = preprocessed_dir+"\\"+sttt+".hdf5"
                file_csv = preprocessed_dir+"\\"+sttt+".csv"
            else:
                file_name = preprocessed_dir+"/"+sttt+".hdf5"
                file_csv = preprocessed_dir+"/"+sttt+".csv"
            
            df = pd.read_csv(file_csv)
            df['start_time'] = pd.to_datetime(df['start_time'])  
            
            mask = (df['start_time'] > detections.iloc[0]['event_start_time']-timedelta(seconds = moving_window)) & (df['start_time'] < detections.iloc[0]['event_start_time']+timedelta(seconds = moving_window))
            df = df.loc[mask]
            dtfl = h5py.File(file_name, 'r')
            dataset = dtfl.get('data/'+df['trace_name'].to_list()[0]) 
            data = np.array(dataset)
                
            cft = recursive_sta_lta(data[:,2], int(2.5 * 100), int(10. * 100))
            on_of = trigger_onset(cft, thr_on, thr_of)
            if len(on_of) >= 1:                    
                p_pick, s_pick = ar_pick(data[:,2], data[:,1], data[:,0], 100, 1.0, 20.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2)
                if (on_of[0][1]+100)/100 > p_pick > (on_of[0][0]-100)/100: 
                   # print('got one')
                    new_picks['traceID'] = df['trace_name'].to_list()[0]
                    new_picks['network'] = dataset.attrs["network_code"]
                    new_picks['station'] = sttt
                    new_picks['instrument_type'] = df['trace_name'].to_list()[0].split('_')[2]
                    new_picks['stlat'] = round(dataset.attrs["receiver_latitude"], 4)
                    new_picks['stlon'] = round(dataset.attrs["receiver_longitude"], 4)
                    new_picks['stelv'] = round(dataset.attrs["receiver_elevation_m"], 2)
                    new_picks['event_start_time'] = datetime.strptime(str(UTCDateTime(dataset.attrs['trace_start_time'].replace(' ', 'T')+'Z')+(on_of[0][0]/100)).replace('T', ' ').replace('Z', ''), '%Y-%m-%d %H:%M:%S.%f')
                    new_picks['event_end_time'] = datetime.strptime(str(UTCDateTime(dataset.attrs['trace_start_time'].replace(' ', 'T')+'Z')+(on_of[0][1]/100)).replace('T', ' ').replace('Z', ''), '%Y-%m-%d %H:%M:%S.%f')
                    new_picks['detection_prob'] = 0.3
                    new_picks['detection_unc'] = 0.6
                    new_picks['p_arrival_time'] = datetime.strptime(str(UTCDateTime(dataset.attrs['trace_start_time'].replace(' ', 'T')+'Z')+p_pick).replace('T', ' ').replace('Z', ''), '%Y-%m-%d %H:%M:%S.%f')
                    new_picks['p_prob'] = 0.3
                    new_picks['p_unc'] = 0.6
                    new_picks['p_snr'] = None
                    new_picks['s_arrival_time'] = None
                    new_picks['s_prob'] = 0.0
                    new_picks['s_unc'] = None
                    new_picks['s_snr'] = None
                    new_picks['amp'] = None
                    detections = detections.append(new_picks , ignore_index=True)      
    return detections                    
                            
                            

def _dbs_associator(start_time, end_time, moving_window, 
                    tbl, pair_n, save_dir, station_list,
                    consider_combination=False):  
    
    if consider_combination==True: 
        if platform.system() == 'Windows':
            Y2000_writer = open(save_dir+"\\"+"Y2000.phs", "w")
        else:
            Y2000_writer = open(save_dir+"/"+"Y2000.phs", "w")
            
        traceNmae_dic = dict()   
        st = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
        et = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S.%f')
        total_t = et-st;
        evid = 0; 
        tt = st
        pbar = tqdm(total= int(np.ceil(total_t.total_seconds()/moving_window)), ncols=100) 
        while tt < et:
            
            detections = tbl[(tbl.event_start_time >= tt) & (tbl.event_start_time < tt+timedelta(seconds = moving_window))]        
            pbar.update()
            if len(detections) >= pair_n:  
                evid += 1
    
                yr = "{:>4}".format(str(detections.iloc[0]['event_start_time']).split(' ')[0].split('-')[0])
                mo = "{:>2}".format(str(detections.iloc[0]['event_start_time']).split(' ')[0].split('-')[1]) 
                dy = "{:>2}".format(str(detections.iloc[0]['event_start_time']).split(' ')[0].split('-')[2]) 
                hr = "{:>2}".format(str(detections.iloc[0]['event_start_time']).split(' ')[1].split(':')[0]) 
                mi = "{:>2}".format(str(detections.iloc[0]['event_start_time']).split(' ')[1].split(':')[1]) 
                sec = "{:>4}".format(str(detections.iloc[0]['event_start_time']).split(' ')[1].split(':')[2]) 
                st_lat_DMS = _decimalDegrees2DMS(float(detections.iloc[0]['stlat']),  "Latitude")
                st_lon_DMS = _decimalDegrees2DMS(float(detections.iloc[0]['stlon']),  "Longitude")
                depth = 5.0
                mag = 0.0

                # QuakeML
                print(detections.iloc[0]['event_start_time'])
                      
                if len(detections)/pair_n <= 2:
                    ch = pair_n
                else:
                    ch = int(len(detections)-pair_n)
                  
                picks = []            
                for ns in range(ch, len(detections)+1):
                    comb = 0
                    for ind in list(combinations(detections.index, ns)):
                        comb+=1
                        selected_detections = detections.loc[ind,:]
                        sorted_detections = selected_detections.sort_values('p_arrival_time')
    
                        Y2000_writer.write("%4d%2d%2d%2d%2d%4.2f%2.0f%1s%4.2f%3.0f%1s%4.2f%5.2f%3.2f\n"%
                                           (int(yr),int(mo),int(dy), int(hr),int(mi),float(sec),float(st_lat_DMS[0]), 
                                            str(st_lat_DMS[1]), float(st_lat_DMS[2]),float(st_lon_DMS[0]), str(st_lon_DMS[1]), 
                                            float(st_lon_DMS[2]),float(depth), float(mag))); 
                                
                        station_buffer=[]; row_buffer=[]; tr_names=[]; tr_names2=[]
                        for _, row in sorted_detections.iterrows():
                            
                            trace_name = row['traceID']+'*'+row['station']+'*'+str(row['event_start_time'])
                            p_unc = row['p_unc']
                            p_prob = row['p_prob']
                            s_unc = row['s_unc']
                            s_prob = row['s_prob']
            
                                
                            station = "{:<5}".format(row['station'])
                            network = "{:<2}".format(row['network']) 
                            try:
                                yrp = "{:>4}".format(str(row['p_arrival_time']).split(' ')[0].split('-')[0])
                                mop = "{:>2}".format(str(row['p_arrival_time']).split(' ')[0].split('-')[1]) 
                                dyp = "{:>2}".format(str(row['p_arrival_time']).split(' ')[0].split('-')[2]) 
                                hrp = "{:>2}".format(str(row['p_arrival_time']).split(' ')[1].split(':')[0]) 
                                mip = "{:>2}".format(str(row['p_arrival_time']).split(' ')[1].split(':')[1]) 
                                sec_p = "{:>4}".format(str(row['p_arrival_time']).split(' ')[1].split(':')[2]) 
                                p = Pick(time=UTCDateTime(row['p_arrival_time']), 
                                         waveform_id=WaveformStreamID(network_code=network,
                                         station_code=station.rstrip()),
                                         phase_hint="P")
                                picks.append(p)
                                
                                if p_unc:                    
                                    Pweihgt = _weighcalculator_prob(p_prob*(1-p_unc))
                                else:
                                    Pweihgt = _weighcalculator_prob(p_prob)
                                try:
                                    Pweihgt = int(Pweihgt)
                                except Exception:
                                    Pweihgt = 4
                                
                            except Exception:
                                sec_p = None

                            try:
                                yrs = "{:>4}".format(str(row['s_arrival_time']).split(' ')[0].split('-')[0])
                                mos = "{:>2}".format(str(row['s_arrival_time']).split(' ')[0].split('-')[1]) 
                                dys = "{:>2}".format(str(row['s_arrival_time']).split(' ')[0].split('-')[2]) 
                                hrs = "{:>2}".format(str(row['s_arrival_time']).split(' ')[1].split(':')[0]) 
                                mis = "{:>2}".format(str(row['s_arrival_time']).split(' ')[1].split(':')[1])                                 
                                sec_s = "{:>4}".format(str(row['s_arrival_time']).split(' ')[1].split(':')[2]) 
                                p = Pick(time=UTCDateTime(row['p_arrival_time']), 
                                         waveform_id=WaveformStreamID(network_code=network, station_code=station.rstrip()),
                                         phase_hint="S")
                                picks.append(p)
                                
                                if s_unc: 
                                    Sweihgt = _weighcalculator_prob(s_prob*(1-s_unc))
                                else:
                                    Sweihgt = _weighcalculator_prob(s_prob)  
                                try:
                                    Sweihgt = int(Sweihgt)
                                except Exception:
                                    Sweihgt = 4
                                
                            except Exception:
                                sec_s = None
                            
                            if row['station'] not in station_buffer:
                                tr_names.append(trace_name)
                                station_buffer.append(row['station'])                      
                                if sec_s:
                                    Y2000_writer.write("%5s%2s  HHE     %4d%2d%2d%2d%2d%5.2f       %5.2fES %1d\n"%
                                                       (station,network,int(yrs),int(mos),int(dys),int(hrs),int(mis),
                                                        float(0.0),float(sec_s), Sweihgt))
                                if sec_p:
                                    Y2000_writer.write("%5s%2s  HHZ IP %1d%4d%2d%2d%2d%2d%5.2f       %5.2f   0\n"%
                                                       (station,network,Pweihgt,int(yrp),int(mop),int(dyp),int(hrp),
                                                        int(mip),float(sec_p),float(0.0)))                        
                            else :
                                tr_names2.append(trace_name)
                                if sec_s:
                                    row_buffer.append("%5s%2s  HHE     %4d%2d%2d%2d%2d%5.2f       %5.2fES %1d\n"%(station,network,
                                                                                                                 int(yrs),int(mos),int(dys),
                                                                                                                 int(hrs),int(mis),0.0,
                                                                                                                 float(sec_s), Sweihgt)); 
                                if sec_p:
                                    row_buffer.append("%5s%2s  HHZ IP %1d%4d%2d%2d%2d%2d%5.2f       %5.2f   0\n"%(station,network,
                                                                                                                 Pweihgt,   
                                                                                                                 int(yrp),int(mop),int(dyp),
                                                                                                                 int(hrp),int(mip),float(sec_p),
                                                                                                                 float(0.0)));                                 
                        Y2000_writer.write("{:<62}".format(' ')+"%10d"%(evid)+'\n');
    
                traceNmae_dic[str(evid)] = tr_names
    
                if len(row_buffer) >= 2*pair_n: 
                    Y2000_writer.write("%4d%2d%2d%2d%2d%4.2f%2.0f%1s%4.2f%3.0f%1s%4.2f%5.2f%3.2f\n"%
                                       (int(yr),int(mo),int(dy),int(hr),int(mi),float(sec), 
                                        float(st_lat_DMS[0]), str(st_lat_DMS[1]), float(st_lat_DMS[2]),
                                        float(st_lon_DMS[0]), str(st_lon_DMS[1]), float(st_lon_DMS[2]),
                                        float(depth), float(mag)));   
                    for rr in row_buffer:
                        Y2000_writer.write(rr);
                 
                    Y2000_writer.write("{:<62}".format(' ')+"%10d"%(evid)+'\n');
                    traceNmae_dic[str(evid)] = tr_names2
                
    
            tt += timedelta(seconds= moving_window)
        
     #   plt.scatter(LTTP, TTP, s=10, marker='o', c='b', alpha=0.4, label='P')
     #   plt.scatter(LTTS, TTS, s=10, marker='o', c='r', alpha=0.4, label='S')
     #   plt.legend('upper right')
     #   plt.show()
    
        print('The Number of Realizations: '+str(evid)+'\n', flush=True)
            
        jj = json.dumps(traceNmae_dic) 
        if platform.system() == 'Windows':
            f = open(save_dir+"\\"+"traceNmae_dic.json","w")
        else:
            f = open(save_dir+"/"+"traceNmae_dic.json","w")
        f.write(jj)
        f.close()

    else:  
        if platform.system() == 'Windows':
            Y2000_writer = open(save_dir+"\\"+"Y2000.phs", "w")
        else:
            Y2000_writer = open(save_dir+"/"+"Y2000.phs", "w")
            
        cat = Catalog()
        traceNmae_dic = dict()    
        st = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
        et = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S.%f')
        total_t = et-st;
        evid = 200000;  evidd = 100000
        tt = st
        pbar = tqdm(total= int(np.ceil(total_t.total_seconds()/moving_window))) 
        while tt < et:
            
            detections = tbl[(tbl.event_start_time >= tt) & (tbl.event_start_time < tt+timedelta(seconds = moving_window))]        
            pbar.update()
            if len(detections) >= pair_n:   
    
                yr = "{:>4}".format(str(detections.iloc[0]['event_start_time']).split(' ')[0].split('-')[0])
                mo = "{:>2}".format(str(detections.iloc[0]['event_start_time']).split(' ')[0].split('-')[1]) 
                dy = "{:>2}".format(str(detections.iloc[0]['event_start_time']).split(' ')[0].split('-')[2]) 
                hr = "{:>2}".format(str(detections.iloc[0]['event_start_time']).split(' ')[1].split(':')[0]) 
                mi = "{:>2}".format(str(detections.iloc[0]['event_start_time']).split(' ')[1].split(':')[1]) 
                sec = "{:>4}".format(str(detections.iloc[0]['event_start_time']).split(' ')[1].split(':')[2]) 
                st_lat_DMS = _decimalDegrees2DMS(float(detections.iloc[0]['stlat']),  "Latitude")
                st_lon_DMS = _decimalDegrees2DMS(float(detections.iloc[0]['stlon']),  "Longitude")
                depth = 5.0
                mag = 0.0
    
                Y2000_writer.write("%4d%2d%2d%2d%2d%4.2f%2.0f%1s%4.2f%3.0f%1s%4.2f%5.2f%3.2f\n"%(int(yr),int(mo),int(dy),
                                                                                             int(hr),int(mi),float(sec), 
                                                                                             float(st_lat_DMS[0]), str(st_lat_DMS[1]), float(st_lat_DMS[2]),
                                                                                             float(st_lon_DMS[0]), str(st_lon_DMS[1]), float(st_lon_DMS[2]),
                                                                                             float(depth), float(mag)));          
                event = Event()
                origin = Origin(time=UTCDateTime(detections.iloc[0]['event_start_time']),
                                longitude=detections.iloc[0]['stlon'],
                                latitude=detections.iloc[0]['stlat'],
                                method="EqTransformer")
                event.origins.append(origin)
                
                station_buffer = []
                row_buffer = []
                sorted_detections = detections.sort_values('p_arrival_time')
                tr_names = []
                tr_names2 = []
                picks = []
                for _, row in sorted_detections.iterrows():
                    trace_name = row['traceID']+'*'+row['station']+'*'+str(row['event_start_time'])
                    p_unc = row['p_unc']
                    p_prob = row['p_prob']
                    s_unc = row['s_unc']
                    s_prob = row['s_prob']
    
                        
                    station = "{:<5}".format(row['station'])
                    network = "{:<2}".format(row['network'])
                    
                    try:
                        yrp = "{:>4}".format(str(row['p_arrival_time']).split(' ')[0].split('-')[0])
                        mop = "{:>2}".format(str(row['p_arrival_time']).split(' ')[0].split('-')[1]) 
                        dyp = "{:>2}".format(str(row['p_arrival_time']).split(' ')[0].split('-')[2]) 
                        hrp = "{:>2}".format(str(row['p_arrival_time']).split(' ')[1].split(':')[0]) 
                        mip = "{:>2}".format(str(row['p_arrival_time']).split(' ')[1].split(':')[1]) 
                        sec_p = "{:>4}".format(str(row['p_arrival_time']).split(' ')[1].split(':')[2]) 
                        p = Pick(time=UTCDateTime(row['p_arrival_time']), 
                                     waveform_id=WaveformStreamID(network_code=network, station_code=station.rstrip()),
                                     phase_hint="P", method_id="EqTransformer")
                        picks.append(p)
                        
                        if p_unc:                    
                            Pweihgt = _weighcalculator_prob(p_prob*(1-p_unc))
                        else:
                            Pweihgt = _weighcalculator_prob(p_prob)
                        try:
                            Pweihgt = int(Pweihgt)
                        except Exception:
                            Pweihgt = 4
                        
                    except Exception:
                        sec_p = None

                    try:
                        yrs = "{:>4}".format(str(row['s_arrival_time']).split(' ')[0].split('-')[0])
                        mos = "{:>2}".format(str(row['s_arrival_time']).split(' ')[0].split('-')[1]) 
                        dys = "{:>2}".format(str(row['s_arrival_time']).split(' ')[0].split('-')[2]) 
                        hrs = "{:>2}".format(str(row['s_arrival_time']).split(' ')[1].split(':')[0]) 
                        mis = "{:>2}".format(str(row['s_arrival_time']).split(' ')[1].split(':')[1])                                 
                        sec_s = "{:>4}".format(str(row['s_arrival_time']).split(' ')[1].split(':')[2]) 
                        p = Pick(time=UTCDateTime(row['s_arrival_time']), 
                                     waveform_id=WaveformStreamID(network_code=network, station_code=station.rstrip()),
                                     phase_hint="S", method_id="EqTransformer")
                        picks.append(p)
                        
                        if s_unc: 
                            Sweihgt = _weighcalculator_prob(s_prob*(1-s_unc))
                        else:
                            Sweihgt = _weighcalculator_prob(s_prob)  
                        try:
                            Sweihgt = int(Sweihgt)
                        except Exception:
                            Sweihgt = 4
                        
                    except Exception:
                        sec_s = None                              
    
                    if row['station'] not in station_buffer:
                        tr_names.append(trace_name)
                        station_buffer.append(row['station'])  
                        if sec_s:
                            Y2000_writer.write("%5s%2s  HHE     %4d%2d%2d%2d%2d%5.2f       %5.2fES %1d\n"%(station,network,
                                                                                                         int(yrs),int(mos),int(dys),
                                                                                                         int(hrs),int(mis),float(0.0),
                                                                                                         float(sec_s), Sweihgt))
                        if sec_p:
                            Y2000_writer.write("%5s%2s  HHZ IP %1d%4d%2d%2d%2d%2d%5.2f       %5.2f   0\n"%(station,network,
                                                                                                         Pweihgt,   
                                                                                                         int(yrp),int(mop),int(dyp),
                                                                                                         int(hrp),int(mip),float(sec_p),
                                                                                                         float(0.0)))                        
                    else :
                        tr_names2.append(trace_name)
                        if sec_s:
                            row_buffer.append("%5s%2s  HHE     %4d%2d%2d%2d%2d%5.2f       %5.2fES %1d\n"%(station,network,
                                                                                                         int(yrs),int(mos),int(dys),
                                                                                                         int(hrs),int(mis),0.0,
                                                                                                         float(sec_s), Sweihgt)); 
                        if sec_p:
                            row_buffer.append("%5s%2s  HHZ IP %1d%4d%2d%2d%2d%2d%5.2f       %5.2f   0\n"%(station,network,
                                                                                                         Pweihgt,   
                                                                                                         int(yrp),int(mop),int(dyp),
                                                                                                         int(hrp),int(mip),float(sec_p),
                                                                                                         float(0.0))); 
                event.picks = picks
                event.preferred_origin_id = event.origins[0].resource_id
                cat.append(event)
    
                evid += 1
                Y2000_writer.write("{:<62}".format(' ')+"%10d"%(evid)+'\n');
                traceNmae_dic[str(evid)] = tr_names
    
                if len(row_buffer) >= 2*pair_n: 
                    Y2000_writer.write("%4d%2d%2d%2d%2d%4.2f%2.0f%1s%4.2f%3.0f%1s%4.2f%5.2f%3.2f\n"%
                                       (int(yr),int(mo),int(dy),int(hr),int(mi),float(sec), 
                                        float(st_lat_DMS[0]), str(st_lat_DMS[1]), float(st_lat_DMS[2]),
                                        float(st_lon_DMS[0]), str(st_lon_DMS[1]), float(st_lon_DMS[2]),
                                        float(depth), float(mag)));   
                    for rr in row_buffer:
                        Y2000_writer.write(rr);
                 
                    evid += 1
                    Y2000_writer.write("{:<62}".format(' ')+"%10d"%(evid)+'\n');
                    traceNmae_dic[str(evid)] = tr_names2
                    
                elif len(row_buffer) < pair_n and len(row_buffer) != 0:
                    evidd += 1
                    traceNmae_dic[str(evidd)] = tr_names2
                    
            elif len(detections) < pair_n and len(detections) != 0:
                tr_names = []
                for _, row in detections.iterrows():
                    trace_name = row['traceID']
                    tr_names.append(trace_name)
                evidd += 1
                traceNmae_dic[str(evidd)] = tr_names                
    
            tt += timedelta(seconds= moving_window)
            
        print('The Number of Associated Events: '+str(evid-200000)+'\n', flush=True)
            
        jj = json.dumps(traceNmae_dic)  
        if platform.system() == 'Windows':
            f = open(save_dir+"\\"+"traceNmae_dic.json","w")
        else:    
            f = open(save_dir+"/"+"traceNmae_dic.json","w")
            
        f.write(jj)
        f.close()
        print(cat.__str__(print_all=True))
        cat.write(save_dir+"/associations.xml", format="QUAKEML")


# ---------------------------------------------------------------------------
# v2: p_arrival_time grouping + overlapping sliding window + deduplication
# Fixes the tumbling-window alignment problem where the same earthquake is
# detected on different stations but their event_start_times land in different
# 60-second bins, causing the pair_n threshold to never be met.
# ---------------------------------------------------------------------------

def run_associator_v2(input_dir, start_time, end_time,
                      moving_window=60, pair_n=3, output_dir='.',
                      consider_combination=False):
    """
    v2 of run_associator.
    Groups detections by p_arrival_time (not event_start_time) and uses a
    sliding window with step = moving_window // 2 to eliminate bin-boundary
    misses.  Duplicate candidates from overlapping windows are merged before
    writing output.
    """
    if os.path.exists("phase_dataset"):
        os.remove("phase_dataset")
    conn = sqlite3.connect("phase_dataset")
    cur = conn.cursor()
    cur.execute('''CREATE TABLE phase_dataset (
                        traceID TEXT, network TEXT, station TEXT, instrument_type TEXT,
                        stlat NUMERIC, stlon NUMERIC, stelv NUMERIC,
                        event_start_time DateTime, event_end_time DateTime,
                        detection_prob NUMERIC, detection_unc NUMERIC,
                        p_arrival_time DateTime, p_prob NUMERIC, p_unc NUMERIC, p_snr NUMERIC,
                        s_arrival_time DateTime, s_prob NUMERIC, s_unc NUMERIC, s_snr NUMERIC,
                        amp NUMERIC)''')

    if platform.system() == 'Windows':
        station_list = sorted({ev for ev in listdir(input_dir) if ev.split("\\")[-1] != ".DS_Store"})
    else:
        station_list = sorted({ev for ev in listdir(input_dir) if ev.split("/")[-1] != ".DS_Store"})

    for st in station_list:
        print(f'reading {st} ...')
        _pick_database_maker(conn, cur, os.path.join(input_dir, st, "X_prediction_results.csv"))

    conn = sqlite3.connect("phase_dataset")
    tbl = pd.read_sql_query("SELECT * FROM phase_dataset", conn)
    tbl['event_start_time'] = tbl['event_start_time'].apply(_date_convertor)
    tbl['event_end_time']   = tbl['event_end_time'].apply(_date_convertor)
    tbl['p_arrival_time']   = tbl['p_arrival_time'].apply(_date_convertor)
    tbl['s_arrival_time']   = tbl['s_arrival_time'].apply(_date_convertor)

    _dbs_associator_v2(start_time, end_time, moving_window, tbl, pair_n,
                       output_dir, station_list, consider_combination)


def _coherent_subset(detections, pair_n, vp=6.0, tolerance=15.0):
    """
    Find the largest coherent subset of stations from the association.

    A subset is coherent if for every pair (A, B) in it:
        |p_arrival_A - p_arrival_B| <= dist(A, B) / Vp + tolerance

    Strategy: start with all stations; iteratively remove the station whose
    P-pick causes the most constraint violations until the remaining set is
    fully coherent or fewer than pair_n stations remain.

    Returns the coherent subset DataFrame, or None if fewer than pair_n
    coherent stations can be found.

    Parameters
    ----------
    vp : float
        P-wave velocity in km/s.
    tolerance : float
        Slack in seconds: accounts for EQT pick error and velocity model
        uncertainty.  15 s covers most cases; raise to 25 s if pick quality
        is poor (e.g. stations with known data issues).
    """
    rows = detections.reset_index(drop=True).copy()

    def _pair_ok(r1, r2):
        dlat = np.radians(float(r2['stlat']) - float(r1['stlat']))
        dlon = np.radians(float(r2['stlon']) - float(r1['stlon']))
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(float(r1['stlat']))) *
             np.cos(np.radians(float(r2['stlat']))) *
             np.sin(dlon / 2) ** 2)
        dist_km = 6371.0 * 2 * np.arcsin(np.sqrt(max(0.0, float(a))))
        max_allowed = dist_km / vp + tolerance
        actual_diff = abs((r1['p_arrival_time'] - r2['p_arrival_time']).total_seconds())
        return actual_diff <= max_allowed

    while len(rows) >= pair_n:
        # Count violations per station
        violations = {idx: 0 for idx in rows.index}
        all_ok = True
        for i in rows.index:
            for j in rows.index:
                if j <= i:
                    continue
                if not _pair_ok(rows.loc[i], rows.loc[j]):
                    violations[i] += 1
                    violations[j] += 1
                    all_ok = False
        if all_ok:
            return rows  # fully coherent
        # Remove the station with the most violations (tie-break: lower detection_prob)
        worst = max(violations, key=lambda k: (violations[k], -rows.loc[k, 'detection_prob']))
        rows = rows.drop(index=worst).reset_index(drop=True)

    return None  # could not find pair_n coherent stations


def _dbs_associator_v2(start_time, end_time, moving_window,
                       tbl, pair_n, save_dir, station_list,
                       consider_combination=False):
    """
    Sliding-window associator that groups by p_arrival_time.

    Algorithm:
      1. Filter to rows with valid p_arrival_time.
      2. Slide a window of width moving_window with step = moving_window // 2.
         For each window keep the highest-probability detection per station.
         If unique_stations >= pair_n → candidate event.
      3. Deduplicate: merge consecutive candidates whose min p_arrival_times
         differ by <= step AND that share >= max(2, pair_n-1) stations.
      4. Write output identical to _dbs_associator (consider_combination=False).

    Origin time in XML = min(p_arrival_time) across confirmed stations — more
    accurate than event_start_time for downstream location and validation.
    """
    if consider_combination:
        raise ValueError(
            "_dbs_associator_v2 does not support consider_combination=True. "
            "Use _dbs_associator instead.")

    step = max(1, moving_window // 2)

    if platform.system() == 'Windows':
        Y2000_writer = open(save_dir + "\\" + "Y2000.phs", "w")
    else:
        Y2000_writer = open(save_dir + "/" + "Y2000.phs", "w")

    cat = Catalog()
    traceNmae_dic = dict()
    st_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    et_dt = datetime.strptime(end_time,   '%Y-%m-%d %H:%M:%S.%f')

    # ---- Phase 1: collect candidates with O(log N) sliding window ----
    valid_tbl = tbl[tbl['p_arrival_time'].notna()].sort_values('p_arrival_time').reset_index(drop=True)

    # Build a numeric offset array for fast searchsorted
    ref = st_dt
    p_offsets = np.array([(dt - ref).total_seconds() for dt in valid_tbl['p_arrival_time']])

    total_steps = int(np.ceil((et_dt - st_dt).total_seconds() / step))
    pbar = tqdm(total=total_steps, ncols=100)

    candidates = []  # list of (min_p_datetime, DataFrame)
    tt_offset = 0.0
    while tt_offset < (et_dt - ref).total_seconds():
        lo = int(np.searchsorted(p_offsets, tt_offset, side='left'))
        hi = int(np.searchsorted(p_offsets, tt_offset + moving_window, side='left'))
        pbar.update()
        if hi - lo >= pair_n:
            win = valid_tbl.iloc[lo:hi]
            best = (win.sort_values('detection_prob', ascending=False)
                       .drop_duplicates('station'))
            if len(best) >= pair_n:
                min_p = best['p_arrival_time'].min()
                candidates.append((min_p, best))
        tt_offset += step
    pbar.close()

    if not candidates:
        print('No associated events found.\n')
        Y2000_writer.close()
        cat.write(save_dir + "/associations.xml", format="QUAKEML")
        return

    candidates.sort(key=lambda x: x[0])

    # ---- Phase 2: deduplicate overlapping-window duplicates ----
    # Two candidates are the same event if they are within step seconds of
    # each other AND share at least max(2, pair_n-1) stations.
    min_shared = max(2, pair_n - 1)
    deduped = []
    for cand_t, cand_det in candidates:
        if deduped:
            prev_t, prev_det = deduped[-1]
            dt_sec = (cand_t - prev_t).total_seconds()
            shared = set(prev_det['station']) & set(cand_det['station'])
            if dt_sec <= step and len(shared) >= min_shared:
                # Same event: take union, best detection per station
                combined = (pd.concat([prev_det, cand_det])
                              .sort_values('detection_prob', ascending=False)
                              .drop_duplicates('station'))
                deduped[-1] = (prev_t, combined)
                continue
        deduped.append((cand_t, cand_det))

    # ---- Phase 3: coherence filter ----
    # For each candidate, find the largest physically-coherent subset of
    # stations (|ΔT_ij| <= dist(A,B)/Vp + tolerance for all pairs).
    # Stations whose P-picks violate physics are dropped from the association;
    # the event is kept as long as >= pair_n coherent stations remain.
    before_events = len(deduped)
    before_total_sta = sum(len(d) for _, d in deduped)
    filtered = []
    for ev_t, det in deduped:
        coherent = _coherent_subset(det, pair_n, vp=6.0, tolerance=15.0)
        if coherent is not None:
            filtered.append((ev_t, coherent))
    deduped = filtered
    after_total_sta = sum(len(d) for _, d in deduped)
    print(f'Coherence filter: {before_events} → {len(deduped)} events; '
          f'stations trimmed {before_total_sta} → {after_total_sta}')

    # ---- Phase 4: write output ----
    evid = 200000
    evidd = 100000

    for ev_time, detections in deduped:
        ev_str = str(ev_time)
        # ev_str is 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD HH:MM:SS.ffffff'
        date_part, time_part = ev_str.split(' ')
        yr  = "{:>4}".format(date_part.split('-')[0])
        mo  = "{:>2}".format(date_part.split('-')[1])
        dy  = "{:>2}".format(date_part.split('-')[2])
        hr  = "{:>2}".format(time_part.split(':')[0])
        mi  = "{:>2}".format(time_part.split(':')[1])
        sec = "{:>4}".format(time_part.split(':')[2])

        st_lat_DMS = _decimalDegrees2DMS(float(detections.iloc[0]['stlat']), "Latitude")
        st_lon_DMS = _decimalDegrees2DMS(float(detections.iloc[0]['stlon']), "Longitude")

        Y2000_writer.write(
            "%4d%2d%2d%2d%2d%4.2f%2.0f%1s%4.2f%3.0f%1s%4.2f%5.2f%3.2f\n" % (
                int(yr), int(mo), int(dy), int(hr), int(mi), float(sec),
                float(st_lat_DMS[0]), str(st_lat_DMS[1]), float(st_lat_DMS[2]),
                float(st_lon_DMS[0]), str(st_lon_DMS[1]), float(st_lon_DMS[2]),
                5.0, 0.0))

        event = Event()
        origin = Origin(
            time=UTCDateTime(ev_time),
            longitude=detections.iloc[0]['stlon'],
            latitude=detections.iloc[0]['stlat'],
            method="EqTransformer")
        event.origins.append(origin)

        station_buffer = []
        row_buffer = []
        tr_names = []
        tr_names2 = []
        picks = []

        for _, row in detections.sort_values('p_arrival_time').iterrows():
            trace_name = row['traceID'] + '*' + row['station'] + '*' + str(row['event_start_time'])
            p_unc   = row['p_unc']
            p_prob  = row['p_prob']
            s_unc   = row['s_unc']
            s_prob  = row['s_prob']
            station = "{:<5}".format(row['station'])
            network = "{:<2}".format(row['network'])

            sec_p = None
            try:
                rp = str(row['p_arrival_time'])
                dp, tp = rp.split(' ')
                yrp = "{:>4}".format(dp.split('-')[0])
                mop = "{:>2}".format(dp.split('-')[1])
                dyp = "{:>2}".format(dp.split('-')[2])
                hrp = "{:>2}".format(tp.split(':')[0])
                mip = "{:>2}".format(tp.split(':')[1])
                sec_p = "{:>4}".format(tp.split(':')[2])
                picks.append(Pick(
                    time=UTCDateTime(row['p_arrival_time']),
                    waveform_id=WaveformStreamID(network_code=network, station_code=station.rstrip()),
                    phase_hint="P", method_id="EqTransformer"))
                Pweihgt = _weighcalculator_prob(p_prob * (1 - p_unc) if p_unc else p_prob)
                try:
                    Pweihgt = int(Pweihgt)
                except Exception:
                    Pweihgt = 4
            except Exception:
                sec_p = None

            sec_s = None
            try:
                rs = str(row['s_arrival_time'])
                ds, ts = rs.split(' ')
                yrs = "{:>4}".format(ds.split('-')[0])
                mos = "{:>2}".format(ds.split('-')[1])
                dys = "{:>2}".format(ds.split('-')[2])
                hrs = "{:>2}".format(ts.split(':')[0])
                mis = "{:>2}".format(ts.split(':')[1])
                sec_s = "{:>4}".format(ts.split(':')[2])
                picks.append(Pick(
                    time=UTCDateTime(row['s_arrival_time']),
                    waveform_id=WaveformStreamID(network_code=network, station_code=station.rstrip()),
                    phase_hint="S", method_id="EqTransformer"))
                Sweihgt = _weighcalculator_prob(s_prob * (1 - s_unc) if s_unc else s_prob)
                try:
                    Sweihgt = int(Sweihgt)
                except Exception:
                    Sweihgt = 4
            except Exception:
                sec_s = None

            if row['station'] not in station_buffer:
                tr_names.append(trace_name)
                station_buffer.append(row['station'])
                if sec_s:
                    Y2000_writer.write(
                        "%5s%2s  HHE     %4d%2d%2d%2d%2d%5.2f       %5.2fES %1d\n" % (
                            station, network, int(yrs), int(mos), int(dys),
                            int(hrs), int(mis), 0.0, float(sec_s), Sweihgt))
                if sec_p:
                    Y2000_writer.write(
                        "%5s%2s  HHZ IP %1d%4d%2d%2d%2d%2d%5.2f       %5.2f   0\n" % (
                            station, network, Pweihgt,
                            int(yrp), int(mop), int(dyp), int(hrp), int(mip),
                            float(sec_p), 0.0))
            else:
                tr_names2.append(trace_name)
                if sec_s:
                    row_buffer.append(
                        "%5s%2s  HHE     %4d%2d%2d%2d%2d%5.2f       %5.2fES %1d\n" % (
                            station, network, int(yrs), int(mos), int(dys),
                            int(hrs), int(mis), 0.0, float(sec_s), Sweihgt))
                if sec_p:
                    row_buffer.append(
                        "%5s%2s  HHZ IP %1d%4d%2d%2d%2d%2d%5.2f       %5.2f   0\n" % (
                            station, network, Pweihgt,
                            int(yrp), int(mop), int(dyp), int(hrp), int(mip),
                            float(sec_p), 0.0))

        event.picks = picks
        event.preferred_origin_id = event.origins[0].resource_id
        cat.append(event)

        evid += 1
        Y2000_writer.write("{:<62}".format(' ') + "%10d" % evid + '\n')
        traceNmae_dic[str(evid)] = tr_names

        if len(row_buffer) >= 2 * pair_n:
            Y2000_writer.write(
                "%4d%2d%2d%2d%2d%4.2f%2.0f%1s%4.2f%3.0f%1s%4.2f%5.2f%3.2f\n" % (
                    int(yr), int(mo), int(dy), int(hr), int(mi), float(sec),
                    float(st_lat_DMS[0]), str(st_lat_DMS[1]), float(st_lat_DMS[2]),
                    float(st_lon_DMS[0]), str(st_lon_DMS[1]), float(st_lon_DMS[2]),
                    5.0, 0.0))
            for rr in row_buffer:
                Y2000_writer.write(rr)
            evid += 1
            Y2000_writer.write("{:<62}".format(' ') + "%10d" % evid + '\n')
            traceNmae_dic[str(evid)] = tr_names2
        elif 0 < len(row_buffer) < pair_n:
            evidd += 1
            traceNmae_dic[str(evidd)] = tr_names2

    Y2000_writer.close()
    print('The Number of Associated Events (v2): ' + str(evid - 200000) + '\n', flush=True)

    jj = json.dumps(traceNmae_dic)
    if platform.system() == 'Windows':
        f = open(save_dir + "\\" + "traceNmae_dic.json", "w")
    else:
        f = open(save_dir + "/" + "traceNmae_dic.json", "w")
    f.write(jj)
    f.close()
    print(cat.__str__(print_all=True))
    cat.write(save_dir + "/associations.xml", format="QUAKEML")

