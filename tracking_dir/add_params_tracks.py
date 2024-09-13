from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA

import datetime
from datetime import timedelta

import scipy as sp

import seaborn as sns

from shapely.geometry import Polygon
import matplotlib.patches as mpatches
from matplotlib import gridspec

from tqdm import tqdm

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from mpl_toolkits.axes_grid1 import make_axes_locatable

import cmaps

from sklearn.cluster import DBSCAN


from pathlib import Path

import sys
import os
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/storage/kubrick/vkoshkina/scripts/vortex_identification')
sys.path.insert(1, './vortex_identification')

from vortex_dir import vortex_processing as vortex
from vortex_dir import show_vortex as show_vx
from vortex_dir import compute_criteria as compute

tracking_type = 'with_CS_speed'
tracking_type = 'track_len_real'
tracking_type = 'track_domain_boundary'



print('circ: ')
circ = input()


cols = ['t', 'datetime' ,'x','y','lat','lon','rad','track_len']

path_data_tracks = f'/storage/kubrick/vkoshkina/data/LoRes_tracks/{tracking_type}'


our_level = 12

months = np.arange(1,13,1)
days = np.arange(1,32,1)

def compute_param_for_track(t, y_in, y_out, x_in, x_out, our_level, param, value_type='mean'):
        
    if value_type == 'max':
        value = np.nanmax(ds[param][t, our_level, 
                              y_in:y_out, x_in:x_out])
    else:
        value = np.nanmedian(ds[param][t, our_level, 
                              y_in:y_out, x_in:x_out])
    
    return value

def open_ds(ls):
    if len(ls) != 0:
        for ii,ifile in enumerate(ls):

            ds = xr.open_dataset(ifile)
        return ds
    
    
def compute_mean_value(ds, t, y_in, y_out, x_in, x_out, our_level):
    if our_level == 0:
        value = np.nanmedian(ds[t,
                          y_in:y_out, x_in:x_out])
    else:
        value = np.nanmedian(ds[t, our_level,
                          y_in:y_out, x_in:x_out])
    return value


def compute_max_value(ds, t, y_in, y_out, x_in, x_out, our_level):
    if our_level == 0:
        value = np.nanmax(ds[t,
                          y_in:y_out, x_in:x_out])
    else:
        value = np.nanmax(ds[t, our_level,
                          y_in:y_out, x_in:x_out])
    return value

def compute_max_R_value(ds, t, y_in, y_out, x_in, x_out, our_level):
    value = np.nanmax(np.abs(ds[t, our_level, y_in:y_out, x_in:x_out]))
    
    if circ == 'AC':
        value = -value
 
    return value


def get_max_wspd(ds, t, y_in, y_out, x_in, x_out):

    u = ds['U10'][t,
                          y_in:y_out, x_in:x_out]
    
    v = ds['V10'][t,
                          y_in:y_out, x_in:x_out]
    
    wspd = np.sqrt(u*u + v*v)
    wspd = np.nanmax(wspd)
    
    return wspd


# def get_mean_delta_theta(ds, start_h, y_in, y_out, x_in, x_out):

#     fin_value_up = compute_mean_value(ds['T'], start_h, y_in, y_out, x_in, x_out, our_level=12)+290
#     fin_value_dwn = compute_mean_value(ds['TH2'], start_h, y_in, y_out, x_in, x_out, our_level=0)
#     fin_delta = fin_value_up - fin_value_dwn
    
#     return fin_delta


def get_mean_delta_theta(ds, t, y_in, y_out, x_in, x_out):

    our_level=12
    
    T_up = ds['T'][t, our_level,
                          y_in:y_out, x_in:x_out] + 290
    
    T_dowm = ds['TH2'][t,
                          y_in:y_out, x_in:x_out]
    
    delta = T_dowm - T_up
    fin_delta = np.nanmedian(delta)
    
    return fin_delta


def get_bounds(ds, x, y, hw):

    y_int = int(y+5)
    x_int = int(x+5) # поправка на обрезание границ
    
    y_in = y_int - hw
    x_in = x_int - hw
    
    y_out = y_int + hw
    x_out = x_int + hw
    
    if y_in < 0:
        y_in = 0
    if x_in < 0:
        x_in = 0
            
    if y_out >= len(ds.west_east):
        y_out = -1
    if x_out >= len(ds.west_east):
        x_out = -1
    return y_in, y_out, x_in, x_out


def add_params(df, path_dir):
    grads = []
    TC_wspd = []
    TC_sst = []
    TC_hf = []
    TC_lf = []
    
#     TC_t2 = []
    TC_th2 = []
    TC_W = []
    
    
    max_R = []

    
    params = ['TH2', 'SST', 'HFX', 'LH', 'W']
    
    
    for i in range(len(df)):
        date = df['datetime'].dt.date[i]
        
        start_h = int(df.hour_idx.values[i])

        x = df.x.values[i]
        y = df.y.values[i]
        hw = int(2*np.round(df.rad.values[i]))
        

        name = f'wrfout_d01_{date}*'

        ls = list(sorted(Path(f"{path_dir}").glob(f'{name}')))
        ds = xr.open_dataset(ls[0])
        
        y_in, y_out, x_in, x_out = get_bounds(ds, x, y, hw)
        
        grad_T = get_mean_delta_theta(ds, start_h, y_in, y_out, x_in, x_out)
        wspd = get_max_wspd(ds, start_h, y_in, y_out, x_in, x_out)
        
            
        th2 = compute_mean_value(ds['TH2'], start_h, y_in, y_out, x_in, x_out, our_level=0)
#         t2 = compute_mean_value(ds['T2'], start_h, y_in, y_out, x_in, x_out, our_level=0)
        sst = compute_mean_value(ds['SST'], start_h, y_in, y_out, x_in, x_out, our_level=0)
        heat = compute_mean_value(ds['HFX'], start_h, y_in, y_out, x_in, x_out, our_level=0)
        latent = compute_mean_value(ds['LH'], start_h, y_in, y_out, x_in, x_out, our_level=0)
        
        W = compute_mean_value(ds['W'], start_h, y_in, y_out, x_in, x_out, our_level=12)
        
        
        
        ####### add RORTEX ####### 
        folder = 'rortex_LoRes_for_tracking'
        path_dir_data = '/storage/kubrick/vkoshkina/data'
        ds = xr.open_dataset(f'{path_dir_data}/{folder}/rortex_criteria_LoRes_{year}.nc')

        ds_t = ds['R_2d'][ds.XTIME == df['datetime'][i]]
        
        x_int = int(x)
        y_int = int(y)
        
        y_in = y_int - hw
        x_in = x_int - hw

        y_out = y_int + hw
        x_out = x_int + hw

        if y_in < 0:
            y_in = 0
        if x_in < 0:
            x_in = 0

        if y_out >= len(ds.west_east):
            y_out = -1
        if x_out >= len(ds.west_east):
            x_out = -1
        
        R = compute_max_R_value(ds_t, 0, y_in, y_out, x_in, x_out, our_level=12) # 500 hPa
        max_R.append(R)
        ####### add RORTEX ####### 
        
        
        
        grads.append(grad_T)
        TC_wspd.append(wspd)
        
#         TC_t2.append(t2)
        TC_th2.append(th2)
        
        TC_sst.append(sst)
        TC_hf.append(heat)
        TC_lf.append(latent) 
        
        TC_W.append(W)  
        
        
        
    df['dT_med'] = grads
    df['wspd10_max'] = TC_wspd
    df['SST_med'] = TC_sst
    df['HFX_med'] = TC_hf
    df['LH_med'] = TC_lf
#     df['T2'] = TC_t2
    df['TH2_med'] = TC_th2
    df['W_med'] = TC_W
    df['R_max'] = max_R
    
    return df



years = np.arange(1979,2019)

i = 0

for year in years:

    print(f'year: {year}')
    
    for month in tqdm(months):
        for day in tqdm(days):
            path_data = f'{path_data_tracks}/{year}/tracks_{circ}'
            path_dir_raw = f'/storage/NAD/NAAD/LoRes/{year}'
            
#             path_data_dir = f'{path_data_tracks}/{year}/tracks_{circ}'
#             path_data = f'{path_data_dir}/{year:04d}-{month:02d}-{day:02d}'

            if os.path.exists(path_data):
                ls = list(sorted(Path(f"{path_data}/").glob(f'*_track_{year:04d}-{month:02d}-{day:02d}*00.csv')))
    #             print(ls)


                if len(ls) != 0:
                    for ii,ifile in enumerate(ls):
                        df = pd.read_csv(ifile, usecols=cols)
                        df['datetime'] = pd.to_datetime(df['datetime'].values)
                        df['hour_idx'] = (df['datetime'].dt.hour/3).values.astype(int)

                        df = add_params(df, path_dir_raw)


                        if len(df['datetime']) > 3:

                            path_data_new = f'{path_data_tracks}/{year}/tracks_{circ}_params'
                            
                            if not os.path.exists(f'{path_data_new}'):
                                os.makedirs(f'{path_data_new}')

                            date_start = str(df['datetime'].values[0])[:-16]
                            df.to_csv(f'{path_data_new}/{i:08d}_track_{date_start}.csv')
                            i+=1

