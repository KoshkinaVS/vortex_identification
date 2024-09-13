import pandas as pd 
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA

import datetime
from datetime import timedelta

import scipy as sp

from shapely.geometry import Polygon

from tqdm import tqdm

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


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

# def get_max_wspd(ds, start_h, y_in, y_out, x_in, x_out):

#     u = compute_max_value(ds['U10'], start_h, y_in, y_out, x_in, x_out, our_level=12)
#     v = compute_max_value(ds['V10'], start_h, y_in, y_out, x_in, x_out, our_level=12)
#     wspd = np.sqrt(u*u + v*v)
    
#     return wspd


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


def add_params(df, path_dir, year, params):
    mean_p = []
    max_R = []
    
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
                
        for param in params:

            if param == 'R_2d':
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
                
            else:
                p = compute_mean_value(ds[param], start_h, y_in, y_out, x_in, x_out, our_level=12)
                mean_p.append(p)
        
        
    df['W_med'] = mean_p
    df['R_max'] = max_R
    
    return df


print('circ: ')
circ = input()


params = ['W', 'R_2d']

# cols = ['t', 'datetime', 'x', 'y','lat','lon','rad','track_len']

years = np.arange(1979,2019)
# years = np.arange(1979,1980)


for year in tqdm(years):
    i = 0

    path_data_dir = f'/storage/kubrick/vkoshkina/data/LoRes_tracks/{tracking_type}/{year}/tracks_{circ}_params'
    path_dir_raw = f'/storage/NAD/NAAD/LoRes/{year}'

    if os.path.exists(path_data_dir):
        ls = list(sorted(Path(f"{path_data_dir}/").glob(f'*_track_{year:04d}-*00.csv')))

        if len(ls) != 0:
            for ii,ifile in tqdm(enumerate(ls)):
                df = pd.read_csv(ifile)
                del df['Unnamed: 0']
                
                df['datetime'] = pd.to_datetime(df['datetime'].values)
#                 df['hour_idx'] = (df['datetime'].dt.hour/3).values.astype(int)

                df = add_params(df, path_dir_raw, year, params)

                date_start = df['datetime'].values[0]
                df.to_csv(f'{path_data_dir}/{i:05d}_track_{date_start}.csv')
                i+=1

