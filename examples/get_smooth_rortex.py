import pandas as pd 
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA

from tqdm import tqdm
import time
import datetime
from datetime import timedelta

from scipy.ndimage import gaussian_filter


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
import sys
import os

# sys.path.insert(3, f'{path_init}/scripts')

from compute_DBSCAN_func import *
from compute_rortex_func import *


years = np.arange(1979,2019)
# # years = np.arange(2000,2010)
years = np.arange(2019,2020)



data_type = 'LoRes'
data_type = 'HiRes'
data_type = 'GPN'
data_type = 'SMP'



path_dir = f'/storage/OPENDATA/NAAD/{data_type}/PressureLevels/'


path_init = f'/storage/thalassa/users/vkoshkina'
path_dir_data = f'{path_init}/data'

if data_type == 'HiRes':
    dist_m = 13897.18
    params = ['ue', 've']
    config_name = 'config_NAAD'
    
    level = 8 # 850 hPa
    level = 12 # 500 hPa
    
elif data_type == 'LoRes'
    dist_m = 77824.23
    path_dir_data = f'{path_dir_data}/LoRes'
    params = ['ue', 've']
    config_name = 'config_NAAD'
    level = 8 # 850 hPa
    level = 12 # 500 hPa

elif data_type == 'GPN':    
    dist_m = 6000.0
    path_dir = f'/storage/buffer/GPN/OUTPUTS/WRF6km/'
    years = np.arange(2022,2023)
    params = ['ua', 'va', 'geopotential', 'HGT']
    config_name = 'config_GPN'
    level = 12 # 1700 m

elif data_type == 'SMP':    
    dist_m = 6000.0
    path_dir = f'/storage/buffer/SMP/OUTPUTS/WRF6km/'
    years = np.arange(2019,2020)
    params = ['ua', 'va', 'geopotential', 'HGT']
    config_name = 'config_GPN'
    level = 10 # 1700 m

    

with open(f'{config_name}.json', 'r') as file:
    config = json.load(file)
    
data = config

# Распаковка данных из JSON файла и присвоение переменным
# path_dir = data["path_dir"]
# data_type = data["data_type"]
# dist_m = data["dist_m"]
eps = data["eps"]
min_samples = data["min_samples"]
size_filter = data["size_filter"]
min_dist = data["min_dist"]

name_crit = data["name_crit"]
level_name = data["level_name"]
x_name = data["x_name"]
y_name = data["y_name"]
time_name = data["time_name"]
time_unit = data["time_unit"]

def get_R2D_nc(u_smooth, v_smooth):
    
    R_2d = get_R2D_ds(u_smooth, v_smooth, dist_m)
    
    #### создание датасета со скоростями ветра на одном уровне высоты ####
    ds = u.to_dataset(name = 'u')
    
    ds[params[0]] = ({time_name: len(ds[time_name]), 
                      level_name: len(ds[level_name]), 
                      y_name: len(ds[y_name]), x_name: len(ds[x_name])}, u_smooth)
    ds[params[1]] = ({time_name: len(ds[time_name]), 
                      level_name: len(ds[level_name]), 
                      y_name: len(ds[y_name]), x_name: len(ds[x_name])}, v_smooth)
    ds[name_crit] = ({time_name: len(ds[time_name]), 
                  level_name: len(ds[level_name]), 
                  y_name: len(ds[y_name]), x_name: len(ds[x_name])}, R_2d.astype(np.float32))
    del ds['u']
 
    return ds

def unification(ds):

    ds[crit_name].attrs['description'] = f'Rortex criterion 2D (at {ds[level_name].values[0]} hPa level)'
    ds[crit_name].attrs['long_name'] = 'Rortex 2D'

    ds.attrs = {}
        
    return ds


# open several days in a row, return dataset and horizontal grid spacing
def open_full_dataset_NAAD_param(path_dir, year, month, value):
    
    ls = list(sorted(Path(f"{path_dir}/").glob(f'{value}/{year}/{value}_{year}-{month:02d}*')))
    # print(ls)
    
    for ii,ifile in tqdm(enumerate(ls)):
        if ii == 0:
            if value == 'geopotential':
                ds = xr.open_dataset(ifile)[value]
                time = xr.open_dataset(ifile).time
                ds = ds.assign_coords({"XTIME": time})
            else:
                ds = xr.open_dataset(ifile)[value]
        else:
            if value == 'geopotential':
                ds_1 = xr.open_dataset(ifile)[value]
                time = xr.open_dataset(ifile).time
                ds_1 = ds_1.assign_coords({"XTIME": time})
            else:
                ds_1 = xr.open_dataset(ifile)[value]
                
            ds = xr.concat([ds, ds_1], time_name)  
            del ds_1
            
    return ds

# open several days in a row, return dataset and horizontal grid spacing
def open_full_dataset_NAAD_param_level(path_dir, year, month, value, level):
    
    ls = list(sorted(Path(f"{path_dir}/").glob(f'{value}/{year}/{value}_{year}-{month:02d}*')))
    
    for ii,ifile in tqdm(enumerate(ls)):
        if ii == 0:
            if value == 'geopotential':
                ds = xr.open_dataset(ifile)[value][:,level:level+1]
                time = xr.open_dataset(ifile).time
                ds = ds.assign_coords({"XTIME": time})
            else:
                ds = xr.open_dataset(ifile)[value][:,level:level+1]
        else:
            if value == 'geopotential':
                ds_1 = xr.open_dataset(ifile)[value][:,level:level+1]
                time = xr.open_dataset(ifile).time
                ds_1 = ds_1.assign_coords({"XTIME": time})
            else:
                ds_1 = xr.open_dataset(ifile)[value][:,level:level+1]
                
            ds = xr.concat([ds, ds_1], time_name)  
            del ds_1
            
    return ds




def add_param(ds, wrfnc, param):
        
    u = wrf.getvar(wrfnc, param, timeidx=wrf.ALL_TIMES)[:,level:level+1]

    ds[param] = (('Time', 'bottom_top', 'south_north', 'west_east'), u.values) 
    ds[param].attrs['description'] = u.attrs['description']
    ds[param].attrs['units'] = u.attrs['units']
    
    return ds
    
start = time.time() ## точка отсчета времени


for year in tqdm(years):
    # for month in tqdm(np.arange(1,13)):
    for month in tqdm(np.arange(8,10)):
        
        
        print(f'\n open {year}-{month:02d}')
    
        if level == 'all':
            ds = open_full_dataset_NAAD_param(path_dir, year, month, 'geopotential')
            print(f'computing at all levels...')
            
        else:
            ds = open_full_dataset_NAAD_param_level(path_dir, year, month, 'geopotential', level)
            print(f'computing at {ds[level_name].values[0]} hPa level...')
        

        for value in params:
#             print(f'\n open {year}-{month:02d} for {value}')
            
            if level == 'all':
                ds_1 = open_full_dataset_NAAD_param(path_dir, year, month, value)
            else:
                ds_1 = open_full_dataset_NAAD_param_level(path_dir, year, month, value, level)
                
            ds = xr.merge([ds, ds_1])


        print(f'ds.dims: {ds.dims}')
        
        ##### определение компонент скорости (на уровне высоты) ######
        #### ВАЖНО: код кушает скорости 4д (т.е. уровень высоты - отдельный dim, не редуцирован) ####
        u = ds[params[0]]
        v = ds[params[1]]
        
        # Применение гауссового сглаживания с сигмой
        sigma = 2
        sigma_2d = (0, 0, sigma, sigma)
        
#         radius = 5
#         truncate = radius/sigma
        
#         u_smooth = gaussian_filter(u.values, sigma=sigma_2d, truncate=truncate)
#         v_smooth = gaussian_filter(v.values, sigma=sigma_2d, truncate=truncate)
        
        u_smooth = gaussian_filter(u.values, sigma=sigma_2d)
        v_smooth = gaussian_filter(v.values, sigma=sigma_2d)

        print(f'\n compute {name_crit}')
        ds_smooth = get_R2D_nc(u_smooth, v_smooth)
        
        print(f'\n compute DBSCAN')
        cluster_ds = get_DBSCAN_ds(ds_smooth, config)


        folder = f'{path_dir_data}/{data_type}/{name_crit}_{data_type}_level_{level}_smoothing/'
        if not os.path.exists(f'{folder}'):
            os.makedirs(f'{folder}')
      
        # собираем в файлик
        ds_smooth.to_netcdf(f'{folder}/sigma_{sigma}_{name_crit}_{data_type}_level_{level}_{year}-{month:02d}.nc', mode='w')
        del ds_smooth
        
        folder = f'{path_dir_data}/{data_type}/DBSCAN_{eps:02d}-{min_samples:02d}-{size_filter:02d}_level_{level}_smoothing'
        if not os.path.exists(folder):
            os.makedirs(f'{folder}')  
            
        cluster_ds.to_netcdf(f'{folder}/sigma_{sigma}_DBSCAN_{data_type}_level_{level}_{year}-{month:02d}.nc', mode='w')
        del cluster_ds 
        

        end = time.time() - start ## собственно время работы программы
        print(f'{end/60} min for year {year}') ## вывод времени
