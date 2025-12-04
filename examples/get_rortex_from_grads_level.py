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

import calendar

from netCDF4 import Dataset
import wrf as wrf

# path_init = f'/storage/thalassa/users/vkoshkina/scripts/CS_processing/CVS_identification'

# sys.path.insert(3, f'{path_init}')

# from compute_DBSCAN_func import *
from compute_rortex_func import *




print('data type: ')
data_type = input() 


if data_type == 'GPN':
    path = f'/storage/buffer/GPN/OUTPUTS/WRF6km/'
    path_dir = f'/storage/thalassa/users/gavr/{data_type}/Coherents/tensor/'
    years = np.arange(2022,2023)
    month = 2
    params = ['ua', 'va', 'geopotential', 'HGT']
else:
    path = f'/storage/OPENDATA/NAAD/{data_type}/'
    path_dir = f'{path}/ModelLevels/tensor/'

    years = np.arange(2010,2011)
    month = 8
    params = ['ue', 've', 'geopotential', 'hgt']
    



dim_type = '2D'
# dim_type = '3D'




our_level = 22 # 5 km
our_level = 12 # 5 km


path_init = f'/storage/thalassa/users/vkoshkina'
path_dir_data = f'{path_init}/data/'



with open(f'config_NAAD.json', 'r') as file:
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

level_name = 'bottom_top'


x_name = data["x_name"]
y_name = data["y_name"]
time_name = data["time_name"]
time_unit = data["time_unit"]


dudx_name, dudy_name, dudz_name = 'dudx', 'dudy', 'dudz'
dvdx_name, dvdy_name, dvdz_name = 'dvdx', 'dvdy', 'dvdz'
dwdx_name, dwdy_name, dwdz_name = 'dwdx', 'dwdy', 'dwdz'



def get_rortex_nc(R_2d, u, name_crit, dim_type):
    """
    Создает xarray Dataset с критерием Rortex.
    
    Параметры:
        R_2d: np.ndarray или xarray.DataArray
            Массив с критерием Rortex
        u: xarray.DataArray
            Массив с одной из компонент скорости (для получения координат)
        name_crit: str
            Имя переменной для сохранения критерия
    """
    # Создаем Dataset из компоненты скорости
    ds = u.to_dataset(name='u')
    
    # Добавляем критерий Rortex
    ds[name_crit] = (
        (time_name, level_name, y_name, x_name),
        R_2d.astype(np.float32)
    )


    
    # # Удаляем временные переменные
    del ds['u']
    
    # Добавляем атрибуты
    ds[name_crit].attrs = {
        'description': f'Rortex criterion {dim_type}',
        'long_name': f'Rortex {dim_type}'
    }
    
    return ds


hgt_name = 'hgt'


if data_type == 'LoRes':
    hgt_file_path = f'{path}/Invariants/NAAD77km_hgt.nc'  # предполагаем, что файл лежит здесь

    if not os.path.exists(hgt_file_path):
        raise FileNotFoundError(f"HGT file not found: {hgt_file_path}")
    
    hgt = xr.open_dataset(hgt_file_path)
elif data_type == 'HiRes':
    hgt_file_path = f'{path}/Invariants/NAAD14km_hgt.nc'  # предполагаем, что файл лежит здесь
    
    if not os.path.exists(hgt_file_path):
        raise FileNotFoundError(f"HGT file not found: {hgt_file_path}")
    
    hgt = xr.open_dataset(hgt_file_path)
else:
    hgt_name = 'HGT'
    


start = time.time() ## точка отсчета времени

# Создаем общий прогресс-бар для месяцев
# monthly_progress = tqdm(range(1, 13), desc="Processing months", position=0)

monthly_progress = tqdm(range(2, 3), desc="Processing months", position=0)


for year in tqdm(years):
    for month in monthly_progress:
        monthly_progress.set_description(f"Processing {dim_type} {year}-{month:02d}")
        start_month = time.time()
        
        num_days = calendar.monthrange(year, month)[1]
        
        # Прогресс-бар для дней с обновлением информации
        daily_progress = tqdm(range(1, num_days + 1), 
                             desc=f"Days in {year}-{month:02d}", 
                             leave=False, 
                             position=1)
        
        for day in daily_progress:
            try:
                name = f'tensor_d01_{year}-{month:02d}-{day:02d}_00.nc'
                ds = xr.open_dataset(f"{path_dir}/{year}/{name}")

                if data_type == 'GPN':
                    raw_file = f"{path}/{year}/wrfout_d01_{year}-{month:02d}-{day:02d}_00:00:00")

                    wrfnc = Dataset(raw_file)
                    u = wrf.getvar(wrfnc, params[0], timeidx=wrf.ALL_TIMES)[:,our_level:our_level+1]
                    v = wrf.getvar(wrfnc, params[1], timeidx=wrf.ALL_TIMES)[:,our_level:our_level+1]
                else:
                    u = xr.open_dataset(f"{path}/ModelLevels/{params[0]}/{year}/{params[0]}_{year}-{month:02d}-{day:02d}.nc")
                    v = xr.open_dataset(f"{path}/ModelLevels/{params[1]}/{year}/{params[1]}_{year}-{month:02d}-{day:02d}.nc")
                
                
                
                # Ваши вычисления
                du_dx, du_dy, du_dz = ds[dudx_name][:,our_level:our_level+1], ds[dudy_name][:,our_level:our_level+1], ds[dudz_name][:,our_level:our_level+1]
                dv_dx, dv_dy, dv_dz = ds[dvdx_name][:,our_level:our_level+1], ds[dvdy_name][:,our_level:our_level+1], ds[dvdz_name][:,our_level:our_level+1]
                dw_dx, dw_dy, dw_dz = ds[dwdx_name][:,our_level:our_level+1], ds[dwdy_name][:,our_level:our_level+1], ds[dwdz_name][:,our_level:our_level+1]
    
                if dim_type == '2D':
                    name_crit = "R2D"
                    r2d = get_R2D_ds_from_grads(du_dx, du_dy, dv_dx, dv_dy)
                    
                elif dim_type == '3D':
                    name_crit = "R3D"
                    r2d = get_R3D_ds_from_grads(du_dx, du_dy, du_dz,
                                              dv_dx, dv_dy, dv_dz,
                                              dw_dx, dw_dy, dw_dz)
                
                rortex_ds = get_rortex_nc(r2d, ds[dudx_name][:,our_level:our_level+1], name_crit, dim_type)
                rortex_ds.attrs = ds.attrs.copy()
                
                current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                rortex_ds.attrs["CREATED"] = current_time
    
                rortex_ds[params[0]] = ((time_name, level_name, y_name, x_name), u[params[0]][:,our_level:our_level+1].values)
                rortex_ds[params[0]].attrs['description'] = u[params[0]].attrs['description']
                rortex_ds[params[0]].attrs['units'] = u[params[0]].attrs['units']
    
                rortex_ds[params[1]] = ((time_name, level_name, y_name, x_name), v[params[0]][:,our_level:our_level+1].values)
                rortex_ds[params[1]].attrs['description'] = v[params[0]].attrs['description']
                rortex_ds[params[1]].attrs['units'] = v[params[0]].attrs['units']
            
                rortex_ds['HGT'] = ((y_name, x_name), hgt['hgt'].values)
                rortex_ds['HGT'].attrs['description'] = hgt['hgt'].attrs['description']
                rortex_ds['HGT'].attrs['units'] = hgt['hgt'].attrs['units']
            
    
                folder = f'{path_dir_data}/{data_type}/rortex_from_grads/R2D_level_{our_level}/{year}'
                os.makedirs(folder, exist_ok=True)
                
                rortex_ds.to_netcdf(f'{folder}/{name_crit}_{data_type}_level_{our_level}_{year}-{month:02d}-{day:02d}.nc', mode='w')
                
                # Обновляем описание прогресс-бара
                daily_progress.set_postfix({
                    'Last processed': f'{year}-{month:02d}-{day:02d}',
                    'File': name
                })
                
                del rortex_ds, ds
                
            except Exception as e:
                tqdm.write(f"Error processing {year}-{month:02d}-{day:02d}: {str(e)}")
                continue
        
        # Вычисляем время выполнения месяца
        month_time = (time.time() - start_month)/60
        monthly_progress.set_postfix({
            'Month time (min)': f"{month_time:.2f}",
            'Last month': f"{year}-{month:02d}"
        })
        
        # Закрываем daily_progress чтобы избежать наложения
        daily_progress.close()