
# from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA
# from scipy import signal

import time

from tqdm import tqdm

import datetime
from datetime import timedelta

# from scipy.ndimage import filters
from tqdm import tqdm

from vortex_dir.load_data import *
from vortex_dir.compute_criteria import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
import os

resolv = 'HiRes'


path_dir = f'/storage/NAADSERVER/NAAD/{resolv}/PressureLevels/'

years = np.arange(1980,2010)
years = np.arange(2000,2010)


# name = input('Alias for final file: ')
name = 'rortex_2d'

if resolv == 'HiRes':
    dist_m = 13897.18
else:    
    dist_m = 77824.23


start = time.time() ## точка отсчета времени

time_name = 'Time'

level = 12

params = ['ue', 've', 'w']
params = ['ue', 've']


# open several days in a row, return dataset and horizontal grid spacing
def open_full_dataset_NAAD_param(path_dir, year, month, value):
    
    ls = list(sorted(Path(f"{path_dir}/").glob(f'{value}/{year}/{value}_{year}-{month:02d}*')))
    
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

for year in tqdm(years):
    for month in tqdm(np.arange(1,13)):
        
        print(f'\n open {year}-{month:02d} for geopotential')
    
        if level == 'all':
            ds = open_full_dataset_NAAD_param(path_dir, year, month, 'geopotential')
        else:
            ds = open_full_dataset_NAAD_param_level(path_dir, year, month, 'geopotential', level)
        

        for value in params:
            print(f'\n open {year}-{month:02d} for {value}')
            
            if level == 'all':
                ds_1 = open_full_dataset_NAAD_param(path_dir, year, month, value)
            else:
                ds_1 = open_full_dataset_NAAD_param_level(path_dir, year, month, value, level)
                
            ds = xr.merge([ds, ds_1])


        print(f'ds.dims: {ds.dims}')
    #     print(ds.data_vars)

        g = 9.80665
        # шаг по вертикали в hPa
#         dz = np.abs(np.gradient(ds.geopotential, 1., axis = [1]))/g 


        grad_tensor_2d = compute_grad_tensor_2d(ds['ue'], ds['ve'], dist_m)
        
        omega_2d = compute_omega_2d(ds['ue'], ds['ve'], dist_m)


#         omega = compute_omega(ds.ue, ds.ve, ds.w, dist_m, dz)


        ################ 2d расчет swirling_strength и rortex ###################
        sw_str_2d = compute_swirling_strength_2d(grad_tensor_2d)

        # ds['sw_str_2d'] = ({'Time': len(ds.Time), 
        #                   'interp_level': len(ds.interp_level), 
        #                   'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, sw_str_2d)

#         R_2d = compute_rortex_2d(sw_str_2d, omega[:,:,:,:,2])
        R_2d = compute_rortex_2d(sw_str_2d, omega_2d)
        

        ds['R_2d'] = ({'Time': len(ds.Time), 
                          'interp_level': len(ds.interp_level), 
                          'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, R_2d)

        path_dir_data = '/storage/kubrick/vkoshkina/data'
        path_data_dir = f'{path_dir_data}/rortex_{resolv}_for_tracking_level_{level}/'

        if not os.path.exists(f'{path_data_dir}'):
            os.makedirs(f'{path_data_dir}')

        # собираем в файлик
        ds.to_netcdf(f'{path_data_dir}/{name}_criteria_{resolv}_level_{level}_{year}-{month:02d}.nc', mode='w')
        del ds

        end = time.time() - start ## собственно время работы программы
        print(f'{end/60} min for year {year}') ## вывод времени
