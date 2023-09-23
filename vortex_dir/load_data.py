import pandas as pd 
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA

import time

from tqdm import tqdm

import datetime
from datetime import timedelta

from tqdm import tqdm



def get_season(year, month, day):
    if month == 1:
        season = 'january'
    elif month == 2:
        season = 'february'
    elif month == 3:
        season = 'march'
    elif month == 4:
        season = 'april'
    elif month == 5:
        season = 'may'
    elif month == 6:
        season = 'june'
    elif month == 7:
        season = 'july'
    elif month == 8:
        season = 'august'
    elif month == 9:
        season = 'september'
    elif month == 10:
        season = 'october'
    elif month == 11:
        season = 'november'
    elif month == 12:
        season = 'december'
    else:
        print('ошибка')
        season = 'year'
    print(f'date: {day}/{season}/{year}')
    return season


# level = int(input('Enter level: '))

# month = int(input('Enter month number: '))

# param = 5 # indent from the horizontal boundaries of the domain


start = time.time() 


# open several days in a row, return dataset and horizontal grid spacing
def open_full_dataset_nc(path_dir, name, step):
    ds = xr.open_dataset(f"{path_dir}/{name}.nc")
    dist_m = 15000 # 15 km ?
    return ds, dist_m

# open period with given time step at 12:00 UTC, return dataset and horizontal grid spacing
def open_step_dataset_nc(path_dir, name, step):
    ds = xr.open_dataset(f"{path_dir}/{name}.nc")
    ds_mini = ds.isel(time=(ds.time.dt.month == 8))
    ds_mini = ds_mini.isel(time=(ds_mini.time.dt.day > 23))
    dist_m = 13897.18
    return ds_mini, dist_m

# open several days in a row, return dataset and horizontal grid spacing
def open_full_dataset_NAAD(path_dir, year, month, day, step, period, param):
        
    date = datetime.date(year, month, day)
    geopotential = xr.open_dataset(path_dir + f'geopotential/{year}/geopotential_' + str(date) + '.nc')
    
    dist_m = geopotential.attrs['DX']
    time = geopotential.time # попытки спасти время
    
    geopotential = geopotential['geopotential'][:,:,param:-param,param:-param]
    
    geopotential = geopotential.assign_coords({"XTIME": time}) #######
    
    
    ue = xr.open_dataset(path_dir + f'ue/{year}/ue_' + str(date) + '.nc').ue[:,:,param:-param,param:-param]
    ve = xr.open_dataset(path_dir + f've/{year}/ve_' + str(date) + '.nc').ve[:,:,param:-param,param:-param]
    w = xr.open_dataset(path_dir + f'w/{year}/w_' + str(date) + '.nc').w[:,:,param:-param,param:-param]

    curr_date = date
    day = 1
    
    pbar = tqdm(desc='load days: ', total=period)
    
    while (day < period):
        
        curr_date = curr_date + timedelta(days=step) 
        
        geopotential_1 = xr.open_dataset(path_dir + f'geopotential/{year}/geopotential_'+ str(curr_date) + '.nc').geopotential[:,:,param:-param,param:-param]
        ue_1 = xr.open_dataset(path_dir + f'ue/{year}/ue_' + str(curr_date) + ".nc").ue[:,:,param:-param,param:-param]
        ve_1 = xr.open_dataset(path_dir + f've/{year}/ve_' + str(curr_date) + ".nc").ve[:,:,param:-param,param:-param]
        w_1 = xr.open_dataset(path_dir + f'w/{year}/w_' + str(curr_date) + ".nc").w[:,:,param:-param,param:-param]

        time_1 = xr.open_dataset(path_dir + 'geopotential/' + str(curr_date.year) + '/geopotential_'+ str(curr_date) + '.nc').time
        geopotential_1 = geopotential_1.assign_coords({"XTIME": time_1}) #######

        geopotential = xr.concat([geopotential, geopotential_1], 'Time')
        ue = xr.concat([ue, ue_1], 'Time')
        ve = xr.concat([ve, ve_1], 'Time')
        w = xr.concat([w, w_1], 'Time')
        
        del geopotential_1, ue_1, ve_1, w_1
        day = (curr_date - date).days
        pbar.update(1)
    pbar.close()
    
    ds = xr.merge([ue, ve, w, geopotential])
    
    del ue, ve, w, geopotential
    
    return ds, dist_m

# open period with given time step at 12:00 UTC, return dataset and horizontal grid spacing
def open_step_dataset_NAAD(path_dir, year, month, day, step, param, period, crit):
    
    date = datetime.date(year, month, day)
    
    geopotential = xr.open_dataset(path_dir + 'geopotential/' + str(year) + '/geopotential_' + str(date) + '.nc')
    
    time = geopotential.time[4:5] # попытки спасти время
    
    dist_m = geopotential.attrs['DX']

    # XTIME[4] = 12:00
    geopotential = geopotential['geopotential'][4:5,:,param:-param,param:-param]
    
    geopotential = geopotential.assign_coords({"XTIME": time}) #######

    ue = xr.open_dataset(path_dir + 'ue/' + str(year) + '/ue_' + str(date) + '.nc').ue[4:5,:,param:-param,param:-param]
    ve = xr.open_dataset(path_dir + 've/' + str(year) + '/ve_' + str(date) + '.nc').ve[4:5,:,param:-param,param:-param]
    w = xr.open_dataset(path_dir + 'w/' + str(year) + '/w_' + str(date) + '.nc').w[4:5,:,param:-param,param:-param]

    curr_date = date
    
    pbar = tqdm(desc='load days: ', total=period)
    day = 1
    
    while (day < period):

        curr_date = curr_date + timedelta(days=step)  
        
        geopotential_1 = xr.open_dataset(path_dir + 'geopotential/' + str(curr_date.year) + '/geopotential_'+ str(curr_date) + '.nc').geopotential[4:5,:,param:-param,param:-param]
        ue_1 = xr.open_dataset(path_dir + 'ue/' + str(curr_date.year) + "/ue_" + str(curr_date) + ".nc").ue[4:5,:,param:-param,param:-param]
        ve_1 = xr.open_dataset(path_dir + 've/' + str(curr_date.year) + "/ve_" + str(curr_date) + ".nc").ve[4:5,:,param:-param,param:-param]
        w_1 = xr.open_dataset(path_dir + 'w/' + str(curr_date.year) + "/w_" + str(curr_date) + ".nc").w[4:5,:,param:-param,param:-param]
        

        time_1 = xr.open_dataset(path_dir + 'geopotential/' + str(curr_date.year) + '/geopotential_'+ str(curr_date) + '.nc').time[4:5]
        geopotential_1 = geopotential_1.assign_coords({"XTIME": time_1}) #######

        geopotential = xr.concat([geopotential, geopotential_1], 'Time')
        ue = xr.concat([ue, ue_1], 'Time')
        ve = xr.concat([ve, ve_1], 'Time')
        w = xr.concat([w, w_1], 'Time')
        
        del geopotential_1, ue_1, ve_1, w_1
        
        pbar.update(1)
        
        day = (curr_date - date).days
    
    pbar.close()
    
    ds = xr.merge([ue, ve, w, geopotential])
    
    del ue, ve, w, geopotential
    
    return ds, dist_m
