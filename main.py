
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


param = 5 # indent from the horizontal boundaries of the domain

path_dir = '/storage/NAADSERVER/NAAD/HiRes/PressureLevels/'


year = int(input('Enter year: '))

month = int(input('Enter month number: '))

day = int(input('Enter start day: '))

period = int(input('Enter period (in days): '))

step = int(input('Enter time step (in days) (step=-1 for for continuous sequence): '))

name = input('Alias for final file: ')

season = get_season(year, month, day)

start = time.time() ## точка отсчета времени

if step == -1:
    ds, dist_m = open_full_dataset_NAAD(path_dir, year, month, day, step=1, period=period, param=param) # подряд
else:
    ds, dist_m = open_step_dataset_NAAD(path_dir, year, month, day, step=step, param=param, period=period, crit='HiRes') # период с шагом

print(f'ds.shape: {ds.ue.shape}')

g = 9.80665
# шаг по вертикали в hPa
dz = np.abs(np.gradient(ds.geopotential, 1., axis = [1]))/g 


grad_tensor = compute_grad_tensor(ds['ue'], ds['ve'], ds['w'], dist_m, dz)

S, A = compute_S_A(grad_tensor)

Q = compute_Q(S, A)

ds['Q'] = ({'Time': len(ds.Time), 'interp_level': len(ds.interp_level), 
            'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, Q)

delta = compute_delta(grad_tensor, S, A)

ds['delta'] = ({'Time': len(ds.Time), 'interp_level': len(ds.interp_level), 
                'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, delta)

lambda2 = compute_lambda2(S, A)

ds['lambda2'] = ({'Time': len(ds.Time), 
                  'interp_level': len(ds.interp_level), 
                  'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, lambda2)


path_dir = '/storage/kubrick/vkoshkina/data'
# собираем в файлик
ds.to_netcdf(f'{path_dir}/{year}/{name}_{season}_criteria_HiRes.nc', mode='w')

end = time.time() - start ## собственно время работы программы

print(f'{end/60} min') ## вывод времени
