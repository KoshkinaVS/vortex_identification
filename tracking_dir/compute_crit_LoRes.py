
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

import sys
import os

sys.path.insert(1, '/storage/2TB/koshkina/scripts/vortex_identification')

from vortex_dir.load_data import *
from vortex_dir.compute_criteria import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


param = 5 # indent from the horizontal boundaries of the domain

# path_dir = '/storage/5TB/NAAD/LoRes/2010'

path_dir = '/storage/2TB/koshkina/data/NAAD/LoRes/'


init_name = 'NAAD_LoRes_2010_big'

name = input('Alias for final file: ')

start = time.time() ## точка отсчета времени

ds, dist_m = open_full_dataset_nc(path_dir, init_name, -1)

ds = ds.isel(y=slice(5,105), x=slice(5,105))

ds = ds.isel(Time=(ds.Time.dt.month > 9))
# ds = ds.isel(Time=(ds.Time.dt.month <= 9))



dist_m = 77824.23

print(f'ds.dims: {ds.dims}')

g = 9.80665
# шаг по вертикали в метрах
# dz = np.abs(np.gradient(ds.interp_level, 1., axis = [0]))*1000
dz = 500


grad_tensor = compute_grad_tensor(ds['ua'], ds['va'], ds['wa'], dist_m, dz)

grad_tensor_2d = compute_grad_tensor_2d(ds['ua'], ds['va'], dist_m)


############# расчет 3 базовых критериев ###################

# S, A = compute_S_A(grad_tensor)

# Q = compute_Q(S, A)

# ds['Q'] = ({'Time': len(ds.Time), 'interp_level': len(ds.interp_level), 
#             'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, Q)

# delta = compute_delta(grad_tensor, S, A)

# ds['delta'] = ({'Time': len(ds.Time), 'interp_level': len(ds.interp_level), 
#                 'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, delta)

# lambda2 = compute_lambda2(S, A)

# ds['lambda2'] = ({'Time': len(ds.Time), 
#                   'interp_level': len(ds.interp_level), 
#                   'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, lambda2)

##################################



############# расчет swirling_strength и rortex ###################

omega = compute_omega(ds['ua'], ds['va'], ds['wa'], dist_m, dz)

# sw_str, sw_vec_reoredered = compute_swirling_strength(grad_tensor)

# ds['sw_str'] = ({'Time': len(ds.Time), 
#                   'interp_level': len(ds.interp_level), 
#                   'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, sw_str)

# R = compute_rortex(sw_str, sw_vec_reoredered, omega)

# ds['R'] = ({'Time': len(ds.Time), 
#                   'interp_level': len(ds.interp_level), 
#                   'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, R)

################ 2d расчет swirling_strength и rortex ###################
sw_str_2d = compute_swirling_strength_2d(grad_tensor_2d)

ds['sw_str_2d'] = ({'Time': len(ds.Time), 
                  'interp_level': len(ds.interp_level), 
                  'y': len(ds.y), 'x': len(ds.x)}, sw_str_2d)

R_2d = compute_rortex_2d(sw_str_2d, omega[:,:,:,:,2])

ds['R_2d'] = ({'Time': len(ds.Time), 
                  'interp_level': len(ds.interp_level), 
                  'y': len(ds.y), 'x': len(ds.x)}, R_2d)

path_dir = '/storage/2TB/koshkina/data/NAAD/LoRes'
# собираем в файлик
ds.to_netcdf(f'{path_dir}/{name}_criteria_LoRes_2010.nc', mode='w')

end = time.time() - start ## собственно время работы программы

print(f'{end/60} min') ## вывод времени
