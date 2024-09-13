from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA

# from matplotlib import animation
import datetime
from datetime import timedelta

import scipy as sp
from scipy.ndimage import label, generate_binary_structure
from scipy import interpolate

import seaborn as sns

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

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

sys.path.insert(2, '/storage/kubrick/vkoshkina/scripts/DBSCAN_tracking/tracking_lib')

# from simple_tracking_scripts import *
# from tracking_with_CS_speed import *

from tracking_domain_boundary import *


tracking_type = 'track_domain_boundary'

name = 'rortex_criteria_LoRes'
path_dir = '/storage/kubrick/vkoshkina/data/rortex_LoRes_for_tracking'

dist_m = 77824.23
our_level = 12


min_samples = 8
eps = 2

CS_points_th = 15

th_Q = 0.
crit = 'R_2d'

x_unit = 'west_east'
y_unit = 'south_north'

u_unit = 'ue'
v_unit = 've'

print('circ: ')
circ = input()

years = np.arange(1979,2019)

for year in years:
    
    print(f'year: {year}')
    
    ds = xr.open_dataset(f"{path_dir}/{name}_{year}.nc")

    path_data_dir = f'/storage/kubrick/vkoshkina/data/LoRes_tracks/{tracking_type}/{year}/tracks_{circ}'
    
    if not os.path.exists(f'{path_data_dir}'):
        os.makedirs(f'{path_data_dir}')

    CS_tracks_list = simple_advection_tracking(ds, 0, circ=circ)

    for TC in tqdm(CS_tracks_list):

        df_TC_Danielle = pd.DataFrame({
                             "t": TC['t'][:-1],
                             "datetime": TC['time'][:-1], 
                             "x": TC['x'][:-1],
                             "y": TC['y'][:-1],
                             "lat": TC['lat'][:-1], 
                             "lon": TC['lon'][:-1], 
                             "rad": TC['rad'][:-1], 
                             "track_len": TC['track_len'][:-1],
    #                          "wspd": TC_wspd,
        })

        start_date = TC['time'][0]
        dist = TC['cluster']
        dt = str(start_date)[:-13]

#         if not os.path.exists(f"{path_data_dir}"):
#             os.makedirs(f"{path_data_dir}")

        df_TC_Danielle.to_csv(f'{path_data_dir}/{int(dist):06d}_track_{dt}.csv')