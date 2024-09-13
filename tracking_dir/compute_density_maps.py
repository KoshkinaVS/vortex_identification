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

year = 2010
# circ = 'AC'

print('circ: ')
circ = input()

params = ['HFX', 'LH']

path_data_dir = f'/storage/kubrick/vkoshkina/data/2010/LoRes_for_tracking/tracks'
path_data_new = f'/storage/kubrick/vkoshkina/data/2010/LoRes_for_tracking/tracks/tracks_{circ}_params_new'

path_data_dir_density = f'/storage/kubrick/vkoshkina/data/2010/LoRes_for_tracking/tracks/density'
if not os.path.exists(f'{path_data_dir_density}'):
    os.makedirs(f'{path_data_dir_density}')

months = np.arange(1,13,1)
days = np.arange(1,32,1)


def load_season_tracks(month_start, month_stop, path_data): 
    CS_tracks_list = []
    
    months = np.arange(month_start,month_stop+1,1)
    
    for month in tqdm(months):

        if os.path.exists(path_data):
        #     ls = list(sorted(Path(f"{path_data_new}/").glob(f'*_track_2010-{month:02d}-{day:02d}*new.csv')))
            ls = list(sorted(Path(f"{path_data}/").glob(f'*_track_2010-{month:02d}-*.csv')))

        #             print(ls)

            if len(ls) != 0:
                for ii,ifile in enumerate(ls):
                    df = pd.read_csv(ifile)
                    df = df.drop(df.columns[0], axis=1)
                    df = df.drop(df.columns[0], axis=1)
                    CS_tracks_list.append(df)
    return CS_tracks_list


def tracks_density_param(CS_tracks_list_summer):
    
    track_density_summer = np.zeros(shape=(100,100))
    track_density_summer_wspd = np.zeros(shape=(100,100))
    track_density_summer_temp = np.zeros(shape=(100,100))
    
    for CS in tqdm(CS_tracks_list_summer):
        for x in range(0,101):
            for y in range(0,101):  
                CS_point = CS[CS['x'] == x][CS['y'] == y]
                if not CS_point.empty:
                    track_density_summer_wspd[y,x] += CS_point[params[0]].values[0]
                    track_density_summer_temp[y,x] += CS_point[params[1]].values[0]
                    track_density_summer[y,x] += 1
                    
    return track_density_summer, track_density_summer_wspd, track_density_summer_temp

CS_tracks_list_winter = load_season_tracks(1, 3, path_data_new)
CS_tracks_list_summer = load_season_tracks(7, 9, path_data_new)

track_density_summer_list = tracks_density_param(CS_tracks_list_summer)

mean_wspd_summer = track_density_summer_list[1]/track_density_summer_list[0]
mean_temp_summer = track_density_summer_list[2]/track_density_summer_list[0]

np.save(f'{path_data_dir_density}/{circ}_track_density_summer_{params[0]}.npy', mean_wspd_summer)
np.save(f'{path_data_dir_density}/{circ}_track_density_summer_{params[1]}.npy', mean_temp_summer)
# np.save(f'{path_data_dir_density}/{circ}_track_density_summer_points.npy', track_density_summer_list[0])

track_density_winter_list = tracks_density_param(CS_tracks_list_winter)

mean_wspd_winter = track_density_winter_list[1]/track_density_winter_list[0]
mean_temp_winter = track_density_winter_list[2]/track_density_winter_list[0]

np.save(f'{path_data_dir_density}/{circ}_track_density_winter_{params[0]}.npy', mean_wspd_winter)
np.save(f'{path_data_dir_density}/{circ}_track_density_winter_{params[1]}.npy', mean_temp_winter)
# np.save(f'{path_data_dir_density}/{circ}_track_density_winter_points.npy', track_density_winter_list[0])
