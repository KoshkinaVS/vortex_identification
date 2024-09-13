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
# from vortex_dir import show_vortex as show_vx
# from vortex_dir import compute_criteria as compute

sys.path.insert(2, '/storage/kubrick/vkoshkina/scripts/DBSCAN_tracking/tracking_lib')

# from simple_tracking_scripts import *
from tracking_with_CS_speed import *



name_ds = 'rortex_criteria_LoRes'

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

print('year: ')
year = int(input())

tracking_type = 'with_CS_speed'
# tracking_type = 'track_domain_boundary'

# path_data_tracks = f'/storage/kubrick/vkoshkina/data/LoRes_tracks/{tracking_type}/{year}/tracks_{circ}'

name_pics = f'tracks_anim_{tracking_type}'


months = np.arange(1,13,1)

ds = xr.open_dataset(f"{path_dir}/{name_ds}_{year}.nc")

def plot_crit(ax, coords_Q, stat_Q, t):


    mc = ax.contourf(ds[x_unit], ds[y_unit], ds.R_2d[t, our_level], 
                cmap='PiYG', 
                  vmin=-np.nanmax(np.abs(ds.R_2d[:, our_level])), vmax=np.nanmax(np.abs(ds.R_2d[:, our_level])),
                 )
#     c = plt.colorbar(mc)

    plt.title(str(ds.XTIME[t].values)[:16])

    plt.xlim(0,100)
    plt.ylim(0,100)

    Q_center = ax.scatter(stat_Q['x'], stat_Q['y'], 
                          s=7, c='deeppink', marker='*', alpha=1, label='crit')

    plt.grid(linestyle = ':', linewidth = 0.5)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=9)

    
# перемещение в новую точку исходя из скорости КС между предыдущими шагами
def get_loc_CS_speed(CS, i):

    hw = (CS['rad'][i])
    
    y = (CS['y'][i])
    x = (CS['x'][i])
    
    y_pr = CS['y'][i-1]
    x_pr = CS['x'][i-1]
    
    # поставить время из ds!!!
    dt = 60*60*3
    
    u = (x - x_pr)/dt
    v = (y - y_pr)/dt
    
#     x_new = x+u/dist_m*dt
#     y_new = y+v/dist_m*dt

    return x, y, u, v, hw

def compute_mean_uv_for_track(t, y_in, y_out, x_in, x_out, our_level):
        
    u = np.nanmean(ds[u_unit][t, our_level, 
                          y_in:y_out, x_in:x_out])
    v = np.nanmean(ds[v_unit][t, our_level, 
                          y_in:y_out, x_in:x_out])
    
    return u, v

def get_mean_wspd(CS, i, level):
    
    hw = int(np.round(CS['rad'][i]))
    
    y = CS['y'][i]
    x = CS['x'][i]

    y_int = int(y)
    x_int = int(x)
    
    y_in = y_int - hw
    x_in = x_int - hw
    
    y_out = y_int + hw
    x_out = x_int + hw
    
    if y_in < 0:
        y_in = 0
    if x_in < 0:
        x_in = 0
            
    if y_out >= len(ds[y_unit]):
        y_out = -1
    if x_out >= len(ds[x_unit]):
        x_out = -1

    u, v = compute_mean_uv_for_track(CS['t'][i], y_in, y_out, x_in, x_out, our_level=level)
    
    return x_int, y_int, u, v, hw

def plot_mean_wspd(ax, x, y, u, v, hw, dist_m, color='blue'):

    dt = 60*60*3
    
    ax.plot([x, x+u/dist_m*dt], [y, y+v/dist_m*dt], color)
    
    p = mpatches.Rectangle(xy=(x - hw/2,y - hw/2), width=hw, height=hw, linewidth=0.5, 
                                     edgecolor='blue', facecolor='none',)
    ax.add_patch(p)

def time_plot_tracks(ax, CS_tracks_list_100, our_time):
    for CS in CS_tracks_list_100:
#         print(CS.columns)
       
        if np.isin(our_time, CS['t']).any():
            
            stop_time = np.argwhere(np.array(CS['t']) == our_time)[0][0]
            
            if np.sum(~np.isnan(CS['x'])) > 6:
                ax.plot(CS['x'][:stop_time+1], CS['y'][:stop_time+1], alpha=0.9, lw=0.5, c='k')
                
                if stop_time == 0:
                    x, y, u, v, hw = get_mean_wspd(CS, stop_time, -3)
                    plot_mean_wspd(ax, x, y, u, v, hw, dist_m=dist_m, color='blue')
                else:
                    x, y, u, v, hw = get_loc_CS_speed(CS, stop_time)
                    plot_mean_wspd(ax, x, y, u, v, hw, dist_m=1, color='blue')

def plot_track_with_DBSCAN(ds, CS_tracks_list, start_time, stop_time, name):
    
    for t in tqdm(range(start_time, stop_time)):
        
        coords_Q = vortex.for_DBSCAN(ds[crit][t, our_level], th_Q, crit)  
        coords_Q, n_clusters_lambda2 = clustering_DBSCAN_rortex_C(coords_Q, min_samples=min_samples, eps=eps, circ=circ)

        coords_Q, n_clusters_Q = vortex.DBSCAN_filter(coords_Q, points=CS_points_th)
        
        stat_Q = get_stat(coords_Q, n_clusters_lambda2, dist_m, circ=circ)
        
        
        fig = plt.figure(figsize=(5, 5), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        
        ax.grid(which='major', linewidth=1.2)
        ax.grid(which='minor', linestyle='--', color='gray', linewidth=0.5)
        
        plot_crit(ax, coords_Q, stat_Q, t)
        
        time_plot_tracks(ax, CS_tracks_list, t)    
        
        if not os.path.exists(f"{path_dir}/pics/{name}/{year}"):
            os.makedirs(f"{path_dir}/pics/{name}/{year}")
        plt.savefig(f"{path_dir}/pics/{name}/{year}/track_{(t-start_time):05d}.png", 
                    dpi=200, bbox_inches="tight", transparent=False)
        plt.close()

def load_season_tracks(year, months, path_data): 
    CS_tracks_list = []
    for month in months:
        if os.path.exists(path_data):
            ls = list(sorted(Path(f"{path_data}/").glob(f'*_track_{year}-{month:02d}-*.csv')))
            if len(ls) != 0:
                for ii,ifile in enumerate(ls):
                    df = pd.read_csv(ifile)
                    df = df.drop(df.columns[0], axis=1)
#                     df = df.drop(df.columns[0], axis=1)
                    CS_tracks_list.append(df)
    return CS_tracks_list

####change!!!!#####
path_data_tracks = f'/storage/kubrick/vkoshkina/data/LoRes_tracks/{tracking_type}/{year}/tracks_{circ}'

    
CS_tracks_list = load_season_tracks(year, months, path_data_tracks)

plot_track_with_DBSCAN(ds, CS_tracks_list, 0, 2918, name_pics)