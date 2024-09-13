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

print('param_sq: ')
param_sq = int(input())

print('circ: ')
circ = input()

# params = ['HFX', 'LH']

path_data_dir = f'/storage/kubrick/vkoshkina/data/LoRes_tracks/'

months = np.arange(1,13,1)
days = np.arange(1,32,1)

def load_season_tracks(year, months, path_data): 
    CS_tracks_list = []
    for month in months:
        if os.path.exists(path_data_new):
            ls = list(sorted(Path(f"{path_data}/").glob(f'*_track_{year}-{month:02d}-*.csv')))
            if len(ls) != 0:
                for ii,ifile in enumerate(ls):
                    df = pd.read_csv(ifile)
                    df = df.drop(df.columns[0], axis=1)
                    df = df.drop(df.columns[0], axis=1)
                    CS_tracks_list.append(df)
    return CS_tracks_list

def density_sq(tracks_data, param_sq=3):    
    # Исходные точки
    points = tracks_data.values
    
    # Размерность массива
    h, w = 100, 100

    # Создаем пустой массив для хранения плотности
    density_map = np.zeros((h, w))

    # Функция для подсчета плотности точек в квадрате
    def count_points_in_square(square_points):
        return len(square_points) / (param_sq*param_sq)

    # Проходим по всем возможным квадратам
    for i in range(h // param_sq):
        for j in range(w // param_sq):
            # Получаем индексы для текущего квадрата
            start_i = i  *  param_sq
            end_i = start_i + param_sq
            start_j = j  *  param_sq
            end_j = start_j + param_sq

            # Фильтруем точки, которые попадают в текущий квадрат
            square_points = points[(points[:,0] >= start_i) & (points[:,0] < end_i) &
                                   (points[:,1] >= start_j) & (points[:,1] < end_j)]

            # Считаем плотность для текущего квадрата
            density = count_points_in_square(square_points)

            # Заполняем плотностью соответствующую ячейку в массиве
            density_map[start_j:end_j, start_i:end_i] = density
            
    return density_map

def compute_density_square_multi(CS_dict, season, param_sq=3):    
    path_data_dir_density = f'{path_data_dir}/density_square_{param_sq}'

    if not os.path.exists(f'{path_data_dir_density}'):
        os.makedirs(f'{path_data_dir_density}')

    density_matrix_multi = np.zeros((100, 100))

    for idx, year in tqdm(enumerate(years)):
        tracks_data_year = CS_dict[f'{season}_tracks'][idx]

        tracks_data = []
        for df in tracks_data_year:
            tracks_data.append(df[['x','y']])

        # Объединяем DataFrame в один
        tracks_data = pd.concat(tracks_data, axis=0)

        density_matrix = density_sq(tracks_data, param_sq=param_sq)

        np.save(f'{path_data_dir_density}/{circ}_track_density_{season}_{year}.npy', density_matrix)
        density_matrix_multi += density_matrix
    density_matrix_multi = density_matrix_multi/len(years)
    
    np.save(f'{path_data_dir_density}/{circ}_track_density_{season}_multi.npy', density_matrix_multi)
    
    return density_matrix_multi

def plot_density(ds, track_density, name, season, param_sq, vmax, cmap=cmaps.WhiteBlueGreenYellowRed, vmin=0, save=False):    
    fig = plt.figure(figsize=(7,7))

    ax = fig.add_subplot(111, projection=ccrs.Stereographic(central_latitude=45.0, central_longitude=-45))
    ax.set_global()
    gl = ax.gridlines(draw_labels=True,
                     linewidth=3, color='k', alpha=0.5, linestyle='--')
    
    cr = ax.contourf(ds.XLONG[0], ds.XLAT[0], track_density, 
                    cmap=cmap, 
#                      vmin=vmin, vmax=vmax,
                     vmin=0, vmax=4,
                     
                    transform=ccrs.PlateCarree())
    

    c = plt.colorbar(cr, shrink=0.7)


    ax.set_extent([np.min(ds.XLONG[0,0])-3, -13, np.min(ds.XLAT)+4, np.max(ds.XLAT[-1])], 
                  ccrs.PlateCarree())


    ax.coastlines(color='k', alpha=0.9, lw=3)
    
    ax.legend(title=f'{season} {name}', loc='lower left')    
    
    path_dir = f'/storage/kubrick/vkoshkina/data/LoRes_tracks'
    
    folder = f"{path_dir}/density_pics"
    
    if save:
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(f"{folder}/{circ}_{name}_{season}_sq_{param_sq}.png", dpi=150, bbox_inches="tight", transparent=True)
        
        plt.close()




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


# years = np.arange(1979,2019)

# CS_dict = {'year': [],
#                'winter_tracks': [],
#                'summer_tracks': [],
#               }
    
# for year in tqdm(years):
#     path_data_new = f'{path_data_dir}/{year}/tracks_{circ}_params'
    
#     CS_tracks_list_winter = load_season_tracks(year, [1, 2, 12], path_data_new)
#     CS_tracks_list_summer = load_season_tracks(year, [6, 7, 8], path_data_new)
    
#     CS_dict['year'].append(year)
#     CS_dict['winter_tracks'].append(CS_tracks_list_winter)
#     CS_dict['summer_tracks'].append(CS_tracks_list_summer)

# density_matrix_multi_winter = compute_density_square_multi(CS_dict, 'winter', param_sq=param_sq)
# density_matrix_multi_summer = compute_density_square_multi(CS_dict, 'summer', param_sq=param_sq)



name_ds = 'rortex_criteria_LoRes_year'
path_dir_ds = '/storage/kubrick/vkoshkina/data/2010/LoRes_for_tracking'

ds = xr.open_dataset(f"{path_dir_ds}/{name_ds}.nc")


path_data_dir_density = f'{path_data_dir}/density_square_{param_sq}'

density_matrix_multi_winter = np.load(f'{path_data_dir_density}/{circ}_track_density_winter_multi.npy')
density_matrix_multi_summer = np.load(f'{path_data_dir_density}/{circ}_track_density_summer_multi.npy')

vmax = np.nanmax([np.abs(density_matrix_multi_winter), np.abs(density_matrix_multi_summer)])

plot_density(ds, density_matrix_multi_winter[:,::-1], 'density', 'winter', vmax=vmax, param_sq=param_sq, save=True)
plot_density(ds, density_matrix_multi_summer[:,::-1], 'density', 'summer', vmax=vmax, param_sq=param_sq, save=True)




# mean_wspd_summer = track_density_summer_list[1]/track_density_summer_list[0]
# mean_temp_summer = track_density_summer_list[2]/track_density_summer_list[0]

# np.save(f'{path_data_dir_density}/{circ}_track_density_summer_{params[0]}.npy', mean_wspd_summer)
# np.save(f'{path_data_dir_density}/{circ}_track_density_summer_{params[1]}.npy', mean_temp_summer)
# # np.save(f'{path_data_dir_density}/{circ}_track_density_summer_points.npy', track_density_summer_list[0])

# track_density_winter_list = tracks_density_param(CS_tracks_list_winter)

# mean_wspd_winter = track_density_winter_list[1]/track_density_winter_list[0]
# mean_temp_winter = track_density_winter_list[2]/track_density_winter_list[0]

# np.save(f'{path_data_dir_density}/{circ}_track_density_winter_{params[0]}.npy', mean_wspd_winter)
# np.save(f'{path_data_dir_density}/{circ}_track_density_winter_{params[1]}.npy', mean_temp_winter)
# # np.save(f'{path_data_dir_density}/{circ}_track_density_winter_points.npy', track_density_winter_list[0])
