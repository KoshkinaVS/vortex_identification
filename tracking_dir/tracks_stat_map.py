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

# tracking_type = 'ordinary'
tracking_type = 'track_len_real'
# tracking_type = 'with_CS_speed'
# tracking_type = 'track_domain_boundary'


precomputed = False
# precomputed = True


variables = {
    'density': {'var_name': 'density', 'vmax':  3, 'vstep':1, 'cmap':cmaps.WhiteBlueGreenYellowRed},
    'wspd': {'var_name': 'wspd', 'vmax':  29, 'vstep':1, 'cmap':cmaps.WhiteBlueGreenYellowRed},
    'rad': {'var_name': 'rad', 'vmax':  8, 'vstep':1, 'cmap':cmaps.WhiteBlueGreenYellowRed},
    'dT': {'var_name': 'dT', 'vmax':  14, 'vstep':1, 'cmap':cmaps.BlueDarkRed18},
    'LH': {'var_name': 'LH', 'vmax':  250, 'vstep':1, 'cmap':cmaps.BlueDarkRed18},
    'HFX': {'var_name': 'HFX', 'vmax':  170, 'vstep':1, 'cmap':cmaps.BlueDarkRed18},
#     'temp': {'var_name': '          tc', 'vmin': -15, 'vmax': 30, 'vstep':2, 'cmap':cmaps.cmp_b2r},
    }
    



years = np.arange(1979,2019)


dist_m = 77824.23


path_data_dir = f'/storage/kubrick/vkoshkina/data/LoRes_tracks'

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


def add_sq_param(square_points, param, sq_type = 'max'):
    if len(square_points) != 0:
        if sq_type == 'max':
            max_wspd = np.nanmax(square_points[param].values)
        else:
            max_wspd = np.nanmedian(square_points[param].values)
    else:
        max_wspd = np.nan
    return max_wspd


def density_sq(tracks_data, param_sq=3):    

    # Размерность массива
    h, w = 100, 100

    # Создаем пустой массив для хранения плотности
    density_map = np.zeros((h, w))
    wspd_map = np.zeros((h, w))
    rad_map = np.zeros((h, w))
    dT_map = np.zeros((h, w))
    HFX_map = np.zeros((h, w))
    LH_map = np.zeros((h, w))
    
#     map_dict = {'density': float, 'wspd': float, 'rad': float, 
#                 'dT': float, 'HFX': float, 'LH': float}   

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
            square_points = tracks_data[(tracks_data['x'] >= start_i) & (tracks_data['x'] < end_i) & 
                                        (tracks_data['y'] >= start_j) & (tracks_data['y'] < end_j)]
            
            # Считаем плотность для текущего квадрата
            density = count_points_in_square(square_points[['x', 'y']].values)
            
            max_wspd = add_sq_param(square_points, 'wspd10_max', sq_type = 'max')
            rad = add_sq_param(square_points, 'rad', sq_type = 'median')
            dT = add_sq_param(square_points, 'dT_med', sq_type = 'median')
            HFX = add_sq_param(square_points, 'HFX_med', sq_type = 'median')
            LH = add_sq_param(square_points, 'LH_med', sq_type = 'median')
            


            # Заполняем плотностью соответствующую ячейку в массиве
            density_map[start_j:end_j, start_i:end_i] = density
            wspd_map[start_j:end_j, start_i:end_i] = max_wspd
            rad_map[start_j:end_j, start_i:end_i] = rad
            dT_map[start_j:end_j, start_i:end_i] = dT
            HFX_map[start_j:end_j, start_i:end_i] = HFX
            LH_map[start_j:end_j, start_i:end_i] = LH
            
            
    map_dict = {'density': density_map, 'wspd': wspd_map, 'rad': rad_map, 
                'dT': dT_map, 'HFX': HFX_map, 'LH': LH_map}        
            
    return map_dict


def compute_density_square_multi(CS_dict, season, params, param_sq=3):    
    path_data_dir_density = f'{path_data_dir}/density_square_{param_sq}'

    if not os.path.exists(f'{path_data_dir_density}'):
        os.makedirs(f'{path_data_dir_density}')

    h, w = 100, 100
    
    t = len(years)
    
    
    map_dict_years = {'density': np.zeros((t, h, w)), 'wspd': np.zeros((t, h, w)), 'rad': np.zeros((t, h, w)), 
                'dT': np.zeros((t, h, w)), 'HFX': np.zeros((t, h, w)), 'LH': np.zeros((t, h, w))}  
    
    map_dict_multi = {'density': np.zeros((h, w)), 'wspd': np.zeros((h, w)), 'rad': np.zeros((h, w)), 
                'dT': np.zeros((h, w)), 'HFX': np.zeros((h, w)), 'LH': np.zeros((h, w))}  

    for idx, year in tqdm(enumerate(years)):

        tracks_data_year = CS_dict[f'{season}_tracks'][idx]

        # Пример использования функции
        tracks_data = []
        for df in tracks_data_year:
            tracks_data.append(df[df.columns[1:]])

        # Объединяем DataFrame в один
        tracks_data = pd.concat(tracks_data, axis=0)

        map_dict = density_sq(tracks_data, param_sq=param_sq)

        for param in params:
            map_dict_years[param][idx] = map_dict[param]
    
    
    for param in params:
        map_dict_multi[param] = np.nanmedian(map_dict_years[param], axis=0)
        np.save(f'{path_data_dir_density}/{tracking_type}_{circ}_track_density_{season}_multi_{param}.npy', map_dict_multi[param])
        
    return map_dict_multi


def plot_density(track_density, name, season, param_sq, vmax, cmap=cmaps.WhiteBlueGreenYellowRed, vmin=0, save=False):    
    fig = plt.figure(figsize=(7,7))

    ax = fig.add_subplot(111, projection=ccrs.Stereographic(central_latitude=45.0, central_longitude=-45))
    ax.set_global()
    gl = ax.gridlines(draw_labels=True,
                     linewidth=3, color='k', alpha=0.5, linestyle='--')

    cr = ax.contourf(ds.XLONG[0], ds.XLAT[0], track_density, 
                    cmap=cmap, 
                     vmin=vmin, vmax=vmax,
                     levels=50,
                    transform=ccrs.PlateCarree())
    
    
    ax.set_extent([np.min(ds.XLONG[0,0])-3, -13, np.min(ds.XLAT)+4, np.max(ds.XLAT[-1])], 
                  ccrs.PlateCarree())


    ax.coastlines(color='k', alpha=0.9, lw=3)
    
    ax.legend(title=f'{season} {name}', loc='lower left')    
    
    
    folder = f"{path_data_dir}/density_pics/{tracking_type}"
    
    if save:
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(f"{folder}/{circ}_{name}_{season}_sq_{param_sq}.png", dpi=150, bbox_inches="tight", transparent=True)
        
        plt.close()



params = ['density', 'wspd', 'rad', 'dT', 'LH', 'HFX']  


    

name_ds = 'rortex_criteria_LoRes_year'
path_dir_ds = '/storage/kubrick/vkoshkina/data/2010/LoRes_for_tracking'

ds = xr.open_dataset(f"{path_dir_ds}/{name_ds}.nc")






if precomputed:
    
    path_data_dir_density = f'{path_data_dir}/density_square_{param_sq}'
    

    h, w = 100, 100

    map_dict_multi_winter = {'density': np.zeros((h, w)), 'wspd': np.zeros((h, w)), 'rad': np.zeros((h, w)), 
                    'dT': np.zeros((h, w)), 'HFX': np.zeros((h, w)), 'LH': np.zeros((h, w))}  
    map_dict_multi_summer = {'density': np.zeros((h, w)), 'wspd': np.zeros((h, w)), 'rad': np.zeros((h, w)), 
                    'dT': np.zeros((h, w)), 'HFX': np.zeros((h, w)), 'LH': np.zeros((h, w))}  

    for param in params:

        map_dict_winter = np.load(f'{path_data_dir_density}/{tracking_type}_{circ}_track_density_winter_multi_{param}.npy')
        map_dict_summer = np.load(f'{path_data_dir_density}/{tracking_type}_{circ}_track_density_summer_multi_{param}.npy')

        map_dict_multi_winter[param] = map_dict_winter
        map_dict_multi_summer[param] = map_dict_summer
else:

    CS_dict = {'year': [],
                   'winter_tracks': [],
                   'summer_tracks': [],
                  }

    for year in tqdm(years):
        path_data_new = f'{path_data_dir}/{tracking_type}/{year}/tracks_{circ}_params'

        CS_tracks_list_winter = load_season_tracks(year, [1, 2, 12], path_data_new)
        CS_tracks_list_summer = load_season_tracks(year, [6, 7, 8], path_data_new)

        CS_dict['year'].append(year)
        CS_dict['winter_tracks'].append(CS_tracks_list_winter)
        CS_dict['summer_tracks'].append(CS_tracks_list_summer)
    
    map_dict_multi_winter = compute_density_square_multi(CS_dict, 'winter', params=params, param_sq=2)
    map_dict_multi_summer = compute_density_square_multi(CS_dict, 'summer', params=params, param_sq=2)
    
def plot_density_multi(ax, track_density, name, season, param_sq, vmax, cmap=cmaps.WhiteBlueGreenYellowRed, vmin=0, save=False):    
    ax.set_global()
    gl = ax.gridlines(draw_labels=True,
                     linewidth=3, color='k', alpha=0.5, linestyle='--')

    
    cr = ax.contourf(ds.XLONG[0], ds.XLAT[0], track_density, 
                    cmap=cmap, 
                     vmin=vmin, vmax=vmax,
                    transform=ccrs.PlateCarree())


    ax.set_extent([np.min(ds.XLONG[0,0])-3, -13, np.min(ds.XLAT)+4, np.max(ds.XLAT[-1])], 
                  ccrs.PlateCarree())


    ax.coastlines(color='k', alpha=0.9, lw=3)
    
    ax.legend(title=f'{season} {name}', loc='lower left')    
    
    
    
def show_multi_density(season, map_dict_multi_winter, save=False):

    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(231, projection=ccrs.Stereographic(central_latitude=45.0, central_longitude=-45))
    ax2 = fig.add_subplot(232, projection=ccrs.Stereographic(central_latitude=45.0, central_longitude=-45))
    ax3 = fig.add_subplot(233, projection=ccrs.Stereographic(central_latitude=45.0, central_longitude=-45))
    ax4 = fig.add_subplot(234, projection=ccrs.Stereographic(central_latitude=45.0, central_longitude=-45))
    ax5 = fig.add_subplot(235, projection=ccrs.Stereographic(central_latitude=45.0, central_longitude=-45))
    ax6 = fig.add_subplot(236, projection=ccrs.Stereographic(central_latitude=45.0, central_longitude=-45))

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    for idx, param in enumerate(params):
#         if param == 'dT' or param == 'HFX' or param == 'LH':
#             cmap=cmaps.BlueDarkRed18
#             vmin = -np.nanmax([np.abs(map_dict_multi_winter[param]), np.abs(map_dict_multi_summer[param])])
#         else:
#             cmap=cmaps.WhiteBlueGreenYellowRed
#             vmin = 0

        vmax = np.nanmax([np.abs(map_dict_multi_winter[param]), np.abs(map_dict_multi_summer[param])])
        
        print(f'{param} vmax: {vmax}')

        if param == 'rad':
            plot_density_multi(axes[idx], map_dict_multi_winter[param]*dist_m*0.001, param, season, vmax=variables[param]['vmax']*dist_m*0.001, vmin=0, cmap=variables[param]['cmap'], param_sq=2, save=False)
        elif param == 'dT' or param == 'HFX' or param == 'LH':
            plot_density_multi(axes[idx], map_dict_multi_winter[param], param, season, vmax=variables[param]['vmax'], vmin=-variables[param]['vmax'], cmap=variables[param]['cmap'], param_sq=2, save=True)
        else:
            plot_density_multi(axes[idx], map_dict_multi_winter[param], param, season, vmax=variables[param]['vmax'], vmin=0, cmap=variables[param]['cmap'], param_sq=2, save=True)



    folder = f"{path_data_dir}/density_pics/{tracking_type}"

    if save:
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(f"{folder}/{circ}_multi_{season}_sq_{param_sq}.png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    
season = 'winter'
show_multi_density(season, map_dict_multi_winter, save=True)

season = 'summer'
show_multi_density(season, map_dict_multi_summer, save=True)