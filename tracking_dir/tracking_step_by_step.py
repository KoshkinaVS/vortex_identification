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


from sklearn.cluster import DBSCAN

import pandas as pd 
import numpy as np
import math
import xarray as xr

import datetime
from datetime import timedelta

from tqdm import tqdm

import sys
import os
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/storage/kubrick/vkoshkina/scripts/vortex_identification')
sys.path.insert(1, './vortex_identification')

from vortex_dir import vortex_processing as vortex
from vortex_dir import show_vortex as show_vx
from vortex_dir import compute_criteria as compute

def get_init_params(data_type):
    min_samples = 8
    eps = 2
    CS_points_th = 25
    
    crit = 'R_2d'
    th = 0.
    
    x_unit = 'west_east'
    y_unit = 'south_north'

    if data_type == 'HiRes':
        dist_m = 13897.18
        our_level = 0
        level = 12
        u_unit = 'ue'
        v_unit = 've'
        time_unit = 'XTIME'

    elif data_type == 'ERA5':
        dist_m = 13897.18
        our_level = 0
        level = 9

        u_unit = 'U'
        v_unit = 'V'
        time_unit = 'time'

    else:    
        dist_m = 77824.23
        our_level = 12 # 500 hPa
        u_unit = 'ue'
        v_unit = 've'
        time_unit = 'XTIME'

    return dist_m, our_level, x_unit, y_unit, u_unit, v_unit, time_unit, crit, th, min_samples, eps, CS_points_th

data_type = 'ERA5'
# data_type = 'HiRes'


dist_m, our_level, x_unit, y_unit, u_unit, v_unit, time_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type)

# примитивная статистика, центр на основе максимума критерия
def get_stat(coords_Q_cluster, n_clusters_Q, dist_m, circ='C', CS_points_th=25):

    centroid_idx = []
    
    centroid_x = []
    centroid_y = []
    
    crit_centroid_x = []
    crit_centroid_y = []
    
    radius_eff = []
    
    for n in range(n_clusters_Q):
    
        coords_Q_c = coords_Q_cluster[coords_Q_cluster['cluster'] == n]
        
        if len(coords_Q_c) >= CS_points_th:
            
            N_points = len(coords_Q_c)
            rad_eff = np.sqrt(N_points/np.pi)  # *dist_m/1000
            radius_eff.append(rad_eff)
            
            centroid_idx.append(n)

            if circ == 'AC':
                crit_max = np.min(coords_Q_c.crit)
            else:
                crit_max = np.max(coords_Q_c.crit)
            center = coords_Q_c[coords_Q_c['crit'] == crit_max]
            crit_centroid_x.append(float(center.x))
            crit_centroid_y.append(float(center.y))


        else:
            
            
            centroid_idx.append(None)
            
            centroid_x.append(None)
            centroid_y.append(None)
            
            crit_centroid_x.append(None)
            crit_centroid_y.append(None)
            
            radius_eff.append(None)
        
    
    stat = pd.DataFrame({'cluster': centroid_idx, 
#                          'x': centroid_x, 'y': centroid_y,              
                         'x': crit_centroid_x, 'y': crit_centroid_y,                         
                         'rad_eff': radius_eff,
                        })
    
    
    stat = stat.dropna()   
    

    return stat

# расчет расстояния между точками по декарту для поиска следующей координаты трека
def dist(p1_x, p1_y, p2_x, p2_y):
    return np.sqrt((p1_x - p2_x)*(p1_x - p2_x) + (p1_y - p2_y)*(p1_y - p2_y))

# перемещение в новую точку исходя из средней скорости потока
def get_new_loc_mean_speed(ds, dt, x, y, hw, t, level):

    y_int = int(np.round(y))
    x_int = int(np.round(x))
    
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

    u, v = compute_mean_uv_for_track(ds, t, y_in, y_out, x_in, x_out, our_level=level)
    
    x_new = x+u/dist_m*dt
    y_new = y+v/dist_m*dt
    
    if y_new < 0:
        y_new = 0
    if x_new < 0:
        x_new = 0
            
    if y_new >= len(ds[y_unit]):
        y_new = len(ds[y_unit])-1
    if x_new >= len(ds[x_unit]):
        x_new = len(ds[x_unit])-1
    
    return x_new, y_new

# расчет средней скорости потока в промежутке hw
def compute_mean_uv_for_track(ds, t, y_in, y_out, x_in, x_out, our_level):
    u = np.nanmean(ds[u_unit][t, our_level, 
                          y_in:y_out, x_in:x_out])
    v = np.nanmean(ds[v_unit][t, our_level, 
                          y_in:y_out, x_in:x_out])
    
    return u, v


# перемещение в новую точку исходя из скорости КС между предыдущими шагами
def get_new_loc_CS_speed(ds, dt, x_pr, y_pr, x_prpr, y_prpr):

    
    u = (x_pr - x_prpr)/dt
    v = (y_pr - y_prpr)/dt
    
    x_new = x_pr+u*dt
    y_new = y_pr+v*dt
    
    if y_new < 0:
        y_new = 0
    if x_new < 0:
        x_new = 0
            
    if y_new >= len(ds[y_unit]):
        y_new = len(ds[y_unit])-1
    if x_new >= len(ds[x_unit]):
        x_new = len(ds[x_unit])-1

    return x_new, y_new

# get pd.df with cluster label for each point
def clustering_DBSCAN_rortex_C(coords_lambda2, eps=10., min_samples=10, metric='euclidean', circ='AC'):
    
    db1 = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=4) # для декартовой сетки 
    
    if circ == 'C':
        coords_lambda2_pos = coords_lambda2[coords_lambda2.crit > 0]
    else:
        coords_lambda2_pos = coords_lambda2[coords_lambda2.crit < 0]
        
    coords_pos = np.array([coords_lambda2_pos.x, coords_lambda2_pos.y]).T
    
    y_db1_pos = db1.fit_predict(coords_pos)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_lambda2_pos = len(set(y_db1_pos)) - (1 if -1 in y_db1_pos else 0)
    n_noise_lambda2_pos = list(y_db1_pos).count(-1)

#     print(f'Estimated number of clusters ({circ}): %d' % n_clusters_lambda2_pos)    
#     print(f'Estimated number of noise points ({circ}): %d' % n_noise_lambda2_pos)


    coords_lambda2_pos['cluster'] = y_db1_pos
       
    return coords_lambda2_pos.reset_index(), n_clusters_lambda2_pos


################################################
################   сам трекинг    ###############
################################################

# инициализация точек-начал треков по ДБСКАНу
def track_init(ds, data_type, stat_Q, CS_tracks_list, our_time, time_unit='XTIME'):

    clstr_len = len(CS_tracks_list)
    
    for i in range(len(stat_Q)):
        
        ts = []
        times = []
        x_coords = []
        y_coords = []
        
        lon_coords = []
        lat_coords = []
        
        rads = []
        track_l = []
        
        ts.append(our_time)
        times.append(ds[time_unit][our_time].values)
    
        x = int(stat_Q.x.values[i])
        y = int(stat_Q.y.values[i])
    
        x_coords.append(x)
        y_coords.append(y)
        
        if data_type == 'ERA5':
            lon_coords.append(float(ds.XLONG[y,x].values))
            lat_coords.append(float(ds.XLAT[y,x].values))
        else:
            lon_coords.append(float(ds.XLONG[our_time,y,x].values))
            lat_coords.append(float(ds.XLAT[our_time,y,x].values))
        
        
        rads.append(stat_Q.rad_eff.values[i])
        track_l.append(0)
        
        CS_coords = {'cluster': clstr_len+i,
                     't': ts,
                     'time': times,
                     'x': x_coords,
                     'y': y_coords,
                     'lon': lon_coords,
                     'lat': lat_coords,
                     'rad': rads,
                     'track_len': track_l,
                    }
        
        CS_tracks_list.append(CS_coords)
                   
    return CS_tracks_list

# трекинг с учетом смещения в новую точку по скорости
def step_of_tracking(CS_tracks_list, ds, data_type, our_time, circ='C', dt_step=3600):
    
    dist_m, our_level, x_unit, y_unit, u_unit, v_unit, time_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type)
    
    coords_Q = vortex.for_DBSCAN(ds[crit][our_time, our_level], th, crit)  
    coords_Q, n_clusters_lambda2 = clustering_DBSCAN_rortex_C(coords_Q, min_samples=min_samples, eps=eps, circ=circ)
    coords_Q, n_clusters_Q = vortex.DBSCAN_filter(coords_Q, points=CS_points_th)
    
    stat_Q = get_stat(coords_Q, n_clusters_lambda2, dist_m, circ=circ, CS_points_th=CS_points_th)

    for i in range(len(CS_tracks_list)):
        
        x_init = CS_tracks_list[i]['x'][-1]
        y_init = CS_tracks_list[i]['y'][-1]
        
        rad_init = CS_tracks_list[i]['rad'][-1]
        
        if ~np.isnan(x_init) and x_init != (len(ds[x_unit]) - 1) and y_init != (len(ds[y_unit]) - 1) and x_init != 0 and y_init != 0:
                
            if len(CS_tracks_list[i]['x']) == 1:
                x, y = get_new_loc_mean_speed(ds, dt_step, x_init, y_init, int(np.round(rad_init)), our_time, 0)
            else:
                x_prpr = CS_tracks_list[i]['x'][-2]
                y_prpr = CS_tracks_list[i]['y'][-2]
                x, y = get_new_loc_CS_speed(ds, dt_step, x_init, y_init, x_prpr, y_prpr)
        
            coords_Q['dist'] = dist(x, y, coords_Q.x, coords_Q.y)

            coords_Q_min_dist  = coords_Q[coords_Q['dist'] < rad_init]
    
            if len(coords_Q_min_dist) > 0:
            
                if circ == 'AC':
                    crit_max = np.min(coords_Q_min_dist['crit'])
                else:
                    crit_max = np.max(coords_Q_min_dist['crit'])
                
                cluster = coords_Q_min_dist[coords_Q_min_dist['crit'] == crit_max]

                stat_Q_c = stat_Q[stat_Q['cluster'] == cluster['cluster'].values[0]]
                            
                # внимание - используется первое значение с мин расстоянием
                x_c = int(cluster.x.values[0])
                y_c = int(cluster.y.values[0])
                
                rad = (stat_Q_c.rad_eff.values[0])
                
                x_pred = CS_tracks_list[i]['x'][-1]
                y_pred = CS_tracks_list[i]['y'][-1]
                
                track_len_new = dist(x_c, y_c, x_pred, y_pred)

                CS_tracks_list[i]['x'].append(x_c)
                CS_tracks_list[i]['y'].append(y_c)
                
                
                if data_type == 'ERA5':
                    CS_tracks_list[i]['lon'].append(float(ds.XLONG[y_c,x_c].values))
                    CS_tracks_list[i]['lat'].append(float(ds.XLAT[y_c,x_c].values))
                else:
                    CS_tracks_list[i]['lon'].append(float(ds.XLONG[our_time,y_c,x_c].values))
                    CS_tracks_list[i]['lat'].append(float(ds.XLAT[our_time,y_c,x_c].values))
                
                CS_tracks_list[i]['rad'].append(rad)
                
                track_l = CS_tracks_list[i]['track_len'][-1]
                
                CS_tracks_list[i]['track_len'].append(track_l + track_len_new)

                CS_tracks_list[i]['t'].append(our_time)
                CS_tracks_list[i]['time'].append(ds[time_unit][our_time].values)
                
                # удаляем строчку с использованными координатами
            
                if len(stat_Q) != 0:
                    stat_Q = stat_Q[stat_Q['cluster'] != cluster['cluster'].values[0]]
                    coords_Q = coords_Q[coords_Q['cluster'] != cluster['cluster'].values[0]]
                    
                
            else:
                CS_tracks_list[i]['x'].append(np.nan)
                CS_tracks_list[i]['y'].append(np.nan)
                
                CS_tracks_list[i]['lon'].append(np.nan)
                CS_tracks_list[i]['lat'].append(np.nan)
                
                CS_tracks_list[i]['rad'].append(np.nan)

                CS_tracks_list[i]['t'].append(np.nan)
                CS_tracks_list[i]['time'].append(np.nan)
                CS_tracks_list[i]['track_len'].append(np.nan)
                
    CS_tracks_list_new = []
    
    for CS in CS_tracks_list:
        if np.isnan(CS['x'][-1]):

            if len(CS['x']) <= 4:
                CS_tracks_list_new.append(CS)

    if len(stat_Q) != 0:
        CS_tracks_list = track_init(ds, data_type, stat_Q.reset_index(), CS_tracks_list_new, our_time, time_unit)

    return CS_tracks_list