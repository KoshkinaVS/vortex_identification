from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import math
import xarray as xr

# from matplotlib import animation
import datetime
from datetime import timedelta

import scipy as sp


# import cartopy.crs as ccrs
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

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


import sys
import os
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/storage/2TB/koshkina/scripts/vortex_identification')

from vortex_dir import vortex_processing as vortex
from vortex_dir import show_vortex as show_vx
from vortex_dir import compute_criteria as compute


name = 'rortex_NAAD_LoRes_2010_big'
path_dir = '/storage/2TB/koshkina/data/NAAD/LoRes'

ds = xr.open_dataset(f"{path_dir}/{name}.nc")


our_level = 9
dist_m = 77824.23

min_samples = 8
eps = 2
CS_points_th = 15

th_Q = 0.
crit = 'R_2d'

name = input('Alias for final folder: ')


def date(our_time):
    date = np.datetime_as_string(ds.Time[our_time].values, unit='m')
    return date

# get coords for clustering after th, X.shape = (y,x), crit - name like in ds Data variables
def for_DBSCAN(X, threshold, crit):
    X_arr = X.to_dataframe().dropna(how='any')
    print(f'points before th: {len(X_arr[crit])}')
    X_arr[crit][np.abs(X_arr[crit]) < threshold] = None
    X_arr = X_arr.dropna(how='any')
    print(f'points after th: {len(X_arr[crit])}')
    
    coords = X_arr[crit].index.to_frame(name=['y', 'x'], index=False)
#     coords['lon'] = X_arr.XLONG.values
#     coords['lat'] = X_arr.XLAT.values
    coords['crit'] = X_arr[crit].values  
    
    return coords


# filter out noise points, CSs < given number
def DBSCAN_filter(coords_lambda2, points=40):
    coords_lambda2 = coords_lambda2.drop(np.where(coords_lambda2['cluster'] == -1)[0])
    clusters = coords_lambda2.groupby('cluster').count()
    true_clusters = clusters[clusters['x'] >= points]
    true_index = true_clusters.index.values
    
    n_clusters = len(true_clusters)
    
    coords_lambda2 = coords_lambda2.where(coords_lambda2.cluster.isin(true_index))
    coords_lambda2 = coords_lambda2.dropna()
    return coords_lambda2, n_clusters


# compute statistic for each cluster: radius, center, and return pd.df for 1 level
def get_stat(coords_Q_cluster, n_clusters_Q, dist_m, mean_center=False):

    centroid_lat = []
    centroid_lon = []
    centroid_idx = []
    
    
    centroid_x = []
    centroid_y = []
    
    crit_centroid_x = []
    crit_centroid_y = []
    
    
    radius_eff = []
    
    
    for n in range(n_clusters_Q):
    
        coords_Q_c = coords_Q_cluster[coords_Q_cluster['cluster'] == n]
        
        if len(coords_Q_c) >= 10:
            
            N_points = len(coords_Q_c)
            rad_eff = np.sqrt(N_points/np.pi)  # *dist_m/1000
            radius_eff.append(rad_eff)
            
            centroid_idx.append(n)
            
#             clst_coords, lon_c, lat_c, bound_coords, radius_median, radius_min, radius_max = get_boundary_polyg(coords_Q_c)

            if mean_center:
                x_c = coords_Q_c.x.mean()
                y_c = coords_Q_c.y.mean()

                centroid_x.append(x_c)
                centroid_y.append(y_c)
            else:
                crit_max = np.max(coords_Q_c.crit)
                center = coords_Q_c[coords_Q_c['crit'] == crit_max]
                
                centroid_x.append(float(center.x))
                centroid_y.append(float(center.y))


        else:
            
            
            centroid_idx.append(None)
#             centroid_lat.append(None)
#             centroid_lon.append(None)
            centroid_x.append(None)
            centroid_y.append(None)
            crit_centroid_x.append(None)
            crit_centroid_y.append(None)
            
            radius_eff.append(None)
        
    
    stat = pd.DataFrame({'cluster': centroid_idx, 
                         'x': centroid_x, 'y': centroid_y,              
#                          'x': crit_centroid_x, 'y': crit_centroid_y,                         
                         
#                          'x_crit': crit_centroid_x, 'y_crit': crit_centroid_y,

                         'rad_eff': radius_eff,
                        })
    
    
    stat = stat.dropna()   
    

    return stat


# get pd.df with cluster label for each point
def clustering_DBSCAN_rortex_C(coords_lambda2, eps=10., min_samples=10, metric='euclidean'):
    
    db1 = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=4) # для декартовой сетки 
    coords_lambda2_pos = coords_lambda2[coords_lambda2.crit > 0]    
    coords_pos = np.array([coords_lambda2_pos.x, coords_lambda2_pos.y]).T
    y_db1_pos = db1.fit_predict(coords_pos)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_lambda2_pos = len(set(y_db1_pos)) - (1 if -1 in y_db1_pos else 0)
    n_noise_lambda2_pos = list(y_db1_pos).count(-1)

    print('Estimated number of clusters (C): %d' % n_clusters_lambda2_pos)
    print('Estimated number of noise points (C): %d' % n_noise_lambda2_pos)

    coords_lambda2_pos['cluster'] = y_db1_pos
            
    return coords_lambda2_pos.reset_index(), n_clusters_lambda2_pos


def dist(p1_x, p1_y, p2_x, p2_y):
    return np.sqrt((p1_x - p2_x)*(p1_x - p2_x) + (p1_y - p2_y)*(p1_y - p2_y))


def track_init(stat_Q, CS_tracks_list, our_time):

    clstr_len = len(CS_tracks_list)
    
    for i in range(len(stat_Q)):
        
        ts = []
        times = []
        x_coords = []
        y_coords = []
        rads = []
        
        ts.append(our_time)
        times.append(ds.Time[our_time].values)
    
        x_coords.append(stat_Q.x.values[i])
        y_coords.append(stat_Q.y.values[i])
        rads.append(stat_Q.rad_eff.values[i])
        
        CS_coords = {'cluster': clstr_len+i,
                     't': ts,
                     'time': times,
                     'x': x_coords,
                     'y': y_coords,
                     'rad': rads,
                     'track_len': 0,
                    }
        
        CS_tracks_list.append(CS_coords)
                   
    return CS_tracks_list



def simple_tracking(ds, our_time):
    CS_tracks_list = []
    
    # init field of CS
    coords_Q = for_DBSCAN(ds[crit][our_time, our_level], th_Q, crit)  
    coords_Q, n_clusters_lambda2 = clustering_DBSCAN_rortex_C(coords_Q, min_samples=min_samples, eps=eps)
    coords_Q, n_clusters_Q = DBSCAN_filter(coords_Q, points=CS_points_th)
    
#     stat_Q = pd.DataFrame({ 
#                              'x': coords_Q.groupby(by='cluster').x.mean().values, 
#                              'y': coords_Q.groupby(by='cluster').y.mean().values,                         
#                             }) 
    
    stat_Q = get_stat(coords_Q, n_clusters_lambda2, dist_m)
    
#     stat_Q = pd.DataFrame({ 
#                              'x': stat_Q.x.values,
#                              'y': stat_Q.y.values,
#                              'rad': stat_Q.rad_eff.values,
#                             }) 
    
    
    CS_tracks_list = track_init(stat_Q, CS_tracks_list, our_time)
        
    for t in tqdm(range(our_time+1,len(ds.Time))):
    # for t in tqdm(range(50)):
        
        coords_Q = for_DBSCAN(ds[crit][t, our_level], th_Q, crit)  
        coords_Q, n_clusters_lambda2 = clustering_DBSCAN_rortex_C(coords_Q, min_samples=min_samples, eps=eps)
        coords_Q, n_clusters_Q = DBSCAN_filter(coords_Q, points=CS_points_th)
        
#         stat_Q = pd.DataFrame({ 
#                          'x': coords_Q.groupby(by='cluster').x.mean().values, 
#                          'y': coords_Q.groupby(by='cluster').y.mean().values,                         
#                         }) 
        
        stat_Q = get_stat(coords_Q, n_clusters_lambda2, dist_m)
    
#         stat_Q = pd.DataFrame({ 
#                              'x': stat_Q.x.values, 
#                              'y': stat_Q.y.values,  
#                              'rad': stat_Q.rad_eff.values,
#                             }) 
    
        for i in tqdm(range(len(CS_tracks_list))):
            
            x_init = CS_tracks_list[i]['x'][-1]
            y_init = CS_tracks_list[i]['y'][-1]
            
            rad_init = CS_tracks_list[i]['rad'][-1]
            
            if ~np.isnan(x_init):
            
                clstr = i

#                 print(f'cluster {clstr}: ', x_init, y_init)

                stat_Q['dist'] = dist(x_init, y_init, stat_Q.x, stat_Q.y)

                min_dist  = np.min(stat_Q['dist'])
                cluster = stat_Q[stat_Q['dist'] == min_dist]
                
                if min_dist < 2*rad_init:

                    # внимание - используется первое значение с мин расстоянием
                    x_c = (cluster.x.values[0])
                    y_c = (cluster.y.values[0])
                    rad = (cluster.rad_eff.values[0])

                    CS_tracks_list[i]['x'].append(x_c)
                    CS_tracks_list[i]['y'].append(y_c)
                    CS_tracks_list[i]['rad'].append(rad)
                    
                    
                    CS_tracks_list[i]['track_len'] += min_dist
                    
                    CS_tracks_list[i]['t'].append(t)
                    CS_tracks_list[i]['time'].append(ds.Time[t].values)
                    

                else:
                    CS_tracks_list[i]['x'].append(np.nan)
                    CS_tracks_list[i]['y'].append(np.nan)
                    CS_tracks_list[i]['rad'].append(np.nan)
                    
                    
                    CS_tracks_list[i]['t'].append(np.nan)
                    CS_tracks_list[i]['time'].append(np.nan)
                
                #удаляем строчку с использованными координатами
                
                warn = False
                # учесть, что может быть несколько индексов!
                if len(stat_Q) != 0:
                    stat_Q = stat_Q.drop([cluster.index[0]])
                else:
                    warn = True
        if ~warn:
            CS_tracks_list = track_init(stat_Q.reset_index(), CS_tracks_list, t)

    return CS_tracks_list


CS_tracks_list = simple_tracking(ds, 0)


for idx, CS in enumerate(CS_tracks_list):
    if CS['track_len'] == 0:
        CS_tracks_list.pop(idx)
print(f'{len(CS_tracks_list)} tracks')


for CS in CS_tracks_list:
    CS['dist'] = dist(CS['x'][0], CS['y'][0], CS['x'][-2], CS['y'][-2])
    
    
def plot_crit(ax, coords_Q, stat_Q, t):


    mc = plt.contourf(ds.x, ds.y, ds.R_2d[t, our_level], 
                cmap='PiYG', 
                  vmin=-np.nanmax(np.abs(ds.R_2d[:, our_level])), vmax=np.nanmax(np.abs(ds.R_2d[:, our_level])),
                 )
#     c = plt.colorbar(mc)

    plt.title(str(ds.Time[t].values)[:16])

    plt.xlim(0,100)
    plt.ylim(0,100)

    # fig.colorbar(Qcr, orientation='vertical')  

    Q_center = ax.scatter(stat_Q['x'], stat_Q['y'], 
                          s=7, c='deeppink', marker='*', alpha=1, label='geom')


    plt.grid(linestyle = ':', linewidth = 0.5)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=9)

    
def time_plot_tracks(ax, CS_tracks_list_100, our_time):
    for CS in CS_tracks_list_100:
        if np.isin(our_time, CS['t']).any():
            
            stop_time = np.argwhere(np.array(CS['t']) == our_time)[0][0]
            
            if np.sum(~np.isnan(CS['x'])) > 12:
                ax.plot(CS['x'][:stop_time+1], CS['y'][:stop_time+1], alpha=0.9, lw=0.5, c='k')
                
            _ = get_mean_wspd(ax, CS, stop_time)
            
            
            
def compute_means_uv(CS, i, y_in, y_out, x_in, x_out, our_level):
    u = np.nanmean(ds['ua'][CS['t'][i], our_level, 
                          y_in:y_out, x_in:x_out])
    v = np.nanmean(ds['va'][CS['t'][i], our_level, 
                          y_in:y_out, x_in:x_out])
    return u, v


def get_mean_wspd(ax, CS, i):
    
    hw = 2*int(CS['rad'][i])
    
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
            
    if y_out >= len(ds.y):
        y_out = -1
    if x_out >= len(ds.x):
        x_out = -1

    u, v = compute_means_uv(CS, i, y_in, y_out, x_in, x_out, our_level=-2)

    wspd = np.sqrt(u*u + v*v)
    tg = np.arctan2(v, u)
    
    print(f'mean wspd: {wspd:.1f} m/s at dir {np.rad2deg(tg):.1f}')
    
    dt = 60*60*3
    
    ax.plot([x, x+u/dist_m*dt], [y, y+v/dist_m*dt], 'blue')
    
    p = mpatches.Rectangle(xy=(x - hw/2,y - hw/2), width=hw, height=hw,linewidth=0.5, 
                                     edgecolor='blue', facecolor='none',)
    ax.add_patch(p)
    
    return wspd, tg



def plot_track_with_DBSCAN(start_time, stop_time, name):
    
    for t in tqdm(range(start_time, stop_time)):
        
        coords_Q = for_DBSCAN(ds[crit][t, our_level], th_Q, crit)  
        coords_Q, n_clusters_lambda2 = clustering_DBSCAN_rortex_C(coords_Q, min_samples=min_samples, eps=eps)

        coords_Q, n_clusters_Q = DBSCAN_filter(coords_Q, points=CS_points_th)
        
        stat_Q = get_stat(coords_Q, n_clusters_lambda2, dist_m)
        
        
        fig = plt.figure(figsize=(5, 5), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        plot_crit(ax, coords_Q, stat_Q, t)
        
        time_plot_tracks(ax, CS_tracks_list, t)    
        
        if not os.path.exists(f"{path_dir}/pics/{name}"):
            os.makedirs(f"{path_dir}/pics/{name}")
        plt.savefig(f"{path_dir}/pics/{name}/track_{(t-start_time):05d}.png", 
                    dpi=200, bbox_inches="tight", transparent=False)
        
        
plot_track_with_DBSCAN(0, 2919, name)