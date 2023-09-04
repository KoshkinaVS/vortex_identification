from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA

import datetime
# import cv2
import scipy as sp
# from scipy.ndimage import label, generate_binary_structure
from scipy import interpolate


# import seaborn as sns

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# from shapely.geometry import Polygon
# import matplotlib.patches as mpatches
from matplotlib import gridspec

from tqdm import tqdm

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from mpl_toolkits.axes_grid1 import make_axes_locatable

import cmaps

from sklearn.cluster import DBSCAN

dist_m = xr.open_dataset("/storage/NAADSERVER/NAAD/HiRes/PressureLevels/w/2009/w_2009-08-10.nc").attrs['DX'] # 13897.18 m


def get_curr_date(ds, our_time):
    date = np.datetime_as_string(ds.XTIME[our_time].values, unit='m')
    return date


def for_DBSCAN(X, threshold, crit):
    
    if crit=='Q':
        X_arr = X.to_dataframe().dropna(how='any')
        print(len(X_arr.Q))
        X_arr.Q[X_arr.Q < threshold] = None
        X_arr = X_arr.dropna(how='any')
        print(len(X_arr.Q))
        
        coords = X_arr.Q.index.to_frame(name=['y', 'x'], index=False)
        coords['lat'] = X_arr.XLAT.values
        coords['lon'] = X_arr.XLONG.values
        coords['crit'] = X_arr.Q.values
        
    if crit=='delta':
        X_arr = X.to_dataframe().dropna(how='any')
        print(len(X_arr.delta))
        X_arr.delta[X_arr.delta < threshold] = None
        X_arr = X_arr.dropna(how='any')
        print(len(X_arr.delta))

        coords = X_arr.delta.index.to_frame(name=['y', 'x'], index=False)        
        coords['lat'] = X_arr.XLAT.values
        coords['lon'] = X_arr.XLONG.values
        coords['crit'] = X_arr.delta.values
        
    if crit=='lambda2':
        X_arr = X.to_dataframe().dropna(how='any')
        print(len(X_arr.lambda2))
        X_arr.lambda2[X_arr.lambda2 < threshold] = None
        X_arr = X_arr.dropna(how='any')
        print(len(X_arr.lambda2))

        coords = X_arr.lambda2.index.to_frame(name=['y', 'x'], index=False)
        coords['lat'] = X_arr.XLAT.values
        coords['lon'] = X_arr.XLONG.values
        coords['crit'] = X_arr.lambda2.values
                
#     plt.scatter(coords['x'], coords['y'], c=coords['crit'], s=1)
#     plt.show()
    
    return coords

# get pd.df with cluster label for each point
def clustering_DBSCAN(coords_lambda2, eps=10., min_samples=10, metric='euclidean'):
    
    db1 = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=4) # для декартовой сетки 
    
    coords = np.array([coords_lambda2.x, coords_lambda2.y]).T
    y_db1 = db1.fit_predict(coords)
#     labels = db1.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_lambda2 = len(set(y_db1)) - (1 if -1 in y_db1 else 0)
    n_noise_lambda2 = list(y_db1).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_lambda2)
    print('Estimated number of noise points: %d' % n_noise_lambda2)
#     print("Silhouette Coefficient: %0.3f"
#           % metrics.silhouette_score(np.radians(coords_lambda2), labels))

    coords_lambda2['cluster'] = y_db1
    
#     plt.scatter(coords_lambda2.x, coords_lambda2.y, c=coords_lambda2.cluster, s=1)
        
    return coords_lambda2, n_clusters_lambda2


def get_boundary_polyg(clst_coords):
    
    x_c = clst_coords.x.mean()
    y_c = clst_coords.y.mean()
    
    clst_coords['rad'] = np.sqrt((clst_coords.x - x_c)**2 + (clst_coords.y - y_c)**2)
    rad = clst_coords.rad.max()  
    
    point_max = clst_coords[clst_coords.rad == rad]
    x_max = point_max.x.values[0]
    y_max = point_max.y.values[0]
    
    phi_0 = np.degrees(np.arcsin((y_max-y_c)/rad))
    
    if rad*np.cos(np.radians(phi_0)) != x_max:
        phi_0 = -phi_0

    x_bound = []
    y_bound = []
    
    lons_circle = []
    lats_circle = []
    
#     x_bound.append(x_max)
#     y_bound.append(y_max)
    
#     lons_circle.append(x_max)
#     lats_circle.append(y_max)
    
    angle = 2
    
    for i in range(360//angle):
    
        phi = phi_0 + i*angle
        x = rad*np.cos(np.radians(phi))
        y = rad*np.sin(np.radians(phi))
        
        lon = x_c + x
        lat = y_c + y
        
        lons_circle.append(lon)
        lats_circle.append(lat)
        
        
#         phi = np.degrees(np.arcsin((lat-y_c)/rad))
        
        radi = rad
        while radi >= 0:   
#             rad_min = 2
            rad_min = 10
    
            x_b = None
            y_b = None
    
            rads = np.sqrt((clst_coords.y - lat)**2 + (clst_coords.x - lon)**2)
            if np.nanmin(rads) <= rad_min:
                x_b = clst_coords.x.iloc[np.argmin(rads)]
                y_b = clst_coords.y.iloc[np.argmin(rads)]

            if x_b == None:
                radi -= np.sqrt(2)
                
                x = radi*np.cos(np.radians(phi))
                y = radi*np.sin(np.radians(phi))

                lon = x_c + x
                lat = y_c + y
            else: break
                
#         rads = np.sqrt((clst_coords.y - lat)**2 + (clst_coords.x - lon)**2)
#         x_b = clst_coords.x.iloc[np.argmin(rads)]
#         y_b = clst_coords.y.iloc[np.argmin(rads)]
        
        if x_b != None:
            x_bound.append(x_b)
            y_bound.append(y_b)    

    bound_coords = np.array([x_bound, y_bound]).T
    radius = np.sqrt((bound_coords[:,1] - y_c)**2 + (bound_coords[:,0] - x_c)**2)
    
#     radius = np.sqrt((clst_coords.y - y_c)**2 + (clst_coords.x - x_c)**2)
    
    radius_median = np.median(radius)
    
    radius_min = np.percentile(radius, 5)
    radius_max = np.percentile(radius, 95)
    
    return clst_coords, x_c, y_c, bound_coords, radius_median, radius_min, radius_max


# compute statistic for each cluster: radius, center, and return pd.df for 1 level
def get_stat(coords_Q_cluster, n_clusters_Q, dist_m):

    centroid_lat = []
    centroid_lon = []
    centroid_idx = []
    
    
    centroid_x = []
    centroid_y = []
    
    crit_centroid_x = []
    crit_centroid_y = []
    
    rad = []
    rad_min = []
    rad_max = []
    geom = []
    elong = []
    
    radius_eff = []
    
    
    for n in range(n_clusters_Q):
    
        coords_Q_c = coords_Q_cluster[coords_Q_cluster['cluster'] == n]
        
        if len(coords_Q_c) >= 10:
            
            N_points = len(coords_Q_c)
            rad_eff = np.sqrt(N_points/np.pi)*dist_m/1000
            radius_eff.append(rad_eff)
            
            centroid_idx.append(n)
            
            clst_coords, lon_c, lat_c, bound_coords, radius_median, radius_min, radius_max = get_boundary_polyg(coords_Q_c)

            x_c = coords_Q_c.x.mean()
            y_c = coords_Q_c.y.mean()
            
#             lon_c = coords_Q_c.lon.mean()
#             lat_c = coords_Q_c.lat.mean()
            
#             centroid_lat.append(lat_c)
#             centroid_lon.append(lon_c)
            
            centroid_x.append(x_c)
            centroid_y.append(y_c)
            
            crit_max = np.max(coords_Q_c.crit)
            center = coords_Q_c[coords_Q_c['crit'] == crit_max]
            crit_centroid_x.append(float(center.x))
            crit_centroid_y.append(float(center.y))
            
            rad.append(radius_median*dist_m/1000) # km
            rad_min.append(radius_min*dist_m/1000)
            rad_max.append(radius_max*dist_m/1000)
            
#             try:
#                 ellipse = cv2.fitEllipse(bound_coords) 
#                 elongation = ellipse[1][1]/ellipse[1][0]
#                 elong.append(elongation)
#             except Exception:   
#                 elong.append(None)
                
            geom.append(bound_coords)

        else:
#             print(f'cluster {n} is too small!')
            geom.append(None)
            rad.append(0)
            rad_max.append(0)
            rad_min.append(0)
            elong.append(0)
            
            
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
#                          'lon': centroid_lon, 'lat': centroid_lat,
                         
                         'x_crit': crit_centroid_x, 'y_crit': crit_centroid_y,
                         'radius': rad,
                         'radius_min': rad_min, 
                         'radius_max': rad_max,
                         'rad_eff': radius_eff,
                        
                         'geometry': geom,
#                          'elongation_CV': elong
                        })
    
    
    stat = stat.dropna()   
    
#     stat['delta_center'] = np.sqrt((stat.x - stat.x_crit)**2 
#                                      + (stat.y - stat.y_crit)**2)

    return stat



def get_spline(ds, year=2010, folder='DBSCAN_7_2_50', grad=False): 

    ths_levels = np.load(f"../data/{year}/{folder}/ths_levels_HiRes_year.npy")  

    x = ds.interp_level.values
    fx = sp.linspace(x[0], x[-1], 50)

    ys = np.nanargmax(ths_levels, axis=2)
    
    if grad:
        grad_ths = np.abs(np.gradient(ths_levels, 1., axis = [2])) 
        ys = np.nanargmax(grad_ths, axis=2)
    
    y = np.nanmean(ys, axis=0)
#     func_4d = approx_season(x, ths_levels, season)
    spl = interpolate.UnivariateSpline(x[::-1], y[::-1], k=5)
    
#     for i in range(len(ths_levels)):
#         plt.scatter(ys[i], x)

    fig = plt.figure(figsize=(5,5), dpi=300)

    ys_std = np.std(ys, axis=0)*0.01
    plt.fill_betweenx(x, y*0.01-ys_std, y*0.01+ys_std, color='r', alpha=0.1)
    
    plt.gca().invert_yaxis()

    plt.tick_params(axis='both', which='both', direction='in', labelsize=9)
    
    plt.ylabel('Высота, гПа')
    plt.xlabel('Пороговое значение')
    plt.grid(linestyle = ':', linewidth = 0.5)
    
#     plt.axvline(x = np.mean(spl(ds.interp_level))*0.01, color = 'darkred', ls='dashed', label = 'mean')
    
    plt.plot(0.01*spl(fx), fx, lw=2, c='r')
#     plt.savefig('../data/func/ths_levels_'+name+'_'+season+'.png', dpi=300, bbox_inches="tight", transparent=True)
    
    return spl, ys

