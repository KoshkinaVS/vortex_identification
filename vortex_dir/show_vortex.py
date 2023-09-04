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

def show_centroids(coords_Q, stat, crit):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(1, 1, 1)

    cmap = plt.cm.get_cmap(cmaps.psgcap, max(coords_Q['cluster'])+1)

    Qcr = ax.scatter(coords_Q['x'], coords_Q['y'], c=coords_Q['cluster'], cmap=cmap,
                     s=2,alpha=0.5)
    #                  c=coords_Q['cluster'], cmap=cmap, 
#                                                  alpha=1, transform=ccrs.PlateCarree())
    # fig.colorbar(Qcr, orientation='vertical')  
    plt.title('Centroids of clusters for '+crit+'-criterion', fontsize=25)
    Q_center = ax.scatter(stat['x'], stat['y'], 
                          s=50, c='deeppink', marker='*', alpha=1, label='geom')

#     ax.set_extent([np.min(ds.XLONG[our_time][0])-3, -13, np.min(ds.XLAT[our_time])+4, np.max(ds.XLAT[our_time][-1])], ccrs.PlateCarree())

    
    for x, y, tex in zip(stat['x']+5, stat['y']+2, stat['cluster']):
        t = plt.text(x, y, round(int(tex), 1), horizontalalignment='center', 
                 verticalalignment='center', fontdict={'color':'b', 'size': 10}, )
#                      transform=ccrs.PlateCarree())
    plt.grid(linestyle = ':', linewidth = 0.5)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=9)
    
    plt.legend()
    plt.show()
    
    
def show_DBSCAN_centers(coords_lambda2, stat_lambda2, ds, our_time):
    fig = plt.figure(figsize=(15, 15))
    
    ax = fig.add_subplot(2, 1, 1, projection=ccrs.Stereographic(central_latitude=45.0, central_longitude=-45))
    ax.set_global()
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, color='grey', alpha=0.7)
    
    n_clusters = len(np.unique(coords_lambda2['cluster']))
    
    cmap = plt.cm.get_cmap(cmaps.psgcap, n_clusters+1)
    Qcr = plt.scatter(coords_lambda2.lon, coords_lambda2.lat, c=coords_lambda2.cluster, s=1, cmap=cmap,
                     transform=ccrs.PlateCarree())
#     Qcr = plt.hexbin(coords_lambda2.x, coords_lambda2.y, C=coords_lambda2.cluster, cmap=cmap)
    Q_center = plt.scatter(stat_lambda2['lon'], stat_lambda2['lat'], 
                          s=40, 
                           ec='k',
                           c='yellow', marker='*',
                           alpha=1,
                          transform=ccrs.PlateCarree())
    
#     fig.colorbar(Qcr, orientation='vertical') 
    # plt.title('Estimated number of clusters for $\lambda_2$-criterion: %d' % n_clusters_lambda2, fontsize=18)
    plt.grid(linestyle = ':', linewidth = 0.5)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=9)
    plt.legend(title='$N=$'+str(n_clusters), title_fontsize=15, loc='lower left')
    ax.set_extent([np.min(ds.XLONG[our_time][0])-3, -13, np.min(ds.XLAT[our_time])+4, np.max(ds.XLAT[our_time][-1])], ccrs.PlateCarree())
    
    # plt.title(f'$\lambda_2$>'+str(th_lambda2)[:5], fontsize=20)
#     plt.savefig(f"../data/CCA{area:02d}-DBS{min_samples:02d}-{eps:02d}.png", dpi=500, bbox_inches="tight", transparent=True)



    
  