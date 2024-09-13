import sys

sys.path.insert(2, '/storage/kubrick/vkoshkina/scripts/DBSCAN_tracking/tracking_lib')

# from simple_tracking_scripts import *
# from tracking_with_CS_speed import *

from tracking_step_by_step import *


tracking_type = 'tracking_step_by_step'

data_type = 'ERA5'
# data_type = 'HiRes'


# print('dataset: ')
# data_type = input()

print('circ: ')
circ = input()


dist_m, our_level, x_unit, y_unit, u_unit, v_unit, time_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type)

if data_type == 'HiRes':
    level = 12
elif data_type == 'ERA5':
    level = 9
    
path_dir_data = '/storage/kubrick/vkoshkina/data'
sub_dir = f'rortex_{data_type}_for_tracking_level_{level}'
name = f'rortex_2d_criteria_{data_type}_level_{level}'

years = np.arange(2010,2011)
months = np.arange(6,10)

for year in years:
    
    path_data_dir = f'/storage/kubrick/vkoshkina/data/{data_type}_tracks/{tracking_type}/DBSCAN_{min_samples:02d}-{eps:02d}-{CS_points_th:02d}/{year}/tracks_{circ}'

    if not os.path.exists(f'{path_data_dir}'):
            os.makedirs(f'{path_data_dir}')
    
    print(f'year: {year}')
    
    CS_tracks_list = []
    
    for month in months:
        
        ds = xr.open_dataset(f"{path_dir_data}/{sub_dir}/{name}_{year}-{month:02d}.nc")
        
        # init field of CS
        coords_Q = vortex.for_DBSCAN(ds[crit][0, our_level], th, crit)  
        coords_Q, n_clusters_lambda2 = clustering_DBSCAN_rortex_C(coords_Q, min_samples=min_samples, eps=eps, circ=circ)
        coords_Q, n_clusters_Q = vortex.DBSCAN_filter(coords_Q, points=CS_points_th)

        stat_Q = get_stat(coords_Q, n_clusters_lambda2, dist_m, circ=circ, CS_points_th=CS_points_th)

        CS_tracks_list = track_init(ds, data_type, stat_Q, CS_tracks_list, 0, time_unit)

        dt_step = np.timedelta64(ds[time_unit].values[1] - ds[time_unit].values[0], 's').astype(int) 
            
        for t in tqdm(range(1,len(ds[time_unit]))):

            CS_tracks_list = step_of_tracking(CS_tracks_list, ds, data_type, t, circ=circ, dt_step=dt_step)

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
#                                "wspd": TC_wspd,
        })

        start_date = TC['time'][0]
        dist = TC['cluster']
        dt = str(start_date)[:-13]

        df_TC_Danielle.to_csv(f'{path_data_dir}/{int(dist):06d}_track_{dt}.csv')