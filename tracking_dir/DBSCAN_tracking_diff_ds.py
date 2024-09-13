import sys

sys.path.insert(2, '/storage/kubrick/vkoshkina/scripts/DBSCAN_tracking/tracking_lib')

# from simple_tracking_scripts import *
# from tracking_with_CS_speed import *

from tracking_domain_boundary_diff_ds import *


tracking_type = 'track_domain_boundary'

data_type = 'ERA5'
data_type = 'HiRes'


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
    
    print(f'year: {year}')
    
    for month in months:
        
        ds = xr.open_dataset(f"{path_dir_data}/{sub_dir}/{name}_{year}-{month:02d}.nc")

        path_data_dir = f'/storage/kubrick/vkoshkina/data/{data_type}_tracks/{tracking_type}/DBSCAN_{min_samples:02d}-{eps:02d}-{CS_points_th:02d}/{year}/tracks_{circ}'

        if not os.path.exists(f'{path_data_dir}'):
            os.makedirs(f'{path_data_dir}')

        CS_tracks_list = simple_advection_tracking(ds, data_type, 0, circ=circ)

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

            df_TC_Danielle.to_csv(f'{path_data_dir}/{int(dist):06d}_track_{dt}.csv')