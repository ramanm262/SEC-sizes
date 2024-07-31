from sec import *
import preprocessing
from multiprocessing import Pool

# 10x35 SEC grid and 14x32 POI grid for 2009-2019 with num_mp_procs=16 took 256 hours


syear, eyear = 2009, 2019
n_sec_lat, n_sec_lon = 10, 35  # Number of rows and columns respectively of SECSs that will exist in the grid
n_poi_lat, n_poi_lon = 14, 32  # Number of rows and columns respectively of POIs that will exist in the grid
R_I = 100000.+6378100.  # Radius of constructed current surface
r = 6378100.  # Radius from the center of the Earth to a station
B_param = "dbn_geo"
stations_list = ['YKC', 'BLC', 'MEA', 'SIT', 'BOU', 'VIC', 'NEW', 'OTT', 'GIM', 'DAW', 'FCC', 'FMC',
                 'FSP', 'SMI', 'ISL', 'PIN', 'RAL', 'RAN', 'CMO', 'IQA', 'C04', 'C06', 'C10', 'T36']
station_coords_list = [np.array([62.48, 64.33, 54.62, 57.07, 40.13, 48.52, 48.27, 45.4, 56.38, 64.05, 58.76,
                                 56.66, 61.76, 60.02, 53.86, 50.2, 58.22, 62.82, 64.87, 63.75, 50.06, 53.35,
                                 47.66, 54.71]),
                       np.array([245.52, 263.97, 246.65, 224.67, 254.77, 236.58, 242.88, 284.45, 265.36, 220.89,
                                 265.92, 248.79, 238.77, 248.05, 265.34, 263.96, 256.32, 267.89, 212.14, 291.48,
                                 251.74, 247.03, 245.79, 246.69])]
poi_coords_list = [np.linspace(45, 55, n_poi_lat), np.linspace(230, 280, n_poi_lon)]
sec_coords_list = [np.linspace(40.5, 60.5, n_sec_lat), np.linspace(220.5, 290.5, n_sec_lon)]
epsilon = 0.09323151264778985
load_scaling_factors = False
num_mp_procs = 16

station_geocolats = np.pi / 2 - np.pi / 180 * station_coords_list[0]
station_geolons = np.pi / 180 * station_coords_list[1]
sec_geocolats = np.pi / 2 - np.pi / 180 * sec_coords_list[0]
sec_geolons = np.pi / 180 * sec_coords_list[1]
poi_geocolats = np.pi / 2 - np.pi / 180 * poi_coords_list[0]
poi_geolons = np.pi / 180 * poi_coords_list[1]

# Generate full list of coords for every SEC in row-major format
all_sec_colats, all_sec_lons = [], []
for colat in sec_geocolats:
    for lon in sec_geolons:
        all_sec_colats.append(colat)
        all_sec_lons.append(lon)

# Generate full list of coords for every POI in row-major format
all_poi_colats, all_poi_lats, all_poi_lons = [], [], []
for colat in poi_geocolats:
    for lon in poi_geolons:
        all_poi_colats.append(colat)
        all_poi_lats.append(90-colat*180/np.pi)
        all_poi_lons.append(lon)
poi_lons_mesh, poi_lats_mesh = np.meshgrid(poi_coords_list[1], poi_coords_list[0])

# Load SuperMAG data from file
n_mag_data = preprocessing.load_supermag(stations_list, syear, eyear, "dbn_geo", saving=False)
e_mag_data = preprocessing.load_supermag(stations_list, syear, eyear, "dbe_geo", saving=False)
mag_data_index = n_mag_data.index
# Create a matrix whose rows are timesteps and columns are Z vectors described in Amm & Viljanen 1999
Z_matrix = [0] * 2 * len(n_mag_data.columns)
station_num = 0
while station_num < len(n_mag_data.columns):
    Z_matrix[2 * station_num] = n_mag_data.iloc[:, station_num]
    Z_matrix[2 * station_num + 1] = e_mag_data.iloc[:, station_num]
    station_num += 1
Z_matrix = pd.concat(Z_matrix, axis=1)
Z_matrix = Z_matrix.to_numpy()

if load_scaling_factors:
    I_interp_df = pd.read_hdf(f"I_interps_{n_sec_lat}by{n_sec_lon}_{syear}-{eyear}.h5", key="I_interp_df")
else:
    I_interp_df = gen_current_data(Z_matrix, station_coords_list, sec_coords_list, epsilon=epsilon)
    I_interp_df.to_hdf(f"I_interps_{n_sec_lat}by{n_sec_lon}_{syear}-{eyear}.h5", key="I_interp_df")

all_B_interps = [pd.Series(np.zeros((len(I_interp_df),)))] * len(poi_coords_list[0]) * len(poi_coords_list[1])
all_B_interps = pd.concat(all_B_interps, axis=1)
for timestep in tqdm.trange(len(I_interp_df), desc="Generating B-field interpolation"):
    multiproc_args = [(I_interp_df.iloc[timestep], B_param, all_poi_colats[poi_num],
                       all_poi_lons[poi_num], all_sec_colats, all_sec_lons, r, R_I)
                      for poi_num in range(len(poi_coords_list[0]) * len(poi_coords_list[1]))]
    with Pool(processes=num_mp_procs) as pool:
        B_poi_interps = pool.starmap(predict_B_timestep, multiproc_args)
    all_B_interps.iloc[timestep] = pd.Series(B_poi_interps, name=timestep)

# Save all_B_interps to h5
all_B_interps.to_hdf(f"all_B_interps_{n_sec_lat}by{n_sec_lon}_{syear}-{eyear}.h5", key=B_param)

print("Done")

