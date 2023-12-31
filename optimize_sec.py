import numpy as np

from sec import *
import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
import optuna

syear, eyear = 2016, 2016
R_I = 100000.+6378100.  # Radius of constructed current surface
r = 6378100.  # Radius from the center of the Earth to a stati
B_param = "dbn_geo"

np.random.seed(0)

stations_list = ['YKC', 'CBB', 'BLC', 'SIT', 'BOU', 'VIC', 'NEW', 'OTT', 'FRD', 'GIM', 'FCC', 'FMC', 'FSP',
                 'SMI', 'ISL', 'PIN', 'RAL', 'INK', 'CMO', 'IQA', 'LET',
                 'T16', 'T32', 'T33', 'T36']
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

def SEC_fit(trial, plot=False):
    stations_list = ['YKC', 'CBB', 'BLC', 'SIT', 'BOU', 'VIC', 'NEW', 'OTT', 'FRD', 'GIM', 'FCC', 'FMC', 'FSP',
                     'SMI', 'ISL', 'PIN', 'RAL', 'INK', 'CMO', 'IQA', 'LET',
                     'T16', 'T32', 'T33', 'T36']
    station_coords_list = [np.array([62.48, 69.1, 64.33, 57.07, 40.13, 48.52, 48.27, 45.4, 38.2, 56.38, 58.76, 56.66,
                                     61.76, 60.02, 53.86, 50.2, 58.22, 68.25, 64.87, 63.75, 49.64, 39.19, 49.4, 54.0,
                                     54.71]),
                           np.array([245.52, 255.0, 263.97, 224.67, 254.77, 236.58, 242.88, 284.45, 282.63, 265.36,
                                     265.92, 248.79, 238.77, 248.05, 265.34, 263.96, 256.32, 226.7, 212.14, 291.48,
                                     247.13, 240.2, 277.7, 259.1, 246.69])]
    station_geocolats = np.pi / 2 - np.pi / 180 * station_coords_list[0]
    station_geolons = np.pi / 180 * station_coords_list[1]
    epsilon = trial.suggest_float(name="epsilon", low=1e-6, high=1e-1, log=True)
    n_sec_lat, n_sec_lon = trial.suggest_int("n_sec_lat", 4, 28), trial.suggest_int("n_sec_lon", 5, 50)  # Number of rows and columns respectively of SECSs that will exist in the grid
    print(f"\nStarting trial with epsilon={epsilon}, n_sec_lat={n_sec_lat}, n_sec_lon={n_sec_lon}\n"
          f"Go read a book. This could take a while.")
    sec_coords_list = [np.linspace(45.5, 55.5, n_sec_lat), np.linspace(230.5, 280.5, n_sec_lon)]
    sec_geocolats = np.pi / 2 - np.pi / 180 * sec_coords_list[0]
    sec_geolons = np.pi / 180 * sec_coords_list[1]
    # Generate full list of coords for every SEC in row-major format
    all_sec_colats, all_sec_lons = [], []
    for colat in sec_geocolats:
        for lon in sec_geolons:
            all_sec_colats.append(colat)
            all_sec_lons.append(lon)

    rmses = []
    for station_num in tqdm.trange(len(stations_list), desc="Analyzing errors at each station"):
        this_station_geocolat, this_station_lon = station_geocolats[station_num], station_geolons[station_num]
        coords_to_be_reduced = station_coords_list.copy()
        reduced_station_coords_list = np.zeros((2, len(station_coords_list[0]) - 1))
        reduced_station_coords_list[0], reduced_station_coords_list[1] = np.delete(coords_to_be_reduced[0], station_num), np.delete(coords_to_be_reduced[1], station_num)
        true_mag = Z_matrix[:, station_num]
        this_Z_matrix = np.delete(Z_matrix, [2*station_num, 2*station_num+1], axis=1)

        I_interp_df = gen_current_data(this_Z_matrix, reduced_station_coords_list, sec_coords_list, epsilon=epsilon,
                                       disable_tqdm=True)

        interps = np.zeros(len(I_interp_df))
        for timestep in range(len(interps)):
            interps[timestep] = predict_B_timestep(I_interp_df.iloc[timestep], B_param,
                                                         poi_colat=this_station_geocolat,
                                                         poi_lon=this_station_lon,
                                                         all_sec_colats=all_sec_colats, all_sec_lons=all_sec_lons,
                                                         r=r, R_I=R_I)

        rmses.append(np.sqrt(mean_squared_error(true_mag, interps)))

    return np.mean(rmses)


study = optuna.create_study()
study.optimize(SEC_fit, n_trials=50)
study.trials_dataframe().to_csv("results.csv")