from sec import *
import preprocessing
from sklearn.metrics import mean_squared_error
import optuna
from multiprocessing import Pool


syear, eyear = 2009, 2019
R_I = 100000.+6378100.  # Radius of constructed current surface
r = 6378100.  # Radius from the center of the Earth
B_param = "dbn_geo"  # "dbn_geo" or "dbe_geo", the vector component of dB to use

np.random.seed(0)
num_mp_procs = 16  # Number of multiprocessing procs you want to use. More is not always better!
num_optuna_trials = 50  # Number of trials to run during optimization

stations_list = ['YKC', 'BLC', 'MEA', 'SIT', 'BOU', 'VIC', 'NEW', 'OTT', 'GIM', 'DAW', 'FCC', 'FMC',
                 'FSP', 'SMI', 'ISL', 'PIN', 'RAL', 'RAN', 'CMO', 'IQA', 'C04', 'C06', 'C10', 'T36']
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


def calculate_station_error(station_num, station_geocolats, station_geolons, station_coords_list, sec_coords_list,
                            epsilon, all_sec_colats, all_sec_lons):
    """
    This function calculate the interpolated magnetic field at one station given the magnetic field measurement at
    all other stations. Then, it calculates the error between the interpolated magnetic field and the actual magnetic
    field measured at that station. It returns the RMSE across all timesteps in the dataset.

    :param station_num: The index of the station (in the list of stations) at which the field is being interpolated.
    :param station_geocolats: A list-like of the geographic colatitudes of all stations, in radians.
    :param station_geolons: A list-like of the geographic longitudes of all stations, in radians.
    :param station_coords_list: A list-like containing two list-likes: The geographic latitudes of all stations in
     degrees, and the geographic longitudes of all stations in degrees, in that order.
    :param sec_coords_list: A list-like containing two list-likes: The geographic latitudes of all SEC poles in
     degrees, and the geographic longitudes of all SEC poles in degrees, in that order.
    :param epsilon: The cutoff value used for regularization during SVD. This is determined by running optimize_sec.py .
    :param all_sec_colats: A full list of SEC colatitudes in row-major format, as generated SEC_fit() .
    :param all_sec_lons: A full list of SEC longitudes in row-major format, as generated SEC_fit() .
    :return: The RMSE between all the real observations at the station and all of the interpolations at the station.
    """

    this_station_geocolat, this_station_lon = station_geocolats[station_num], station_geolons[station_num]
    # Make a copy of station_coords_list() so we don't remove entries from the variable in the SEC_fit() scope
    coords_to_be_reduced = station_coords_list.copy()
    reduced_station_coords_list = np.zeros((2, len(station_coords_list[0]) - 1))
    reduced_station_coords_list[0], reduced_station_coords_list[1] = (np.delete(coords_to_be_reduced[0], station_num),
                                                                      np.delete(coords_to_be_reduced[1], station_num))
    true_mag = Z_matrix[:, station_num]
    reduced_Z_matrix = np.delete(Z_matrix, [2 * station_num, 2 * station_num + 1], axis=1)

    # Obtain the SEC scaling factors produced by the reduced set of magnetometers
    I_interp_df = gen_current_data(reduced_Z_matrix, reduced_station_coords_list, sec_coords_list, epsilon=epsilon,
                                   disable_tqdm=True)

    # Obtain the magnetic field interpolations at just the location of the station of interest
    interps = np.zeros(len(I_interp_df))
    for timestep in range(len(interps)):
        interps[timestep] = predict_B_timestep(I_interp_df.iloc[timestep], B_param,
                                               poi_colat=this_station_geocolat,
                                               poi_lon=this_station_lon,
                                               all_sec_colats=all_sec_colats, all_sec_lons=all_sec_lons,
                                               r=r, R_I=R_I)

    return np.sqrt(mean_squared_error(true_mag, interps))


def SEC_fit(trial):
    """
    This is the objective function of the Optuna optimization. It carries out the leave-one-out procedure at each
    magnetometer station and returns the mean RMSE across all those stations as the loss to the optimizer. This function
    can take a very long time to run, so be as conservative as you can with SEC hyperparameters, the size of the
    dataset you fit it on, and the number of trials you use for optimization.
    :param trial: The trial variable required by Optuna to keep track of progress.
    :return: The mean RMSE across all station. This is the loss that the optimization will minimize.
    """

    station_coords_list = [np.array([62.48, 64.33, 54.62, 57.07, 40.13, 48.52, 48.27, 45.4, 56.38, 64.05, 58.76,
                                     56.66, 61.76, 60.02, 53.86, 50.2, 58.22, 62.82, 64.87, 63.75, 50.06, 53.35,
                                     47.66, 54.71]),
                           np.array([245.52, 263.97, 246.65, 224.67, 254.77, 236.58, 242.88, 284.45, 265.36, 220.89,
                                     265.92, 248.79, 238.77, 248.05, 265.34, 263.96, 256.32, 267.89, 212.14, 291.48,
                                     251.74, 247.03, 245.79, 246.69])]
    station_geocolats = np.pi / 2 - np.pi / 180 * station_coords_list[0]  # Geographic colatitudes of stations
    station_geolons = np.pi / 180 * station_coords_list[1]  # Geographic longitudes of stations
    epsilon = trial.suggest_float(name="epsilon", low=1e-6, high=1e-1, log=True)
    # Define the number of rows and columns respectively of SECSs that will exist in the grid
    n_sec_lat, n_sec_lon = trial.suggest_int("n_sec_lat", 5, 20), trial.suggest_int("n_sec_lon", 20, 40)
    print(f"\nStarting trial with epsilon={epsilon}, n_sec_lat={n_sec_lat}, n_sec_lon={n_sec_lon}\n"
          f"Go read a book. This could take a while.")
    sec_coords_list = [np.linspace(40.5, 60.5, n_sec_lat), np.linspace(220.5, 290.5, n_sec_lon)]
    sec_geocolats = np.pi / 2 - np.pi / 180 * sec_coords_list[0]
    sec_geolons = np.pi / 180 * sec_coords_list[1]
    # Generate full list of coords for every SEC in row-major format
    all_sec_colats, all_sec_lons = [], []
    for colat in sec_geocolats:
        for lon in sec_geolons:
            all_sec_colats.append(colat)
            all_sec_lons.append(lon)

    # Calculate the RMSEs at each station in parallel for this trial
    multiproc_args = [(station_num, station_geocolats, station_geolons, station_coords_list, sec_coords_list,
                            epsilon, all_sec_colats, all_sec_lons) for station_num in range(len(station_geocolats))]
    with Pool(processes=num_mp_procs) as pool:
        rmses = pool.starmap(calculate_station_error, multiproc_args)

    return np.mean(rmses)


study = optuna.create_study()
study.optimize(SEC_fit, n_trials=num_optuna_trials)
study.trials_dataframe().to_csv(f"results_{syear}-{eyear}.csv")
