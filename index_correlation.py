# A script for computing correlations between perimeter size or aspect ratio and the geomagnetic index of your choice
# over the full solar cycle or over all storms in solar maximum or minimum. For specific storms, use stats_analysis.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stats_analysis import calculate_perimeter, calculate_aspect_ratio, correlation_with_index
from skimage import measure
import tqdm
import preprocessing


def find_index_correlation(config_dict):
    syear, eyear = config_dict["syear"], config_dict["eyear"]
    stations_list = config_dict["stations_list"]
    n_sec_lat, n_sec_lon = config_dict["n_sec_lat"], config_dict["n_sec_lon"]
    n_poi_lat, n_poi_lon = config_dict["n_poi_lat"], config_dict["n_poi_lon"]
    poi_coords_list = config_dict["poi_coords_list"]
    B_param = config_dict["B_param"]
    contour_level = config_dict["contour_level"]
    omni_feature = config_dict["omni_feature"]
    solar_cycle_phase = config_dict["solar_cycle_phase"]

    all_B_interps = pd.read_hdf(f"all_B_interps_{n_sec_lat}by{n_sec_lon}_{syear}-{eyear}.h5", B_param)

    poi_geocolats = np.pi / 2 - np.pi / 180 * poi_coords_list[0]
    poi_geolons = np.pi / 180 * poi_coords_list[1]

    # Generate full list of coords for every POI in row-major format
    all_poi_colats, all_poi_lats, all_poi_lons = [], [], []
    for colat in poi_geocolats:
        for lon in poi_geolons:
            all_poi_colats.append(colat)
            all_poi_lats.append(90-colat*180/np.pi)
            all_poi_lons.append(lon)

    # Load SuperMAG and geomagnetic index data for comparison to interpolations
    mag_data = preprocessing.load_supermag(stations_list, syear, eyear, B_param, saving=False)
    mag_data.drop_duplicates(inplace=True)  # Remove timestamps that are double-counted by immediately successive storms
    all_B_interps.drop_duplicates(inplace=True)
    assert len(mag_data) == len(all_B_interps)
    all_B_interps.index = mag_data.index
    index_data = preprocessing.load_omni(syear, eyear, "/data/ramans_files/omni-feather/", feature=omni_feature)

    if solar_cycle_phase == "minimum":
        print("Only using data from the low part of the solar cycle")
        all_B_interps = all_B_interps.loc[(all_B_interps.index < pd.to_datetime("2010-01-01")) |
                                          (all_B_interps.index >= pd.to_datetime("2018-01-01"))]
    elif solar_cycle_phase == "maximum":
        print("Only using data from the high part of the solar cycle")
        all_B_interps = all_B_interps.loc[(all_B_interps.index >= pd.to_datetime("2012-01-01")) &
                                          (all_B_interps.index < pd.to_datetime("2014-07-01"))]
    else:
        print("Using data from the entire solar cycle")
    mag_data = mag_data.loc[all_B_interps.index]
    # index_data is pared to the same index later in this script

    perimeters = []
    aspect_ratios = []
    for timestep in tqdm.trange(len(all_B_interps),
                                desc=f'Generating heatmaps for solar cycle phase "{solar_cycle_phase}"'):
        heatmap_data = np.abs(np.array(all_B_interps.iloc[timestep]).reshape((n_poi_lat, n_poi_lon)))
        contours = measure.find_contours(heatmap_data, level=contour_level)
        # Each element of the resulting list is a list of coords in row, col format (not degrees)

        this_perimeters = []
        this_aspect_ratios = []
        for contour in contours:
            if contour[0, 0] == contour[-1, 0] and contour[0, 1] == contour[-1, 1]:  # If contour is closed
                p = calculate_perimeter(contour.copy(), poi_coords_list=poi_coords_list)
                ar = calculate_aspect_ratio(contour.copy(), poi_coords_list=poi_coords_list)
                this_perimeters.append(p)
                this_aspect_ratios.append(ar)
        perimeters.append(this_perimeters)
        aspect_ratios.append(this_aspect_ratios)

    timestamps = mag_data.index
    perimeter_maxes = [0]*len(perimeters)
    ar_maxes = [0]*len(aspect_ratios)  # aspect_ratios and perimeters should have the same length
    for timestep in range(len(perimeters)):
        if len(perimeters[timestep]) > 0:
            perimeter_maxes[timestep] = np.max(perimeters[timestep])
            ar_maxes[timestep] = np.max(aspect_ratios[timestep])  # There will be ARs whenever there are perimeters
    perimeter_maxes = pd.DataFrame(perimeter_maxes, index=timestamps)
    ar_maxes = pd.DataFrame(ar_maxes, index=timestamps)
    index_data = index_data.loc[perimeter_maxes.index]

    return correlation_with_index(perimeter_maxes, index_data), correlation_with_index(ar_maxes, index_data)


if __name__ == "__main__":
    syear, eyear = 2009, 2019
    stations_list = ['YKC', 'BLC', 'MEA', 'SIT', 'BOU', 'VIC', 'NEW', 'OTT', 'GIM', 'DAW', 'FCC', 'FMC',
                     'FSP', 'SMI', 'ISL', 'PIN', 'RAL', 'RAN', 'CMO', 'IQA', 'C04', 'C06', 'C10', 'T36']
    n_sec_lat, n_sec_lon = 10, 35  # # of rows and columns respectively of SECSs that will exist in the grid
    n_poi_lat, n_poi_lon = 14, 32  # # of rows and columns respectively of POIs that will exist in the grid
    poi_coords_list = [np.linspace(45, 55, n_poi_lat), np.linspace(230, 280, n_poi_lon)]
    B_param = "dbn_geo"
    contour_level = 25.95
    omni_feature = "SYM_H"
    plot_every_n_interps = 5000
    solar_cycle_phase = "full"  # "minimum", "maximum", or "full"
    stats_plots_location = "stats_plots/"
    interp_plots_location = "interp_plots/"

    config_dict = {"syear": syear, "eyear": eyear, "stations_list": stations_list,
                   "n_sec_lat": n_sec_lat, "n_sec_lon": n_sec_lon, "n_poi_lat": n_poi_lat, "n_poi_lon": n_poi_lon,
                   "poi_coords_list": poi_coords_list, "B_param": B_param, "contour_level": contour_level,
                   "omni_feature": omni_feature, "solar_cycle_phase": solar_cycle_phase}

    corr_perim_full, corr_ar_full = find_index_correlation(config_dict)

    config_dict["solar_cycle_phase"] = "minimum"
    corr_perim_min, corr_ar_min = find_index_correlation(config_dict)

    config_dict["solar_cycle_phase"] = "maximum"
    corr_perim_max, corr_ar_max = find_index_correlation(config_dict)

    print(f"Correlation between perimeter size and {omni_feature} (full solar cycle): {corr_perim_full}")
    print(f"Correlation between aspect ratio and {omni_feature} (full solar cycle): {corr_ar_full}")
    print(f"Correlation between perimeter size and {omni_feature} (solar minimum): {corr_perim_min}")
    print(f"Correlation between aspect ratio and {omni_feature} (solar minimum): {corr_ar_min}")
    print(f"Correlation between perimeter size and {omni_feature} (solar maximum): {corr_perim_max}")
    print(f"Correlation between aspect ratio and {omni_feature} (solar maximum): {corr_ar_max}")
