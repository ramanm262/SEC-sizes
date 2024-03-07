import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import matplotlib.patches as patches
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from scipy.stats import pearsonr
from skimage import measure
import tqdm
import preprocessing


# Config variables
syear, eyear = 2009, 2019
stations_list = ['YKC', 'BLC', 'MEA', 'SIT', 'BOU', 'VIC', 'NEW', 'OTT', 'GIM', 'DAW', 'FCC', 'FMC',
                 'FSP', 'SMI', 'ISL', 'PIN', 'RAL', 'RAN', 'CMO', 'IQA', 'C04', 'C06', 'C10', 'T36']
station_coords_list = [np.array([62.48, 64.33, 54.62, 57.07, 40.13, 48.52, 48.27, 45.4, 56.38, 64.05, 58.76,
                                 56.66, 61.76, 60.02, 53.86, 50.2, 58.22, 62.82, 64.87, 63.75, 50.06, 53.35,
                                 47.66, 54.71]),
                       np.array([245.52, 263.97, 246.65, 224.67, 254.77, 236.58, 242.88, 284.45, 265.36, 220.89,
                                 265.92, 248.79, 238.77, 248.05, 265.34, 263.96, 256.32, 267.89, 212.14, 291.48,
                                 251.74, 247.03, 245.79, 246.69])]
n_sec_lat, n_sec_lon = 10, 35  # # of rows and columns respectively of SECSs that will exist in the grid
n_poi_lat, n_poi_lon = 14, 32  # # of rows and columns respectively of POIs that will exist in the grid
w_lon, e_lon, s_lat, n_lat = 215., 295., 30., 65.
poi_coords_list = [np.linspace(45, 55, n_poi_lat), np.linspace(230, 280, n_poi_lon)]
epsilon = 0.09323151264778985
B_param = "dbn_geo"
omni_feature = "AE_INDEX"
plot_interps = True
plot_every_n_interps = 5000
solar_cycle_phase = "full"  # "minimum", "maximum", or "full"
stats_plots_location = "stats_plots/"
interp_plots_location = "interp_plots/"


def calculate_perimeter(contour, poi_coords_list, r=6378100):
    rad_between_rows = np.pi * (poi_coords_list[0][1] - poi_coords_list[0][0]) / 180  # Given a regular rectangular grid
    rad_between_cols = np.pi * (poi_coords_list[1][1] - poi_coords_list[1][0]) / 180
    contour[:, 0], contour[:, 1] = (contour[:, 0] * rad_between_rows + poi_coords_list[0][0] * np.pi / 180,
                                    contour[:, 1] * rad_between_cols + poi_coords_list[1][0] * np.pi / 180)
    hd_matrix = haversine_distances(contour, contour)  # Contains the distances between each pair of vertices
    # Use only the distances between each successive vertex
    perimeter_km = np.sum(np.diag(hd_matrix, k=1)) * r / 1000
    return perimeter_km


def calculate_aspect_ratio(contour, poi_coords_list):
    rad_between_rows = np.pi * (poi_coords_list[0][1] - poi_coords_list[0][0]) / 180  # Given a regular rectangular grid
    rad_between_cols = np.pi * (poi_coords_list[1][1] - poi_coords_list[1][0]) / 180
    contour[:, 0], contour[:, 1] = (contour[:, 0] * rad_between_rows + poi_coords_list[0][0] * np.pi / 180,
                                    contour[:, 1] * rad_between_cols + poi_coords_list[1][0] * np.pi / 180)
    # Aspect ratio = longitudinal extent / latitudinal extent
    aspect_ratio = (max(contour[:, 1]) - min(contour[:, 1])) / (max(contour[:, 0]) - min(contour[:, 0]))
    return aspect_ratio


def correlation_with_index(param_series, index_series):
    combined_df = pd.concat([param_series, index_series], axis=1)
    combined_df = combined_df[combined_df[0] != 0]
    return pearsonr(combined_df.iloc[:, 0], combined_df.iloc[:, 1])[0]


if __name__ == "__main__":
    all_B_interps = pd.read_hdf(f"all_B_interps_{n_sec_lat}by{n_sec_lon}_{syear}-{eyear}.h5", B_param)

    station_geocolats = np.pi / 2 - np.pi / 180 * station_coords_list[0]
    station_geolons = np.pi / 180 * station_coords_list[1]
    poi_geocolats = np.pi / 2 - np.pi / 180 * poi_coords_list[0]
    poi_geolons = np.pi / 180 * poi_coords_list[1]


    # Generate full list of coords for every POI in row-major format
    all_poi_colats, all_poi_lats, all_poi_lons = [], [], []
    for colat in poi_geocolats:
        for lon in poi_geolons:
            all_poi_colats.append(colat)
            all_poi_lats.append(90-colat*180/np.pi)
            all_poi_lons.append(lon)
    poi_lons_mesh, poi_lats_mesh = np.meshgrid(poi_coords_list[1], poi_coords_list[0])


    projection = ccrs.AlbersEqualArea(central_latitude=50, central_longitude=255)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5), sharex=True, sharey=True, subplot_kw={"projection": projection})
    ax1.set_extent((w_lon, e_lon, s_lat, n_lat))
    # ax2.set_extent((w_lon, e_lon, s_lat, n_lat))
    fig.subplots_adjust(wspace=0.05, left=0.075, right=0.88)
    # fig.suptitle(f'd$B_{B_param[2]}$', fontsize=20)
    plt.cla()
    color_map = cm.coolwarm
    divider = make_axes_locatable(ax2)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    norm = plt.cm.colors.TwoSlopeNorm(vcenter=0, vmin=np.percentile(all_B_interps, 5),
                                      vmax=np.percentile(all_B_interps, 95))
    scalarmappable = plt.cm.ScalarMappable(norm=norm, cmap=color_map)
    station_scatter = ax1.scatter(station_coords_list[1], station_coords_list[0],
                                  c=np.random.rand(len(station_coords_list[0])), s=80, cmap=color_map,
                                  transform=ccrs.PlateCarree())
    ax1.set_title("Locations of Stations", fontsize=16)

    # Load SuperMAG and geomagnetic index data for comparison to interpolations
    mag_data = preprocessing.load_supermag(stations_list, syear, eyear, B_param, saving=False)
    mag_data.drop_duplicates(inplace=True)  # Remove timestamps that are double-counted by immediately successive storms
    all_B_interps.drop_duplicates(inplace=True)
    assert len(mag_data) == len(all_B_interps)
    all_B_interps.index = mag_data.index
    index_data = preprocessing.load_omni(syear, eyear, "/data/ramans_files/omni-feather/", feature=omni_feature)

    if solar_cycle_phase == "minimum":
        print("Only using data from the low part of the solar cycle")
        all_B_interps = all_B_interps.loc[(all_B_interps.index < pd.to_datetime("2011-01-01")) |
                                          (all_B_interps.index >= pd.to_datetime("2016-01-01"))]
        syear, eyear = "Solar minimum", "Solar minimum"  # Hacky way to make plot titles and axis labels say the right thing
    elif solar_cycle_phase == "maximum":
        print("Only using data from the high part of the solar cycle")
        all_B_interps = all_B_interps.loc[(all_B_interps.index >= pd.to_datetime("2011-01-01")) &
                                          (all_B_interps.index < pd.to_datetime("2016-01-01"))]
        syear, eyear = "Solar maximum", "Solar maximum"
    else:
        print("Using data from the entire solar cycle")
    mag_data = mag_data.loc[all_B_interps.index]
    # index_data is pared to the same index later in this script

    anomaly_coords = []
    perimeters = []
    aspect_ratios = []
    contour_level = 49.01  # np.percentile(max_list, 90)
    for timestep in tqdm.trange(len(all_B_interps), desc="Generating heatmaps"):
        heatmap_data = np.abs(np.array(all_B_interps.iloc[timestep]).reshape((n_poi_lat, n_poi_lon)))

        if plot_interps and (timestep % plot_every_n_interps == 0):
            ax1.cla()
            ax2.cla()
            ax1.scatter(station_coords_list[1], station_coords_list[0],
                        c=mag_data.iloc[timestep],
                        s=80, cmap=color_map, norm=norm, transform=ccrs.PlateCarree())
            ax1.gridlines(draw_labels=False)
            ax1.coastlines()
            ax2.scatter(np.array(all_poi_lons) * 180 / np.pi, all_poi_lats, c=-all_B_interps.iloc[timestep],
                        cmap=color_map, s=80, marker='s', norm=norm, transform=ccrs.PlateCarree())
            ax2.set_extent((w_lon, e_lon, s_lat, n_lat))
            ax2.gridlines(draw_labels=False)
            ax2.coastlines()
            ax2.set_title(f"Interpolation {mag_data.index[timestep]} ($\\epsilon$ = {epsilon:.4f})", fontsize=16)
            cb = fig.colorbar(scalarmappable, ax=(ax1, ax2), cax=cax, orientation='vertical')
            cb.ax.set_title('nT', fontsize=16)
            cb.ax.tick_params(labelsize=14)

        contours = measure.find_contours(heatmap_data, level=contour_level)
        # Each element of the resulting list is a list of coords in row, col format (not degrees)

        this_perimeters = []
        this_aspect_ratios = []
        for contour in contours:
            this_contour_lons = poi_lons_mesh[contour[:, 0].astype(int), contour[:, 1].astype(int)]
            this_contour_lats = poi_lats_mesh[contour[:, 0].astype(int), contour[:, 1].astype(int)]
            if plot_interps and (timestep % plot_every_n_interps == 0):  # Only plot if plotting is enabled
                ax2.plot(this_contour_lons, this_contour_lats, linewidth=4, color='black', transform=ccrs.PlateCarree())
            if contour[0, 0] == contour[-1, 0] and contour[0, 1] == contour[-1, 1]:  # If contour is closed
                p = calculate_perimeter(contour.copy(), poi_coords_list=poi_coords_list)
                ar = calculate_aspect_ratio(contour.copy(), poi_coords_list=poi_coords_list)
                this_perimeters.append(p)
                this_aspect_ratios.append(ar)
                if 2300 < p < 2900:
                    anomaly_coords.append([np.mean(this_contour_lats), np.mean(this_contour_lons)])
                else:
                    anomaly_coords.append([np.nan, np.nan])

        if plot_interps and (timestep % plot_every_n_interps == 0) and (len(this_perimeters) != 0):
            plt.savefig(interp_plots_location + f"interpolated_values_{timestep}.png")
        perimeters.append(this_perimeters)
        aspect_ratios.append(this_aspect_ratios)

    if plot_interps:
        print(f"Your interpolation heatmaps are ready in {stats_plots_location}")

    num_of_perimeters_list = [len(timestep) for timestep in perimeters]
    all_perimeter_sizes_list = [perimeter for timestep in perimeters for perimeter in timestep]
    all_ars_list = [ar for timestep in aspect_ratios for ar in timestep]

    plt.figure()
    plt.hist(num_of_perimeters_list, bins=np.arange(10-0.5))
    plt.yscale("log")
    plt.title("Number Identified LMPs", fontsize=16)
    plt.xlabel("# of LMPs", fontsize=14)
    plt.ylabel(f"# of occurrences in {syear}{('-'+str(eyear))*(syear!=eyear)}", fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(stats_plots_location + "num_of_blobs.png")

    plt.figure()
    plt.hist(all_perimeter_sizes_list, bins=int(np.max(all_perimeter_sizes_list)/50), align="left")
    plt.yscale("log")
    plt.title(f"LMP Sizes {syear}{('-'+str(eyear))*(syear!=eyear)}", fontsize=16)
    plt.xlabel("LMP Perimeter (km)", fontsize=14)
    plt.ylabel(f"# of occurrences in {syear}{('-'+str(eyear))*(syear!=eyear)}", fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    # plt.xlim(2300,2900)
    plt.tight_layout()
    plt.savefig(stats_plots_location + f"blob_sizes_{n_sec_lat}by{n_sec_lon}.png")

    plt.figure()
    plt.hist(np.log10(all_ars_list), bins=50, align="left")
    plt.yscale("log")
    plt.title(f"LMP Aspect Ratios {syear}{('-'+str(eyear))*(syear!=eyear)}", fontsize=16)
    plt.xlabel("$log_{10}($Aspect Ratio$)$", fontsize=14)
    plt.ylabel(f"# of occurrences in {syear}{('-'+str(eyear))*(syear!=eyear)}", fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(stats_plots_location + f"aspect_ratios_{n_sec_lat}by{n_sec_lon}.png")

    try:
        stime, etime = pd.to_datetime("2015-07-05 00:00:00"), pd.to_datetime("2015-07-07 00:00:00")
        timestamps = mag_data.index
        plt.figure()
        fig, perim_ax = plt.subplots(1, 1, figsize=(10, 5))
        index_ax = perim_ax.twinx()
        perimeter_maxes = [0]*len(perimeters)
        for timestep in range(len(perimeters)):
            if len(perimeters[timestep]) > 0:
                perimeter_maxes[timestep] = np.max(perimeters[timestep])
        perimeter_maxes = pd.DataFrame(perimeter_maxes, index=timestamps)
        timestamps = timestamps[(timestamps >= stime) & (timestamps <= etime)]
        perimeter_maxes = perimeter_maxes[(perimeter_maxes.index >= stime) & (perimeter_maxes.index <= etime)]
        index_data = index_data.loc[perimeter_maxes.index]
        perim_ax.plot(timestamps, perimeter_maxes, c="black", label="Largest LMP Perimeter")
        index_ax.plot(timestamps, index_data, c="green", label="AE Index")
        fig.suptitle(f"LMP Sizes (Example Storm) (r={correlation_with_index(perimeter_maxes, index_data)})", fontsize=16)
        perim_ax.set_xlabel("Time", fontsize=14)
        perim_ax.set_ylabel(f"Perimeter of largest LMP (km)", fontsize=14)
        index_ax.set_ylabel("AE Index (nT)", fontsize=14, color="green")
        perim_ax.tick_params(axis='x', labelsize=14, labelrotation=40)
        perim_ax.tick_params(axis='y', labelsize=14)
        index_ax.tick_params(axis='y', labelcolor="green", labelsize=14)
        # plt.xlim(stime, etime)
        # plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(stats_plots_location + "blob_size_timeseries.png")
    except ValueError:
        print('#'*8+"\nWarning!\nSkipping example storm plot because there is no data for it.\n"
                    "Ensure the storm dates solar_cycle_phase are set correctly.\n"+'#'*8)

    plt.cla()
    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), subplot_kw={"projection": projection})
    ax.set_extent((w_lon+10, e_lon-10, s_lat+10, n_lat-5))
    ax.set_title(f'Locations of Anomalies', fontsize=20)
    centroids = ax.scatter([coords[1] for coords in anomaly_coords],
                           [coords[0] for coords in anomaly_coords],
                           c="orange", s=20, transform=ccrs.PlateCarree(),
                           label="Anomaly Centroids")
    station_scatter = ax.scatter(station_coords_list[1], station_coords_list[0], c="green", marker='*',
                                 s=160, transform=ccrs.PlateCarree(), label="Magnetometer Stations")
    rect = ax.add_patch(patches.Rectangle((230.5, 45.5), 50, 10, facecolor="#1f77b4",
                                          alpha=0.2, transform=ccrs.PlateCarree(), label="Current System Grid Extent"))
    plt.legend(loc="upper left", prop={'size': 14}, handles=[centroids, station_scatter, rect])
    ax.gridlines(draw_labels=False)
    ax.coastlines()
    plt.savefig(stats_plots_location + "anomaly_locs.png")

    print(f"Your statistics plots are ready in {stats_plots_location}")
