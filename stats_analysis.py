from shutil import unregister_archive_format

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from scipy.stats import pearsonr, mode
from skimage import measure
import tqdm
import preprocessing


def calculate_perimeter(contour, poi_coords_list, r=6378100):
    """
    Computes the perimeter of a single LGMD.
    """
    rad_between_rows = np.pi * (poi_coords_list[0][1] - poi_coords_list[0][0]) / 180  # Given a regular rectangular grid
    rad_between_cols = np.pi * (poi_coords_list[1][1] - poi_coords_list[1][0]) / 180
    contour[:, 0], contour[:, 1] = (contour[:, 0] * rad_between_rows + poi_coords_list[0][0] * np.pi / 180,
                                    contour[:, 1] * rad_between_cols + poi_coords_list[1][0] * np.pi / 180)
    hd_matrix = haversine_distances(contour, contour)  # Contains the distances between each pair of vertices
    # Use only the distances between each successive vertex
    perimeter_km = np.sum(np.diag(hd_matrix, k=1)) * r / 1000
    return perimeter_km


def calculate_aspect_ratio(contour, poi_coords_list):
    """
    Computes the aspect ratio of a single LGMD.
    """
    rad_between_rows = np.pi * (poi_coords_list[0][1] - poi_coords_list[0][0]) / 180  # Given a regular rectangular grid
    rad_between_cols = np.pi * (poi_coords_list[1][1] - poi_coords_list[1][0]) / 180
    contour[:, 0], contour[:, 1] = (contour[:, 0] * rad_between_rows + poi_coords_list[0][0] * np.pi / 180,
                                    contour[:, 1] * rad_between_cols + poi_coords_list[1][0] * np.pi / 180)
    # Aspect ratio = longitudinal extent / latitudinal extent
    lon_easternmost, lon_westernmost = np.max(contour[:, 1]), np.min(contour[:, 1])
    lat_easternmost, lat_westernmost = contour[np.argmax(contour[:, 1]), 0], contour[np.argmin(contour[:, 1]), 0]
    lat_northernmost, lat_southernmost = np.max(contour[:, 0]), np.min(contour[:, 0])
    e_w_average = (lat_easternmost + lat_westernmost) / 2
    aspect_ratio = (lon_easternmost - lon_westernmost) * np.cos(e_w_average) / (lat_northernmost - lat_southernmost)

    return aspect_ratio


def correlation_with_index(param_series, index_series):
    """
    Computes the linear correlation between a geomagnetic index timeseries and a timeseries of an LGMD attribute.
    """
    combined_df = pd.concat([param_series, index_series], axis=1)
    combined_df = combined_df[combined_df[0].notna()]
    return pearsonr(combined_df.iloc[:, 0], combined_df.iloc[:, 1])[0]


def kl_divergence(max_series, min_series, bins, integrated=True):
    """
    Calculates the symmetrized Kullback-Leibler divergence between two series.
    """
    to_keep = np.where((min_series != 0) & (max_series != 0))
    max_series, min_series = max_series[to_keep], min_series[to_keep]
    assert len(bins) > 1
    bin_width = bins[1] - bins[0]
    bins = bins[to_keep]
    if integrated:
        # return 0.5 * np.sum(bin_width * np.abs(max_series - min_series)), bins  # Uncomment to use the total variation instead
        return (np.sum(bin_width * max_series * np.log(max_series / min_series)) +
                np.sum(bin_width * min_series * np.log(min_series / max_series)), bins)
    else:
        # return 0.5 * bin_width * (max_series - min_series), bins  # Uncomment to use the total variation instead
        return bin_width * (max_series * np.log(max_series / min_series) + min_series * np.log(min_series / max_series)), bins


def plot_num_of_blobs(num_blobs_full=[], num_blobs_min=[], num_blobs_max=[], log_y=True):
    num_of_variables = (len(num_blobs_full) > 0) + (len(num_blobs_min) > 0) + (len(num_blobs_max) > 0)
    assert num_of_variables > 0
    plt.figure(figsize=(5.5, 4))
    plt.hist([num_blobs_full, num_blobs_min, num_blobs_max], bins=np.arange(8),
             label=["Full Solar Cycle", "Solar Minimum", "Solar Maximum"], align="left", density=True)
    if log_y:
        plt.yscale("log")
    plt.title("Number of Identified LGMDs", fontsize=16)
    plt.xlabel("# of LGMDs", fontsize=14)
    plt.ylabel("Probability Density", fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(stats_plots_location + "num_of_blobs.jpg")
    full, bins, _patches = plt.hist(num_blobs_full, bins=np.arange(7), density=True)
    smin, bins, _patches = plt.hist(num_blobs_min, bins=np.arange(7), density=True)
    smax, bins, _patches = plt.hist(num_blobs_max, bins=np.arange(7), density=True)
    plt.figure()
    kl_non_integrated, kl_bins = kl_divergence(smax, smin, bins, integrated=False)
    kl, _ = kl_divergence(smax, smin, bins)
    plt.bar(kl_bins, kl_non_integrated, width=kl_bins[1]-kl_bins[0], align="center")
    kl_smax_uniform, _ = kl_divergence(smax, np.ones_like(smax)/(len(smax)+1), bins)
    kl_smin_uniform, _ = kl_divergence(smin, np.ones_like(smin)/(len(smin)+1), bins)
    print("Number of LGMDs")
    print(f"KL Divergence between Solar Maximum and a uniform distribution: {kl_smax_uniform:.4f}")
    print(f"KL Divergence between Solar Minimum and a uniform distribution: {kl_smin_uniform:.4f}")
    # plt.yscale("log")
    plt.title("Solar Cycle Phase Difference in Number of LGMDs"
              "\n$J_{KL}$"+f"={kl:.4f}", fontsize=16)
    plt.ylabel("Non-integrated $J_{KL}(S_{max}||S_{min})$", fontsize=14)
    plt.xlabel("# of LGMDs", fontsize=14)
    plt.savefig(stats_plots_location + "num_of_blobs_diffs.pdf")


def plot_blob_sizes(sizes_full=[], sizes_min=[], sizes_max=[], log_y=True):
    num_of_variables = (len(sizes_full) > 0) + (len(sizes_min) > 0) + (len(sizes_max) > 0)
    assert num_of_variables > 0
    plt.figure(figsize=(6.5, 4), dpi=300)
    try:
        num_bins = int(np.max(sizes_full)/50)
    except:
        try:
            num_bins = int(np.max(sizes_min)/50)
        except:
            num_bins = int(np.max(sizes_max)/50)
    plot_range = (0, 12500)
    plt.hist(sizes_full, bins=num_bins, histtype="step", label="Full Solar Cycle", align="left", density=True, range=plot_range)
    plt.hist(sizes_min, bins=num_bins, histtype="step", label="Solar Minimum", align="left", density=True, range=plot_range)
    plt.hist(sizes_max, bins=num_bins, histtype="step", label="Solar Maximum", align="left", density=True, range=plot_range)
    if log_y:
        plt.yscale("log")
    plt.title("LGMD Sizes", fontsize=16)
    plt.xlabel("LGMD Perimeter (km)", fontsize=14)
    plt.ylabel("Probability Density", fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(stats_plots_location + f"blob_sizes_{n_sec_lat}by{n_sec_lon}.jpg")
    full, bins, _patches = plt.hist(sizes_full, bins=num_bins, density=True)
    smin, bins, _patches = plt.hist(sizes_min, bins=num_bins, density=True)
    smax, bins, _patches = plt.hist(sizes_max, bins=num_bins, density=True)
    plt.figure()
    kl_non_integrated, kl_bins = kl_divergence(smax, smin, bins, integrated=False)
    kl, _ = kl_divergence(smax, smin, bins)
    plt.bar(kl_bins, kl_non_integrated, width=kl_bins[1]-kl_bins[0], align="edge")
    kl_smax_uniform, _ = kl_divergence(smax, np.ones_like(smax)/len(smax), bins)
    kl_smin_uniform, _ = kl_divergence(smin, np.ones_like(smin)/len(smin), bins)
    print("LGMD Sizes")
    print(f"KL Divergence between Solar Maximum and a uniform distribution: {kl_smax_uniform:.4f}")
    print(f"KL Divergence between Solar Minimum and a uniform distribution: {kl_smin_uniform:.4f}")
    # plt.yscale("log")
    plt.title("Solar Cycle Phase $J_{KL}$"+f"={kl:.4f}", fontsize=16)
    plt.ylabel("Non-integrated $J_{KL}(S_{max}||S_{min})$", fontsize=14)
    plt.xlabel("LGMD Perimeter (km)", fontsize=14)
    plt.savefig(stats_plots_location + f"blob_size_diffs_{n_sec_lat}by{n_sec_lon}.pdf")


def plot_aspect_ratios(ars_full=[], ars_min=[], ars_max=[], log_y=True):
    num_of_variables = (len(ars_full) > 0) + (len(ars_min) > 0) + (len(ars_max) > 0)
    assert num_of_variables > 0
    plt.figure(figsize=(6.2, 4), dpi=300)
    plot_range = (-0.6, 1.3)
    plt.hist(np.log10(ars_full), bins=100, histtype="step", label="Full Solar Cycle", align="left", density=True, range=plot_range)
    plt.hist(np.log10(ars_min), bins=100, histtype="step", label="Solar Minimum", align="left", density=True, range=plot_range)
    plt.hist(np.log10(ars_max), bins=100, histtype="step", label="Solar Maximum", align="left", density=True, range=plot_range)
    if log_y:
        plt.yscale("log")
    plt.title("LGMD Aspect Ratios", fontsize=16)
    plt.xlabel("$log_{10}($Aspect Ratio$)$", fontsize=14)
    plt.ylabel("Probability Density", fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(stats_plots_location + f"aspect_ratios_{n_sec_lat}by{n_sec_lon}.jpg")
    full, bins, _patches = plt.hist(np.log10(ars_full), bins=100, density=True)
    smin, bins, _patches = plt.hist(np.log10(ars_min), bins=100, density=True)
    smax, bins, _patches = plt.hist(np.log10(ars_max), bins=100, density=True)
    plt.cla()
    plt.figure()
    kl_non_integrated, kl_bins = kl_divergence(smax, smin, bins, integrated=False)
    kl, _ = kl_divergence(smax, smin, bins)
    plt.bar(kl_bins, kl_non_integrated, width=kl_bins[1]-kl_bins[0], align="edge")
    kl_smax_uniform, _ = kl_divergence(smax, np.ones_like(smax)/len(smax), bins)
    kl_smin_uniform, _ = kl_divergence(smin, np.ones_like(smin)/len(smin), bins)
    print("Aspect Ratios")
    print(f"KL Divergence between Solar Maximum and a uniform distribution: {kl_smax_uniform:.4f}")
    print(f"KL Divergence between Solar Minimum and a uniform distribution: {kl_smin_uniform:.4f}")
    # plt.yscale("log")
    plt.title("Solar Cycle Phase Difference in LGMD Aspect Ratios"
            "\n$J_{KL}$"+f"={kl:.4f}", fontsize=16)
    plt.ylabel("Non-integrated $J_{KL}(S_{max}||S_{min})$", fontsize=14)
    plt.xlabel("$log_{10}($Aspect Ratio$)$", fontsize=14)
    plt.savefig(stats_plots_location + f"aspect_ratio_diffs_{n_sec_lat}by{n_sec_lon}.pdf")


def plot_num_and_sizes(num_full, sizes_full, log_y=True):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4), width_ratios=[1, 1.5], dpi=300)
    num_bins = 100
    axs[0].hist(num_full, bins=np.arange(8), align="left", density=True)
    axs[0].set_title("Number of Identified LGMDs", fontsize=16)
    axs[0].set_xlabel("# of LGMDs", fontsize=14)
    axs[0].set_ylabel("Frequency of Occurrence", fontsize=14)
    axs[1].hist(sizes_full, bins=int(np.max(sizes_full)/50), align="left", density=True)
    axs[1].set_title("LGMD Sizes", fontsize=16)
    axs[1].set_xlabel("LGMD Perimeter (km)", fontsize=14)
    axs[1].set_ylabel("Frequency of Occurrence", fontsize=14)
    if log_y:
        axs[0].set_yscale("log")
        axs[1].set_yscale("log")
    plt.tight_layout()
    # Label the subplots a and b
    axs[0].text(.9, .9, "a", transform=axs[0].transAxes, fontsize=20, fontweight='bold', va='top')
    axs[1].text(.9, .9, "b", transform=axs[1].transAxes, fontsize=20, fontweight='bold', va='top')
    plt.savefig(stats_plots_location + "combined_num_and_sizes.jpg")


def plot_gm_index_histogram(lgmd_attribute, index_at_interp_time, attribute_name, rolling=False):
    plt.figure(figsize=(6, 4))
    combined_df = pd.concat([pd.DataFrame(lgmd_attribute, index=index_at_interp_time.index), index_at_interp_time], axis=1)
    combined_df.dropna(inplace=True)
    lgmd_attribute, index_at_interp_time = combined_df.iloc[:, 0], combined_df.iloc[:, 1]
    bins = 100
    if attribute_name == "Perimeter":
        plt.ylim(500, 3400)
    elif attribute_name == "log(A)":
        plt.ylim(0.2, 0.8)
    elif attribute_name == "Number":
        bins = (100, 6)
    else:
        raise ValueError("Invalid attribute name (must be one of 'Number', 'Perimeter', or 'log(A)')")
    plt.hist2d(index_at_interp_time, lgmd_attribute, bins=bins)
    plt.colorbar()
    plt.title(f"Correlation: {pearsonr(lgmd_attribute, index_at_interp_time)[0]:.4f}", fontsize=16)
    plt.xlabel(f"{omni_feature}", fontsize=14)
    plt.ylabel(attribute_name, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlim(0, 700)
    plt.tight_layout()
    plt.savefig(stats_plots_location + f"{omni_feature}_{attribute_name}"+"_rolling"*rolling+"_histogram.png")


def stats_analysis(config_dict):
    syear, eyear = config_dict["syear"], config_dict["eyear"]
    stations_list = config_dict["stations_list"]
    station_coords_list = config_dict["station_coords_list"]
    n_sec_lat, n_sec_lon = config_dict["n_sec_lat"], config_dict["n_sec_lon"]
    n_poi_lat, n_poi_lon = config_dict["n_poi_lat"], config_dict["n_poi_lon"]
    w_lon, e_lon, s_lat, n_lat = config_dict["w_lon"], config_dict["e_lon"], config_dict["s_lat"], config_dict["n_lat"]
    poi_coords_list = config_dict["poi_coords_list"]
    epsilon = config_dict["epsilon"]
    B_param = config_dict["B_param"]
    contour_level = config_dict["contour_level"]
    omni_feature = config_dict["omni_feature"]
    plot_interps = config_dict["plot_interps"]
    plot_every_n_interps = config_dict["plot_every_n_interps"]
    solar_cycle_phase = config_dict["solar_cycle_phase"]
    stats_plots_location = config_dict["stats_plots_location"]
    interp_plots_location = config_dict["interp_plots_location"]

    # Use the right dataset depending on your preference indicated in B_param
    if B_param == "dbn_geo" or B_param == "dbe_geo":
        all_B_interps = pd.read_hdf(f"all_B_interps_{n_sec_lat}by{n_sec_lon}_{syear}-{eyear}.h5", B_param)
    elif B_param == "HORIZONTAL":
        all_BN_interps = pd.read_hdf(f"all_B_interps_{n_sec_lat}by{n_sec_lon}_{syear}-{eyear}.h5",
                                     "dbn_geo")
        all_BE_interps = pd.read_hdf(f"all_B_interps_{n_sec_lat}by{n_sec_lon}_{syear}-{eyear}.h5",
                                     "dbe_geo")
        all_B_interps = np.sqrt(all_BN_interps**2 + all_BE_interps**2)
        del all_BN_interps, all_BE_interps

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

    # Set up the figure and axes that will be used to generate the interpolation plots in the interp_plots/ directory
    projection = ccrs.AlbersEqualArea(central_latitude=50, central_longitude=255)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5), sharex=True, sharey=True,
                                   subplot_kw={"projection": projection})
    ax1.set_extent((w_lon, e_lon, s_lat, n_lat))
    # ax2.set_extent((w_lon, e_lon, s_lat, n_lat))
    fig.subplots_adjust(wspace=0.05, left=0.075, right=0.88)
    # fig.suptitle(f'd$B_{B_param[2]}$', fontsize=20)
    plt.cla()
    divider = make_axes_locatable(ax2)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    if B_param == "dbn_geo" or B_param == "dbe_geo":
        color_map = cm.coolwarm
        norm = plt.cm.colors.TwoSlopeNorm(vcenter=0, vmin=np.percentile(all_B_interps, 5),
                                          vmax=np.percentile(all_B_interps, 95))
    elif B_param == "HORIZONTAL":
        color_map = cm.viridis
        norm = plt.cm.colors.TwoSlopeNorm(vcenter=np.percentile(all_B_interps, 30),
                                          vmin=np.percentile(all_B_interps,0), vmax=np.percentile(all_B_interps, 95))
    scalarmappable = plt.cm.ScalarMappable(norm=norm, cmap=color_map)
    station_scatter = ax1.scatter(station_coords_list[1], station_coords_list[0],
                                  c=np.random.rand(len(station_coords_list[0])), s=80, cmap=color_map,
                                  transform=ccrs.PlateCarree())
    ax1.set_title("Locations of Stations", fontsize=16)

    # Load SuperMAG and geomagnetic index data for comparison to interpolations
    mag_data = preprocessing.load_supermag(stations_list, syear, eyear, B_param, saving=False)
    # all_B_interps = all_B_interps.diff().dropna()  # Uncomment to use dB/dt instead of dB
    # mag_data = mag_data.diff().dropna()  # Uncomment to use dB/dt instead of dB
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
    mag_data = mag_data.loc[all_B_interps.index]  # index_data is pared to the same index later in this script


    # Loop through each timestep. Plot some of the heatmaps and extract information on LGMDs.
    perimeters = []
    aspect_ratios = []
    centroids = []
    for timestep in tqdm.trange(len(all_B_interps),
                                desc=f'Generating heatmaps for solar cycle phase "{solar_cycle_phase}"'):
        # We take the absolute value of the heatmaps so that contours are computed at the same level for pos and neg
        heatmap_data = np.abs(np.array(all_B_interps.iloc[timestep]).reshape((n_poi_lat, n_poi_lon)))

        # Plot the interpolation, but not always, since we don't want too many plots
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
            ax2.set_xlabel("Geographic Longitude (degrees)", fontsize=14, labelpad=1)
            ax2.set_ylabel("Geographic Latitude (degrees)", fontsize=14, labelpad=1)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.coastlines()
            ax2.set_title(f"Interpolation {mag_data.index[timestep]} ($\\epsilon$ = {epsilon:.4f})", fontsize=16)
            cb = fig.colorbar(scalarmappable, ax=(ax1, ax2), cax=cax, orientation='vertical')
            cb.ax.set_title('nT', fontsize=16)
            cb.ax.tick_params(labelsize=14)

        # Now we find the contours! This is where the good stuff happens.
        contours = measure.find_contours(heatmap_data, level=contour_level)
        # Each element of the resulting list is a list of coords in row, col format (not degrees)

        # Plot the contours on the interpolation plot and calculate their perimeter and aspect ratios
        this_perimeters = []
        this_aspect_ratios = []
        this_centroids = []
        for contour in contours:
            this_contour_lons = poi_lons_mesh[contour[:, 0].astype(int), contour[:, 1].astype(int)]
            this_contour_lats = poi_lats_mesh[contour[:, 0].astype(int), contour[:, 1].astype(int)]
            if plot_interps and (timestep % plot_every_n_interps == 0):  # Only plot if plotting is enabled
                ax2.plot(this_contour_lons, this_contour_lats, linewidth=4, color='black', transform=ccrs.PlateCarree())
            if contour[0, 0] == contour[-1, 0] and contour[0, 1] == contour[-1, 1]:  # If contour is closed
                contour[:, 0] = contour[:, 0].astype(int)
                contour[:, 1] = contour[:, 1].astype(int)
                contour_copy = contour.copy()
                p = calculate_perimeter(contour_copy, poi_coords_list=poi_coords_list)
                contour_copy = contour.copy()
                ar = calculate_aspect_ratio(contour_copy, poi_coords_list=poi_coords_list)
                this_perimeters.append(p)
                this_aspect_ratios.append(ar)
                centroid_lat = np.mean(this_contour_lats)
                centroid_lon = np.mean(this_contour_lons)
                this_centroids.append((centroid_lat, centroid_lon))


        if plot_interps and (timestep % plot_every_n_interps == 0) and (len(this_perimeters) != 0):
            plt.savefig(interp_plots_location + f"interpolated_values_{timestep}.pdf")
        perimeters.append(this_perimeters)
        aspect_ratios.append(this_aspect_ratios)
        centroids.append(this_centroids)

    num_of_perimeters_list = [len(timestep) for timestep in perimeters]
    all_perimeter_sizes_list = [perimeter for timestep in perimeters for perimeter in timestep]
    all_ars_list = [ar for timestep in aspect_ratios for ar in timestep]
    all_log_ars_list = [np.log10(ar) for timestep in aspect_ratios for ar in timestep]

    # Make a list of perimeters where there are only 1, 2, 3, 4, 5, or 6 LGMDs
    one_perimeter_sizes = [perimeter for timestep in perimeters if len(timestep) == 1 for perimeter in timestep]
    two_perimeter_sizes = [perimeter for timestep in perimeters if len(timestep) == 2 for perimeter in timestep]
    three_perimeter_sizes = [perimeter for timestep in perimeters if len(timestep) == 3 for perimeter in timestep]
    four_perimeter_sizes = [perimeter for timestep in perimeters if len(timestep) == 4 for perimeter in timestep]
    five_perimeter_sizes = [perimeter for timestep in perimeters if len(timestep) == 5 for perimeter in timestep]
    six_perimeter_sizes = [perimeter for timestep in perimeters if len(timestep) == 6 for perimeter in timestep]

    # Make a list of aspect ratios where there are only 1, 2, 3, 4, 5, or 6 LGMDs
    one_perimeter_ars = [np.log10(ar) for timestep in aspect_ratios if len(timestep) == 1 for ar in timestep]
    two_perimeter_ars = [np.log10(ar) for timestep in aspect_ratios if len(timestep) == 2 for ar in timestep]
    three_perimeter_ars = [np.log10(ar) for timestep in aspect_ratios if len(timestep) == 3 for ar in timestep]
    four_perimeter_ars = [np.log10(ar) for timestep in aspect_ratios if len(timestep) == 4 for ar in timestep]
    five_perimeter_ars = [np.log10(ar) for timestep in aspect_ratios if len(timestep) == 5 for ar in timestep]
    six_perimeter_ars = [np.log10(ar) for timestep in aspect_ratios if len(timestep) == 6 for ar in timestep]

    # for each centroid, calculate the distance to the nearest station
    # centroid_distances = []
    # for timestep in tqdm.tqdm(centroids, desc="Calculating distances between centroids and magnetometers"):
    #     this_distances = []
    #     for centroid in timestep:
    #         distances = []
    #         for station_lat, station_lon in zip(station_coords_list[0], station_coords_list[1]):
    #             dist_to_station = haversine_distances([[centroid[0], centroid[1]], [station_lat, station_lon]])[0][1]
    #             dist_to_station *= 6378100 * np.pi / (1000 * 180)
    #             distances.append(dist_to_station)
    #         this_distances.append(np.min(distances))
    #     centroid_distances.append(this_distances)
    # all_centroid_dists_list = [dist for timestep in centroid_distances for dist in timestep]

    # Violin plots
    # fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # perimeter_violin = axs[0].violinplot(
    #     [one_perimeter_sizes, two_perimeter_sizes, three_perimeter_sizes, four_perimeter_sizes,
    #      five_perimeter_sizes], showmedians=True, showextrema=True,
    #     quantiles=[[.05, .95] for _ in range(5)])
    # axs[0].set_yscale("log")
    # axs[0].set_ylabel("Perimeter (km)", fontsize=14)
    # axs[0].set_xlabel("Number of LGMDs", fontsize=14)
    # axs[0].yaxis.set_tick_params(labelsize=14)
    # axs[0].xaxis.set_tick_params(labelsize=14)
    # perimeter_violin["cmedians"].set_edgecolor("black")
    # perimeter_violin["cmedians"].set_linewidth(2)
    # perimeter_violin["cquantiles"].set_edgecolor("darkorange")
    # perimeter_violin["cquantiles"].set_linewidth(2)
    # perimeter_violin["cmaxes"].set_edgecolor("#4daf4a")
    # perimeter_violin["cmins"].set_edgecolor("#4daf4a")
    # perimeter_violin["cmaxes"].set_linewidth(2)
    # perimeter_violin["cmins"].set_linewidth(2)
    # legend_lines = [Line2D([0], [0], color="black", lw=2),
    #                 Line2D([0], [0], color="darkorange", lw=2),
    #                 Line2D([0], [0], color="#4daf4a", lw=2)]
    # axs[0].legend(legend_lines, ["Median", "$5^{th}$/$95^{th}$ Percentile", "Minimum/Maximum"], fontsize=10)
    # ar_violin = axs[1].violinplot([one_perimeter_ars, two_perimeter_ars, three_perimeter_ars, four_perimeter_ars,
    #                                five_perimeter_ars], showmedians=True, showextrema=True,
    #                               quantiles=[[.05, .95] for _ in range(5)])
    # axs[1].set_ylabel("log(Aspect Ratio)", fontsize=14)
    # axs[1].set_xlabel("Number of LGMDs", fontsize=14)
    # axs[1].yaxis.set_tick_params(labelsize=14)
    # axs[1].xaxis.set_tick_params(labelsize=14)
    # ar_violin["cmedians"].set_edgecolor("black")
    # ar_violin["cmedians"].set_linewidth(2)
    # ar_violin["cquantiles"].set_edgecolor("darkorange")
    # ar_violin["cquantiles"].set_linewidth(2)
    # ar_violin["cmaxes"].set_edgecolor("#4daf4a")
    # ar_violin["cmins"].set_edgecolor("#4daf4a")
    # ar_violin["cmaxes"].set_linewidth(2)
    # ar_violin["cmins"].set_linewidth(2)
    # axs[1].legend(legend_lines, ["Median", "$5^{th}$/$95^{th}$ Percentile", "Minimum/Maximum"], fontsize=10)
    # plt.tight_layout()
    # plt.savefig(stats_plots_location + "violin_perimeters_ars.pdf")

    # plt.figure()
    # plt.hist(all_centroid_dists_list, bins=100, density=False)
    # plt.title("Distances between LGMD centroids and magnetometers", fontsize=16)
    # plt.xlabel("Distance (km)", fontsize=14)
    # plt.ylabel("Frequency of occurrence", fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.tight_layout()
    # plt.savefig(stats_plots_location + "centroid_distances.pdf")

    # fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    # cdist_vs_size = axs[0].hist2d(all_centroid_dists_list, all_perimeter_sizes_list, bins=[100, 100])
    # axs[0].set_ylabel("Perimeter (km)", fontsize=14)
    # for ax_num in range(len(axs)):
    #     axs[ax_num].xaxis.set_tick_params(labelsize=14)
    #     axs[ax_num].yaxis.set_tick_params(labelsize=14)
    #     axs[0].set_xlabel("Distance to nearest station (km)", fontsize=14)
    # # divider = make_axes_locatable(axs[0])
    # # cax = divider.append_axes('right', size='5%', pad=0.05)
    # # plt.colorbar(cdist_vs_size[3], cax=cax, orientation='vertical')
    # cdist_vs_ar = axs[1].hist2d(all_centroid_dists_list, all_log_ars_list, bins=[100, 100])
    # axs[1].set_ylabel("log(Aspect Ratio)", fontsize=14)
    # plt.savefig(stats_plots_location + "centroid_distances_vs_size_and_ar.pdf")

    # Rolling mean plots
    # min_index_data = index_data.rolling(window=30).min()
    # min_index_data = min_index_data.loc[all_B_interps.index]
    # plot_gm_index_histogram(perimeters_means, min_index_data, attribute_name="Perimeter", rolling=True)
    # plot_gm_index_histogram(ar_means, min_index_data, attribute_name="log(A)", rolling=True)
    #
    # mlt_data = (all_B_interps.index.hour - 6) % 24  # UTC to MST conversion
    # concatenated_data = pd.concat([pd.DataFrame(mlt_data, index=all_B_interps.index),
    #                                   pd.DataFrame(perimeters_means, index=all_B_interps.index)], axis=1)
    # concatenated_data.dropna(inplace=True)
    # mlt_data = concatenated_data.iloc[:, 0]
    # perimeters_means = concatenated_data.iloc[:, 1]
    # plt.figure(figsize=(6, 4))
    # plt.hist2d(mlt_data, perimeters_means, bins=[24, 50])
    # plt.ylim(280, 5000)
    # plt.colorbar()
    # plt.xlabel("MLT", fontsize=14)
    # plt.ylabel("Per-Minute Mean LGMD Perimeter (km)", fontsize=14)
    # plt.savefig(stats_plots_location + "mlt_perimeter_means_histogram.png")
    #
    # concatenated_data = pd.concat([pd.DataFrame(mlt_data, index=all_B_interps.index),
    #                                   pd.DataFrame(ar_means, index=all_B_interps.index)], axis=1)
    # concatenated_data.dropna(inplace=True)
    # mlt_data = concatenated_data.iloc[:, 0]
    # ar_means = concatenated_data.iloc[:, 1]
    # plt.figure(figsize=(6, 4))
    # plt.hist2d(mlt_data, ar_means, bins=[24, 160])
    # plt.ylim(.2, .8)
    # plt.colorbar()
    # plt.xlabel("MLT", fontsize=14)
    # plt.ylabel("Per-Minute Mean LGMD Aspect Ratio (km)", fontsize=14)
    # plt.savefig(stats_plots_location + "mlt_ar_means_histogram.png")


    print("Perimeters:", np.median(one_perimeter_sizes), np.median(two_perimeter_sizes), np.median(three_perimeter_sizes),
          np.median(four_perimeter_sizes), np.median(five_perimeter_sizes), np.median(six_perimeter_sizes))
    print("Aspect Ratios:", np.median(one_perimeter_ars), np.median(two_perimeter_ars), np.median(three_perimeter_ars),
          np.median(four_perimeter_ars), np.median(five_perimeter_ars), np.median(six_perimeter_ars))

    # Print some stats
    print(f"Solar cycle phase: {solar_cycle_phase}")
    print(f"Mean perimeter size: {np.mean(all_perimeter_sizes_list):.2f} km")
    print(f"Median perimeter size: {np.median(all_perimeter_sizes_list):.2f} km")
    print(f"Maximum perimeter size: {np.max(all_perimeter_sizes_list):.2f} km")
    print(f"Mode perimeter size: {mode(np.round(all_perimeter_sizes_list, decimals=-1))[0][0]:.2f} km")
    print(f"Mode aspect ratio: {mode(np.round(all_ars_list, decimals=2))[0][0]:.2f}")
    print(f"Proportion of log(AR)s less than zero: {len([ar for ar in all_ars_list if np.log10(ar) < 0])/len(all_ars_list):.4f}")
    print(f"Proportion of times no LGMDs are found: "
          f"{len([num for num in num_of_perimeters_list if num == 0])/len(num_of_perimeters_list):.4f}")

    try:
        stime, etime = pd.to_datetime("2013-03-01 00:00:00"), pd.to_datetime("2013-03-04 00:00:00")
        timestamps = mag_data.index
        plt.figure()
        fig, var_axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        index_axs = [var_axs[0].twinx(), var_axs[1].twinx(), var_axs[2].twinx()]
        perimeter_means = [np.nan]*len(perimeters)
        ar_means = [np.nan]*len(aspect_ratios)
        for timestep in range(len(perimeters)):
            if len(perimeters[timestep]) > 0:  # If there are any perimeters
                perimeter_means[timestep] = np.mean(perimeters[timestep])
                ar_means[timestep] = np.mean(np.log10(aspect_ratios[timestep]))
        num_storm_lgmds = pd.DataFrame(num_of_perimeters_list, index=timestamps)
        perimeter_means = pd.DataFrame(perimeter_means, index=timestamps)
        ar_means = pd.DataFrame(ar_means, index=timestamps)
        timestamps = timestamps[(timestamps >= stime) & (timestamps <= etime)]
        num_storm_lgmds = num_storm_lgmds[(num_storm_lgmds.index >= stime) & (num_storm_lgmds.index <= etime)]
        perimeter_means = perimeter_means[(perimeter_means.index >= stime) & (perimeter_means.index <= etime)]
        ar_means = ar_means[(ar_means.index >= stime) & (ar_means.index <= etime)]
        index_data = index_data.loc[perimeter_means.index]
        corr_str = correlation_with_index(perimeter_means, index_data)
        var_axs[0].plot(timestamps, num_storm_lgmds, c="black")
        index_axs[0].plot(timestamps, index_data, c="green")
        var_axs[1].plot(timestamps, perimeter_means, c="black")
        index_axs[1].plot(timestamps, index_data, c="green")
        var_axs[2].plot(timestamps, ar_means, c="black")
        index_axs[2].plot(timestamps, index_data, c="green")
        # Add correlation values to legend in a hacky way
        var_axs[0].plot([], [], ' ', label=f"r={correlation_with_index(num_storm_lgmds, index_data):.2f}")
        var_axs[1].plot([], [], ' ', label=f"r={correlation_with_index(perimeter_means, index_data):.2f}")
        var_axs[2].plot([], [], ' ', label=f"r={correlation_with_index(ar_means, index_data):.2f}")
        # fig.suptitle(f"LGMD Sizes (Example Storm) (r={corr_str})",
        #              fontsize=16)
        var_axs[0].set_ylabel("Number", fontsize=14)
        var_axs[1].set_ylabel(f"Mean Perimeter (km)", fontsize=14)
        var_axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        var_axs[2].set_ylabel("Mean $log(A)$", fontsize=14)
        var_axs[2].set_xlabel("Time", fontsize=14)
        # index_axs[0].set_ylabel("SYM-H (nT)", fontsize=14, color="green")
        index_axs[1].set_ylabel("SYM-H (nT)", fontsize=14, color="green")
        # index_axs[2].set_ylabel("SYM-H (nT)", fontsize=14, color="green")
        for this_plot in range(3):
            if this_plot == 2:
                var_axs[this_plot].tick_params(axis='x', labelsize=14, labelrotation=40)
            else:
                var_axs[this_plot].tick_params(axis='x', bottom=False, labelbottom=False)
            var_axs[this_plot].tick_params(axis='y', labelsize=14)
            index_axs[this_plot].tick_params(axis='y', labelcolor="green", labelsize=14)
            var_axs[this_plot].legend(fontsize=14)
        plt.subplots_adjust(hspace=0)
        plt.tight_layout()
        plt.savefig(stats_plots_location + "blob_size_timeseries.pdf")
    except ValueError:
        print('#'*8 +"\nWarning!\nSkipping example storm plot because there is no data for it.\n"
                    "Ensure the storm dates that determine solar_cycle_phase are set correctly.\n"+'#'*8)

    return num_of_perimeters_list, all_perimeter_sizes_list, all_ars_list


if __name__ == "__main__":
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
    B_param = "dbn_geo"  # "dbn_geo", "dbe_geo", or "HORIZONTAL"
    contour_level = 25.95
    omni_feature = "SYM_H"
    plot_interps = True
    plot_every_n_interps = 5000
    solar_cycle_phase = "full"  # "minimum", "maximum", or "full"
    stats_plots_location = "stats_plots/"
    interp_plots_location = "interp_plots/"

    config_dict = {"syear": syear, "eyear": eyear, "stations_list": stations_list,
                   "station_coords_list": station_coords_list,
                   "n_sec_lat": n_sec_lat, "n_sec_lon": n_sec_lon, "n_poi_lat": n_poi_lat, "n_poi_lon": n_poi_lon,
                   "w_lon": w_lon, "e_lon": e_lon, "s_lat": s_lat, "n_lat": n_lat, "poi_coords_list": poi_coords_list,
                   "epsilon": epsilon, "B_param": B_param, "contour_level": contour_level, "omni_feature": omni_feature,
                   "plot_interps": plot_interps, "plot_every_n_interps": plot_every_n_interps,
                   "solar_cycle_phase": solar_cycle_phase, "stats_plots_location": stats_plots_location,
                   "interp_plots_location": interp_plots_location}

    num_perimeters_full, sizes_perimeters_full, ars_full = stats_analysis(config_dict)

    config_dict["solar_cycle_phase"] = "minimum"
    num_perimeters_min, sizes_perimeters_min, ars_min = stats_analysis(config_dict)

    config_dict["solar_cycle_phase"] = "maximum"
    num_perimeters_max, sizes_perimeters_max, ars_max = stats_analysis(config_dict)

    plot_num_of_blobs(num_blobs_full=num_perimeters_full, num_blobs_min=num_perimeters_min, num_blobs_max=num_perimeters_max)
    plot_blob_sizes(sizes_full=sizes_perimeters_full, sizes_min=sizes_perimeters_min, sizes_max=sizes_perimeters_max)
    plot_aspect_ratios(ars_full=ars_full, ars_min=ars_min, ars_max=ars_max)
    plot_num_and_sizes(num_full=num_perimeters_full, sizes_full=sizes_perimeters_full)
