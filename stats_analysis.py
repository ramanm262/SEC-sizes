import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
from skimage import measure
import tqdm


# Config variables
syear, eyear = 2016, 2016
station_coords_list = [np.array([62.48, 69.1, 64.33, 57.07, 40.13, 48.52, 48.27, 45.4, 38.2, 56.38, 58.76, 56.66,
                                 61.76, 60.02, 53.86, 50.2, 58.22, 68.25, 64.87, 63.75, 49.64, 39.19, 49.4, 54.0,
                                 54.71]),
                       np.array([245.52, 255.0, 263.97, 224.67, 254.77, 236.58, 242.88, 284.45, 282.63, 265.36,
                                 265.92, 248.79, 238.77, 248.05, 265.34, 263.96, 256.32, 226.7, 212.14, 291.48,
                                 247.13, 240.2, 277.7, 259.1, 246.69])]
n_sec_lat, n_sec_lon = 16, 7  # Number of rows and columns respectively of SECSs that will exist in the grid
n_poi_lat, n_poi_lon = 21, 40  # Number of rows and columns respectively of POIs that will exist in the grid
poi_coords_list = [np.linspace(40, 60, n_poi_lat), np.linspace(220, 290, n_poi_lon)]
epsilon = 0.09610742146188743
plot_interps = True
plot_every_n_interps = 1000
stats_plots_location = "stats_plots/"
interp_plots_location = "interp_plots/"


def rms_deviation(B_interp_vector):
    # AKA Rq in surface metrology
    return np.sqrt(np.mean(np.square(B_interp_vector)))


def mean_deviation(B_interp_vector):
    # AKA Ra in surface metrology
    return np.mean(np.abs(B_interp_vector))


all_B_interps = pd.read_hdf(f"all_B_interps_{n_sec_lat}by{n_sec_lon}_{syear}-{eyear}.h5", 'dbn_geo')

mean_list, std_list, rms_deviation_list, mean_deviation_list, max_list, min_list = [], [], [], [], [], []

for t in tqdm.trange(len(all_B_interps), desc="Generating statistics"):
    B_interp = all_B_interps.iloc[t]
    mean_list.append(np.mean(B_interp))
    std_list.append(np.std(B_interp))
    rms_deviation_list.append(rms_deviation(B_interp))
    mean_deviation_list.append(mean_deviation(B_interp))
    max_list.append(np.max(B_interp))
    min_list.append(np.min(B_interp))

y_axis_max = 12600

plt.hist(mean_list, bins=int(2*np.max(mean_list)/5), align="left")
plt.title("Spatial Mean of Disturbance")
plt.xlabel("Mean Disturbance (nT)")
plt.ylabel("Number of times the region had this value")
plt.ylim(0, y_axis_max)
plt.tight_layout()
plt.savefig(stats_plots_location + "mean.png")

plt.cla()
plt.hist(std_list, bins=int(np.max(std_list)/10), align="left")
plt.title("Spatial Standard Deviation of Disturbance")
plt.xlabel("Standard Deviation of Disturbance (nT)")
plt.ylabel("Number of times the region had this value")
plt.ylim(0, y_axis_max)
plt.tight_layout()
plt.savefig(stats_plots_location + "std.png")

plt.cla()
plt.hist(mean_deviation_list, bins=int(np.max(mean_deviation_list)/10), align="left")
plt.title("Spatial Mean Deviation of Disturbance")
plt.xlabel("Mean Deviation of Disturbance (nT)")
plt.ylabel("Number of times the region had this value")
plt.ylim(0, y_axis_max)
plt.tight_layout()
plt.savefig(stats_plots_location + "mean_deviation.png")

plt.cla()
plt.hist(rms_deviation_list, bins=int(np.max(rms_deviation_list)/10), align="left")
plt.title("Spatial RMS Deviation of Disturbance")
plt.xlabel("RMS Deviation of Disturbance (nT)")
plt.ylabel("Number of times the region had this value")
plt.ylim(0, y_axis_max)
plt.tight_layout()
plt.savefig(stats_plots_location + "rms_deviation.png")

plt.cla()
plt.hist(max_list, bins=int(np.max(max_list)/50), align="left")
plt.title("Spatial Maximum of Disturbance")
plt.xlabel("Maximum Disturbance (nT)")
plt.ylabel("Number of times the region had this value")
plt.ylim(0, y_axis_max)
plt.tight_layout()
plt.savefig(stats_plots_location + "max.png")

plt.cla()
plt.hist(min_list, bins=int(np.abs(np.min(min_list))/50), align="left")
plt.title("Spatial Minimum of Disturbance")
plt.xlabel("Minimum Disturbance (nT)")
plt.ylabel("Number of times the region had this value")
plt.ylim(0, y_axis_max)
plt.tight_layout()
plt.savefig(stats_plots_location + "min.png")


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


plt.cla()
color_map = "bwr"
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(13, 5))
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
norm = plt.cm.colors.Normalize(vmin=np.min(all_B_interps), vmax=np.max(all_B_interps))
scalarmappable = plt.cm.ScalarMappable(norm=norm, cmap=color_map)
station_scatter = ax1.scatter(station_coords_list[1], station_coords_list[0])
ax1.set_xlim(200, 310)
ax1.set_ylim(35, 75)
ax1.set_title("Locations of Stations and Current System Poles")


perimeters = []
for timestep in tqdm.trange(len(all_B_interps), desc="Generating heatmaps"):
    heatmap_data = np.abs(np.array(all_B_interps.iloc[timestep]).reshape((n_poi_lat, n_poi_lon)))

    if plot_interps and (timestep % plot_every_n_interps == 0):
        ax2.cla()
        ax2.scatter(np.array(all_poi_lons) * 180 / np.pi, all_poi_lats, c=all_B_interps.iloc[timestep], cmap=color_map)
        station_scatter.set_color(np.random.rand(len(station_coords_list[0]), 3))
        ax2.set_xlim(210, 300)
        ax2.set_ylim(35, 75)
        ax2.set_title(f"Interpolation (epsilon = {epsilon})")
        cb = fig.colorbar(scalarmappable, ax=(ax1, ax2), cax=cax, orientation='vertical')

    contours = measure.find_contours(heatmap_data)  # Each element of list is a list coords in row, col format

    this_perimeters = []
    for contour in contours:
        this_contour_lons = poi_lons_mesh[contour[:, 0].astype(int), contour[:, 1].astype(int)]
        this_contour_lats = poi_lats_mesh[contour[:, 0].astype(int), contour[:, 1].astype(int)]
        if plot_interps and (timestep % plot_every_n_interps == 0):  # Only plot if plotting is enabled
            ax2.plot(this_contour_lons, this_contour_lats, linewidth=2, color='black')
        contour_mask = np.zeros_like(heatmap_data)
        contour_mask[contour[:, 0].astype(int), contour[:, 1].astype(int)] = 1
        if contour[0, 0] == contour[-1, 0] and contour[0, 1] == contour[-1, 1]:  # If contour is closed
            this_perimeters.append(measure.perimeter(contour_mask))

    if plot_interps and (timestep % plot_every_n_interps == 0):
        plt.savefig(interp_plots_location + f"interpolated_values_{timestep}.png")
    perimeters.append(this_perimeters)

num_of_perimeters_list = [len(timestep) for timestep in perimeters]
all_perimeter_sizes_list = [perimeter for timestep in perimeters for perimeter in timestep]

plt.figure()
plt.hist(num_of_perimeters_list)
plt.title("Number of dB 'Blobs'")
plt.xlabel("Number of blobs")
plt.ylabel("Number of times the region had this value")
plt.tight_layout()
plt.savefig(stats_plots_location + "num_of_blobs.png")

plt.figure()
plt.hist(all_perimeter_sizes_list)
plt.title("dB Blob Sizes")
plt.xlabel("dB Blob Size")
plt.ylabel("Number of times the region had this value")
plt.tight_layout()
plt.savefig(stats_plots_location + "blob_sizes.png")
