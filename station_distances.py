from stats_analysis import *


station_coords_list = [np.array([62.48, 64.33, 54.62, 57.07, 40.13, 48.52, 48.27, 45.4, 56.38, 64.05, 58.76,
                                 56.66, 61.76, 60.02, 53.86, 50.2, 58.22, 62.82, 64.87, 63.75, 50.06, 53.35,
                                 47.66, 54.71, 38.2]),
                       np.array([245.52, 263.97, 246.65, 224.67, 254.77, 236.58, 242.88, 284.45, 265.36, 220.89,
                                 265.92, 248.79, 238.77, 248.05, 265.34, 263.96, 256.32, 267.89, 212.14, 291.48,
                                 251.74, 247.03, 245.79, 246.69, 77.37+180])]
stations_list = ['YKC', 'BLC', 'MEA', 'SIT', 'BOU', 'VIC', 'NEW', 'OTT', 'GIM', 'DAW', 'FCC', 'FMC',
                 'FSP', 'SMI', 'ISL', 'PIN', 'RAL', 'RAN', 'CMO', 'IQA', 'C04', 'C06', 'C10', 'T36', "FRD"]

station_coords_list = [np.radians(station_coords_list[0]), np.radians(station_coords_list[1])]
coords_pairs_list = [(station_coords_list[0][i], station_coords_list[1][i]) for i in range(len(station_coords_list[0]))]
all_distances = haversine_distances(coords_pairs_list) * 6371

max_distances = np.unique(np.sort(all_distances.flatten()))[-5:]
min_distances = np.unique(np.sort(all_distances[all_distances != 0].flatten()))[:15]
max_indices = np.argwhere(np.isin(all_distances, max_distances))
min_indices = np.argwhere(np.isin(all_distances, min_distances))
print(f"Max distances: {max_distances}")
print(f"Min distances: {min_distances}")
print(f"Max indices: {max_indices}")
print(f"Min indices: {min_indices}")

plt.hist(np.unique(all_distances.flatten()), bins=40)
plt.xlabel("Distance (km)")
plt.ylabel("Frequency")
plt.title("Histogram of Distances Between Stations")
plt.show()

# Find the distance between each station and its closest neighbor
closest_distances = np.min([row[row != 0] for row in all_distances], axis=1)
# Print the names of the stations with the closest distances
print(pd.concat([pd.Series(stations_list), pd.Series(closest_distances)], axis=1).sort_values(by=1))