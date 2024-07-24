import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
from scipy.stats import mode


# Config variables
syear, eyear = 2009, 2019
n_sec_lat, n_sec_lon = 10, 35  # # of rows and columns respectively of SECSs that will exist in the grid
B_param = "dbn_geo"
stats_plots_location = "stats_plots/"


def rms_deviation(B_interp_vector):
    # AKA Rq in surface metrology
    return np.sqrt(np.mean(np.square(B_interp_vector)))


def mean_deviation(B_interp_vector):
    # AKA Ra in surface metrology
    return np.mean(np.abs(B_interp_vector))


if __name__ == "__main__":

    if B_param == "dbn_geo" or B_param == "dbe_geo":
        all_B_interps = pd.read_hdf(f"all_B_interps_{n_sec_lat}by{n_sec_lon}_{syear}-{eyear}.h5", B_param)
    elif B_param == "HORIZONTAL":
        all_BN_interps = pd.read_hdf(f"all_B_interps_{n_sec_lat}by{n_sec_lon}_{syear}-{eyear}.h5","dbn_geo")
        all_BE_interps = pd.read_hdf(f"all_B_interps_{n_sec_lat}by{n_sec_lon}_{syear}-{eyear}.h5", "dbe_geo")
        all_B_interps = np.sqrt(all_BN_interps**2 + all_BE_interps**2)
        del all_BN_interps, all_BE_interps    # Remove timestamps that are double-counted by immediately successive storms
    all_B_interps.drop_duplicates(inplace=True)

    mean_list, std_list, rms_deviation_list, mean_deviation_list, max_list, min_list = [], [], [], [], [], []

    for t in tqdm.trange(len(all_B_interps), desc="Generating statistics"):
        B_interp = all_B_interps.iloc[t]
        mean_list.append(np.mean(B_interp))
        std_list.append(np.std(B_interp))
        rms_deviation_list.append(rms_deviation(B_interp))
        mean_deviation_list.append(mean_deviation(B_interp))
        max_list.append(np.max(B_interp))
        min_list.append(np.min(B_interp))

    plt.hist(mean_list, bins=int(2*np.max(mean_list)/5), align="left")
    plt.yscale("log")
    plt.title("Spatial Mean of Disturbance", fontsize=16)
    plt.xlabel("Mean Disturbance (nT)", fontsize=14)
    plt.ylabel(f"# of occurrences in {syear}{('-'+str(eyear))*(syear!=eyear)}", fontsize=14)
    peak_location = mode(np.round(mean_list, decimals=0))[0][0]
    plt.axvline(peak_location, color='k', linestyle='dashed', label=f"Mode: {peak_location:.0f} nT")
    # plt.ylim(0, 10000)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(stats_plots_location + "mean.pdf")

    plt.cla()
    plt.hist(std_list, bins=int(np.max(std_list)/5), align="left")
    plt.yscale("log")
    plt.title("Spatial Standard Deviation of Disturbance", fontsize=16)
    plt.xlabel("Standard Deviation of Disturbance (nT)", fontsize=14)
    plt.ylabel(f"# of occurrences in {syear}{('-'+str(eyear))*(syear!=eyear)}", fontsize=14)
    peak_location = mode(np.round(std_list, decimals=0))[0][0]
    plt.axvline(peak_location, color='k', linestyle='dashed', label=f"Mode: {peak_location:.0f} nT")
    # plt.ylim(0, 5000)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(stats_plots_location + "std.pdf")

    plt.cla()
    plt.hist(mean_deviation_list, bins=int(np.max(mean_deviation_list)/5), align="left")
    plt.yscale("log")
    plt.title("Spatial Mean Deviation of Disturbance", fontsize=16)
    plt.xlabel("Mean Deviation of Disturbance (nT)", fontsize=14)
    plt.ylabel(f"# of occurrences in {syear}{('-'+str(eyear))*(syear!=eyear)}", fontsize=14)
    peak_location = mode(np.round(mean_deviation_list, decimals=0))[0][0]
    plt.axvline(peak_location, color='k', linestyle='dashed', label=f"Mode: {peak_location:.0f} nT")
    # plt.ylim(0, 5000)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(stats_plots_location + "mean_deviation.pdf")

    plt.cla()
    plt.hist(rms_deviation_list, bins=int(np.max(rms_deviation_list)/5), align="left")
    plt.yscale("log")
    plt.title("Spatial RMS Deviation of Disturbance", fontsize=16)
    plt.xlabel("RMS Deviation of Disturbance (nT)", fontsize=14)
    plt.ylabel(f"# of occurrences in {syear}{('-'+str(eyear))*(syear!=eyear)}", fontsize=14)
    peak_location = mode(np.round(rms_deviation_list, decimals=0))[0][0]
    plt.axvline(peak_location, color='k', linestyle='dashed', label=f"Mode: {peak_location:.0f} nT")
    # plt.ylim(0, 5000)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(stats_plots_location + "rms_deviation.pdf")

    plt.cla()
    plt.hist(max_list, bins=int(np.max(max_list)/5), align="left")
    plt.yscale("log")
    plt.title("Spatial Maximum of Disturbance", fontsize=16)
    plt.xlabel("Maximum Disturbance (nT)", fontsize=14)
    plt.ylabel(f"# of occurrences in {syear}{('-'+str(eyear))*(syear!=eyear)}", fontsize=14)
    # percentile = np.percentile(max_list, 50)
    # plt.axvline(percentile, color='k', linestyle='dashed', label=f"Median: {percentile:.2f}")
    peak_location = mode(np.round(max_list, decimals=0))[0][0]
    plt.axvline(peak_location, color='k', linestyle='dashed', label=f"Mode: {peak_location:.0f} nT")
    # plt.ylim(0, 2200)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.savefig(stats_plots_location + "max.pdf")

    plt.cla()
    if B_param == "HORIZONTAL":
        min_bins = int(np.abs(np.max(max_list))/5)
    else:
        min_bins = int(np.abs(np.min(min_list))/5)
    plt.hist(min_list, bins=min_bins, align="left")
    plt.yscale("log")
    plt.title("Spatial Minimum of Disturbance", fontsize=16)
    plt.xlabel("Minimum Disturbance (nT)", fontsize=14)
    plt.ylabel(f"# of occurrences in {syear}{('-'+str(eyear))*(syear!=eyear)}", fontsize=14)
    # percentile = np.percentile(min_list, 50)
    # plt.axvline(percentile, color='k', linestyle='dashed', label=f"Median: {percentile:.2f}")
    peak_location = mode(np.round(min_list, decimals=0))[0][0]
    plt.axvline(peak_location, color='k', linestyle='dashed', label=f"Mode: {peak_location:.0f} nT")
    # plt.ylim(0, 2200)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.savefig(stats_plots_location + "min.pdf")

    print(f"Your metrology plots are ready in {stats_plots_location}")
