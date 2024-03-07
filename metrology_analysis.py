import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm


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

    all_B_interps = pd.read_hdf(f"all_B_interps_{n_sec_lat}by{n_sec_lon}_{syear}-{eyear}.h5", B_param)
    # Remove timestamps that are double-counted by immediately successive storms
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
    # plt.ylim(0, 10000)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(stats_plots_location + "mean.png")

    plt.cla()
    plt.hist(std_list, bins=int(np.max(std_list)/5), align="left")
    plt.yscale("log")
    plt.title("Spatial Standard Deviation of Disturbance", fontsize=16)
    plt.xlabel("Standard Deviation of Disturbance (nT)", fontsize=14)
    plt.ylabel(f"# of occurrences in {syear}{('-'+str(eyear))*(syear!=eyear)}", fontsize=14)
    # plt.ylim(0, 5000)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(stats_plots_location + "std.png")

    plt.cla()
    plt.hist(mean_deviation_list, bins=int(np.max(mean_deviation_list)/5), align="left")
    plt.yscale("log")
    plt.title("Spatial Mean Deviation of Disturbance", fontsize=16)
    plt.xlabel("Mean Deviation of Disturbance (nT)", fontsize=14)
    plt.ylabel(f"# of occurrences in {syear}{('-'+str(eyear))*(syear!=eyear)}", fontsize=14)
    # plt.ylim(0, 5000)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(stats_plots_location + "mean_deviation.png")

    plt.cla()
    plt.hist(rms_deviation_list, bins=int(np.max(rms_deviation_list)/5), align="left")
    plt.yscale("log")
    plt.title("Spatial RMS Deviation of Disturbance", fontsize=16)
    plt.xlabel("RMS Deviation of Disturbance (nT)", fontsize=14)
    plt.ylabel(f"# of occurrences in {syear}{('-'+str(eyear))*(syear!=eyear)}", fontsize=14)
    # plt.ylim(0, 5000)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(stats_plots_location + "rms_deviation.png")

    plt.cla()
    plt.hist(max_list, bins=int(np.max(max_list)/5), align="left")
    plt.yscale("log")
    plt.title("Spatial Maximum of Disturbance", fontsize=16)
    plt.xlabel("Maximum Disturbance (nT)", fontsize=14)
    plt.ylabel(f"# of occurrences in {syear}{('-'+str(eyear))*(syear!=eyear)}", fontsize=14)
    percentile = np.percentile(max_list, 50)
    plt.axvline(percentile, color='k', linestyle='dashed', label=f"{percentile:.2f}")
    # plt.ylim(0, 2200)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.savefig(stats_plots_location + "max.png")

    plt.cla()
    plt.hist(min_list, bins=int(np.abs(np.min(min_list))/5), align="left")
    plt.yscale("log")
    plt.title("Spatial Minimum of Disturbance", fontsize=16)
    plt.xlabel("Minimum Disturbance (nT)", fontsize=14)
    plt.ylabel(f"# of occurrences in {syear}{('-'+str(eyear))*(syear!=eyear)}", fontsize=14)
    percentile = np.percentile(min_list, 50)
    plt.axvline(percentile, color='k', linestyle='dashed', label=f"{percentile:.2f}")
    # plt.ylim(0, 2200)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.savefig(stats_plots_location + "min.png")

    print(f"Your metrology plots are ready in {stats_plots_location}")
