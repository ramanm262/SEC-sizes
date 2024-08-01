import datetime as dt
import pandas as pd
import numpy as np
import tqdm

"""
This file contains functions that format the data correctly but do not clean it. Therefore, it must be run AFTER
prepare_supermag_and_omni.py
"""


def storm_extract(df, storm_list, lead, recovery):
    """
    This function takes in the entire SuperMAG dataset and returns only the sections associated with storm time.
    :param df: A pandas dataframe containing the prepared SuperMAG dataset.
    :param storm_list: A pandas dataframe contining the dates and times (one column) of the SYM-H minima of all storms
    you wish to analyze.
    :param lead: An int representing the number of hours before SYM-H minimum at which the storm is considered to start.
    :param recovery: An int representing the number of hours before SYM-H minimum at which the storm is considered to
    end.
    :return: storms, a pandas dataframe containing the same information as df except just those data during storm time.
    """
    storms = []
    stime, etime = [], []  # will store the resulting time stamps here then append them to the storm time df
    for date in storm_list:
        stime.append((dt.datetime.strptime(date, '%m-%d-%Y %H:%M'))-pd.Timedelta(hours=lead))
        etime.append((dt.datetime.strptime(date, '%m-%d-%Y %H:%M'))+pd.Timedelta(hours=recovery))
    # adds the time stamp lists to the storm_list dataframes
    storm_list['stime'] = stime
    storm_list['etime'] = etime
    for start, end in tqdm.tqdm(zip(storm_list['stime'], storm_list['etime']), total=len(storm_list),
                                desc="Extracting storms"):
        storm = df[(df.index >= start) & (df.index <= end)]
        storm = storm.dropna()
        if len(storm) != 0:
            storms.append(storm)			# creates a list of smaller storm time dataframes
    storms = pd.concat(storms, axis=0, ignore_index=False)

    return storms


def load_supermag(stations_list, syear, eyear, B_param="dbn_geo", storm_time_only=True, saving=True,
                  data_path="/data/ramans_files/mag-feather/"):
    """
    This function loads the SuperMAG data from the feather files and returns a pandas dataframe containing the data.
    :param stations_list: A list of strings containing the names of the stations you wish to analyze.
    :param syear: An int representing the start year of the data you wish to analyze.
    :param eyear: An int representing the end year of the data you wish to analyze.
    :param B_param: String, either "dbn_geo" "dbe_geo". This represents whether you want to load the geographic
    northward or geographic eastward component, respectively, of the magnetic perturbation. If you want to load both,
    call this function twice.
    :param storm_time_only: A bool determining whether the loaded SuperMAG data should be pared down to just storm
    time. Default True.
    :param saving: A bool determining whether the data should be saved to a .h5 file. If false, the data will be loaded
    from a .nh5 file instead. Default True.
    :param data_path: A string describing the path to where the deata should be saved (or loaded, if saving==False).
    :return: mag_data, a pandas dataframe of magnetometer data from each station and from each timestep in storm time,
    ready for analysis!
    """
    if saving:
        if B_param == "HORIZONTAL":
            raise ValueError("'B_param' cannot be 'HORIZONTAL' when 'saving' is True."
                             "Use 'dbn_geo' or 'dbe_geo' for 'B_param' instead.")
        mag_data = []
        for station_num in tqdm.trange(len(stations_list), desc=f"Loading SuperMAG {B_param[2]} data"):

            station_name = stations_list[station_num]

            # Load SuperMAG data
            this_mag_data = pd.read_feather(data_path+f"magData-{station_name}_{syear}-"
                                            f"{eyear}.feather")
            this_mag_data = this_mag_data.set_index("Date_UTC", drop=True)
            this_mag_data = this_mag_data[B_param]
            this_mag_data.rename(station_name+'_'+B_param, inplace=True)
            this_mag_data.index = pd.to_datetime(this_mag_data.index)
            mag_data.append(this_mag_data)
        mag_data = pd.concat(mag_data, axis=1, ignore_index=False)

        if storm_time_only:
            storm_list = pd.read_csv("stormList.csv", header=None, names=["dates"])
            mag_data = storm_extract(mag_data, storm_list["dates"], B_param, lead=12, recovery=24)

        mag_data.to_hdf(f"supermag_processed_{syear}-{eyear}.h5", key=B_param)

    else:
        if B_param == "dbn_geo" or B_param == "dbe_geo":
            mag_data = pd.read_hdf(f"supermag_processed_{syear}-{eyear}.h5", key=B_param)
        elif B_param == "HORIZONTAL":
            mag_n_data = pd.read_hdf(f"supermag_processed_{syear}-{eyear}.h5", key="dbn_geo")
            mag_e_data = pd.read_hdf(f"supermag_processed_{syear}-{eyear}.h5", key="dbe_geo")
            # Remove column names so the dataframes can be added elementwise
            mag_n_data.columns, mag_e_data.columns = (np.arange(len(mag_n_data.columns)),
                                                      np.arange(len(mag_e_data.columns)))
            mag_data = np.sqrt(mag_n_data**2 + mag_e_data**2)
            # Put column names back in
            mag_data.columns = [station_name + "_HORIZONTAL" for station_name in stations_list]
            del mag_n_data, mag_e_data

    return mag_data


def load_omni(syear, eyear, data_path, feature="SYM_H"):
    """
    Use this function to load a geomagnetic index preprocessed OMNI data.
    :param syear: Int. The start year of the dataset you wish to load.
    :param eyear: Int. The end year of the dataset you wish to load.
    :param data_path: String. The path to the directory containing the OMNI data.
    :param feature: String. The name of the geomagnetic index you wish to load. Can be 'SYM_H' 'AE_INDEX'. Default
    'SYM_H'.
    :return: omni_data, a one-column pandas dataframe containing the geomagnetic index data.
    """
    omni_data = pd.read_feather(data_path + f"omniData-{syear}-{eyear}-interp-None.feather")
    omni_data = omni_data.rename(columns={"Epoch": "Date_UTC"})
    omni_data.set_index("Date_UTC", inplace=True, drop=True)
    omni_data = omni_data[[feature]]  # Only use these features
    return omni_data


if __name__ == "__main__":
    syear = 2009
    eyear = 2019
    stations_list = ['YKC', 'BLC', 'MEA', 'SIT', 'BOU', 'VIC', 'NEW', 'OTT', 'GIM', 'DAW', 'FCC', 'FMC',
                     'FSP', 'SMI', 'ISL', 'PIN', 'RAL', 'RAN', 'CMO', 'IQA', 'C04', 'C06', 'C10', 'T36']
    load_supermag(stations_list, syear, eyear, B_param="dbn_geo", saving=True)
    print("Done")
