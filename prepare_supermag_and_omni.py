import datetime as dt
import tqdm
import glob
from preprocessing_fns import *

"""
This script prepares SuperMAG and OMNI data just before they are fully preprocessed. If you wish to use this, modify
the file paths so that they point to the locations where you wish to load and save your data.
You should only need to run this script once.
"""


def preprocess_supermag(stime, etime, mag_files_path, stations, savefile_name, interpolate=True, method="linear",
                        limit=None):
    """
    This function loads your downloaded SuperMAG csv files and does some basic preparation and cleaning.
    :param stime: A pandas timestamp with 1-minute precision representing the date and time of the earliest point in
    your dataset.
    :param etime: A pandas timestamp with 1-minute precision representing the date and time of the latest point in
    your dataset.
    :param mag_files_path: A string which is the path in which your SuperMAG files are kept. In this directory,
    they should be stored in subdirectories which are each named as that station's IAGA code.
    :param stations: A list of strings containing the IAGA codes of each station you are using.
    :param savefile_name: A string which provides some extra identifying information in the names of the feather files
    created by this function. It is recommended to leave this as an empty string or to use it to specify the
    range of years your dataset covers.
    :param interpolate: Bool saying whether to interpolate missing data. Default True. It's very rare that
    you would want this to be False.
    :param method: String. Interpolation method for missing data. Default 'linear'.
    :param limit: Int. Limit to the interpolation (in minutes) None for unlimited, 0 for no interpolation.
    Default 'None'.
    :return: None
    """

    for station in stations:
        mag_files = glob.glob(mag_files_path + f"{station}/{station}-*-supermag-baseline.csv")

        m = []  # Initialize a list that will store all the data from files associated with this station

        for entry in tqdm.tqdm(sorted(mag_files), desc=f"Preprocessing SuperMag data for {station}"):
            df = pd.read_csv(entry)
            df.drop('IAGA', axis=1, inplace=True)
            df['Date_UTC'] = pd.to_datetime(df['Date_UTC'])
            df.set_index('Date_UTC', inplace=True, drop=True)
            df = df.reindex(
                pd.date_range(start=dt.datetime(df.index[0].year, 1, 1),
                              end=dt.datetime(df.index[0].year, 12, 31, 23, 59),
                              freq='1 Min'), copy=True, fill_value=np.NaN)
            df['Date_UTC'] = df.index

            m.append(df)

        # Concatenate all dataframes for this station into a single dataframe
        mag_data = pd.concat(m, axis=0, ignore_index=True)

        del m

        mag_data.set_index('Date_UTC', inplace=True, drop=True)

        mag_data = mag_data[stime:etime]

        if interpolate:
            for param in mag_data.columns:
                mag_data[param] = mag_data[param].interpolate(method=method, limit=limit)

        # Export to feather format
        mag_data.reset_index(drop=False).to_feather(f"/data/ramans_files/mag-feather/magData-{station}_"+savefile_name+
                                                    ".feather")

        del mag_data

    print(f"Done preprocessing SuperMAG from {syear} to {eyear}\n!")


def preprocess_omni(syear, eyear, data_dir,
                     interpolate=True, method='linear', limit=None, to_drop=[]):
    """
    This function loads your downloaded OMNI data and does some basic preparation and cleaning.
    :param syear: An int representing the start year of the file you want to load.
    :param eyear: An int representing the end year of the file you want to load.
    :param data_dir: A string which is the path in which your OMNI files are kept.
    :param interpolate: Bool saying whether to interpolate missing data. Default True. It's very rare that
    you would want this to be False.
    :param method: String. Interpolation method for missing data. Default 'linear'.
    :param limit: Int. Limit to the interpolation (in minutes) None for unlimited, 0 for no interpolation.
    :param to_drop: List of strings. Columns to drop from the dataframe. Default [], that is, to drop nothing.
    :return: omni_data, a pandas dataframe containing the prepared data.
    """
    start_time = pd.Timestamp(syear, 1, 1)
    end_time = pd.Timestamp(eyear, 12, 31, 23, 59, 59)

    omni_files = glob.glob(data_dir + '*/*.cdf', recursive=True)
    o = []
    for fil in tqdm.tqdm(sorted(omni_files), desc="Loading OMNI files"):
        cdf = omnicdf2dataframe(fil)
        o.append(cdf)

    omni_data = pd.concat(o, axis=0, ignore_index=True)
    omni_data.index = omni_data.Epoch

    del o

    # Select the temporal subset to export
    omni_data = omni_data[start_time:end_time]

    # Rename some variables to avoid conflicts
    omni_data.rename(columns={'E': 'E_Field'}, inplace=True)
    omni_data.rename(columns={'F': 'B_Total'}, inplace=True)

    # Convert bad numbers to np.nan
    bad_omni_to_nan(omni_data)

    # Drop unwanted columns
    omni_data = omni_data.drop(to_drop, axis=1)

    # Process the data. Right now interpolation on all missing data. May change later
    if interpolate:
        for param in omni_data.columns:
            omni_data[param] = omni_data[param].interpolate(method=method, limit=limit)

    # Export to feather format
    print(omni_data.info())
    omni_data.reset_index(drop=True).to_feather(data_dir + f'ramans_files/omni-feather/omniData-{syear}-{eyear}-interp-'
                                                           f'{limit}.feather')

    print(f'Finished pre-processing OMNI data from {syear} to {eyear}\n')
    return omni_data


if __name__ == "__main__":
    stations_list = ['YKC', 'BLC', 'MEA', 'SIT', 'BOU', 'VIC', 'NEW', 'OTT', 'FRD', 'GIM', 'DAW', 'FCC', 'FMC',
                     'FSP', 'SMI', 'ISL', 'PIN', 'RAL', 'RAN', 'CMO', 'IQA', 'C04', 'C06', 'C10', 'T36']
    syear, eyear = 2009, 2019
    stime = pd.Timestamp(syear, 1, 1)
    etime = pd.Timestamp(eyear, 12, 31, 23, 59, 59)
    mag_files_path = "/data/supermag/baseline/"
    omni_files_path = "/data/omni/hro_1min/"

    preprocess_supermag(stime, etime, mag_files_path, stations_list, f"{syear}-{eyear}", interpolate=True,
                        method="linear", limit=10)
    preprocess_omni(syear, eyear, omni_files_path, interpolate=True, method='linear', limit=None, to_drop=[])
