import glob
import datetime as dt
import pandas as pd
import numpy as np
import tqdm
import glob
from preprocessing_fns import *


def preprocess_supermag(stime, etime, stations, savefile_name, interpolate=True, method="linear", limit=None):
    """
    method	: Interpolation method for missing data. Default 'linear'
    limit	: Limit to the interpolation (in minutes) None for unlimited, 0 for no interpolation.
              Default 'None'
    """


    for station in stations:
        mag_files = glob.glob(f"/data/supermag/baseline/{station}/{station}-*-supermag-baseline.csv")

        m = []

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
        mag_data.reset_index(drop=False).to_feather(f"/data/ramans_files/mag-feather/magData-{station}_"+savefile_name+".feather")

        del mag_data

    print(f"Done preprocessing SuperMAG from {syear} to {eyear}\n!")


def preprocess_omni(syear, eyear, data_dir,
                     interpolate=True, method='linear', limit=None, to_drop=[]):
    start_time = pd.Timestamp(syear, 1, 1)
    end_time = pd.Timestamp(eyear, 12, 31, 23, 59, 59)

    omni_files = glob.glob(data_dir + 'omni/hro_1min/*/*.cdf', recursive=True)
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
    bad_omni_to_nan(omni_data);

    # Drop unwanted columns
    omni_data = omni_data.drop(to_drop, axis=1)

    # Process the data. Right now interpolation on all missing data. May change later
    if interpolate:
        for param in omni_data.columns:
            omni_data[param] = omni_data[param].interpolate(method=method, limit=limit)

    # Export to feather format
    print(omni_data.info())
    omni_data.reset_index(drop=True).to_feather(data_dir + f'ramans_files/omni-feather/omniData-{syear}-{eyear}-interp-{limit}.feather')

    print(f'Finished pre-processing OMNI data from {syear} to {eyear}\n')
    return omni_data


if __name__ == "__main__":
    stations_list = ['YKC', 'BLC', 'MEA', 'SIT', 'BOU', 'VIC', 'NEW', 'OTT', 'GIM', 'DAW', 'FCC', 'FMC',
                     'FSP', 'SMI', 'ISL', 'PIN', 'RAL', 'RAN', 'CMO', 'IQA', 'C04', 'C06', 'C10', 'T36']
    syear, eyear = 2009, 2019
    stime = pd.Timestamp(syear, 1, 1)
    etime = pd.Timestamp(eyear, 12, 31, 23, 59, 59)
    # preprocess_supermag(stime, etime, stations_list, f"{syear}-{eyear}", interpolate=True, method="linear", limit=None)
    preprocess_omni(syear, eyear, "/data/", interpolate=True, method='linear', limit=None, to_drop=[])