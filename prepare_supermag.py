import glob
import datetime as dt
import pandas as pd
import numpy as np
import tqdm
import cdflib


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

    print("Done preprocessing!")


if __name__ == "__main__":
    stations_list = ['YKC', 'BLC', 'MEA', 'SIT', 'BOU', 'VIC', 'NEW', 'OTT', 'GIM', 'DAW', 'FCC', 'FMC',
                     'FSP', 'SMI', 'ISL', 'PIN', 'RAL', 'RAN', 'CMO', 'IQA', 'C04', 'C06', 'C10', 'T36']
    syear, eyear = 2009, 2019
    stime = pd.Timestamp(syear, 1, 1)
    etime = pd.Timestamp(eyear, 12, 31, 23, 59, 59)
    preprocess_supermag(stime, etime, stations_list, f"{syear}-{eyear}", interpolate=True, method="linear", limit=None)

