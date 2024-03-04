import datetime as dt
import pandas as pd
import tqdm


def storm_extract(df, storm_list, B_param, lead, recovery):
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
    if saving:
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
        mag_data = pd.read_hdf(f"supermag_processed_{syear}-{eyear}.h5", key=B_param)

    return mag_data


def load_omni(syear, eyear, data_path, feature="SYM_H"):
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
