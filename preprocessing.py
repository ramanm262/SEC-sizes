import pandas as pd
import datetime as dt
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


if __name__ == "__main__":
    stations_list = ['YKC', 'CBB', 'BLC', 'SIT', 'BOU', 'VIC', 'NEW', 'OTT', 'FRD', 'GIM', 'FCC', 'FMC', 'FSP',
                 'SMI', 'ISL', 'PIN', 'RAL', 'INK', 'CMO', 'IQA', 'LET',
                 'T16', 'T32', 'T33', 'T36']
    a=load_supermag(stations_list, 2016, 2016, B_param="dbe_geo", saving=True)
    print(a)
