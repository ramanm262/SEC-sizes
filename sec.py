"""
Functions for applying the Spherical Elementary Current Systems method to magnetometer data
"""

from scipy.linalg import svd
import numpy as np
import pandas as pd
import tqdm


def B_theta(I_0, r, theta, R_I):
    """
    Calculates the magnitude of the horizontal component of the magnetic field produced by a current system; for
    example, as in (Amm & Viljanen 1999), Eq. 10.
    :param I_0: The scaling factor in the equation; that is, the strength of the current system. When calculating the
    transfer matrix, this should be 1.
    :param r: The radius of the Earth's surface in meters.
    :param theta: The sphere-interior angle between the SEC and a point of interest. That is, this is theta when the
    SEC is at the North Pole. Refer to (Amm & Viljanen 1999), section 3.
    :param R_I: The radius, in meters, of the model ionospheric current sheet. Should be roughly 1 RE + 100 km.
    :return: A float which is the egocentric theta component of the ground magnetic field produced by an SEC with
    value I_0.
    """
    return -10**-7*I_0/(r*np.sin(theta))*((r/R_I-np.cos(theta))/np.sqrt(1-(2*r*np.cos(theta))/R_I+(r/R_I)**2) +
                                          np.cos(theta))*10**9


def ego_to_geo(Btheta_ego, colat_l, lon_l, colat_k, lon_k):
    """
    Transforms a magnetic field vector in the egocentric system into the vector in the geographic system.
    Uses Spherical Law of Cosines to find theta_ego and Btheta_geo and Spherical Law of Sines Bphi_geo.
    :param Btheta_ego: The vector in the egocentric coordinate system that needs to be transformed. Because we only care
    about the ground-parallel component of B, this vector has only one nonzero element, which points along the polar
    axis. Therefore, this parameter is a float, not an array-like.
    :param colat_k: The geographic colatitude, in radians, of the magnetometer station or other point of interest at
    which the magnetic field is measured.
    :param lon_k: The geographic longitude, in radians, of the aforementioned point of interest.
    :param colat_l: The geographic colatitude, in radians, of the SECS which generates the magnetic field of interest.
    :param lon_l: The geographic longitude, in radians, of the aforementioned SECS.
    :return: Btheta_geo, Bphi_geo: Respectively the magnitudes of the theta and phi components of the input vector
    Btheta_ego, but viewed in the geographic coordinate system instead of in the egocentric coordinate system.
    """
    theta_ego = np.arccos(np.cos(colat_k) * np.cos(colat_l) + np.sin(colat_k) * np.sin(colat_l) *
                          np.cos(lon_k - lon_l))
    cosC = (np.cos(colat_l) - np.cos(colat_k) * np.cos(theta_ego)) / (np.sin(colat_k) * np.sin(theta_ego))
    sinC = np.sin(colat_l) * np.sin(lon_l - lon_k) / np.sin(theta_ego)

    Btheta_geo = Btheta_ego * cosC
    Bphi_geo = Btheta_ego * sinC

    return Btheta_geo, Bphi_geo


def calculate_T(station_geocolats, station_geolons, sec_geocolats, sec_geolons, r=6378100., R_I=100000.+6378100.):
    """
    Generates the transfer matrix needed to find the SECS scaling factors from magnetometer observations. The format of
    this matrix is as in (Amm & Viljanen 1999) section 4.
    :param station_geocolats: A vector whose elements are the geographic colatitudes (in radians) of each of the
    relevant magnetometer stations. The length of this vector should be equal to the number of stations.
    :param station_geolons: A vector whose elements are the geographic longitudes (in radians) of each of the
    relevant magnetometer stations. The length of this vector should be equal to the number of stations.
    :param sec_geocolats: A vector whose elements are the possible geographic colatitudes (in radians) of the SECS. Note
    that the length of this vector should be equal to the number of ROWS OF SECs, not the number of SECs.
    :param sec_geolons: A vector whose elements are the possible geographic longitudes (in radians) of the SECS. Note
    that the length of this vector should be equal to the number of COLUMNS OF SECs, not the number of SECs.
    :param r: The radius of the Earth's surface in meters.
    :param R_I: The radius, in meters, of the model ionospheric current sheet. Should be roughly 1 R_E + 100 km.
    :return: T, a numpy array of numpy arrays, each of which is a row in the transfer matrix. T has a number of rows
    equal to twice the number of magnetometer stations and a number of columns equal to the total number of SECs.
    """
    T = []
    for colat_k in station_geocolats:  # For each magnetometer station
        lon_k = station_geolons[station_geocolats.tolist().index(colat_k)]  # Obtain its colat and lon
        T_k_theta = []  # Instantiate the two rows of T which correspond to this magnetometer station
        T_k_phi = []
        for colat_l in sec_geocolats:  # Loop through both the SEC colats and the SEC lons, i.e. loop though each SEC
            for lon_l in sec_geolons:
                theta_ego = np.arccos(np.cos(colat_k) * np.cos(colat_l) + np.sin(colat_k) * np.sin(colat_l) *
                                      np.cos(lon_k - lon_l))  # The separation angle between the station and the SEC
                Btheta_ego = B_theta(I_0=1., r=r, theta=theta_ego, R_I=R_I)  # Calculate B vector in egocentric coords
                Btheta_geo, Bphi_geo = ego_to_geo(Btheta_ego, colat_l, lon_l, colat_k, lon_k)  # Transform to geographic
                T_k_theta.append(Btheta_geo)  # theta component of effect of SECS l on station k
                T_k_phi.append(Bphi_geo)  # phi component of effect of SECS l on station k
        T.append(T_k_theta)  # Add the rows for this station to T
        T.append(T_k_phi)  # Comment this line if you only want B_n component, or comment above line if only want B_e
        # Note: This could probably be sped up by instantiating T with np.zeros() and replacing elements individually
    return np.array(T)


def gen_current_timestep(Z, T, epsilon=1e-2):
    """
    This function calculates the scaling factors for all SECs for a single timestep given the magnetic field at all
    magnetometers, expressed in Z vector notation as in (Amm & Viljanen 1999).
    :param Z: The Z vector, a list-like of floats, which is a row of the Z matrix corresponding to a single timestep.
    This should have length equal to twice the number of magnetometer stations, or if you're only using magnetic field
    component, its length should be equal to the number of magnetometer stations.
    :param T: The transfer matrix, an array of lists of floats, as described in calculate_T() .
    :param epsilon: A float which is the cutoff coefficient for singular values in order to condition T.
    :return: An array of floats. This is I, the vector of scaling factors for all systems.
    """
    U, s, V = svd(T, full_matrices=False)
    s_max = np.max(s)
    w = np.zeros((len(s), len(s)))
    for elem in range(len(s)):
        if np.abs(s[elem]) > epsilon * s_max:
            w[elem, elem] = 1./s[elem]  # All singular values lower than the threshold remain zero

    return np.matmul(np.matmul(np.matmul(V.T, w), U.T), Z)


def gen_current_data(mag_data, station_coords_list, secs_coords_list, epsilon=1e-2, disable_tqdm=False):
    """
    This function generates the scaling factors for all SECs given the magnetic field at all magnetometer. It
    does this once for each timestep. You should save the output of this function since it has a decently long runtime
    on any dataset that isn't small. Then, you can simply load those scaling factors from a file and pass them
    one timestep at a time to predict_B_timestep() .
    :param mag_data: A numpy array which is the full dataset of magnetic measurement data. Its 0th axis should be time
    (i.e., each row is a timestep), and its first axis should magnetic observations in different components. Therefore,
    each row should look like the magnetic observation vector Z as described in (Amm & Viljanen 1999).
    :param station_coords_list: A list-like containing two list-likes: The geographic latitudes of all stations in
     degrees, and the geographic longitudes of all stations in degrees, in that order.
    :param secs_coords_list: A list-like containing two list-likes: The geographic latitudes of all SECs in
     degrees, and the geographic longitudes of all SECs in degrees, in that order.
    :param epsilon: A float which is the cutoff coefficient for singular values in order to condition T.
    :param disable_tqdm: A bool representing whether you want a tqdm bar showing this function's progress through the
    set of timestep. It is recommended that this be True unless you calling this function within a loop that either
    is rapidly-iterating or already uses a tqdm bar at a higher level in the loop.
    :return: I_frame, a pandas dataframe whose rows are timesteps and whose columns are the SEC scaling factors.
    """
    station_geolats = station_coords_list[0]
    station_geolons = station_coords_list[1]
    sec_geolats = secs_coords_list[0]
    sec_geolons = secs_coords_list[1]

    station_geocolats = np.pi/2 - np.pi/180*station_geolats
    station_geolons = np.pi/180 * station_geolons
    sec_geocolats = np.pi/2 - np.pi/180*sec_geolats
    sec_geolons = np.pi/180 * sec_geolons

    T = calculate_T(station_geocolats, station_geolons, sec_geocolats, sec_geolons)

    # Initialize list of SECS scaling factors
    I_frame = pd.DataFrame(np.zeros((len(mag_data), len(sec_geolats)*len(sec_geolons))))
    for timestep in tqdm.trange(len(mag_data), desc="Generating SECS scaling factors", disable=disable_tqdm):
        Z = mag_data[timestep]  # Initialize the magnetic observation vector Z as described in (Amm & Viljanen 1999)
        I_frame.iloc[timestep] = gen_current_timestep(Z, T, epsilon)

    return I_frame


def predict_B_timestep(I_vec, B_param, poi_colat, poi_lon, all_sec_colats, all_sec_lons, r=6378100.,
                       R_I=100000.+6378100.):
    """
    This function calculates the magnetic field at a single point of interest given the SEC scaling factors and the
    locations of the SEC poles. This is accomplished by summing the magnetic field contributions of each SEC at the
    point of interest. You may want to consider calling this function both within a loop through many POIs and a loop
    over many timesteps in order to make movies of magnetic field perturbation heatmaps. Just be aware that this can
    become expensive with higher numbers of timesteps and POIs.
    :param I_vec: A vector of floats, which are each the scaling factors for the current systems, in order.
    :param B_param: String, either "dbn_geo" "dbe_geo". This represents whether you want to calculate the geographic
    northward or geographic eastward component, respectively, of the magnetic perturbation. If you want both, call
    this function twice.
    :param poi_colat: The geographic colatitude, in radians, of the point you would like to interpolate at.
    :param poi_lon: The geographic longitude, in radians, of the point you would like to interpolate at.
    :param all_sec_colats: A full list of SEC geographic colatitudes in radians.
    :param all_sec_lons: A full list of SEC geographic longitudes in radians and with the corresponding SECs in th
    same order as in all_sec_colats.
    :param r: The radius of the planet, in meters.
    :param R_I: The distance between the current distribution defined by the SECSs and the center of the planet, in
    meters.
    :return: B_comp, a float representing the north or east component of the interpolated magnetic perturbation at the
    point of interest and at the given time.
    """
    B_comp = 0  # Starts at zero and contributions from each SEC are added to it

    for sec_num in range(len(all_sec_colats)):
        sec_colat, sec_lon = all_sec_colats[sec_num], all_sec_lons[sec_num]
        # theta_ego is the polar angle of this SEC's magnetic field contribution in the egocentric coordinate system
        theta_ego = np.arccos(np.cos(poi_colat) * np.cos(sec_colat) + np.sin(poi_colat) * np.sin(sec_colat) *
                              np.cos(sec_lon - poi_lon))
        # Btheta_ego is the magnitude of the magnetic contribution itself.
        # Its direction is always along the axis defined by the egocentric polar unit vector.
        Btheta_ego = B_theta(I_vec[sec_num], r=r, theta=theta_ego, R_I=R_I)
        # However, Btheta_ego may have some components along the geographic polar and azimuthal directions
        Btheta_geo, Bphi_geo = ego_to_geo(Btheta_ego, sec_colat, sec_lon, poi_colat, poi_lon)

        if B_param == "dbn_geo":
            B_comp -= Btheta_geo
        elif B_param == "dbe_geo":
            B_comp += Bphi_geo
        else:
            raise ValueError("'B_param' is not valid")

    return B_comp
