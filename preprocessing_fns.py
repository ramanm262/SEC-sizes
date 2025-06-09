import cdflib
import pandas as pd
import numpy as np

# Used to suppress annoying commandline complaints about repeatedly inserting columns into a DataFrame
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def omnicdf2dataframe(file):
    """
    Function by Victor Pinto
    Load a CDF File and convert it in a Pandas DataFrame. Used to extract geomagnetic indices.

    WARNING: This will not return the CDF Attributes, just the variables.
    WARNING: Only works for CDFs of the same array length (OMNI)
    """
    cdf = cdflib.CDF(file)
    cdfdict = {}

    for key in cdf.cdf_info().zVariables:
        cdfdict[key] = cdf[key]

    cdfdf = pd.DataFrame(cdfdict)

    if 'Epoch' in cdf.cdf_info().zVariables:
        cdfdf['Epoch'] = pd.to_datetime(cdflib.cdfepoch.encode(cdfdf['Epoch'].values))

    return cdfdf


def bad_omni_to_nan(df):
    """
    Function by Victor Pinto
    Remove filling numbers for missing data in OMNI data (1 min) and replace
    them with np.nan values
    """
    # IMF
    df.loc[df['B_Total'] >= 9999, 'B_Total'] = np.nan
    df.loc[df['BX_GSE'] >= 9999, 'BX_GSE'] = np.nan
    df.loc[df['BY_GSE'] >= 9999, 'BY_GSE'] = np.nan
    df.loc[df['BZ_GSE'] >= 9999, 'BZ_GSE'] = np.nan
    df.loc[df['BY_GSM'] >= 9999, 'BY_GSM'] = np.nan
    df.loc[df['BZ_GSM'] >= 9999, 'BZ_GSM'] = np.nan

    # Speed
    df.loc[df['flow_speed'] >= 99999, 'flow_speed'] = np.nan
    df.loc[df['Vx'] >= 99999, 'Vx'] = np.nan
    df.loc[df['Vy'] >= 99999, 'Vy'] = np.nan
    df.loc[df['Vz'] >= 99999, 'Vz'] = np.nan

    # Particles
    df.loc[df['proton_density'] >= 999, 'proton_density'] = np.nan
    df.loc[df['T'] >= 9999999.0, 'T'] = np.nan
    df.loc[df['Pressure'] >= 99, 'Pressure'] = np.nan

    # Other
    df.loc[df['E_Field'] >= 999, 'E_Field'] = np.nan
    df.loc[df['Beta'] >= 999, 'Beta'] = np.nan
    df.loc[df['Mach_num'] >= 999, 'Mach_num'] = np.nan
    df.loc[df['Mgs_mach_num'] >= 99, 'Mgs_mach_num'] = np.nan

    # Indices
    df.loc[df['AE_INDEX'] >= 99999, 'AE_INDEX'] = np.nan
    df.loc[df['AL_INDEX'] >= 99999, 'AL_INDEX'] = np.nan
    df.loc[df['AU_INDEX'] >= 99999, 'AU_INDEX'] = np.nan
    df.loc[df['SYM_D'] >= 99999, 'SYM_D'] = np.nan
    df.loc[df['SYM_H'] >= 99999, 'ASY_D'] = np.nan
    df.loc[df['ASY_H'] >= 99999, 'ASY_H'] = np.nan
    df.loc[df['PC_N_INDEX'] >= 999, 'PC_N_INDEX'] = np.nan

    return df
