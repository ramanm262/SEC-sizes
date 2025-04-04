# SEC-sizes [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14758094.svg)](https://doi.org/10.5281/zenodo.14758094)

Code repository for the paper "Localized Geomagnetic Disturbances: A Statistical Analysis 
of Spatial Scale" by R. Mukundan, A. Keesee, J. Marchezi, V. Pinto, M. Coughlan, and D. 
Hampton (2025). For any questions regarding this code, please contact
[raman.mukundan@unh.edu](mailto:raman.mukundan@unh.edu) .

## How to use

You  can run this code for replication or for your own purposes. There are several .py 
scripts that should be run to replicate the results from the paper. These scripts can
be run in your favorite IDE or from the command line, however, you may want to edit
the files themselves in order to change their configuration variables, such as the list
of magnetometers to use or the start and end year of the dataset. The scripts are
listed and described here in the order in which they should be run:

### 1. [`prepare_supermag_and_omni.py`](prepare_supermag_and_omni.py)
This script  does a little data cleaning. This does not download the required
data! you must do that yourself. Baseline-subtracted SuperMAG data are available 
year-by-year at [https://supermag.jhuapl.edu/mag/](https://supermag.jhuapl.edu/mag/)
and OMNI data are available at
[https://omniweb.gsfc.nasa.gov/ow_min.html](https://omniweb.gsfc.nasa.gov/ow_min.html).
### 2. [`preprocessing.py`](preprocessing.py)
This preprocesses the clean data, including selecting parameters and paring the data
range down to just storm time.  
### 3. [`optimize_sec.py`](optimize_sec.py)
This script determines the optimal SECS hyperparameters: number of rows in the Current
Systems grid, number of columns in the grid, and the cutoff coefficient. The output
from all the Optuna trials are saved to a file.

It is optional to run this script. If you trust that default hyperparameters are the
best ones, then you can skip this script and go straight to running script #4.
### 4. [`calculate_interps.py`](calculate_interps.py)
This script uses Spherical Elementary Current Systems (SECS) to create heatmaps,
whose data are saved to a file. This file can fairly large (~10 GB depending on 
configuration variables), and running this script can take quite a long time, so 
be sure you have enough disk space before you start.
### 5. [`stats_analysis.py`](stats_analysis.py)
This script analyzes the file produced in the last step. It locates localized
geomagnetic disturbance (LGMDs) and calculates their perimeter and aspect ratio.
From these, it produces the main statistics that are shown in the paper.
### 6. [`metrology_analysis.py`](metrology_analysis.py)
This script generates plots about the surface metrology of the heatmaps, many of which
are shown in the paper's Supporting Information.

## Other Files

### [`preprocessing_fns.py`](preprocessing_fns.py)
Contains some functions used by `preprocessing.py` to clean and prepare OMNI data.
This is not a standalone script and should not be run by itself.

### [`sec.py`](sec.py) (See also [this gist](https://gist.github.com/ramanm262/4ccd662721ae59b62378b1b728d09979))
A set of utility functions for carrying current shell reconstruction and magnetic
field interpolation with the Spherical Elementary Current Systems method. This includes
Current System scaling factor calculation, magnetic field computation from those 
scaling factors, and coordinate system transformations for vectors.

### [`station_distances.py`](station_distances.py)
A small, independent routine that calculates the distances between all pairs of
magnetometer stations in the dataset. The output of this file is not used by any
other script; it is just for ensuring that the inter-station separations are
sufficiently small compared to the typical LGMD size. Run this script separately
after running `stats_analysis.py` to make sure that this is the case.

### [`stormList.csv`](stormList.csv)
A list of all geomagnetic storms from 1995 through 2019. The times contained within are
the SYM-H minima of those storms. This list is the same as the one used in
[Pinto et al. (2022)](https://doi.org/10.3389/fspas.2022.8697402). This list is used by
`storm_extract()` in `preprocessing.py` to trim the dataset in this study.
## Requirements

Required Python packages can be found in `requirements.txt` and in the list below. The code 
may work with later versions of these packages, but it is not guaranteed.

    cartopy==0.21.1
    cdflib==1.2.4
    joblib==1.3.2
    matplotlib==3.7.5
    numpy==1.24.3
    optuna==3.4.0
    pandas==2.0.3
    pyarrow==15.0.0
    scikit-image==0.21.0
    scikit-learn==1.3.2
    scipy==1.10.1
    tables==3.8.0
    threadpoolctl==3.2.0
    tqdm==4.63.1
