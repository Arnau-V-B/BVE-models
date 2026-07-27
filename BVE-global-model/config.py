"""
Configuration file for the global nonlinear non-divergent barotropic vorticity equation (BVE) simulation.
All the initial conditions of the simulation are defined in this file. The code is prepared to read the
netCDF file of an ERA5 reanalysis relative vorticity field, but it can easily be adapted for any other
dataset options.
This script has to be placed in the same direcotry as 'main.py'
"""


import xarray as xr
import numpy as np


# EARTH PARAMETERS ==========================================================================================
R = 6371 * 10**3                      # Earth radius (in m)
Omega = 2.0 * np.pi / (24*3600)       # Earth rangular velocity (in rad/s)
g0 = 9.806                            # Mean gravitational acceleration (in m/s^2)


# TIME INTEGRATION ==========================================================================================
ti = 0                                # Initial time (in s)
tf = 48 * 3600		                  # Final time (in s)
dt = 1800                             # Time step (in s)
tau_d = 3 * 3600                      # Hyperdiffusion time scale (in s)

# RAW filter
nu = 0.1                              # Damping factor   
alpha = 0.5                           # Displacement factor


# SPECTRAL SCHEME ===========================================================================================
lmax = 85                             # Spectral triangular truncation (maximum wavenumber represented)
gridtype = 'GLQ'                      # Spectral transform algorithm:
                                      #     Gauss-Legendre (GLQ) --> Gaussian latitudes grid (N+1)x(2N+1)
                                      #                              (slower but exact, RECOMMENDED)
                                      #     Discroll-Healy (DH) --> Equally sampled grid Nx2N
                                      #                             (faster but needs more gridpoints for a given truncation)


# OUTPUT ====================================================================================================
output_name = "exp_test"              # Name of the output folder
save_time = 3 * 3600                  # Time interval between saves (in s)
PLOT = True                           # Switch on/off plotting 


# INITIAL CONDITIONS ========================================================================================

# Initial relative vorticity field
dataset_name = "reanalysis/forecast_15-04-2026_glob.nc"     # Name of the dataset file

# We open and read the relative vorticity dataset
ds = xr.open_dataset(dataset_name, engine='netcdf4')
start_date = ds['valid_time'].values[0]
press_lvl = ds['pressure_level'].values[0]

# We get the initial relative vortiticy field
zeta0 = ds['vo'].sel(valid_time=start_date, pressure_level=press_lvl).values

# We get the latitude and longitude coordinates (in ERA5: nlat = N+1, nlon = 2N)
lat = ds['latitude'].values
lon = ds['longitude'].values

ds.close()