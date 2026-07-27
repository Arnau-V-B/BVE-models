"""
Configuration file for the global non-divergent barotropic vorticity equation (BVE) simulation.
All the initial conditions of the simulation are defined in this file.
This script has to be placed in the same direcotry as 'main.py'
"""


import numpy as np


# EARTH PARAMETERS ==========================================================================================
R = 6371 * 10**3                      				# Earth radius (in m)
Omega = 2.0 * np.pi / (24*3600)       				# Earth rangular velocity (in rad/s)
g0 = 9.806                            				# Mean gravitational acceleration (in m/s^2)


# GRID PARAMETERS ===========================================================================================
nlat = 129                                          # Number of latitude grid points (odd)
nlon = 256                                          # Number of longitude grid points (2*(nlat-1))
lat = np.linspace(90, -90, nlat)                    # Vector of latitudes
lon = np.linspace(0, 360, nlon, endpoint=False)     # Vector of longitudes
lons, lats = np.meshgrid(lon, lat)                  # lon x lat grid


# TIME INTEGRATION ==========================================================================================
ti = 0                  							# Initial time (in s)
tf = 10 * 24*3600									# Final time (in s)
dt = 1800                							# Time step (in s)
tau_d = 3 * 3600        							# Hyperdiffusion time scale (in s)

# RAW filter
nu = 0.1           									# Damping factor   
alpha = 0.5        									# Displacement factor


# SPECTRAL SCHEME ===========================================================================================
lmax = 85                             # Spectral triangular truncation (maximum wavenumber represented)
gridtype = 'GLQ'                      # Spectral transform algorithm:
                                      #     Gauss-Legendre (GLQ) --> Gaussian latitudes grid (N+1)x(2N+1)
                                      #                              (slower but exact, RECOMMENDED)
                                      #     Discroll-Healy (DH) --> Equally sampled grid Nx2N
                                      #                             (faster but needs more gridpoints for a given truncation)


# OUTPUT ====================================================================================================
output_name = "exp_test"       						# Name of the output folder
save_time = 6 * 3600         						# Time interval between saves (in s)
PLOT = True         								# Switch on/off plotting
MODE = 'linear'        								# Equation to solve: 'linear' or 'nonlinear' 


# INITIAL CONDITIONS ========================================================================================

# We define the initial streamfunction perturbation
def pert0(lon, lat):
	A = 5 * 10**7		                                # Perturbation amplitude (in m^2/s)
	lon0 = 180		                                    # Central longitude (in deg)
	lat0 = 45		                                    # Central latitude (in deg)
	r = 7 	                                    		# Width radius (in deg)

	dlon = np.deg2rad(lon - lon0)   					# Longitudinal section
	dlat = np.deg2rad(lat - lat0)                       # Latitudinal section

	r2 = dlon**2 + dlat**2     							# Perturbation position

	return A * np.exp(-r2 / (np.deg2rad(r)**2))


psi0 = pert0(lons, lats)


# We also define the background zonal velocity flow

# Uniform flow
# def zonal(lon, lat):
#       return 20 * np.ones_like(lon)

# Jet-like flow
def zonal(lon, lat):
	U0 = 30				                                # Background zonal velocity (in m/s)
	lat0 = 30			                                # Jet latitude (in deg)
	r = 15                                 				# Jet width (in deg)

	lat2 = (np.deg2rad(lat - lat0))**2	            	# Position factor
	
	return U0 * np.exp(-lat2/np.deg2rad(r)**2)


U = zonal(lons, lats)