"""
Configuration file for the beta channel non-divergent barotropic vorticity equation (BVE) simulation.
All the initial conditions of the simulation are defined in this file.
This script has to be placed in the same direcotry as the main program main_BVE_beta.py
"""


import numpy as np


# EARTH PARAMETERS ==========================================================================================
R_earth = 6371 * 10**3                      # Earth radius (in m)
Omega_earth = 2*np.pi / (24*3600)           # Earth rangular velocity (in rad/s)
phi1 = np.deg2rad(0)                        # Lowest latitude value (in rad)
phi2 = np.deg2rad(90)                       # Highest latitude value (in rad)
phi0 = np.deg2rad(45)                       # Mean latitude value (in rad)
g0 = 9.806                                  # Mean gravitational acceleration (in m/s^2)
H = 8.5 * 10**3                             # Atmospheric scale height (in m)


# GRID PARAMETERS ===========================================================================================
Lx = 2*np.pi * R_earth * np.cos(phi0)       # Londitudinal length (in m)
Ly = R_earth * (phi2 - phi1)      			# Latitudinal length (in m)
nx = 256                					# Number of longitudinal grid points
ny = 64                						# Number of latitudinal grid points


# TIME INTEGRATION ==========================================================================================
ti = 0                  					# Initial time (in s)
tf = 10 * 24*3600							# Final time (in s)

# RAW filter
nu = 0.1            						# Damping factor
alpha = 0.5         						# Displacement factor


# OUTPUT ====================================================================================================
output_name = "exp_test"       				# Name of the output folder
save_time = 6 * 3600        				# Time interval between saves (in s)
PLOT = True         						# Switch on/off plotting


# INITIAL CONDITIONS ========================================================================================

# We define the initial streamfunction perturbation

# 1) Pure wave
# def pert0(x,y):
# 	A = 5 * 10**7     				     # Perturbation amplitude (in m^2/s)
# 	k = 2.0*np.pi / Lx				     # Longitudinal wave number (in rad/m)
# 	m = np.pi / Ly	    			     # Latitudinal wave number (in rad/m)

# 	return A * np.sin(k*x) * np.sin(m*y)
	
# 2) Gaussian wave
def pert0(x,y):
	A = 5 * 10**7						# Perturbation amplitude (in m^2/s)
	x0 = Lx / 2.0						# Central longitude (in m)
	y0 = Ly / 2.0						# Central latitude (in m)
	r = 700 * 10**3						# Width radius (in m)

	r2 = (x - x0)**2 + (y - y0)**2      # Perturbation position

	return A * np.exp(-r2 / (r**2))

# 3) Uniform field
# def pert0(x,y):
# 	A = 5 * 10**7						 # Background streamfunction field (in m^2/s)

# 	return A * np.ones_like(x)

# 4) Meridional wave
# def pert0(x,y):
#     A = 200 * 10**8					 # Maximum streamfunction value (in m^2/s)
#     L = 500 * 10**3		       		 # Perturbation width radius (in m)
#     m = np.pi / Ly					 # Latitudinal wave number (in rad/m)
#     xm = Lx / 2.0						 # Central longitude (in m)

#     return A / (1 + ((x - xm) / L)**2) * np.sin(m*y)


# We also define the background zonal velocity flow

# 1) Constant background velocity
def zonal(x,y):
	U0 = 0								# Background zonal velocity (in m/s)

	return U0 * np.ones_like(x)

# 2) Jet-like background velocity
# def zonal(x,y):
# 	U0 = 30								 # Background zonal velocity (in m/s)
# 	y0 = Ly / 2.5						 # Jet latitude (in m)
# 	r = 2000 * 10**3					 # Jet width (in m)

# 	y2 = (y - y0)**2					 # Position factor
	
# 	return U0 * np.exp(-y2/r**2)


# TOPOGRAPHY ================================================================================================

USE_TOPOGRAPHY = False						# Switch on/off topography

# 1) Gaussian mountain
def topography(x,y):
		h0 = 6000							# Maximum height (in m)
		x0 = Lx / 2.0						# Central longitude
		y0 = Ly / 2.0						# Central latitude
		r = 500 * 10**3						# Width radius (in m)

		r2 = (x - x0)**2 + (y - y0)**2		# Mountain position

		return h0 * np.exp(- r2 / (2*r**2))