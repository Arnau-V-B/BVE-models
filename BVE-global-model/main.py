"""
Main program that runs the non-divergent barotropic vorticity equation (BVE) simulation.
The initial conditions can be set in the config.py file located in the same directory.
This version of the program computes the non-linear non-divergent BVE globally in spherical coordinates
using spherical harmonics transforms.
At the end, an output folder is generated to save the results in netCDF format and simple figures with
GIFs to keep track of the conservation properties of the model and visualize the evolution of the
streamfunction and vorticity fields.

@Author: Arnau Vicente Bou
"""


# Python libraries to import
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pyshtools as pysh
import time
import sys
import os


# First, we import all the functions and routines for the simulation
from core import *

# We set a time counter to keep track of the total execution time of the code through the terminal
start_time = time.time()

# We generate an output folder to save the results of the simulation
output_dir = "output/"
os.makedirs(output_dir, exist_ok=True)


# PARAMETER DEFINITION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

print("Obtaining model parameters ...\n")

# We start importing all the initial parameters defined in the config.py file
from config import *

# We set the grid coordinates (different for GLQ and DH algorithms)
if gridtype == 'GLQ':
	# GLQ scheme needs a quadratic grid: 2*nlat = 3*(lmax + 1) ; though a linear grid would be sufficient
	nlat_grid = int(3/2 * (lmax + 1))
	glq_nodes, glq_weights = pysh.expand.SHGLQ(nlat_grid)	# Hermite polynomials zeros
	lat_grid = np.degrees(np.arcsin(glq_nodes))				# Gaussian latitudes
	lon_grid = np.linspace(0, 360, 2*nlat_grid+1)			# 0º to 360º (included)
	lons_grid, lats_grid = np.meshgrid(lon_grid, lat_grid)
elif gridtype == 'DH':
	# DH scheme needs a cubic grid: 2*nlat = 4*(lmax + 1)
	nlat_grid = int(2 * lmax + 2)
	glq_nodes, glq_weights = None, None
	lat_grid = np.linspace(90, -90, nlat_grid)					# 90º to -90º
	lon_grid = np.linspace(0, 360, 2*nlat_grid, endpoint=False) # 0º to 360º (not included)
	lons_grid, lats_grid = np.meshgrid(lon_grid, lat_grid)

# We precompute some useful parameters
f = 2 * Omega * np.sin(np.pi * lats_grid / 180)		# Coriolis parameter
derfact = 1 / np.cos(np.pi * lats_grid / 180)		# Spherical scale factor
tan_lat = np.tan(np.pi * lats_grid / 180)			# Tangent of latitude
if gridtype == 'DH':				# Avoid division by zero
	derfact[0] = derfact[1]
	derfact[-1] = derfact[-2]
	tan_lat[0] = tan_lat[1]
	tan_lat[-1] = tan_lat[-2]

# We build the laplacian operators in the spectral space (with desired truncation lmax)
l = np.arange(lmax + 1).reshape(1, -1, 1)		# Polynomial degrees (shape = [1, lmax+1, 1])
lap = - l * (l + 1) / R**2						# Normal laplacian
lap2 = lap**2									# Squared laplacian (for hyperdiffiusion n=2)
inv_lap = np.zeros_like(lap)					# Inverse laplacian
inv_lap[l>0] = 1 / lap[l>0]			# Avoid division by zero

# We compute the hyperdiffusion coefficient (scaled accordingly)
kmax = (lmax * (lmax + 1)) / R**2				# Maximum wave mode
eta = 1 / (tau_d * kmax**2)						# Hyperdiffusion coefficient
hyp_denom1 = 1 / (1 + dt * eta * lap2)			# Implicit operator for Euler scheme
hyp_denom2 = 1 / (1 + 2 * dt * eta * lap2)		# Implicit operator for Leapfrog scheme

# We generate empty lists to keep track of the conserved quantities:
energies = []					# kinetic energy
enstrophies = []				# enstrophy
mean_vorticities = []			# mean relative vorticity
# And also to save the evolution fields
times = []						# time
streamfunctions = []			# streamfunction
vorticities = []				# relative vorticity

# We create a folder in 'output' to save the results of the specific experiment
os.makedirs(output_dir + output_name, exist_ok=True)

# Once all the parameters are defined, we create a dictionary to store the most frequently used
params = {
	'radius': R,
	'gridtype': gridtype,
	'lmax': lmax,
	'sampl': nlat_grid,
	'glq_nodes': glq_nodes,
	'glq_weights': glq_weights,
	'f': f,
	'derfact': derfact,
	'tan_lat': tan_lat,
	'start': start_date
}


# INITIAL FIELDS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

print("Generating initial fields ...\n")

# We configure the interpolator from the original grid to the custom grid
interp_z = RegularGridInterpolator((lat, lon), zeta0, bounds_error=False, fill_value=None)

# We interpolate the original vorticity field to the custom grid
zeta0_grid = interp_z((lats_grid, lons_grid))

# We convert it to spectral space
zeta0_spec = grid2spec(zeta0_grid, params)

# We obtain the streamfunction field from the relative vorticity
psi0_spec = inv_lap * zeta0_spec
psi0 = spec2grid(psi0_spec, params)

# And we extract the horizontal velocity fields from the streamfunction
u0_spec, v0_spec = compute_vel(psi0_spec, params)
u0 = spec2grid(u0_spec, params) * derfact
v0 = spec2grid(v0_spec, params) * derfact

# We compute all the conserved values
energy, enstrophy, zetamean = compute_conserved_values(u0, v0, zeta0_grid, params)

# And save them in the lists
energies.append(energy)
enstrophies.append(enstrophy)
mean_vorticities.append(zetamean)

# We also save the initial fields
times.append(ti)
if gridtype == 'GLQ':		# Remove the extra 360º longitude band
	streamfunctions.append(psi0[:,:-1].copy())
	vorticities.append(zeta0_grid[:,:-1].copy())
else: 
	streamfunctions.append(psi0.copy())
	vorticities.append(zeta0_grid.copy())

# Finally we set the next save time
next_save_time = save_time


# TIME INTEGRATION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

print("Starting time integration ...")

# We compute the initial advection term
adv0_spec = compute_adv(u0, v0, zeta0_grid, params)

# And perform a forward Euler step in time for the first integration
zeta_spec = zeta0_spec
zetaold_spec = zeta0_spec
# We apply the hyperdiffusion implicitly (i.e. (1 + dt*hyp)zeta_i+1 = rhs_i))
zetanew_spec = (zeta_spec + dt * adv0_spec) * hyp_denom1
zetanew = spec2grid(zetanew_spec, params)

# We can also extract the new streamfunction field
psi_spec = inv_lap * zetanew_spec

# And get the new velocity fields
u_spec, v_spec = compute_vel(psi_spec, params)
u = spec2grid(u_spec, params) * derfact
v = spec2grid(v_spec, params) * derfact

# Again, we comptute the conserved values
energy, enstrophy, zetamean = compute_conserved_values(u, v, zetanew, params)

# And save them in the lists
energies.append(energy)
enstrophies.append(enstrophy)
mean_vorticities.append(zetamean)


# Now, we can start the main integration loop
for t in range(ti+2*dt, tf+dt, dt):

	# We display in the terminal an updating counter showing the elapsed time of computation and 
	# the simulation time to visualize the program progress
	elapsed = time.time() - start_time
	sys.stdout.write(f"\rElapsed time: {elapsed:.2f}s | Simulation time: {t/3600:.2f}h")
	sys.stdout.flush()

	# We update the vorticity fields from last iteration 
	zetaold_spec = zeta_spec
	zeta_spec = zetanew_spec
	zeta = zetanew
	
	# We compute the advection term
	adv_spec = compute_adv(u, v, zeta, params)

		# Now, a leapfrog step with a RAW filter is performed to integrate in time
	zeta_spec, zetanew_spec = leapfrog_raw_step(zeta_spec, zetaold_spec, adv_spec, hyp_denom2, dt, nu, alpha)
	zetanew = spec2grid(zetanew_spec, params)

	# Now we can extract the new streamfunction field
	psi_spec = inv_lap * zetanew_spec

	# And compute the new velocity fields
	u_spec, v_spec = compute_vel(psi_spec, params)
	u = spec2grid(u_spec, params) * derfact
	v = spec2grid(v_spec, params) * derfact

	# Finally, we comptute the conserved values
	energy, enstrophy, zetamean = compute_conserved_values(u, v, zetanew, params)
	
	# And save them in the lists
	energies.append(energy)
	enstrophies.append(enstrophy)
	mean_vorticities.append(zetamean)


	# After each iteration we check if the simulation is diverging
	if np.isinf(zetanew).any():
		raise ValueError(f"Infinity detected in vorticity field at t = {t/3600:.2f}h")
	elif np.isnan(zetanew).any():
		raise ValueError(f"NaN detected in vorticity field at t = {t/3600:.2f}h")

	# Every time we get to the save interval, we save a snapshot of the psi and zeta fields
	if t >= next_save_time:
		times.append(t)
		psi = spec2grid(psi_spec, params)
		if gridtype == 'GLQ':	# We remove the extra 360º longitude band
			streamfunctions.append(psi[:,:-1].copy())
			vorticities.append(zetanew[:,:-1].copy())
		else:
			streamfunctions.append(psi.copy())
			vorticities.append(zetanew.copy())
		next_save_time += save_time


# SAVING RESULTS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

print('')
print("\nSaving simulation results...")

from saving import *

# We first copy the config.py file used in a '.txt' file
with (open("config.py", 'r') as file, 
	open(output_dir + output_name + f"/params_{output_name}.txt", 'w') as file_copy):
	file_copy.write(file.read())

# Then, we save the simulation data into netCDF Datasets
# We start by creating the directory where the data will be saved
data_dir = output_dir + output_name + "/data/"
os.makedirs(data_dir, exist_ok=True)

# First we save the conserved values
save_conserved_values(data_dir, energies, enstrophies, mean_vorticities, output_name)
# And then the relative vorticity and streamfunction snapshots
save_fields_evolution(data_dir, streamfunctions, vorticities, lon_grid, lat_grid, times, output_name, params)


# PLOTTING RESULTS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

if PLOT:
	print("\nGenerating figures...")

	from plotting import *

	# We create the directory where the figures will be saved
	im_dir = output_dir + output_name + "/figures/"
	os.makedirs(im_dir, exist_ok=True)
	os.makedirs(im_dir + "temp_frames/", exist_ok=True)

	# First we plot a graph of the behaviour of the conserved values
	plot_conserved_values(data_dir, output_name, im_dir)
	# Then we plot the evolution of the vorticity and streamfunction fields
	plot_fields_evolution(data_dir, output_name, im_dir)