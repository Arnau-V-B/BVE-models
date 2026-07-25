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


# FUNCTIONS DEFINED FOR THE SIMULATION ================================================================================

def grid2spec(field, grid):
	"""
	This function takes a 2D gridded field and transforms it into a set of real spherical harmonics
	coefficients up to a maximum order of triangular truncation lmax.

	It accepts both Discroll-Healy (DH) (equally sampled) and Gauss-Legendre quadrature (GLQ)
	(gaussian latitudes) grids.

	PARAMETERS:
	(input) -->
		field       : 2D array (nlat, nlon) of grid field
		grid        : type of grid quadrature: DH or GLQ
	(internal) -->
		lmax        : maximum wavenumber for spectral triangular truncation
		glq_weights : weights for each of the roots of Legendre polynomial of roder lmax
		glq_nodes   : roots of Legendre polynomial of order lmax
	(output) -->
		field_spec  : 3D array (2, lmax+1, lmax+1) of real spherical harmonics coefficients,
					  [0,:,:] for cosine coefficients and [1,:,:] for sine coefficients
	"""

	if grid == 'DH':
		field_spec = pysh.expand.SHExpandDH(field, sampling=2, lmax_calc=lmax)

	elif grid == 'GLQ':
		field_spec = pysh.expand.SHExpandGLQ(field, w=glq_weights, zero=glq_nodes, lmax_calc=lmax)

	return field_spec


def spec2grid(field_spec, grid):
	"""
	This function takes a 3D array of real spherical harmonics with triangular truncation lmax
	and shape (2,lmax+1,lmax+1) and transforms it into a 2D gridded field with sampling sampl.

	It accepts both Discroll-Healy (DH) (equally sampled) and Gauss-Legendre quadrature (GLQ)
	(gaussian latitudes) grids.

	PARAMETERS:
	(input) -->
		field_spec : 3D array (2, lmax+1, lmax+1) of real spherical harmonics coefficients,
					 [0,:,:] for cosine coefficients and [1,:,:] for sine coefficients
		grid       : type of grid quadrature: DH or GLQ
	(internal) -->
		sampl      : sampling of the resulting gridded field
		glq_nodes  : roots of Legendre polynomial of order lmax
	(output) -->
		field      : 2D array (nlat, nlon) of the gridded field
	"""

	if grid == 'DH':
		field = pysh.expand.MakeGridDH(field_spec, sampling=2, lmax=sampl)

	elif grid == 'GLQ':
		field = pysh.expand.MakeGridGLQ(field_spec, zero=glq_nodes, lmax=sampl)

	return field


def lambda_derivative(field_spec):
	"""
	This function computes the derivative in longitude for a given field in the real spherical
	harmonics spectral space. The spectral field coefficients of triangular truncation lmax need to be
	shaped as (2,lmax+1,lmax+1).

	The following recursion relation between spectral coefficients is used to compute the derivative:
	-->	d/dlambda f_lm = i * m * f_lm
		Cosine coeffs (C): d/dlambda C_lm = m * S_lm
		Sine coeffs (S): d/dlambda S_lm = - m * C_lm

	PARAMETERS:
	(input) -->
		field_spec : 3D array (2, lmax+1, lmax+1) of real spherical harmonics coefficients,
					 [0,:,:] for cosine coefficients and [1,:,:] for sine coefficients
	(internal) -->
		R          : Earth radius (for correct scaling)
	(output) -->
		dlambda    : 3D array (2, lmax+1, lmax+1) with the coefficients of the derivative in longitude
	"""

	# We obtain the spectral truncation of the coefficients
	lmax = field_spec.shape[1] - 1

	# We generate a vector of m's from 0 to lmax
	m = np.arange(lmax + 1)
	
	# We compute the derivative with the recursion relation (applied to all orders l and m)
	dlambda_C = m[None, :] * field_spec[1]
	dlambda_S = - m[None, :] * field_spec[0]

	# Finally we combine both cosine and sine coefficients into a (2,lmax+1,lmax+1) array
	# And divide by the Earth radius to get the correct units
	return np.stack((dlambda_C, dlambda_S), axis=0) / R


def theta_derivative(field_spec):
	"""
	This function computes the derivative in latitude for a given field in the real spherical
	harmonics spectral space multiplied by the cosine of latitude. The spectral field coefficients of
	triangular truncation lmax need to be shaped as (2,lmax+1,lmax+1).

	The following recursion relation between spectral coefficients is used to compute the derivative:
	-->	cos(theta) * d/dtheta f_lm = (l+2) * eps_l+1,m * f_l+1,m - (l-1) * eps_lm * f_l-1,m
		eps_lm = sqrt((l^2 - m^2) / (4*l^2 - 1))

	PARAMETERS:
	(input) -->
		field_spec : 3D array (2, lmax+1, lmax+1) of real spherical harmonics coefficients
					 [0,:,:] for cosine coefficients and [1,:,:] for sine coefficients
	(internal) -->
		R          : Earth radius (for correct scaling)
	(output) -->
		dtheta     : 3D array (2, lmax+1, lmax+1) with the coefficients of the derivative in lat * cos(lat)
	"""

	# We obtain the spectral truncation of the coefficients
	lmax = field_spec.shape[1] - 1

	# We generate an empty array to store the derivative coefficients
	dtheta = np.zeros_like(field_spec)

	# We generate a vector of m's from 0 to lmax
	m = np.arange(lmax + 1)

	# We iterate over all wavenumbers l from 0 to lmax
	for l in range(lmax + 1):
		# We get the m orders from 0 to l+1
		m_valid = m[:l+1]

		# We precompute the epsilon values
		eps_l = np.sqrt((l**2 - m_valid**2) / (4*l**2 - 1))
		eps_lp1 = np.sqrt(((l+1)**2 - m_valid**2) / (4*(l+1)**2 - 1))

		# We apply the recursion relation
		if l == 0:
			dtheta[:, l, :l+1] = (l+2) * eps_lp1 * field_spec[:, l+1, :l+1]
		elif l == lmax:
			dtheta[:, l, :l+1] = - (l-1) * eps_l * field_spec[:, l-1, :l+1]
		else:
			dtheta[:, l, :l+1] = ((l+2) * eps_lp1 * field_spec[:, l+1, :l+1]
								- (l-1) * eps_l * field_spec[:, l-1, :l+1])

	# Finally we divide by the Earth radius to get the correct units
	return dtheta / R


def compute_adv(u, v, vort):
	"""
	This function computes the advection term of the BVE in the spectral space given the horizontal
	velocity fields and the vorticity field in the grid space.
	Assuming a non-divergent flow:
	---> adv = - V * Nabla(zeta+f) = - Nabla * (V * (zeta+f)) = - div(V * (zeta+f))
	In spherical coordinates the horizontal divergence of a field A is given by:
	---> div(A) = 1/(R*cos(theta)) * d/dlambda (A) + 1/R * d/dtheta (A) - 1/R * tan(theta) * A

	To avoid convolutions, products are computed in the grid and then transformed to spectral
	space to perform the horizontal divergence of the product.
	
	PARAMETERS:
	(input) -->
		u        : 2D array (nlat, nlon) of grid zonal velocity field
		v        : 2D array (nlat, nlon) of grid meridional velocity field
		vort     : 2D array (nlat, nlon) of grid vorticity field
	(internal) -->
		f        : 2D array (nlat, nlon) of grid Coriolis parameter (f=2*Omega*sin(latitude))
		derfact  : 2D array (nlat, nlon) of grid inverse cosine of latitude (derfact=1/cos(latitude))
		tan_lat  : 2D array (nlat, nlon) of grid tangent of latitude (tan_lat=tan(latitude))
	(output) -->
		adv_spec : 3D array (2, lmax+1, lmax+1) with the spectral coefficients of the advection term
	"""

	# We first compute the product V·(zeta+f) in the grid space
	u_zeta = u * (vort + f)
	v_zeta = v * (vort + f)

	# Then we transform the product into the spectral space
	u_zeta_spec = grid2spec(u_zeta, gridtype)
	v_zeta_spec = grid2spec(v_zeta, gridtype)

	# We compute the spectral derivatives
	u_zeta_lambda_spec = lambda_derivative(u_zeta_spec)
	v_zeta_theta_spec = theta_derivative(v_zeta_spec)

	# We obtain the flat part of the horizontal divergence
	div_spec = u_zeta_lambda_spec + v_zeta_theta_spec
	# And transform it to grid space dividing by cos(theta)
	div = spec2grid(div_spec, gridtype) * derfact

	# We apply the curvature term and change sign to obtain the advection
	adv = - (div - v_zeta * tan_lat / R)
	# And we transform the advection to the spectral space
	adv_spec = grid2spec(adv, gridtype)

	return adv_spec


def compute_vel(stream_spec):
	"""
	This function computes the horizontal velocity fields in the spectral space multiplied by cos(lat)
	given the spectral coefficients of the stream function. The spectral field coefficients of triangular
	truncation lmax need to be shaped as (2,lmax+1,lmax+1).

	To do so, it computes the curl of the stream function:
	--> V = Nabla x psi
		cos(theta) * u = - cos(theta)/R * d/dtheta psi
		cos(theta) * v = 1/R * d/dlambda psi

	PARAMETERS:
	(input) -->
		stream_spec : 3D array (2, lmax+1, lmax+1) of real spherical harmonics coefficients,
					  [0,:,:] for cosine coefficients and [1,:,:] for sine coefficients
	(output) -->
		u_spec      : 3D array (2, lmax+1, lmax+1) with the spectral coefficients of the zonal velocity field
					  multiplied by cos(latitude)
		v_spec      : 3D array (2, lmax+1, lmax+1) with the spectral coefficients of the meridional velocity field
					  multiplied by cos(latitude)
	"""

	# We compute the spectral curl of the stream function
	u_spec = - theta_derivative(stream_spec)
	v_spec = lambda_derivative(stream_spec)

	return u_spec, v_spec


# MAIN PROGRAM PIPELINE ===============================================================================================

if __name__ == '__main__':
	# We set a time counter to keep track of the total execution time of the code through the terminal
	start_time = time.time()

	# We generate an output folder to save the results of the simulation
	output_dir = "output/"
	os.makedirs(output_dir, exist_ok=True)

	# We define all the main parameters that will be used in the simulation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	print("Obtaining model parameters ...\n")

	# We first import all the initial parameters defined in the config.py file
	from config import *

	# We begin verifying that truncation is lower than the grid sampling
	if lmax > sampl:
		raise ValueError("lmax must be smaller or equal to sampl")
	
	# We set the grid coordinates (different for DH and GLQ)
	if gridtype == 'DH':
		# DH grid scheme has less spectral resolution: nlat_grid ~ 2*sampling
		nlat_grid = 2 * sampl + 2
		lat_grid = np.linspace(90, -90, nlat_grid)					# 90º to -90º
		lon_grid = np.linspace(0, 360, 2*nlat_grid, endpoint=False) # 0º to 360º (not included)
		lons_grid, lats_grid = np.meshgrid(lon_grid, lat_grid)
	elif gridtype == 'GLQ':
		# GLQ grid scheme has highest spectral resolution: nlat_grid ~ sampling
		nlat_grid = sampl
		glq_nodes, glq_weights = pysh.expand.SHGLQ(nlat_grid)	# Hermite polynomials zeros
		lat_grid = np.degrees(np.arcsin(glq_nodes))				# Gaussian latitudes
		lon_grid = np.linspace(0, 360, 2*nlat_grid+1)			# 0º to 360º (included)
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
		'gridtype': gridtype,
		'lmax': lmax,
		'start': start_date
	}


	# Now, we generate the initial fields >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	print("Generating initial fields ...\n")

	# We configure the interpolator from the original grid to the custom grid
	interp_z = RegularGridInterpolator((lat, lon), zeta0, bounds_error=False, fill_value=None)

	# We interpolate the original vorticity field to the custom grid
	zeta0_grid = interp_z((lats_grid, lons_grid))

	# We convert it to spectral space
	zeta0_spec = grid2spec(zeta0_grid, gridtype)

	# We obtain the streamfunction field from the relative vorticity
	psi0_spec = inv_lap * zeta0_spec
	psi0 = spec2grid(psi0_spec, gridtype)

	# And we extract the horizontal velocity fields from the streamfunction
	u0_spec, v0_spec = compute_vel(psi0_spec)
	u0 = spec2grid(u0_spec, gridtype) * derfact
	v0 = spec2grid(v0_spec, gridtype) * derfact

	# We compute all the conserved values
	energy = np.mean(0.5 * (u0**2 + v0**2))
	enstrophy = np.sum((zeta0_grid + f)**2 / 2.0)
	zetamean = np.mean(zeta0_grid)
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


	# We start the time integration >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	print("Starting time integration ...")

	# We compute the advection term
	adv0_spec = compute_adv(u0, v0, zeta0_grid)

	# And perform a forward Euler step in time for the first integration
	zeta_spec = zeta0_spec
	zetaold_spec = zeta0_spec
	# We apply the hyperdiffusion implicitly (i.e. (1 + dt*hyp)zeta_i+1 = rhs_i))
	zetanew_spec = (zeta_spec + dt * adv0_spec) * hyp_denom1
	zetanew = spec2grid(zetanew_spec, gridtype)

	# We can also extract the new streamfunction field
	psi_spec = inv_lap * zetanew_spec
	
	# And get the new velocity fields
	u_spec, v_spec = compute_vel(psi_spec)
	u = spec2grid(u_spec, gridtype) * derfact
	v = spec2grid(v_spec, gridtype) * derfact

	# Again, we comptute the conserved values
	energy = np.mean(0.5 * (u**2 + v**2))
	enstrophy = np.sum((zetanew + f)**2 / 2.0)
	zetamean = np.mean(zetanew)
	# And save them in the lists
	energies.append(energy)
	enstrophies.append(enstrophy)
	mean_vorticities.append(zetamean)


	# Now, we can start the main integration loop >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
		adv_spec = compute_adv(u, v, zeta)

		# Now, a Leapfrog scheme is used to perform the time integration
		# Again, we apply the hyperdiffusion implicitly (i.e. (1 + 2dt*hyp)zeta_i+1 = rhs_i))
		zetanew_spec = (zetaold_spec + 2 * dt * adv_spec) * hyp_denom2

		# After the time step, we apply a Robert-Asselin-Williams (RAW) filter to reduce the
		# computational mode amplitude and reaching up to third order precision
		# We compute the correcting term (a centered difference)
		delta = zetanew_spec - 2.0*zeta_spec + zetaold_spec
		# And then we apply this correction to the current and new vorticity fields with the RAW filter
		# damping it with nu and displacing zeta forwards and zetanew backwards with alpha
		zeta_spec += nu*alpha/2.0 * delta
		zetanew_spec += - nu*(1-alpha)/2.0 * delta
		zetanew = spec2grid(zetanew_spec, gridtype)

		# Now we can extract the new streamfunction field
		psi_spec = inv_lap * zetanew_spec

		# And compute the new velocity fields
		u_spec, v_spec = compute_vel(psi_spec)
		u = spec2grid(u_spec, gridtype) * derfact
		v = spec2grid(v_spec, gridtype) * derfact

		# Finally, we comptute the conserved values
		energy = np.mean(0.5 * (u**2 + v**2))
		enstrophy = np.sum((zetanew + f)**2 / 2.0)
		zetamean = np.mean(zetanew)
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
			psi = spec2grid(psi_spec, gridtype)
			if gridtype == 'GLQ':	# We remove the extra 360º longitude band
				streamfunctions.append(psi[:,:-1].copy())
				vorticities.append(zetanew[:,:-1].copy())
			else:
				streamfunctions.append(psi.copy())
				vorticities.append(zetanew.copy())
			next_save_time += save_time


	# In the end, we save the results of the simulation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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


	# FIGURE PLOTTING =================================================================================================

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