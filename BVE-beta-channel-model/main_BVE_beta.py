"""
Main program that runs the non-divergent barotropic vorticity equation (BVE) simulation.
The initial conditions can be set in the config.py file located in the same directory.
This version of the program computes the non-divergent BVE linearized over a mean zonal flow
in the beta-plane aproximation.
At the end, an output folder is generated to save the results in netCDF format and simple figures with
GIFs to keep track of the conservation properties of the model and visualize the evolution of the
streamfunction and vorticity fields.

@Author: Arnau Vicente Bou
"""


# Python libraries to import
import numpy as np
import scipy as sp
import xarray as xr
import time
import sys
import os


# FUNCTIONS DEFINED FOR THE SIMULATION ================================================================================

def laplace(func,dx,dy):
	"""
	This function returns the 2D laplacian of a given discrete function field: Nabla^2 func.

	For the interior points centered differences are used and for the boundaries forward
	and backward differences are considered accordingly.

	PARAMETERS:
	(input) -->
		func   : 2D array (ny, nx) of the function field
		dx, dy : grid spacing in x and y-direction
	(output) -->
		lap    : 2D array (ny, nx) with the laplacian of the function field
	"""

	# We inicialize the laplacian solution field
	lap = np.zeros_like(func)

	# We precompute the multiplying factors
	dx2 = dx**2
	dy2 = dy**2

	# Interior (centered X and Y)
	lap[1:-1,1:-1] = ((func[1:-1,2:] - 2*func[1:-1,1:-1] + func[1:-1,:-2]) / dx2 +
					  (func[2:,1:-1] - 2*func[1:-1,1:-1] + func[:-2,1:-1]) / dy2)
	
	# S boundary (centered X and forward Y)
	lap[0,1:-1] = ((func[0,2:] - 2*func[0,1:-1] + func[0,:-2]) / dx2 +
				   (func[2,1:-1] - 2*func[1,1:-1] + func[0,1:-1]) / dy2)
	# N boundary (centered X and backward Y)
	lap[-1,1:-1] = ((func[-1,2:] - 2*func[-1,1:-1] + func[-1,:-2]) / dx2 +
				   (func[-3,1:-1] - 2*func[-2,1:-1] + func[-1,1:-1]) / dy2)
	# W boundary (forward X and centered Y)
	lap[1:-1,0] = ((func[1:-1,2] - 2*func[1:-1,1] + func[1:-1,0]) / dx2 +
				   (func[2:,0] - 2*func[1:-1,0] + func[:-2,0]) / dy2)
	# E boundary (backward X and centered Y)
	lap[1:-1,-1] = ((func[1:-1,-3] - 2*func[1:-1,-2] + func[1:-1,-1]) / dx2 +
				   (func[2:,-1] - 2*func[1:-1,-1] + func[:-2,-1]) / dy2)
	
	# SW corner (forward X and Y)
	lap[0,0] = ((func[0,2] - 2*func[0,1] + func[0,0]) / dx2 +
				   (func[2,0] - 2*func[1,0] + func[0,0]) / dy2)
	# NW corner (forward X and backward Y)
	lap[-1,0] = ((func[-1,2] - 2*func[-1,1] + func[-1,0]) / dx2 +
				   (func[-3,0] - 2*func[-2,0] + func[-1,0]) / dy2)
	# SE corner (backward X and forward Y)
	lap[0,-1] = ((func[0,-3] - 2*func[0,-2] + func[0,-1]) / dx2 +
				   (func[2,-1] - 2*func[1,-1] + func[0,-1]) / dy2)
	# NE corner (backward X and Y)
	lap[-1,-1] = ((func[-1,-3] - 2*func[-1,-2] + func[-1,-1]) / dx2 +
				   (func[-3,-1] - 2*func[-2,-1] + func[-1,-1]) / dy2)
	
	return lap


def poisson_fft(zeta, kx, dy):
	"""
	This function solves the 2D Poisson equation psi = Nabla^-2 zeta with periodic BC in x
	and Dirichlet BC (psi=0) in y.

	To do so, it uses FFT for x and a discretized centered differences scheme for y, which result
	in the following equation:
	--> psi_hat_j-1 - (2 + k^2*dy^2) psi_hat_j + psi_hat_j+1 = dy^2 * zeta_j

	Given the symetry of the system, a tridiagonal matrix solver is used with a banded matrix layout.
	
	PARAMETERS:
	(input) -->
		zeta : 2D array (ny, nx) of relative vorticity field
		kx   : 1D array (nx/2 + 1) with Fourier spectral grid wavenumbers in x-direction
		dy   : grid spacing in y-direction
	(output) -->
		psi  : 2D array (ny, nx) of streamfunction field
	"""

	# We obtain the grid dimensions in y
	ny = zeta.shape[0]

	# We compute the transformed vorticity field (only the real part)
	zeta_hat = sp.fft.rfft(zeta, axis=1)

	# We initialize the transformed streamfunction field
	psi_hat = np.zeros_like(zeta_hat, dtype=complex)

	# We precompute the multiplying factors to reduce operations in tha main loop 
	kx2 = kx**2
	dy2 = dy**2
	kx2dy2 = kx2*dy2

	# Now we generate the tridiagonal matrix that represents the equations system
	# ---> A_b * psi_hat = b 
	# For a more optimized calculation, we use of a banded matrix representation:

	# Firts we pre-allocate the banded matrix
	# For tridiagonal matrix, we need a (3, n) array (instead of a full n*n matrix) where:
	# 	row 0: upper diagonal (super-diagonal)
	# 	row 1: main diagonal
	# 	row 2: lower diagonal (sub-diagonal)
	A_b = np.zeros((3, ny-2), dtype=complex)
	
	# Then we fill the constant parts of the banded matrix
	# Upper diagonal (index 0): ones
	A_b[0, 1:] = 1.0  	# Note: first element of upper diagonal is not used
	
	# Lower diagonal (index 2): ones
	A_b[2, :-1] = 1.0  	# Note: last element of lower diagonal is not used
	
	# Once the banded matrix is built, we solve the tridiagonal system for each Fourier mode in X
	for i in range(len(kx)):
		
		# Firstly, the main diagonal term for current kx is set
		A_b[1, :] = -(2.0 + kx2dy2[i])
		
		# Then the RHS vector term is also set (only for interior points)
		b = dy2 * zeta_hat[1:-1, i]
		
		# Now we can solve the system using the solve_banded method from scipy (very optimized)
		# 	l = 1 (number of non-zero diagonals below main diagonal)
		# 	u = 1 (number of non-zero diagonals above main diagonal)
		psi_hat[1:-1, i] = sp.linalg.solve_banded((1, 1), A_b, b)
		
		# Finally, we set the boundary conditions at y boundaries
		psi_hat[0, i] = 0.0
		psi_hat[-1, i] = 0.0
	
	# In the end we recover the real streamfunction field by applying an inverse Fourier transform
	psi = sp.fft.irfft(psi_hat, axis=1)
	
	return psi


def jacobian(psi,zeta,kx,beta,U,Z_y):
	"""
	This function computes the RHS of the non-divergent BVE linearized over a mean zonal flow U
	in the beta-plane aproximation:
	--> d(zeta)/dt = - U * d(zeta)/dx - d(psi)/dx * (dZ/dy + beta)

	To do so, it assumes that relative vorticity and streamfunction are periodic in x and
	transforms both fields into the spectral space using the Fast Fourier Transform
	(FFT) method, solving the problem as follows:
	--> RHS = - U * Re{i*kx(z)*zeta_hat} - Re{i*kx(p)*psi_hat} * (dZ/dy + beta)
	
	PARAMETERS:
	(input) -->
		psi  : 2D array (ny, nx) of streamfunction field
		zeta : 2D array (ny, nx) of vorticity field
		kx   : 1D array (nx/2 + 1) with Fourier spectral grid wavenumbers in x-direction
		beta : Rossby parameter df/dy
		U    : 2D array (ny, nx) of mean zonal velocity field
		Z_y  : 2D array (ny, nx) of meridional derivative of mean relative vorticity field
	(output) -->
		RHS  : 2D array (ny, nx) with the RHS term of the BVE
	"""

	# We transform the streamfunction and vorticity fields to spectral space in x
	psi_hat = sp.fft.rfft(psi, axis=1)
	zeta_hat = sp.fft.rfft(zeta, axis=1)

	# We precompute the imaginary coefficients
	ikx = 1j * kx

	# We compute the transformed derivatives
	psi_hat_x = ikx * psi_hat
	zeta_hat_x = ikx * zeta_hat

	# And we return to the real space to compute the RHS
	psi_x = sp.fft.irfft(psi_hat_x, axis=1)
	zeta_x = sp.fft.irfft(zeta_hat_x, axis=1)

	RHS = - U * zeta_x - psi_x * (Z_y + beta)
	
	return RHS


def jacobian_with_topography(psi,zeta,kx,beta,U,Z_y,h,f0,H):
	"""
	This function computes the RHS of the non-divergent BVE linearized over a mean zonal flow U
	in the beta-plane aproximation including a source of vorticity from topography h: 
	--> d(zeta)/dt = - U * d(zeta)/dx - d(psi)/dx * (dZ/dy + beta) - f0/H * U * d(h)/dx

	To do so, it assumes that relative vorticity, streamfunction and topography are periodic in x and
	transforms both fields into the spectral space using the Fast Fourier Transform (FFT) method,
	solving the problem as follows:
	--> RHS = - U * Re{i*kx(z)*zeta_hat} - Re{i*kx(p)*psi_hat} * (dZ/dy + beta) - f0/H * U * Re{i*kx(h)*h_hat}
	
	PARAMETERS:
	(input) -->
		psi  : 2D array (ny, nx) of streamfunction field
		zeta : 2D array (ny, nx) of relative vorticity field
		kx   : 1D array (nx/2 + 1) with Fourier spectral grid wavenumbers in x-direction
		beta : Rossby parameter df/dy
		U    : 2D array (ny, nx) of mean zonal velocity field
		Z_y  : 2D array (ny, nx) of meridional derivative of mean relative vorticity field
		h    : 2D array (ny, nx) of topography function field
		f0   : Coriolis parameter at mid-latitudes
		H    : atmospheric scale height
	(output) -->
		RHS  : 2D array (ny, nx) with the RHS term of the BVE with topography
	"""

	# We transform the streamfunction, vorticity and topography fields to spectral space in x
	psi_hat = sp.fft.rfft(psi, axis=1)
	zeta_hat = sp.fft.rfft(zeta, axis=1)
	h_hat = sp.fft.rfft(h, axis=1)

	# We precompute the imaginary coefficients
	ikx = 1j * kx

	# We compute the transformed derivatives
	psi_hat_x = ikx * psi_hat
	zeta_hat_x = ikx * zeta_hat
	h_hat_x = ikx * h_hat

	# And we return to the real space to compute the RHS
	psi_x = sp.fft.irfft(psi_hat_x, axis=1)
	zeta_x = sp.fft.irfft(zeta_hat_x, axis=1)
	h_x = sp.fft.irfft(h_hat_x, axis=1)

	RHS = - U * zeta_x - psi_x * (Z_y + beta) - f0/H * U * h_x

	return RHS


def find_vel(psi,kx,ky):
	"""
	This function obtains the horizontal velocity field from a given 2D streamfunction field:
	--> u = - d(psi)/dy ; v = d(psi)/dx

	To do so, it transforms all the fields into the spectral space using the Fast Fourier Transform
	(FFT) method and solves the problem as follows:
	--> u_hat = -i * ky * psi_hat ; v_hat = i * kx * psi_hat
	
	PARAMETERS:
	(input) -->
		psi    : 2D array (ny, nx) of streamfunction field
		kx, ky : 1D arrays (nx/2 + 1) with Fourier spectral grid wavenumbers in x and y-direction
	(output) -->
		u, v   : 2D arrays (ny, nx) of the zonal and meridional velocity fields
	"""

	# We transform the streamfunction field to the spectral space
	psi_hat = sp.fft.rfft2(psi)

	# We compute the transformed zonal velocity field 'u_hat'
	u_hat = -1j * ky * psi_hat

	# We compute the transformed meridional velocity field 'v_hat'
	v_hat = 1j * kx * psi_hat

	# In the end we recover the real velocity fields by applying an inverse Fourier transform
	u = sp.fft.irfft2(u_hat)
	v = sp.fft.irfft2(v_hat)

	return u,v


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

	# We generate the coordinates and the spectral grids
	dx = float(Lx/(nx-1))
	dy = float(Ly/(ny-1))
	x = np.linspace(0,Lx,nx)
	y = np.linspace(0,Ly,ny)
	xs,ys = np.meshgrid(x,y)

	kx = 2.0 * np.pi * sp.fft.rfftfreq(nx, dx)	# kx = 2\pi*m/Lx ; m = [0,1,2,...,Nx/2]
	ky = 2.0 * np.pi * sp.fft.fftfreq(ny, dy)
	kxs,kys = np.meshgrid(kx,ky)
	
	# We precompute some useful parameters
	f0 = 2.0 * Omega_earth * np.sin(phi0)					# Coriolis parameter at central latitude
	beta = 2.0 * Omega_earth * np.cos(phi0) / R_earth		# Rossby parameter
	fs = f0 + beta*ys										# Coriolis parameter at each latitude

	# We generate empty lists to keep track of the conservated quantities:
	energies = []					# kinetic energy
	enstrophies = []				# enstrophy
	vorticity_means = []			# mean relative vorticity
	# And also to save the evolution fields
	times = []						# time
	streamfunctions = []			# streamfunction
	vorticities = []				# relative vorticity

	# We create a folder in 'output' to save the results of the specific experiment
	os.makedirs(output_dir + output_name, exist_ok=True)


	# Now, we generate the initial fields >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	print("Generating initial fields ...\n")
	
	# First we generate the streamfunction perturbation field
	psi0 = pert0(xs,ys)
	# We get rid of its residual mean
	psi0 -= np.mean(psi0)

	# And we convert the streamfunction field into an initial vorticity field perturbation
	zeta0 = laplace(psi0,dx,dy)

	# Then, we generate the background zonal velocity field U
	Us = zonal(xs,ys)

	# And obtain the corresponding background vorticity field advection term (i.e. d(Zeta)/dy = -d2U/dy2)
	Zs_y = np.zeros_like(Us)
	Zs_y[1:-1,:] = - (Us[2:,:] - 2*Us[1:-1,:] + Us[:-2,:]) / dy**2		# centered differences
	Zs_y[0,:] = - (Us[2,:] - 2*Us[1,:] + Us[0,:]) / dy**2				# forward differences
	Zs_y[-1,:] = - (Us[-2,:] - 2*Us[-1,:] + Us[0,:]) / dy**2			# backward differences

	# Also, we generate the topography field (if required)
	if USE_TOPOGRAPHY:
		hs = topography(xs,ys)

	# With all the initial fields generated, we can compute the first conserved values
	u0, v0 = find_vel(psi0, kxs, kys)
	energy = np.mean(0.5 * (u0**2 + v0**2))
	enstrophy = np.sum((zeta0 + fs)**2 / 2.0)
	zetamean = np.mean(zeta0)

	# Then we append the results in the corresponding lists
	energies.append(energy)
	enstrophies.append(enstrophy)
	vorticity_means.append(zetamean)

	# We also save the initial fields
	times.append(ti)
	streamfunctions.append(psi0.copy())
	vorticities.append(zeta0.copy())

	# Finally we set the next save time
	next_save_time = save_time


	# We start the time integration >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	print("Starting time integration ...")

	# Now we compute the initial RHS term of the BVE (topography is considered if required)
	if USE_TOPOGRAPHY:
		rhs0 = jacobian_with_topography(psi0,zeta0,kxs,beta,Us,Zs_y,hs,f0,H)
	else:
		rhs0 = jacobian(psi0,zeta0,kxs,beta,Us,Zs_y)

	# And we assign the initial and first vorticity fields to the 3 step Leapfrog scheme
	zeta = zeta0
	zeta_old = zeta0

	# Before we start the time integration, we compute the value of the first time interval dt
	# so it is consistent with the CFL condition given by the initial perturbation.
	# We first compute the maximum horizontal speed achieved by the field
	U_max = np.max(np.sqrt(u0**2 + v0**2)) + np.max(np.abs(Us))
	print(f"U_max = {U_max:.2f} m/s")
	# Then, we compute the highest dt that satisfies the CFL condition
	dt = int(min(dx, dy) / U_max)
	print(f"CFL time step: dt = {dt} s")

	# Once dt is computed, we begin the time integration by using a forward differences
	# Euler scheme (first order) for the first time step
	zetanew = zeta + dt*rhs0

	# We obtain the new streamfunction field
	psi = poisson_fft(zetanew,kx,dy)

	# Again, we calculate the conserved values with the new fields
	u, v = find_vel(psi, kxs, kys)
	energy = np.mean(0.5 * (u**2 + v**2))
	enstrophy = np.sum((zetanew + fs)**2 / 2.0)
	zetamean = np.mean(zetanew)

	# And save all the conserved values
	energies.append(energy)
	enstrophies.append(enstrophy)
	vorticity_means.append(zetamean)
	

	# Now, we can start the main integration loop >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	for t in range(ti+2*dt,tf+dt,dt):

		# We display in the terminal an updating counter showing the elapsed time of computation and 
		# the simulation time to visualize the program progress
		elapsed = time.time() - start_time
		sys.stdout.write(f"\rElapsed time: {elapsed:.2f}s | Simulation time: {t/3600:.2f}h")
		sys.stdout.flush()

		# We update the vorticity fields from last iteration 
		zetaold = zeta
		zeta = zetanew

		# We compute the RHS term
		if USE_TOPOGRAPHY:
			rhs = jacobian_with_topography(psi,zeta,kxs,beta,Us,Zs_y,hs,f0,H)
		else:
			rhs = jacobian(psi,zeta,kxs,beta,Us,Zs_y)

		# Now, a Leapfrog scheme is used to perform the time integration (i.e. second order centered diffreneces)
		zetanew = zetaold + 2.0*dt*rhs

		# After the time step, we apply a Robert-Asselin-Williams (RAW) filter to reduce the
		# computational mode amplitude and reaching up to third order precision
		# We compute the correcting term (a centered difference)
		delta = zetanew - 2.0*zeta + zetaold
		# And then we apply this correction to the current and new vorticity fields with a RAW filter
		# damping it with nu and displacing zeta forwards and zetanew backwards with alpha
		zeta += nu*alpha/2.0 * delta
		zetanew += - nu*(1-alpha)/2.0 * delta

		# We compute the new streamfunction field
		psi = poisson_fft(zetanew,kx,dy)

		# Finally, at the end of each iteration we keep track of the conserved magnitudes
		u, v = find_vel(psi,kxs,kys)
		energy = np.mean(0.5 * (u**2 + v**2))
		enstrophy = np.sum((zetanew + fs)**2 / 2.0)
		zetamean = np.mean(zetanew)

		# And save them in the corresponding lists
		energies.append(energy)
		enstrophies.append(enstrophy)
		vorticity_means.append(zetamean)

		# Every time we get to the save interval, we save a snapshot of the psi and zeta fields
		if t >= next_save_time:
			times.append(t)
			streamfunctions.append(psi.copy())
			vorticities.append(zetanew.copy())
			next_save_time += save_time


	# In the end, we save the results of the simulation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	print('')
	print("Saving simulation results...")

	# We first copy the config.py file used in a '.txt' file
	with (open("config.py", 'r') as file, 
		open(output_dir + output_name + f"/params_{output_name}.txt", 'w') as file_copy):
		file_copy.write(file.read())

	# Then, we save the simulation data into netCDF Datasets
	# We start by creating the directory where the data will be saved
	data_dir = output_dir + output_name + "/data/"
	os.makedirs(data_dir, exist_ok=True)

	# First we save the conserved values
	cons = xr.Dataset(
		{
			'kinetic_energy': (['iteration'], energies),
			'enstrophy': (['iteration'], enstrophies),
			'mean_vorticity': (['iteration'], vorticity_means)
		},
		coords={
			'iteration': np.arange(len(energies))
		}
	)

	cons.attrs['description'] = 'Evolution of the conserved values during the simulation'
	cons['kinetic_energy'].attrs = {
		'description': 'Mean kinetic energy of the of the field at each iteration',
		'units': 'm^2/s^2',
		'long_name': 'Kinetic energy'
	}
	cons['enstrophy'].attrs = {
		'description': 'Mean enstrophy of the of the field at each iteration',
		'units': '1/s^2',
		'long_name': 'Enstrophy'
	}
	cons['mean_vorticity'].attrs = {
		'description': 'Mean vorticity of the of the field at each iteration',
		'units': '1/s',
		'long_name': 'Mean vorticity',
		'positive': 'Cyclonic'
	}
	cons['iteration'].attrs['description'] = 'Iteration number in the simulation'
	
	cons_file = f"conserved_values_{output_name}.nc"
	cons.to_netcdf(data_dir + cons_file)
	cons.close()

	# And then the relative vorticity and streamfunction snapshots
	evo = xr.Dataset(
		{
			'streamfunction': (['time', 'y', 'x'], np.stack(streamfunctions)),
			'vorticity': (['time', 'y', 'x'], np.stack(vorticities))
		},
		coords={
			'time': times,
			'y': y,
			'x': x
		}
	)

	evo.attrs = {
		'description': 'Evolution of the 500 hPa relative vorticity and streamfunction fields in a linear beta-channel BVE simulation',
		'Conventions': 'CF-1.7',
		'history': f'Created on {time.ctime()}',
		'source': 'Beta-channel barotropic vorticity equation simulation at 500 hPa in Python'
	}
	evo['streamfunction'].attrs = {
		'description': '2D simulated streamfunction perturbation field',
		'units': 'm**2 s**-1',
		'long_name': 'Stream function',
		'standard_name': 'streamfunction'
	}
	evo['vorticity'].attrs = {
		'description': '2D simulated relative vorticity perturbation field',
		'units': 's**-1',
		'long_name': 'Vorticity (relative)',
		'standard_name': 'vorticity',
		'positive': 'Cyclonic'
	}
	evo['x'].attrs = {
		'description': 'Eastward distance from Greenwich meridian',
		'units': 'm'
	}
	evo['y'].attrs = {
		'description': 'Northward distance from lowest latitude band',
		'units': 'm'
	}
	evo['time'].attrs = {
		'units': 's',
		'long_name': 'time',
		'standard_name': 'time'
	}

	evo_file = f"fields_evolution_{output_name}.nc"
	evo.to_netcdf(data_dir + evo_file)
	evo.close()


	# FIGURE PLOTTING =================================================================================================

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