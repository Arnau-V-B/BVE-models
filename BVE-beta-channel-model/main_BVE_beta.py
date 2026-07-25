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

# Once all the parameters are defined, we create a dictionary to store the most frequently used
params = {
    'step': (dx, dy),
	'k': (kx, ky),
	'ks': (kxs, kys),
	'f0': f0,
	'fs': fs,
	'beta': beta,
	'H': H
}


# INITIAL FIELDS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
print("Generating initial fields ...\n")

# First we generate the streamfunction perturbation field
psi0 = pert0(xs,ys)
# We get rid of its residual mean
psi0 -= np.mean(psi0)

# And we convert the streamfunction field into an initial vorticity field perturbation
zeta0 = laplace(psi0, params)

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
u0, v0 = find_vel(psi0, params)

# With all the initial fields generated, we can compute the first conserved values
energy, enstrophy, zetamean = compute_conserved_values(u0, v0, zeta0, params)

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


# TIME INTEGRATION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

print("Starting time integration ...")

# Now we compute the initial RHS term of the BVE (topography is considered if required)
if USE_TOPOGRAPHY:
	rhs0 = jacobian_with_topography(psi0, zeta0, Us, Zs_y, hs, params)
else:
	rhs0 = jacobian(psi0, zeta0, Us, Zs_y, params)

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
psi = poisson_fft(zetanew, params)

# Again, we calculate the conserved values with the new fields
u, v = find_vel(psi, params)

# Again, we calculate the conserved values with the new fields
energy, enstrophy, zetamean = compute_conserved_values(u, v, zetanew, params)

# And save all the conserved values
energies.append(energy)
enstrophies.append(enstrophy)
vorticity_means.append(zetamean)


# Now, we can start the main integration loop
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
		rhs = jacobian_with_topography(psi, zeta, Us, Zs_y, hs, params)
	else:
		rhs = jacobian(psi, zeta, Us, Zs_y, params)

	# Now, a leapfrog step with a RAW filter is performed to integrate in time
	zeta, zetanew = leapfrog_raw_step(zeta, zetaold, rhs, dt, nu, alpha)

	# We compute the new streamfunction field
	psi = poisson_fft(zetanew, params)

	# And extract the corresponding velocity fields
	u, v = find_vel(psi, params)

	# Finally, at the end of each iteration we keep track of the conserved magnitudes
	energy, enstrophy, zetamean = compute_conserved_values(u, v, zetanew, params)

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
save_conserved_values(data_dir, energies, enstrophies, vorticity_means, output_name)
# And then the relative vorticity and streamfunction snapshots
save_fields_evolution(data_dir, streamfunctions, vorticities, x, y, times, output_name)


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