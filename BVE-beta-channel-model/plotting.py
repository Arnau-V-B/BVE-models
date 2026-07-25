"""
Module with the plotting functions to represent the results obtained after
the BVE simulation, which are saved in the 'ouput/exp_()/data/' folder.
The figures generated will be saved in the 'output/exp_()/figures/' folder.
This script has to be placed in the same directory as 'main.py'
"""


# We import the required libraries for the plots
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import BoundaryNorm
import imageio


# CONSERVED VALUES ==========================================================================================

def plot_conserved_values(data_dir, output_name, im_dir):

	# We first read and extract the information in the corresponding Dataset
	cons_file = f"conserved_values_{output_name}.nc"
	cons = xr.open_dataset(data_dir + cons_file, engine='netcdf4')

	energies = cons['kinetic_energy']
	enstrophies = cons['enstrophy']
	vorticity_means = cons['mean_vorticity']
	iterations = cons['iteration']

	cons.close()

	# We establish a fixed format for the y axis
	y_formatter = ticker.ScalarFormatter(useOffset=True, useMathText=True)

	# Then we plot the evolution of all the conserved magnitudes in a triple figure
	fig, axs = plt.subplots(3,1, figsize=(8,8), sharex=True)
	ax1, ax2, ax3 = axs

	ene=ax1.plot(iterations,energies, label='Mean kinetic energy')
	ax1.set_title("Evolution of conserved magnitudes")
	ax1.set_ylabel('Mean kinetic energy (J/kg)')
	ax1.set_xlim(iterations[0],iterations[-1])
	ax1.yaxis.set_major_formatter(y_formatter)
	ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useOffset=True)

	ens=ax2.plot(iterations,enstrophies, label='Enstrophy')
	ax2.set_ylabel(r'Enstrophy (1/s$^2$)')
	ax2.set_xlim(iterations[0],iterations[-1])
	ax2.yaxis.set_major_formatter(y_formatter)
	ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useOffset=True)

	zet=ax3.plot(iterations,vorticity_means, label='Mean vorticity')
	ax3.set_ylabel('Mean vorticity (1/s)')
	ax3.set_xlabel('Nº iterations')
	ax3.set_xlim(iterations[0],iterations[-1])
	ax3.yaxis.set_major_formatter(y_formatter)
	ax3.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useOffset=True)

	fig.tight_layout()
	plt.savefig(im_dir + cons_file[:-3] + ".png", dpi=150)
	plt.close(fig)
	

# VORTICITY AND STREAMFUNCTION EVOLUTION ====================================================================

def plot_fields_evolution(data_dir, output_name, im_dir):
	
	# Again, we read and extract all the information contained in the Dataset
	evo_file = f"fields_evolution_{output_name}.nc"
	evo = xr.open_dataset(data_dir + evo_file, engine='netcdf4')

	streamfunctions = evo['streamfunction']
	vorticities = evo['vorticity']
	x = evo['x']
	y = evo['y']
	xs, ys = np.meshgrid(x, y)
	times = [int(time/3600) for time in evo['time'].values]

	# First we plot the vorticity field evolution

	# We pick the levels of the colormap that best fit our data
	# To do so, we use the percentiles to know the different scales of the data
	all_data = vorticities.values.flatten()

	# We compute the percentiles
	p1 = np.percentile(all_data, 1)
	p99 = np.percentile(all_data, 99)

	# And we create a symmetric level scale based on the percentiles to give importance to
	# the range of values where most of the data is
	max_abs = max(abs(p1), abs(p99))
	levels = np.concatenate([
			np.linspace(-max_abs, -max_abs/2, 5),
			np.linspace(-max_abs/2, max_abs/2, 11),
			np.linspace(max_abs/2, max_abs, 5)
			])
	levels = np.unique(levels)
	norm = BoundaryNorm(levels, 256)

	# Then we plot each of the saved snapshots and generate a GIF
	images = []

	for i in range(len(times)):

		fig, ax = plt.subplots(figsize=(12,4))
		mesh = ax.contourf(xs, ys, vorticities[i], cmap='coolwarm', 
					 		norm=norm, levels=levels, extend='both')
		cbar = fig.colorbar(mesh, ax=ax, extend='both', label='Vorticity (1/s)')
		cbar.set_ticks(levels)
		ax.set_title(f'Vorticity field at t = {times[i]}h')
		ax.set_xlabel('x (m)')
		ax.set_ylabel('y (m)')
		fig.tight_layout()

		fig_name = f"vorticity_field_{output_name}_t{times[i]}h.png"
		fig.savefig(im_dir + "temp_frames/" + fig_name, dpi=150)
		plt.close(fig)

		images.append(imageio.v2.imread(im_dir + "temp_frames/" + fig_name))
	
	gif_name = f"vorticity_field_{output_name}_evolution.gif"
	imageio.mimsave(im_dir + gif_name, images, duration=250, loop=0)

	# Finally, we plot the streamfunction field evolution
	images = []

	for i in range(len(times)):

		fig, ax = plt.subplots(figsize=(12,4))
		mesh = ax.contour(xs,ys,streamfunctions[i],cmap='coolwarm')
		cbar = fig.colorbar(mesh, ax=ax, label=r'Stream function ($\mathrm{m^2/s}$)')
		ax.set_title(f'Stream function field at t = {times[i]}h')
		ax.set_xlabel('x (m)')
		ax.set_ylabel('y (m)')
		fig.tight_layout()

		fig_name = f"streamfunction_field_{output_name}_t{times[i]}h.png"
		fig.savefig(im_dir + "temp_frames/" + fig_name, dpi=150)
		plt.close(fig)

		images.append(imageio.v2.imread(im_dir + "temp_frames/" + fig_name))
	
	gif_name = f"streamfunction_field_{output_name}_evolution.gif"
	imageio.mimsave(im_dir + gif_name, images, duration=250, loop=0)

	evo.close()