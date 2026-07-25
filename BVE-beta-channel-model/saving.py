"""
Module with the functions to save the results obtained after the BVE simulation.
The data is saved in NetCDF4 files inside the 'ouput/exp_()/data/' folder.
This script has to be placed in the same directory as 'main.py'
"""


import xarray as xr
import numpy as np
import time


# CONSERVED VALUES ==========================================================================================

def save_conserved_values(data_dir, energies, enstrophies, vorticity_means, output_name):

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

	
# VORTICITY AND STREAMFUNCTION FIELDS =======================================================================

def save_fields_evolution(data_dir, streamfunctions, vorticities, x, y, times, output_name):

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