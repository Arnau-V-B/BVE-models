"""
Module with the functions to save the results obtained after the global BVE simulation
in the linear or nonlinear approximations.
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

def save_fields_evolution(data_dir, streamfunctions, vorticities, lon, lat, times, output_name, args):

	MODE = args['mode']
	gridtype = args['gridtype']
	lmax = args['lmax']

	# First, we create the latitude and longitude coordinates
	if gridtype == 'GLQ':
		lon = lon[:-1]	# We remove the extra 360º longitude band

	time_coord = xr.DataArray(
		times,
		dims='time',
		attrs={
			'units': 's',
			'long_name': 'time',
			'standard_name': 'time'
		}
	)
	lat_coord = xr.DataArray(
		lat,
		dims='lat',
		attrs={
			'units': 'degrees_north',
			'long_name': 'latitude',
			'standard_name': 'latitude',
			'stored_direction': 'decreasing'
		}
	)
	lon_coord = xr.DataArray(
		lon,
		dims='lon',
		attrs={
			'units': 'degrees_east',
			'long_name': 'longitude',
			'standard_name': 'longitude'
		}
	)

	# Then we create the dataset and save the fields with their corresponding attributes
	evo = xr.Dataset(
		{
			'streamfunction': (['time', 'lat', 'lon'], np.stack(streamfunctions)),
			'vorticity': (['time', 'lat', 'lon'], np.stack(vorticities))
		},
		coords={
			'time': time_coord,
			'lat': lat_coord,
			'lon': lon_coord
		}
	)

	if MODE == 'linear':
		desc = 'Evolution of 500 hPa relative vorticity and streamfunction perturbation fields in a linear BVE simulation'
	else:
		desc = 'Evolution of 500 hPa relative vorticity and streamfunction perturbation fields in a nonlinear BVE simulation'

	evo.attrs = {
		'description': desc,
		'Conventions': 'DF-1.7',
		'history': f'Created on {time.ctime()}',
		'source': 'Global barotropic vorticity equation simulation at 500 hPa in Python'
	}
	evo['streamfunction'].attrs = {
		'description': '2D simulated streamfunction perturbation field',
		'units': 'm**2 s**-1',
		'long_name': 'Stream function',
		'standard_name': 'streamfunction',
		'gridType': f'{gridtype} (T{lmax})'
	}
	evo['vorticity'].attrs = {
		'description': '2D simulated relative vorticity perturbation field',
		'units': 's**-1',
		'long_name': 'Vorticity (relative)',
		'standard_name': 'vorticity',
		'positive': 'Cyclonic',
		'gridType': f'{gridtype} (T{lmax})'
	}

	evo_file = f"fields_evolution_{output_name}.nc"
	evo.to_netcdf(data_dir + evo_file)
	evo.close()