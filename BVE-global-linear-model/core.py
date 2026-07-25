"""
This module contains all the functions and routines required to run the BVE simulation
globally over the sphere linearizing the equation over a mean zonal flow or including
the nonlinear terms.
This script has to be placed in the same directory as 'main.py'
"""


import numpy as np
import pyshtools as pysh


def grid2spec(field, args):
	"""
	This function takes a 2D gridded field and transforms it into a set of real spherical harmonics	
	coefficients up to a maximum order of triangular truncation lmax.

	It accepts both Discroll-Healy (DH) (equally sampled) and Gauss-Legendre quadrature (GLQ)
	(gaussian latitudes) grids.

	PARAMETERS:
	(input) -->
		field       : 2D array (nlat, nlon) of grid field
		args        : extra general arguments
	(internal) -->
		grid        : type of grid quadrature: DH or GLQ
		lmax        : maximum wavenumber for spectral triangular truncation
		weights     : weights for each of the roots of Legendre polynomial of roder lmax
		nodes       : roots of Legendre polynomial of order lmax
	(output) -->
		field_spec  : 3D array (2, lmax+1, lmax+1) of real spherical harmonics coefficients,
					  [0,:,:] for cosine coefficients and [1,:,:] for sine coefficients
	"""

	grid = args['gridtype']
	lmax = args['lmax']
	weights = args['glq_weights']
	nodes = args['glq_nodes']

	if grid == 'DH':
		field_spec = pysh.expand.SHExpandDH(field, sampling=2, lmax_calc=lmax)

	elif grid == 'GLQ':
		field_spec = pysh.expand.SHExpandGLQ(field, w=weights, zero=nodes, lmax_calc=lmax)

	return field_spec


def spec2grid(field_spec, args):
	"""
	This function takes a 3D array of real spherical harmonics with triangular truncation lmax
	and shape (2,lmax+1,lmax+1) and transforms it into a 2D gridded field with sampling sampl.

	It accepts both Discroll-Healy (DH) (equally sampled) and Gauss-Legendre quadrature (GLQ)
	(gaussian latitudes) grids.

	PARAMETERS:
	(input) -->
		field_spec : 3D array (2, lmax+1, lmax+1) of real spherical harmonics coefficients,
					 [0,:,:] for cosine coefficients and [1,:,:] for sine coefficients
		args       : extra general arguments
	(internal) -->
		grid       : type of grid quadrature: DH or GLQ
		sampl      : sampling of the resulting gridded field
		nodes      : roots of Legendre polynomial of order lmax
	(output) -->
		field      : 2D array (nlat, nlon) of the gridded field
	"""

	grid = args['gridtype']
	sampl = args['sampl']
	nodes = args['glq_nodes']

	if grid == 'DH':
		field = pysh.expand.MakeGridDH(field_spec, sampling=2, lmax=sampl)

	elif grid == 'GLQ':
		field = pysh.expand.MakeGridGLQ(field_spec, zero=nodes, lmax=sampl)

	return field


def lambda_derivative(field_spec, args):
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
		args       : extra general arguments
	(internal) -->
		R          : Earth radius (for correct scaling)
	(output) -->
		dlambda    : 3D array (2, lmax+1, lmax+1) with the coefficients of the derivative in longitude
	"""

	R = args['radius']

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


def theta_derivative(field_spec, args):
	"""
	This function computes the derivative in latitude for a given field in the real spherical
	harmonics spectral space multiplied by the cosine of latitude. The spectral field coefficients of
	triangular truncation lmax need to be shaped as (2,lmax+1,lmax+1).

	The following recursion relation between spectral coefficients is used to compute the derivative:
	-->	cos(theta) * d/dtheta f_lm = ((l+2) * eps_l+1,m * f_l+1,m - (l-1) * eps_lm * f_l-1,m)
		eps_lm = sqrt((l^2 - m^2) / (4*l^2 - 1))

	PARAMETERS:
	(input) -->
		field_spec : 3D array (2, lmax+1, lmax+1) of real spherical harmonics coefficients,
					 [0,:,:] for cosine coefficients and [1,:,:] for sine coefficients
		args       : extra general arguments
	(internal) -->
		R          : Earth radius (for correct scaling)
	(output) -->
		dtheta     : 3D array (2, lmax+1, lmax+1) with the coefficients of the derivative in lat * cos(lat)
	"""

	R = args['radius']

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


def compute_adv_linear(v, vort, U_grid, Z_abs_theta, args):
	"""
	This function computes the advection term of the linearized BVE in the spectral space given the 
	perturbed meridional velocity (v) and vorticity (zeta) fields in the grid space. It also contains
	the constant background zonal flow U and its associated vorticity Z. 
	---> adv = - 1/(R cos(theta)) * U * d/dlambda (zeta) - 1/R * v * d/dtheta (Z + f)

	To avoid convolutions, the products are computed in the grid and then transformed to spectral space.
	
	PARAMETERS:
	(input) -->
		v           : 2D array (nlat, nlon) of grid perturbed meridional velocity field
		vort        : 2D array (nlat, nlon) of grid perturbed vorticity field
		U_grid      : 2D array (nlat, nlon) of grid background zonal velocity field
		Z_abs_theta : 2D array (nlat, nlon) of grid meridional derivative of absolute background vorticity
		args        : extra general arguments
	(internal) -->
		derfact     : 2D array (nlat, nlon) of grid inverse cosine of latitude (derfact=1/cos(latitude))
	(output) -->
		adv_spec    : 3D array (2, lmax+1, lmax+1) with the spectral coefficients of the advection term
	"""

	derfact = args['derfact']

	# We first compute d(zeta)/dx
	vort_spec = grid2spec(vort, args)
	zeta_lambda_spec = lambda_derivative(vort_spec, args)
	zeta_lambda = spec2grid(zeta_lambda_spec, args) * derfact

	# Then we compute the products in the real space
	U_zeta = U_grid * zeta_lambda
	v_Z_abs = v * Z_abs_theta

	# Transform them into the spectral space
	U_zeta_spec = grid2spec(U_zeta, args)
	v_Z_abs_spec = grid2spec(v_Z_abs, args)

	# And we obtain the advection term
	adv_spec = - (U_zeta_spec + v_Z_abs_spec)

	return adv_spec


def compute_adv_nonlinear(u, v, vort, U_grid, Z_theta, args):
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
		U_grid   : 2D array (nlat, nlon) of grid background zonal velocity field
		Z_theta  : 2D array (nlat, nlon) of grid meridional derivative of mean flow vorticity field
		args     : extra general arguments
	(internal) -->
		R        : Earth radius (for correct scaling)
		f        : 2D array (nlat, nlon) of grid Coriolis parameter (f=2*Omega*sin(latitude))
		derfact  : 2D array (nlat, nlon) of grid inverse cosine of latitude (derfact=1/cos(latitude))
		tan_lat  : 2D array (nlat, nlon) of grid tangent of latitude (tan_lat=tan(latitude))
	(output) -->
		adv_spec : 3D array (2, lmax+1, lmax+1) with the spectral coefficients of the advection term
	"""

	R = args['radius']
	f = args['f']
	derfact = args['derfact']
	tan_lat = args['tan_lat']

	# We first compute the product V·(zeta+f) in the grid space
	u_zeta = u * (vort + f)
	v_zeta = v * (vort + f)

	# Then we transform the product into the spectral space
	u_zeta_spec = grid2spec(u_zeta, args)
	v_zeta_spec = grid2spec(v_zeta, args)
	zeta_spec = grid2spec(vort, args)

	# We compute the spectral derivatives
	u_zeta_lambda_spec = lambda_derivative(u_zeta_spec, args)
	v_zeta_theta_spec = theta_derivative(v_zeta_spec, args)
	zeta_lambda_spec = lambda_derivative(zeta_spec, args)

	# We obtain the flat part of the horizontal divergence
	div_spec = u_zeta_lambda_spec + v_zeta_theta_spec
	# And transform it to grid space dividing by cos(theta)
	div = spec2grid(div_spec, args) * derfact
	zeta_lambda = spec2grid(zeta_lambda_spec, args) * derfact

	# We apply the curvature term and change sign to obtain the advection
	adv = - (div - v_zeta * tan_lat / R) - U_grid * zeta_lambda - v * Z_theta
	# And we transform the advection into the spectral space
	adv_spec = grid2spec(adv, args)

	return adv_spec


def compute_vel(stream_spec, args):
	"""
	This function computes the horizontal velocity fields in the spectral space multiplied by cos(latitude)
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
		args        : extra general arguments
	(output) -->
		u_spec      : 3D array (2, lmax+1, lmax+1) with the spectral coefficients of the zonal velocity field
					  multiplied by cos(latitude)
		v_spec      : 3D array (2, lmax+1, lmax+1) with the spectral coefficients of the meridional velocity field
					  multiplied by cos(latitude)
	"""

	# We compute the spectral curl of the stream function
	u_spec = - theta_derivative(stream_spec, args)
	v_spec = lambda_derivative(stream_spec, args)

	return u_spec, v_spec


def compute_conserved_values(u, v, zeta, args):
	"""
	This function computes the three main conserved values in the non-divergent BVE simulation:
	--> kinetic energy: KE = mean(1/2 * (u^2 + v^2))
		enstrophy     : ε = 1/2 * (zeta + f)^2
		mean vorticity: ζ = mean(zeta)
		
	PARAMETERS:
	(input) -->
		u         : 2D array (nlat, nlon) of grid zonal velocity field
		v         : 2D array (nlat, nlon) of grid meridional velocity field
		zeta      : 2D array (nlat, nlon) of grid vorticity field
	(internal) -->
		f         : 2D array (nlat, nlon) of grid Coriolis parameter (f=2*Omega*sin(latitude))
	(output) -->
		energy    : 2D array (nlat, nlon) of grid kinetic energy field
		enstrophy : 2D array (nlat, nlon) of grid enstrophy field
		zetamean  : 2D array (nlat, nlon) of grid mean relative vorticity field
	"""
	
	f = args['f']

	energy = np.mean(0.5 * (u**2 + v**2))
	enstrophy = np.sum(0.5 * (zeta + f)**2)
	zetamean = np.mean(zeta)
	
	return energy, enstrophy, zetamean


def leapfrog_raw_step(zeta_spec, zetaold_spec, adv_spec, hyp, dt, nu, alpha):
	"""
	This function performs a leapfrog time integration step applying a Robert-Asselin-Williams
	(RAW) filter with a damping coefficient ν and a displacement coefficient α.
	It works on in the spherical harmonics spectral space and applies an implicit hyperdiffiusion:
	--> (1 + 2dt*hyp) * ζ(t+dt) = RHS(t)

	The RAW filter consists on the following three steps:
	--> ζ(t+dt) = ζ''(t-dt) + 2dt * RHS'(t)
		ζ''(t) = ζ'(t) + να/2 * (ζ(t+dt) - 2*ζ'(t) + ζ''(t-dt))
		ζ'(t+dt) = ζ(t+dt) - ν(1-α)/2 * (ζ(t+dt) - 2*ζ'(t) + ζ''(t-dt))
	where the tildes show the amount of filtering passes.

	PARAMETERS:
	(input) -->
		zeta_spec    : 3D array (2, lmax+1, lmax+1) with the spectral coefficients of the current vorticity field
		zetaold_spec : 3D array (2, lmax+1, lmax+1) with the spectral coefficients of the old vorticity field
		adv_spec     : 3D array (2, lmax+1, lmax+1) with the spectral coefficients of the advection term
		hyp          : 3D array (2, lmax+1, lmax+1) with the implicit hyperdiffusion operator
		dt           : time step of the integration
		nu           : damping coefficient of the RAW filter
		alpha        : displacement coefficient of the RAW filter
	(internal) -->
		delta        : 3D array (2, lmax+1, lmax+1) with the spectral coefficients of the difference between the new, current and old vorticity fields
	(output) -->
		zeta_spec    : 3D array (2, lmax+1, lmax+1) with the spectral coefficients of the current vorticity field (modified)
		zetanew_spec : 3D array (2, lmax+1, lmax+1) with the spectral coefficients of the new vorticity field
	"""

	# First we perform a leapfrog step
	zetanew_spec = (zetaold_spec + 2 * dt * adv_spec) * hyp

	# After the time step, we apply a Robert-Asselin-Williams (RAW) filter to reduce the
	# computational mode amplitude and reaching up to third order precision
	# We compute the correcting term (a centered difference)
	delta = zetanew_spec - 2.0*zeta_spec + zetaold_spec
	
	# And then we apply this correction to the current and new vorticity fields with a RAW filter
	# damping it with nu and displacing zeta forwards and zetanew backwards with alpha
	zeta_spec += nu*alpha/2.0 * delta
	zetanew_spec += - nu*(1-alpha)/2.0 * delta

	return zeta_spec, zetanew_spec