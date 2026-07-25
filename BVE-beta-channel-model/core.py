"""
This module contains all the functions and routines required to run the BVE simulation
in the beta-plane aproximation.
This script has to be placed in the same directory as 'main.py'
"""


import numpy as np
import scipy as sp


def laplace(func, args):
	"""
	This function returns the 2D laplacian of a given discrete function field: Nabla^2 func.

	For the interior points centered differences are used and for the boundaries forward
	and backward differences are considered accordingly.

	PARAMETERS:
	(input) -->
		func   : 2D array (ny, nx) of the function field
		args   : extra general arguments
	(internal) -->
		dx, dy : grid spacing in x and y-direction
	(output) -->
		lap    : 2D array (ny, nx) with the laplacian of the function field
	"""

	dx, dy = args['step'][0], args['step'][1]

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


def poisson_fft(zeta, args):
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
		args : extra general arguments
	(internal) -->
		kx   : 1D array (nx/2 + 1) with Fourier spectral grid wavenumbers in x-direction
		dy   : grid spacing in y-direction
	(output) -->
		psi  : 2D array (ny, nx) of streamfunction field
	"""

	kx = args['k'][0]
	dy = args['step'][1]

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


def jacobian(psi, zeta, U, Z_y, args):
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
		U    : 2D array (ny, nx) of mean zonal velocity field
		Z_y  : 2D array (ny, nx) of meridional derivative of mean relative vorticity field
		args : extra general arguments
	(internal) -->
		kx   : 1D array (nx/2 + 1) with Fourier spectral grid wavenumbers in x-direction
		beta : Rossby parameter df/dy
	(output) -->
		RHS  : 2D array (ny, nx) with the RHS term of the BVE
	"""

	kx = args['k'][0]
	beta = args['beta']

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


def jacobian_with_topography(psi, zeta, U, Z_y, h, args):
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
		U    : 2D array (ny, nx) of mean zonal velocity field
		Z_y  : 2D array (ny, nx) of meridional derivative of mean relative vorticity field
		h    : 2D array (ny, nx) of topography function field
		args : extra general arguments
	(internal) -->
		kx   : 1D array (nx/2 + 1) with Fourier spectral grid wavenumbers in x-direction
		beta : Rossby parameter df/dy
		f0   : Coriolis parameter at mid-latitudes
		H    : atmospheric scale height
	(output) -->
		RHS  : 2D array (ny, nx) with the RHS term of the BVE with topography
	"""

	kx = args['k'][0]
	beta = args['beta']
	f0 = args['f0']
	H = args['H']

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


def find_vel(psi, args):
	"""
	This function obtains the horizontal velocity field from a given 2D streamfunction field:
	--> u = - d(psi)/dy ; v = d(psi)/dx

	To do so, it transforms all the fields into the spectral space using the Fast Fourier Transform
	(FFT) method and solves the problem as follows:
	--> u_hat = -i * ky * psi_hat ; v_hat = i * kx * psi_hat
	
	PARAMETERS:
	(input) -->
		psi      : 2D array (ny, nx) of streamfunction field
		args     : extra general arguments
	(internal) -->
		kxs, kys : 2D arrays (nx/2 + 1, nx/2 + 1) with Fourier spectral grid wavenumbers in x and y-direction
	(output) -->
		u, v     : 2D arrays (ny, nx) of the zonal and meridional velocity fields
	"""

	kxs, kys = args['ks'][0], args['ks'][1]

	# We transform the streamfunction field to the spectral space
	psi_hat = sp.fft.rfft2(psi)

	# We compute the transformed zonal velocity field 'u_hat'
	u_hat = -1j * kys * psi_hat

	# We compute the transformed meridional velocity field 'v_hat'
	v_hat = 1j * kxs * psi_hat

	# In the end we recover the real velocity fields by applying an inverse Fourier transform
	u = sp.fft.irfft2(u_hat)
	v = sp.fft.irfft2(v_hat)

	return u,v


def compute_conserved_values(u, v, zeta, args):
	"""
	This function computes the three main conserved values in the non-divergent BVE simulation:
	--> kinetic energy: KE = mean(1/2 * (u^2 + v^2))
		enstrophy     : ε = 1/2 * (zeta + f)^2
		mean vorticity: ζ = mean(zeta)
		
	PARAMETERS:
	(input) -->
		u         : 2D array (ny, nx) of zonal velocity field
		v         : 2D array (ny, nx) of meridional velocity field
		zeta      : 2D array (ny, nx) of relative vorticity field
		args      : extra general arguments
	(internal) -->
		f         : 2D array (ny, nx) of Coriolis parameter field
	(output) -->
		energy    : 2D array (ny, nx) of kinetic energy field
		enstrophy : 2D array (ny, nx) of enstrophy field
		zetamean  : 2D array (ny, nx) of mean relative vorticity field
	"""
	
	f = args['fs']

	energy = np.mean(0.5 * (u**2 + v**2))
	enstrophy = np.sum(0.5 * (zeta + f)**2)
	zetamean = np.mean(zeta)
	
	return energy, enstrophy, zetamean


def leapfrog_raw_step(zeta, zetaold, rhs, dt, nu, alpha):
	"""
	This function performs a leapfrog time integration step applying a Robert-Asselin-Williams
	(RAW) filter with a damping coefficient ν and a displacement coefficient α.

	The RAW filter consists on the following three steps:
	--> ζ(t+dt) = ζ''(t-dt) + 2dt * RHS'(t)
		ζ''(t) = ζ'(t) + να/2 * (ζ(t+dt) - 2*ζ'(t) + ζ''(t-dt))
		ζ'(t+dt) = ζ(t+dt) - ν(1-α)/2 * (ζ(t+dt) - 2*ζ'(t) + ζ''(t-dt))
	where the tildes show the amount of filtering passes.

	PARAMETERS:
	(input) -->
		zeta    : 2D array (ny, nx) of the current vorticity field
		zetaold : 2D array (ny, nx) of the old vorticity field
		rhs     : 2D array (ny, nx) of the RHS of the BVE equation
		dt      : time step of the integration
		nu      : damping coefficient of the RAW filter
		alpha   : displacement coefficient of the RAW filter
	(internal) -->
		delta   : 2D array (ny, nx) of the difference between the new, current and old vorticity fields
	(output) -->
		zeta    : 2D array (ny, nx) of the current vorticity field (modified)
		zetanew : 2D array (ny, nx) of the new vorticity field
	"""

	# First we perform a leapfrog step
	zetanew = zetaold + 2.0*dt*rhs

	# After the time step, we apply a Robert-Asselin-Williams (RAW) filter to reduce the
	# computational mode amplitude and reaching up to third order precision
	# We compute the correcting term (a centered difference)
	delta = zetanew - 2.0*zeta + zetaold
	
	# And then we apply this correction to the current and new vorticity fields with a RAW filter
	# damping it with nu and displacing zeta forwards and zetanew backwards with alpha
	zeta += nu*alpha/2.0 * delta
	zetanew += - nu*(1-alpha)/2.0 * delta

	return zeta, zetanew