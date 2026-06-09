# Global non-divergent Barotropic Vorticity Equation (BVE) $$\beta$$-channel solver

This model solves the non-divergent BVE on the $$\beta$$-plane aproximation in cartesian coordinates using Fast Fourier Transforms (FFT) in x and a centered differences scheme in y. It is designed to simulate Rossby wave propagation and dispersion in the most simple aproximation (a $$\beta$$-channel) so that the results can be directly compared to the analytic solutions that appear after solving the system. It also includes the option to add an very simple extra forcing term to the equation that takes into account the effect of topography through changes in depth.

The model solves the equation in a latitudinally enclosed domain, with periodic boundary conditions in x and Dirichlet in y (in particular, no flow conditions: $$\psi_{y=0} = \psi_{y=N_y} = 0$$), and doesn't include any diffusion terms for numerical stability. The time integration follows a leapfrog scheme with a Robert–Asselin–Williams (RAW) filter to control its associated computational mode and improve the accuracy to third order (i.e. $$O(\Delta t^3)$$), and the time step is set to 30 min (but it can be lowered if numerical instability appears).

## Mathematical formalism

### The $$beta$$-channel approximation

In a latitudinally restricted domain, the sphericity of the Earth's can have negligible effects so the surface can be considered as locally flat in pretty good approximation. In this case, the Coriolis parameter $$f$$ becomes constant with latitude, so the planetary vorticity advection effect disappears completely and Rossby waves as they are known can no longer exist. In order to avoid this, a more accurate approximation can be made by expanding the Coriolis parameter in a Taylor series around the central latitude $\theta_0$ up to the first order (i.e., linearly):

$$f \approx f_0 + \beta y \quad \mathrm{; with} \quad \beta = \left.\frac{\partial f}{\partial y}\right|_{\theta_0} = \frac{2\Omega}{R}\cos\theta_0 = ctt;$$

where $$\Omega$$ is the angular rotation speed of the Earth and $$R$$ its radius. In this case, given a background flow with the following properties: $$u = \overline{U} + u' \ ; \ v = v'$$ (where $$\overline{}$$ denotes the mean state and ' the perturbation), the non-divergent BVE can be linearized as follows:

$$\frac{\partial \zeta'}{\partial t} \approx - \overline{U} \frac{\partial \zeta'}{\partial x} - v'\left(\beta + \frac{\partial \overline{\zeta}}{\partial y}\right) \quad \mathrm{; where} \quad \overline{\zeta} = - \frac{\partial \overline{U}}{\partial y};$$

which is the equation solved in `main_BVE_beta.py`.

### The Fast Fourier Transforms (FFT)

Given a field $$\psi$$ that is continuous and periodic in the x direction on an interval $$0\leq x \leq L$$, it can be decomposed into an infinite series of sinusoidal functions known as Fourier series:

$$\psi(x) = \sum_{k=-\inf}^{\inf} \hat{\psi}_k e^{i2\pi k x/L} \quad \mathrm{; with} \quad \hat{\psi}_k = \frac{1}{L}\int_0^L \psi(x)e^{-i2\pi k x/L}dx$$

By definition, this operations can be very computationally slow, but thanks to algorithms such as the Fast Fourier Transform (FFT), the system of equations can be factorized and this expansion process can be done very efficiently. The advantage of representing the fields as series of waves is that in the Fourier space the derivatives become analytic and can be solved very easily:

$$\frac{\partial \hat{\psi}_k}{\partial x} = ik \hat{\psi}_k;$$

and the solution obtained is exact, unlike the one obtained with finite differences schemes.

### The time integration

Finally, to integrate in time, a first order Euler scheme is used at the first iteration:

$$\zeta_{i+1} = \zeta_i + \Delta t RHS_i,$$

and then a third order scheme based on a leapfrog with RAW filter is used for the following iterations:

$$\zeta_{i+1} = \zeta_{i-1} + 2\Delta t RHS_i$$

$$\zeta_i = \zeta_i + \frac{\nu \alpha}{2}(\zeta_{i+1} - 2\zeta_i + \zeta_{i-1})$$

$$\zeta_{i+1} = \zeta_{i+1} - \frac{\nu (1-\alpha)}{2}(\zeta_{i+1} - 2\zeta_i + \zeta_{i-1});$$

where $$\nu=0.1$$ and $$\alpha=0.5$$ seem to conserve kinetic energy, enstrophy and mean vorticity the most.

## Requirements

All the code has been written in Python v3.13.7 on a Windows 11 laptop with both clarity and efficienty in mind, so there should be no problem running it in any computer hardware as long as it has Python installed. In the general [`README.md`](../README.md) file, there is a [`requirements.txt`](../requirements.txt) file containing the packages needed for all models ready to be automatically installed with `pip`. However, if the user wants to install them manually, here are the Python packages that need to be installed for this specific model:
- [numpy (v2.4.6)](https://pypi.org/project/numpy/)
- [scipy (v1.17.1)](https://pypi.org/project/scipy/)
- [xarray (v2026.4.0)](https://docs.xarray.dev/en/stable/)
- [netcdf4 (v1.7.4)](https://pypi.org/project/netCDF4/)
- [matplotlib (v3.10.9)](https://pypi.org/project/matplotlib/)  (*not necessary if plotting is disabled*)
- [imageio (v2.37.3)](https://pypi.org/project/ImageIO/)    (*not necessary if plotting is disabled*)

> [!NOTE]
> The versions shown are only orientative, as newer or older ones will most likely work just fine.

## License

This program is completely open source and was made in colaboration with Universitat de Barcelona as part of a master's degree final thesis.
