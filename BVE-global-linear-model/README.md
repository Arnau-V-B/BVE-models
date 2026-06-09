# Global non-divergent Barotropic Vorticity Equation (BVE) linear and nonlinear solvers

This subproject contains two programs that solve the non-divergent BVE on the sphere using spherical harmonics transforms. `main_BVE_glob_linear.py` linearizes the equation over a mean zonal flow $$\overline{U}$$ (i.e., only mean flow - perturbation interactions are simulated), while `main_BVE_glob_nonlinear.py` includes all the terms (i.e., both mean flow - perturbation and perturbation - perturbation interactions are simulated). They are designed to simulate simple atmospheric dynamics under multiple idealized circulation scenarios and compare the effects of including or not the nonlinearity of the equations, so they can serve as toy models for quick testing. The workflow of the code is based on [technical documentation](https://www.gfdl.noaa.gov/wp-content/uploads/files/user_files/pjp/barotropic.pdf) from the NOAA Geophysical Fluid Dynamics Laboratory (GFDL).

Both models use a spectral transform method with triangular truncation, support both equally sampled (Driscoll–Healy) and Gauss–Legendre quadrature grids (see SHTOOLS [documentation](https://github.com/SHTOOLS/SHTOOLS.git)), and include implicit hyperdiffusion for numerical stability. The time integration follows a leapfrog scheme with a Robert–Asselin–Williams (RAW) filter to control its associated computational mode and improve the accuracy to third order (i.e. $$O(\Delta t^3)$$), and the time step is set to 30 min (but it can be lowered if numerical instability appears).

## Mathematical formalism

### The linear and nonlinear BVE

Given a background flow with the following properties: $$u = \overline{U} + u' \ ; \ v = v'$$ (where $$\overline{}$$ denotes the mean state and ' the perturbation), the non-divergent BVE can be linearized as follows:

$$\frac{\partial \zeta'}{\partial t} \approx - \overline{U} \frac{\partial \zeta'}{\partial x} - v'\left(\beta + \frac{\partial \overline{\zeta}}{\partial y}\right) \quad \mathrm{; where} \quad \overline{\zeta} = - \frac{\partial \overline{U}}{\partial y};$$

which is the equation solved in `main_BVE_glob_linear.py`. However, if all crossed products are considered, the non-divergent BVE becomes:

$$\frac{\partial \zeta'}{\partial t} \approx - \overline{U} \frac{\partial \zeta'}{\partial x} - v'\left(\beta + \frac{\partial \overline{\zeta}}{\partial y}\right) - \vec{v}'\cdot\vec{\nabla}\zeta';$$

where the new term that appears is nonlinear in the sense that there are products between perturbations. This is the equation solved in `main_BVE_glob_nonlinear.py`.

### The spherical harmonics transforms

On the spherical Earth surface, fields are periodic in longitude $\lambda$ (i.e. azimuthal angle $\phi$) and enclosed in latitude $\theta$ (i.e. copolar angle $-\Theta$), so they can be represented as series of spherical harmonics up to a given order $l_{max}$ (the truncation wavenumber):

$$\psi(\lambda,\theta) = \sum_{l=0}^{l_{max}}\sum_{m=-l}^{l} \psi_{lm} P_{lm}(sin\theta) e^{im\lambda};$$

where $$P_{lm}$$ is the associated Legendre polynomial of order $$m$$ and degree $$l$$. In this "spectral" space, the spatial derivatives ($$\frac{\partial}{\partial x} \equiv \frac{1}{R\cos(\theta)}\frac{\partial}{\partial \lambda}$$ and $$\frac{\partial}{\partial y} \equiv \frac{1}{R}\frac{\partial}{\partial \theta}$$) become analytic and can be solved through recursive relations such as:

$$\frac{\partial \hat{\psi}_{lm}}{\partial \lambda} = i m \hat{\psi}_{lm}$$

$$\cos(\theta) \frac{\partial \hat{\psi}_{lm}}{\partial \theta} = (l+2) \epsilon_{l+1,m} \hat{\psi}_{l+1,m} - (l-1) \epsilon_{l,m} \hat{\psi}_{l-1,m} \quad \mathrm{; where} \quad \epsilon_{lm} = \sqrt{\frac{l^2 - m^2}{4l^2 - 1}}$$

### The hyperdiffusion term

To prevent the simulation from being contaminated with aliasing of the highest modes (i.e. the smallest waves) resulting from the products between fields, an extra term has to be added to the BVE:

$$\frac{\partial \zeta}{\partial t} = - \vec{v}\cdot\vec{\nabla}(\zeta + f) - \eta \nabla^4 \zeta \quad \mathrm{; with} \quad \eta = \frac{R^4}{\tau (l_{max}(l_{max} + 1))^2};$$

where $$R$$ is the Earth radius, $$\tau$$ the decay rate of the target waves and $$l_{max}$$ the highest wavenumber. This is known as the hyperdiffusion term and only dissipates the response at the smallest unresolved scales, so that they disappear soon after they are formed and also transfer of eddy energy from large to small scales is partially simulated. By default, the decay rate is set to 3 hours, but it can be increased to for better resolution or decreased for even more smoothing.

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
- [pyshtools (v4.14.1)](https://pypi.org/project/pyshtools/)
- [matplotlib (v3.10.9)](https://pypi.org/project/matplotlib/)  (*not necessary if plotting is disabled*)
- [imageio (v2.37.3)](https://pypi.org/project/ImageIO/)    (*not necessary if plotting is disabled*)

> [!NOTE]
> The versions shown are only orientative, as newer or older ones will most likely work just fine.

## License

This program is completely open source and was made in colaboration with Universitat de Barcelona as part of a master's degree final thesis.
