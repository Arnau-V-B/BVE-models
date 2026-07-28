# Global non-divergent Barotropic Vorticity Equation (BVE) solver

> [!NOTE]
> This model was first a GitHub project on its one and it has been moved here together with the other two models [BVE-beta-channel-model](../BVE-beta-channel-model) and [BVE-global-linear-model](../BVE-global-linear-model). To see the old development history of this program please visit the original archived repository: [BVE-global-model](https://github.com/Arnau-V-B/BVE-global-model.git).

This program solves the non-divergent BVE on the sphere using spherical harmonics transforms. It is designed for global atmospheric dynamics simulations, such as 500 hPa flow evolution, with realistic initial conditions of single-level relative vorticity fields from reanalysis data (e.g., ERA5). The workflow of the code is based on [technical documentation](https://www.gfdl.noaa.gov/wp-content/uploads/files/user_files/pjp/barotropic.pdf) from the NOAA Geophysical Fluid Dynamics Laboratory (GFDL).

The model uses a spectral transform method with triangular truncation, supports both equally sampled (Driscoll–Healy) and Gauss–Legendre quadrature grids (see SHTOOLS [documentation](https://github.com/SHTOOLS/SHTOOLS.git)), and includes implicit hyperdiffusion for numerical stability. The time integration follows a leapfrog scheme with a Robert–Asselin–Williams (RAW) filter to control its associated computational mode and improve the accuracy up to third order (i.e. $$O(\Delta t^3)$$), and the time step is set to 30 min (but it can be lowered if numerical instability appears).

[`reanalysis/`](./reanalysis/) folder contains some examples of reanalysis relative vorticity fields from [ERA5 hourly data on pressure levels from 1940 to present](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview) in netCDF4 format to use as initial conditions to run the model. Nevertheless, the user may add any other `.nc` files as initial conditions in this folder.

## Mathematical formalism

### The linear and nonlinear BVE

For a barotropic atmosphere with constant height, the evolution of relative vorticity can be described pretty well with the non-divergent barotropic vorticity equation:

$$\frac{\partial \zeta}{\partial t} = - \vec{v}\cdot\vec{\nabla}(\zeta + f) \quad \longleftrightarrow \quad \frac{\partial}{\partial t}(\nabla^2\psi) = - J(\psi, \nabla^2\psi + f);$$

where $$\vec{v}$$ is the horizontal wind velocity, $$\zeta$$ is the relative vorticity (defined as $$\zeta = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$$), $\psi$ is the streamfuncion (defined as $$\vec{v} = \hat{k} \times \vec{\nabla}\psi$$ or $$\zeta = \nabla^2\psi$$), $$f = 2\Omega\sin(\varphi)$$ is the Coriolis parameter and $$J(A,B) = \frac{\partial A}{\partial x} \frac{\partial B}{\partial y} - \frac{\partial A}{\partial y}\frac{\partial B}{\partial x}$$ is the Jacobi operator.

### The spherical harmonics transforms

On the spherical Earth surface, fields are periodic in longitude $$\lambda$$ (i.e. azimuthal angle $$\phi$$) and enclosed in latitude $$\varphi$$ (i.e. copolar angle $$90^\circ-\theta$$), so they can be represented as series of spherical harmonics up to a given order $$l_{max}$$ (the truncation wavenumber):

$$\psi(\lambda,\varphi) = \sum_{l=0}^{l_{max}}\sum_{m=-l}^{l} \psi_{lm} P_{lm}(sin\varphi) e^{im\lambda};$$

where $$P_{lm}$$ is the associated Legendre polynomial of order $$m$$ and degree $$l$$. In this "spectral" space, the spatial derivatives ($$\frac{\partial}{\partial x} \equiv \frac{1}{R\cos(\varphi)}\frac{\partial}{\partial \lambda}$$ and $$\frac{\partial}{\partial y} \equiv \frac{1}{R}\frac{\partial}{\partial \varphi}$$) become analytic and can be solved through recursive relations such as:

$$\frac{\partial \hat{\psi}_{lm}}{\partial \lambda} = i m \hat{\psi}_{lm}$$

$$\cos(\varphi) \frac{\partial \hat{\psi}_{lm}}{\partial \varphi} = (l+2) \epsilon_{l+1,m} \hat{\psi}_{l+1,m} - (l-1) \epsilon_{l,m} \hat{\psi}_{l-1,m} \quad \mathrm{; where} \quad \epsilon_{lm} = \sqrt{\frac{l^2 - m^2}{4l^2 - 1}}$$

### The hyperdiffusion term

To prevent the simulation from being contaminated with aliasing from interactions between the highest modes (i.e. the smallest waves) resulting from the products between fields, an extra term has to be added to the BVE:

$$\frac{\partial \zeta}{\partial t} = - \vec{v}\cdot\vec{\nabla}(\zeta + f) - \eta \nabla^4 \zeta \quad \mathrm{; with} \quad \eta = \frac{R^4}{\tau (l_{max}(l_{max} + 1))^2};$$

where $$R$$ is the Earth radius, $$\tau$$ the decay rate of the target waves and $$l_{max}$$ the highest wavenumber. This is known as the hyperdiffusion term and only dissipates the response at the smallest unresolved scales, so that they disappear soon after they are formed and also transfer of eddy energy from large to small scales is partially simulated. By default, the decay rate is set to 3 hours, but it can be increased to for better resolution or decreased for even more smoothing.

### The Robert-Asselin-Williams (RAW) filter

Finally, to integrate in time, a first order Euler scheme is used at the first iteration:

$$\zeta_{i+1} = \zeta_i + \Delta t \ F(\zeta_i),$$

where $$F(\zeta_i)$$ is the RHS of the BVE. Then a seconf order leapfrog scheme is used modified with a RAW filter, which is composed of the following three steps:

$$\zeta_{i+1} = \tilde{\tilde{\zeta}}_{i-1} + 2\Delta t \ F(\tilde{\zeta}_i)$$

$$\tilde{\tilde{\zeta}}_i = \tilde{\zeta}_i + \frac{\nu \alpha}{2}(\zeta_{i+1} - 2\tilde{\zeta}_i + \tilde{\tilde{\zeta}}_{i-1})$$

$$\tilde{\zeta}_{i+1} = \zeta_{i+1} - \frac{\nu (1-\alpha)}{2}(\zeta_{i+1} - 2\tilde{\zeta}_i + \tilde{\tilde{\zeta}}_{i-1});$$

where $$i$$ denotes the time iteration, the tilde symbol shows the amount of filtering passes, $$\nu\in[0,1]$$ damps the computational mode and $$\alpha\in[0.5,1]$$ displaces the solutions to improve integration acuracy. The combination $$\nu=0.1$$ and $$\alpha=0.5$$ seem conserve kinetic energy, enstrophy and mean vorticity the most.

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

This program is completely open source and was made in colaboration with Universitat de Barcelona as part of a master's thesis.
