# Non-divergent Barotropic Vorticity Equation `Python` solvers

This repository contains a collection of numerical models developed integrally in `Python` that solve the non-divergent BVE in different geometries and scenarios:

$$\frac{\partial \zeta}{\partial t} = \vec{v}\cdot\vec{\nabla}(\zeta + f)$$

where $$\vec{v} = (u,v)$$ is the horizontal wind velocity, $$\zeta$$ is the relative vorticity (defined as $$\zeta = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$$), $$f = 2\Omega\sin\theta$$ is the Coriolis parameter (being $$\Omega$$ the Earth's angular rotation speed and $$\theta$$ the latitude) and $$\vec{\nabla} = \left(\frac{\partial}{\partial x}, \frac{\partial}{\partial y}\right)$$ the horizontal gradient operator.

In this approximation, the atmosphere is treated as a barotropic incompressible fluid with constant depth and only advection and propagation by the flow in one single level is contemplated. Therefore, there are no sources or sinks of vorticity other than the state of circulation in the simulated layer.

## Models

(in increasing complexity)

### [BVE-beta-channel-model](BVE-beta-channel-model/README.md)

Contains `main_BVE_beta.py`, which solves the non-divergent BVE linearized over a mean zonal flow $$\overline{U}$$ in cartesian coordinates using the $$\beta$$-plane aproximation (i.e., $$f \approx f_0 + \beta y$$ with $$f_0,\beta = ctt.$$). It is useful for analysing Rossby wave dispersion with analytic mathematical expressions.

### [BVE-global-linear-model](BVE-global-linear-model/README.md)

Contains two programs: `main_BVE_glob_linear.py` and `main_BVE_glob_nonlinear.py`. The former solves the non-divergent BVE linearized over a mean flow $$\overline{U}$$ globally in spherical coordinates (i.e, $$f,\beta \neq ctt.$$), while the later solves the full equation including the nonlinear terms. Both are initialized with an arbitrary streamfunction perturbation and mean zonal flow. This model is very useful to simulate Rossby wave propagation and dispersion in more complex ideal scenarios and compare the results obtained by including or not the nonlinear effects in the equations.

### [BVE-global-model](BVE-global-model/README.md)

Contains `main_BVE_glob.py`, which solves the non-divergent BVE globally in spherical coordinates for any real relative vorticity field obtained from ERA5 reanalysis. Nevertheless, it could be easily adapted to accept fields from other reanalysis or even self-crafted ones. It is useful to run qualitative predictions of real atmospheric circulation at mid-high tropospheric levels where the barotropic aproximation is best fulfilled.

## Programs general structure

The execution of all programs relies on two files that need to be in the same directory:
1. `config.py`. Here all the initial conditions and parameters of the simulation are specified. The script can be easily modified to accomodate it to the user needs.
2. `main_(model).py`. This is the main program. It contains all the internal functions, the execution pipeline and the saving and plotting routines of the corresponding model. The final plotting routine can be disabled to save time if more customized plots are desired.

Once the initial conditions are properly set in `config.py`, to run the simulation script `main_(model).py` has to be executed. For example, on Windows 11 this can be done by typing in the following line in the CMD terminal:
```
python main_(model).py
```

During the execution an `output/` folder is created in the same directory where all the simulation output is saved. In it a file named `params_exp_().txt` contains a copy of the `config.py` file with the initial conditions used in the simulation, and two more folders are created `data/` and `figures/`. The former contains two netCDF files with the register of conserved values (i.e. knietic energy, enstrophy and mean vorticity) in `conserved_values_exp_().nc` and the streamfunction and relative vorticity fields evolution in `fields_evolution_exp_().nc`. The later, contains a figure showing the evolution of the three conserved values (`conserved_values_exp_().png`), two GIFs with the evolution of the fields (`.gif`) and the single frames used for both of them.

## Requirements

All the code has been written in Python v3.13.7 on a Windows 11 laptop with both clarity and efficienty in mind, so there should be no problem running it in any computer hardware as long as it has Python installed. The Python packages required for each of the models are specified on their corresponding `README.md` files.

> [!WARNING]
> All programs have only been tested with Windows 11, so Windows 10, MacOS and Linux users could possibly experience some problems running them.

## License

This project is completely open source and was made in colaboration with Universitat de Barcelona as part of a master's degree final thesis.