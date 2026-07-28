# Non-divergent Barotropic Vorticity Equation `Python` solvers

This repository contains a collection of numerical models developed integrally in `Python` that solve the non-divergent BVE in different geometries and scenarios:

$$\frac{\partial \zeta}{\partial t} = \vec{v}\cdot\vec{\nabla}(\zeta + f)$$

where $$\vec{v} = (u,v)$$ is the horizontal wind velocity, $$\zeta$$ is the relative vorticity (defined as $$\zeta = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$$), $$f = 2\Omega\sin\varphi$$ is the Coriolis parameter (being $$\Omega$$ the Earth's angular rotation speed and $$\varphi$$ the latitude) and $$\vec{\nabla} = \left(\frac{\partial}{\partial x}, \frac{\partial}{\partial y}\right)$$ the horizontal gradient operator.

In this approximation, the atmosphere is treated as a barotropic incompressible fluid with constant depth and only advection and propagation by the flow in one single level is contemplated. Therefore, there are no sources or sinks of vorticity other than the state of circulation in the simulated layer.

## Models

(*in increasing complexity...*)

### [BVE-beta-channel-model](BVE-beta-channel-model/README.md)

Solves the non-divergent BVE linearized over a mean zonal flow $$\overline{U}$$ in cartesian coordinates using the $$\beta$$-plane aproximation (i.e., $$f \approx f_0 + \beta y$$ with $$f_0,\beta = ctt.$$). It is useful for analysing the basics of Rossby wave propagation and dispersion with a direct comparison to analytic mathematical expressions.

### [BVE-global-linear-model](BVE-global-linear-model/README.md)

Solves the non-divergent BVE globally in spherical coordinates (i.e, $$f,\beta \neq ctt.$$). It can use both the equation linearized over a mean flow $$\overline{U}$$ or the full equation including the nonlinear terms. The model is initialized with an arbitrary streamfunction perturbation and mean zonal flow. It is very useful to simulate Rossby wave propagation and dispersion in more complex ideal scenarios and compare the results obtained by including or not the nonlinear effects in the equations.

### [BVE-global-model](BVE-global-model/README.md)

Solves the non-divergent BVE globally in spherical coordinates for any real relative vorticity field obtained from ERA5 reanalysis. Nevertheless, it could be easily adapted to accept fields from other reanalysis or even self-crafted ones. It is useful to run qualitative predictions of real atmospheric circulation at mid to high tropospheric levels where the barotropic aproximation is best fulfilled.

## Programs general structure

The execution of all programs relies on five files that need to be located in the same directory:
1. `config.py`. Here all the initial conditions and parameters of the simulation are specified. This script can be easily modified to accomodate it to the user needs.
2. `core.py`. It contains all the functions and routines needed to run the BVE simulation. Each of these functions is completely documented so the user can understand what each of them does.
3. `saving.py`. This small script contains the routines responsible of saving the simulation results into NetCDF4 files.
4. `plotting.py`. It contains the routines with which the figures are generated at the end of the simulation. It can be disabled by setting the flag `PLOT` as **False** in `config.py`.
5. `main.py`. This is the main program. It contains the execution pipeline that imports all the above scripts and runs the simulation. It is also completely documented so the user can follow the entire workflow.

Once the initial conditions are properly set in `config.py`, to run the simulation the script `main.py` has to be executed. For example, on Windows this can be done by typing in the following line in the CMD terminal inside the model directory:
```
python main.py
```

During the execution an `output/` folder is created in the same directory where all the simulation output is saved. In it a file named `params_exp_().txt` contains a copy of the `config.py` file with the initial conditions used in the simulation, and two more folders are created `data/` and `figures/`. The former contains two netCDF files with the register of conserved values (i.e. knietic energy, enstrophy and mean vorticity) in `conserved_values_exp_().nc` and the streamfunction and relative vorticity fields evolution in `fields_evolution_exp_().nc`. The later, contains a figure showing the evolution of the three conserved values (`conserved_values_exp_().png`), two GIFs with the evolution of the fields (`.gif`) and a folder `frames/` with the single frames used to create both of them.

## Requirements

All the code has been written in Python v3.13.7 on a Windows 11 laptop with both clarity and efficienty in mind, so there should be no problem running it in any computer hardware as long as it has Python installed. The Python packages required for all the models are specified in the `requirements.txt` file so they can be automatically installed through `pip` in a fresh virtual environment:

1. First, create a new virtual environment with Python `venv` in the directory that contains all the models. For Windows users this can be done by typing the following line in the CMD or PowerShell terminal:
```
python -m venv my_env
```
Where `my_env` is the name you want to give to your environment.

2. Then, activate the new environment. Windows users need to type in the same terminal:
```
my_env\Scripts\activate
```

3. Finally, install all the required packages automatically using `pip` and providing it with the `requirements.txt` file:
```
pip install -r requirements.txt
```

In case the user wants to install the packages manually, each of the models `README.md` file shows a list of the required packages for each program.

> [!WARNING]
> All programs have only been tested with Windows 10 and 11, so MacOS and Linux users could possibly experience some problems running them.

## License

This project is completely open source and was made in colaboration with Universitat de Barcelona as part of a master's thesis.
