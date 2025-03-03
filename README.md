# Comparison of Discrete and Continuous Shape Derivatives

## Description
Modification of the helmholtz-x repository from Ekrem Ekici for the purpose of calculation of discrete shape derivatives on the simple duct, Rijke tube and Kornilov case.  
- compute continuous and discrete shape derivatives
- build figures
### Folder Architecture
- /helmholtz-x: contains a slightly modified version of the HelmholtzX toolbox from Ekrem Ekici
- /RijkeTube: 2D Rijke tube geometry test case
- /KornilovCase: Kornilov test case geometry test case
#### Subfolders of Test Cases
use paraview to analyse .xdmf files within:
- /Meshes: mesh files
- /Results: direct and adjoint eigenvectors and eigenvalues as .txt file
- /InputFunctions: h,w,T,rho,c functions and Free-Form-Deformation field Vffd
- /Figures: building datasets and plotting figures and results
- /FTFMatrices: data for Kornilov Flame Transfer Function in state space format
### Files
the `environ.yml` is used to set up conda environment as explained below
the main class is `test_case.py`, which builds and computes the requested test case  
the standard physical problem parameters are specified in `*params.py`  
#### HelmholtzX Toolbox
overview of the helmholtzX utilities:
- `acoustic_matrices.py`: assemble matrices A,B,C for discrete Helmholtz Equation: A+wB+w**2C=D(w)
- `distribute_params.py`: distribute parameters onto the mesh file
- `dolfinx_utils.py`: utils for assembly of flame matrix
- `eigensolvers.py`: contains Newton solver for eigenvalue problem using eps solver
- `eigenvectors.py`: normalizations for eigenvectors
- `flame_matrices.py`: assemble matrix D(w)
- `flame_transfer_function.py`: defines FTF as n-tau model or state-space model
- `io_utils.py`: read and write functions
- `parameter_utils.py`: convert temperature, sound speed and gamma functions
- `petsc4py_utils.py`: vector and matrice computation methods
- `shape_derivatives.py`: functions to compute continuous shape derivative
- `solver_utils.py`: logging utilities


## Set Up
set up the code on your machine:  
quick install the suitable conda environment using the `environ.yml` file with the command:  
`~$ conda env create -f environ.yml`  
then activate with:  
`~$ conda activate helmholtzx-env`  
create subfolders called "/Meshes", "/Results", "/InputFunctions" within each testcase folder:  
`~$ mkdir KornilovCase/{Meshes,Results,InputFunctions} RijkeTube/{Meshes,Results,InputFunctions}`  
also add the helmholtz-x-TFD directory to python path by adding this command  
`export PYTHONPATH="${PYTHONPATH}:/.../helmholtz-x-TFD"`
in your `~/.bashrc` file, where you replace `/...` with the path to your directory


## Usage
to compute derivatives for a testcase head to the corresponding folder:  
`~$ cd KornilovCase` or `~$ cd RijkeTube`  
for shape derivative using continuous formula from Ekici dissertation:  
`~$ python3 continuous_shape_derivative_*`  
for shape derivative using discrete formula derived by Varillon run:  
`~$ python3 discrete_shape_derivative_*`  
adjust parameters in the `*params.py` files if needed or directly when building the object from class in `test_case.py`  
### Examine Input Functions
to examine the input functions h,w,T,rho,c and the free form deformation field, run the function `write_input_function()` and examine .xdmf files using paraview  
### Figures
to reproduce the figures from the thesis paper this method can be used:  
- `build_data_fig*.py`: computes the results and saves them as `data_fig*.txt` file
- `plot_fig*.py`: plots the figure using python matplotlib from data in `data_fig*.txt` file


