# Comparison of Discrete and Continuous Shape Derivatives

## Description
Modification of the [helmholtz-x](https://github.com/ekremekc/helmholtz-x/tree/main) repository for the purpose of computing of discrete shape derivatives on the acoustic duct, Rijke tube and Kornilov case.  
- compute continuous, discrete and analytic shape derivatives
- build figures
### Folder Architecture
- /helmholtz_x: contains a slightly modified version of the HelmholtzX toolbox
- /RijkeTube: 2D Rijke tube geometry test case
- /KornilovCase: Kornilov test case geometry test case
- /Figures: building datasets and plotting figures and results
#### Subfolders of Test Cases
use paraview to analyse .xdmf files within:
- /Meshes: contains mesh files
- /Results: direct and adjoint eigenvectors and eigenvalues as .txt file
- /InputFunctions: h,w,T,rho,c functions and Free-Form-Deformation field Vffd
- /FTFMatrices: data for Kornilov Flame Transfer Function in state space format
### Files
the `environ.yml` is used to set up conda environment as explained below  
the `requirements.txt` file contains the required dependencies for the docker as explained below  
the main class is `test_case.py`, which builds and computes the requested test case  
the standard physical problem parameters are specified in `*params.py`  
#### HelmholtzX Toolbox
overview of the helmholtzX utilities:
- `acoustic_matrices.py`: assemble matrices A,B,C for discrete Helmholtz Equation: A+wB+w**2C=D(w)
- `distribute_params.py`: functions to distribute parameters onto the mesh file
- `dolfinx_utils.py`: utils for assembly of flame matrix
- `eigensolvers.py`: contains Newton solver for eigenvalue problem using eps solver
- `eigenvectors.py`: normalizations for eigenvectors
- `flame_matrices.py`: assemble matrix D(w)
- `flame_transfer_function.py`: defines FTF as n-tau model or state-space model
- `io_utils.py`: read and write functions
- `parameter_utils.py`: convert temperature, sound speed and gamma functions
- `petsc4py_utils.py`: vector and matrix computation methods
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
to test computing derivatives for a testcase, head to the corresponding folder:  
`~$ cd KornilovCase` or `~$ cd RijkeTube`  
for shape derivative using continuous formula:  
`~$ python3 continuous_shape_derivative_*`  
for shape derivative using discrete formula:  
`~$ python3 discrete_shape_derivative_*`  
adjust parameters in the `*params.py` files if needed or directly when building the object from class `test_case.py`  
### Examine Input Functions
to examine the input functions h,w,T,rho,c and the free form deformation field, run the function `write_input_function()` and examine .xdmf files using paraview  
### Figures
to access, modify and print figures from the `data_fig*.txt` files with the jupyter notebooks `Fig*.ipynb` use this link to create an isolated browser environment:  
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kspc0/helmholtz-x-TFD/HEAD)
 (loading might take a while)  
changes made in this environment will not apply to the repository and will get lost when closing the browser session  

also, to reproduce the figure data from scratch, you can run `build_data_fig*.py`  
it computes the test case and save the data as `data_fig*` files
