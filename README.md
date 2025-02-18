
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
the main class is `test_case.py`, which builds and computes the requested test case  
the standard physical problem parameters are specified in `*params.py`  
these are distributed onto the meshfile using the `distribute_params.py` functions  

## Set Up
set up the code on your machine:  
quick install the suitable conda environment using the `environ.yml` file with the command:  
`~$ conda env create -f environ.yml`  
then activate with:  
`~$ conda activate helmholtzx-env`  
create subfolders called "/Meshes", "/Results", "/InputFunctions" within each testcase folder:  
`~$ mkdir KornilovCase/{Meshes,Results,InputFunctions} RijkeTube/{Meshes,Results,InputFunctions}`  

## Usage
to compute derivatives for a testcase head to the corresponding folder:  
`~$ cd KornilovCase` or `~$ cd RijkeTube`  
for shape derivative using continuous formula from Dr. Ekrem Ekici dissertation:  
`~$ python3 continuous_shape_derivative_*`  
for shape derivative using discrete formula derived by Dr. Gregoire Varillon run:  
`~$ python3 discrete_shape_derivative_*`  
adjust parameters in the `*params.py` files if needed or directly when building the object from class in `test_case.py`  
### Examine Input Functions
run the function `write_input_function()` and examine using paraview  
### Figures
figures are build using:  
- `build_data_fig*.py`: computes the results and saves them as `data_fig*.txt` file
- `plot_fig*.py`: plots the figure using matplotlib from data in `data_fig*.txt` file


