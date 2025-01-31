Modification of the helmholtz-x repo from Ekrem Ekici for the purpose of calculation of numeric shape derivatives on the Kornilov case

# Folder Archictecture
## Tools and Data
- /helmholtz-x: contains a slightly modified version of the HelmholtzX toolbox from Ekrem Ekici
- /FTFMatrices: data for Kornilov Flame Transfer Function in state space format
## Test Cases (prefix)
- /RijkeTube: 2D Rijke tube geometry (r)
- /Duct: rectangular 2D duct geometry (d)
- /KornilovCase: Kornilov test case geometry (k)

# Programs
every test case has 3 programs:  
- ()main_continuous: calculates shape derivative using continuous formula from Ekici dissertation
- ()main_discrete: calculates shape derivative using discrete formula derived by Gregoire Varillon
- ()params: defines up input functions and parameters for the testcase

# Set Up
to set up the code on your machine:
- install correct conda environment
- create subfolders called "Meshes", "Results", "InputFunctions" within each testcase folder
