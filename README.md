Modification of the helmholtz-x repo from Ekrem Ekici for the purpose of calculation of numeric shape derivatives on the Kornilov case

# Folder Architecture
## Tools and Data
- /helmholtz-x: contains a slightly modified version of the HelmholtzX toolbox from Ekrem Ekici
- /FTFMatrices: data for Kornilov Flame Transfer Function in state space format
## Test Cases
- /RijkeTube: 2D Rijke tube geometry (running)
- /KornilovCase: Kornilov test case geometry (still in development)


# Rijke Tube Programs
in order to run a shape derivative computation, execute the desired program and parameters in:
- continuous_shape_derivative: calculates shape derivative using continuous formula from Ekici dissertation
- discrete_shape_derivative: calculates shape derivative using discrete formula derived by Gregoire Varillon
-> these create a test case object from the class TestCase and execute computation using the parameters from rparams


# Set Up
to set up the code on your machine:
- install correct conda environment
- create subfolders called "Meshes", "Results", "InputFunctions" within each testcase folder
