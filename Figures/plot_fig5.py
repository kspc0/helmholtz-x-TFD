import matplotlib.pyplot as plt
import numpy as np
import os

# set path to data
path = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(path, 'data_fig5.txt')

# read file
with open(file, 'r') as f:
    lines = f.readlines()

# create empty lists to store the data
duct = []
eigenvalues = []
continuous = []
discrete = []

# read the data from the file
for line in lines[1:]: # skip the first line
    duc, eig, con, dis = line.strip().split(',')
    duct.append(round(float(duc),1))
    eigenvalues.append(complex(eig)/2/np.pi)
    continuous.append(complex(con))
    discrete.append(complex(dis))

# transform the lists into numpy arrays to perform operations
duct = np.array(duct)
eigenvalues = np.array(eigenvalues)
continuous = np.array(continuous)
discrete = np.array(discrete)

# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(20, 6))

# Plot eigenvalues on the complex plane
ax.scatter(eigenvalues.real, eigenvalues.imag, s=100 ,color='blue', label='Eigenvalues')
# Add labels to each point
for i, txt in enumerate(duct):
    ax.annotate(f'{txt}', (eigenvalues[i].real, eigenvalues[i].imag), fontsize=22,
                 textcoords="offset points", xytext=(0,15), ha='center')

scale = 18
offset = 0.01
# Add arrows for continuous and discrete shape derivatives
for i in range(len(eigenvalues)):
    ax.quiver(eigenvalues[i].real, eigenvalues[i].imag+offset, continuous[i].real/scale, continuous[i].imag/scale,
               angles='xy', scale_units='xy', scale=1, color='red', label='Continuous' if i == 0 else "", width=0.0025)
    ax.quiver(eigenvalues[i].real, eigenvalues[i].imag-offset, discrete[i].real/scale, discrete[i].imag/scale,
               angles='xy', scale_units='xy', scale=1, color='green', label='Discrete' if i == 0 else "", width=0.0025)
    #ax.arrow(eigenvalues[i].real, eigenvalues[i].imag, continuous[i].real/scale, continuous[i].imag/scale, head_width=0.01, head_length=1)
    #ax.arrow(eigenvalues[i].real, eigenvalues[i].imag, discrete[i].real/scale, discrete[i].imag/scale, head_width=0.01, head_length=1)

# Set the range of the y axis
ax.set_ylim(-0.5, 0.5)

# Set the fontsize of the tick labels
ax.tick_params(axis='both', which='major', labelsize=24)
# Set labels and title
ax.set_xlabel('Real Part', fontsize=26)
ax.set_ylabel('Imaginary Part', fontsize=26)
ax.legend(fontsize=26)
ax.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()
