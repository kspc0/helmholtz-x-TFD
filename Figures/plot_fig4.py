import os
import numpy as np
import matplotlib.pyplot as plt
fontsize = 26
path = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(path, 'data_fig4.txt')

with open(file, 'r') as f:
    lines = f.readlines()
perturbations = []
discrete_shape_derivatives = []

# read the data from the file
for line in lines[1:]: # skip the first line
    p, der = line.strip().split(',')
    perturbations.append(float(p))
    discrete_shape_derivatives.append(float(der))
# transform the lists into numpy arrays to perform operations
perturbations = np.array(perturbations)

# create figure and axes
fig, ax = plt.subplots(figsize=(20, 8))

# Fit a linear line to the first few data points
# real part
slope, intercept = np.polyfit(perturbations[:5], discrete_shape_derivatives[:5], 1)
linear_fit = slope * perturbations + intercept
print('Slope:', slope)
ax.plot(perturbations, linear_fit, 'r--', label='Linear Fit')

# plot the real part of shape derivatives
ax.plot(perturbations, discrete_shape_derivatives, color='green', marker='o', label='Discrete Shape Derivative')
# set labels and title
ax.set_xlabel('Perturbation', fontsize=fontsize)
ax.set_ylabel('Discrete Derivatives', fontsize=fontsize)
ax.legend(loc='upper right')
ax.legend(fontsize=fontsize)
ax.grid(True)


# Increase the font size of the axis numbers
ax.tick_params(axis='both', which='major', labelsize=24)

plt.tight_layout() # make plot look better
plt.show() # show the plot
