import matplotlib.pyplot as plt
import numpy as np
import os

fontsize=26
os.chdir(os.path.join(os.getcwd()))
file = os.path.join('data_fig2.txt')

# read file
with open(file, 'r') as f:
    lines = f.readlines()

# create empty lists to store the data
duct = []
frequ = []
analytic = []
continuous = []
discrete = []
discrete_refined = []

# read the data from the file
for line in lines[1:]: # skip the first line
    duc, fre, ana, con, f_dis, c_dis = map(float, line.strip().split(','))
    duct.append(duc)
    frequ.append(fre)
    analytic.append(ana)
    continuous.append(con)
    discrete_refined.append(f_dis)
    discrete.append(c_dis)

# transform the lists into numpy arrays to perform operations
duct = np.array(duct)
frequ = np.array(frequ)
analytic = np.array(analytic)
continuous = np.array(continuous)
discrete = np.array(discrete)
discrete_refined = np.array(discrete_refined)

# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(20, 16))

# Plot all normalized data in one graph
#ax.plot(duct, frequ, marker='o', label='Frequency') # if frequency is needed
ax.plot(duct, analytic, marker='x', label='Analytic Shape Derivative', linewidth=2, linestyle='--', color='red')
ax.plot(duct, continuous, marker='x', label='Continuous Shape Derivative')
ax.plot(duct, discrete_refined, marker='x', label='Discrete Shape Derivative')
ax.plot(duct, discrete, marker='x', label='Discrete Shape Derivative (coarser mesh)')

# Set labels and title
ax.set_xlabel('Duct length [m]', fontsize=fontsize)
ax.set_ylabel('Shape Derivatives [1/(ms)]', fontsize=fontsize)
ax.legend(fontsize=fontsize)
ax.grid(True)
ax.tick_params(axis='both', which='major', labelsize=24)

# Adjust layout
plt.tight_layout()
plt.show()
