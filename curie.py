import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Read data from a document file (e.g., 'data.txt')
data = np.loadtxt('3546curie')

# Separate columns into T and R
T = data[:, 0]
R = data[:, 1]

# Define the step size (n-step differentiation)
n = 1  # You can adjust this to the desired step size


# Compute the derivative using a larger step size
dR_dT_long_step = (R[n:] - R[:-n]) / (T[n:] - T[:-n])

# Create a new T array that matches the length of dR/dT with the longer step
T_long_step = (T[:-n] + T[n:]) / 2

# Remove infinities and NaN values from the data
valid_indices = np.isfinite(dR_dT_long_step)
T_clean = T_long_step[valid_indices]
dR_dT_clean = dR_dT_long_step[valid_indices]

# Create a figure with two subplots (Before filtering T)
plt.figure(figsize=(10, 8))

# First subplot: Plot T vs R
plt.subplot(2, 1, 1)
plt.plot(T, R, marker='o', linestyle='-', color='g')
plt.xlabel('T(°C)')
plt.ylabel('R(Ohm)')
plt.title('Plot of T vs R (Before filtering)')
plt.grid(True)

# Second subplot: Plot T vs dR/dT (with longer step)
plt.subplot(2, 1, 2)
plt.plot(T_clean, dR_dT_clean, marker='o', linestyle='-', color='b', label='dR/dT')
plt.xlabel('T')
plt.ylabel('dR/dT (Long Step)')
plt.title(f'Plot of T vs dR/dT (n={n} steps, Before filtering)')
plt.grid(True)

# Show the plots before filtering
plt.tight_layout()
plt.show()

# Now filter out values where T is lower than 200 or higher than 400
valid_range = (T_clean >= 200) & (T_clean <= 400)
T_filtered = T_clean[valid_range]
dR_dT_filtered = dR_dT_clean[valid_range]

# Apply a Gaussian filter to the dR/dT data
sigma = 4  # Standard deviation for the Gaussian filter
dR_dT_gaussian = gaussian_filter1d(dR_dT_filtered, sigma=sigma)

# Create a new plot after filtering
plt.figure(figsize=(8, 6))

# Plot T vs dR/dT (after filtering)
plt.plot(T_filtered, dR_dT_filtered, marker='o', linestyle='-', color='r', label='dR/dT')
plt.plot(T_filtered, dR_dT_gaussian, linestyle='-', color='blue', label='Gaussian Filtered Curve')
plt.xlabel('T')
plt.ylabel('dR/dT')
#plt.title(f'Plot of T vs dR/dT (n={n} steps, After filtering T between 200 and 400)')
plt.grid(True)
plt.legend()

# Show the plot after filtering
plt.tight_layout()
plt.show()

