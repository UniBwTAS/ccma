"""
Example :: Primitive Outlier Detection

Note :: This primitive example serves as a basic illustration to highlight the application of the CCMA in outlier detection.

Steps:
    1. Given noisy data with outliers.
    2. Smooth data using a flat CCMA kernel for global emphasis.
        - Utilize a flat kernel, such as uniform, to prevent excessive focus on individual data points.
        - The primary goal is to find a rough estimate of the original shape -- not smoothing
    3. Calculate the standard deviation between smoothed points and noisy points.
    4. Remove all points where the deviation exceeds "c * std," where c is a constant.
    5. Smooth the data again with local emphasis (e.g., Pascal triangle).
"""

import matplotlib.pyplot as plt
import numpy as np
from ccma import CCMA
import random


# Create perfect ellipse
n = 50
points = np.array([np.cos(np.linspace(0, 2 * np.pi, n)),
                   1.5 * np.sin(np.linspace(0, 2 * np.pi, n))]).T

# Create noise and add to original points
sigma_noise = 0.05
noise = np.random.normal(0, sigma_noise, (n, 2))
noisy_points = points + noise

# Create outliers and add to noisy points
sigma_outlier = 0.5
n_o = int(n * 0.1)       # Define 10% outliers
random_indices = random.choices(list(range(n)), k=n_o)
outlier_noise = np.random.normal(0, sigma_outlier, (n_o, 2))
noisy_points[random_indices, :] = noisy_points[random_indices, :] + outlier_noise

# Smooth path via flat-CCMA (smaller rho values -> makes kernel flat)
ccma = CCMA(w_ma=7, w_cc=3, distrib="normal", rho_ma=0.8, rho_cc=0.9)

# TRY-IT-YOURSELF :: uncomment line below, and you will see the effect of a non-flat kernel
# ccma = CCMA(w_ma=5, w_cc=3, distrib="pascal")

smoothed_path = ccma.filter(noisy_points, mode="wrapping")

# Detect outliers & get points w/o outliers
cutoff = 1.5        # Parameter for outlier detection (outlier if cutoff * std > deviation)
deviations = np.linalg.norm(noisy_points - smoothed_path, axis=1)
std = np.std(deviations)
noisy_points_wo_outliers = noisy_points[deviations < std * cutoff, :]
outliers = noisy_points[deviations > std * cutoff, :]

# Smooth path w/o outliers via CCMA (this time use Pascal triangle)
ccma = CCMA(w_ma=7, w_cc=3)
smoothed_path_wo_outliers = ccma.filter(noisy_points_wo_outliers, mode="wrapping")


# Visualize results

fig, axs = plt.subplots(1, 3)

axs[0].plot(*points.T, 'b-', alpha=0.25, linewidth=3, label="Original points")
axs[0].plot(*noisy_points.T, 'r-o', alpha=0.25, markersize=4, linewidth=1, label="Nosy points")
axs[0].set_aspect("equal")
axs[0].grid(True)
axs[0].legend()
axs[0].set_title("Ellipse with outliers")

axs[1].plot(*points.T, 'b-', alpha=0.25, linewidth=3)
axs[1].plot(*noisy_points.T, 'r-o', alpha=0.25, markersize=4, linewidth=1)
axs[1].plot(*smoothed_path.T, 'k-', alpha=0.5, linewidth=3, label="CCMA-smoothed w/ outliers")
axs[1].set_aspect("equal")
axs[1].grid(True)
axs[1].legend()
axs[1].set_title("CCMA-smoothed (with outliers)")

axs[2].plot(*points.T, 'b-', alpha=0.25, linewidth=3)
axs[2].plot(*noisy_points.T, 'r-o', alpha=0.25, markersize=4, linewidth=1)
axs[2].plot(*outliers.T, 'mo', alpha=1., markersize=6, label="Outliers")
axs[2].plot(*smoothed_path_wo_outliers.T, 'k-', alpha=0.5, linewidth=3, label="CCMA-smoothed w/o outliers")

# TRY-IT-YOURSELF :: Uncomment lines and see how it would look like with outlier detection.
# smoothed_path_w_outliers = ccma.filter(noisy_points, mode="wrapping")
# axs[2].plot(*smoothed_path_w_outliers.T, '-', color='orange', alpha=0.5, linewidth=3, label="CCMA-smoothed w outliers")

axs[2].set_aspect("equal")
axs[2].grid(True)
axs[2].legend()
axs[2].set_title("CCMA-smoothed (w/o outliers)")

plt.show()
