import numpy as np
from ccma import CCMA
import matplotlib.pyplot as plt


# Define custom kernel
def get_triangle_kernel(width):
    ramp = np.array(range(1, width + 1))
    half_width = width // 2 + 1
    ramp[-half_width:] = ramp[:half_width][::-1]
    triangle_kernel = ramp / np.sum(ramp)
    return triangle_kernel


# Define ccma-object with custom kernel
ccma = CCMA(w_ma=5, w_cc=3, distrib=get_triangle_kernel)

# Create some data to test the kernel
n = 120
sigma = 0.1
points = np.array([2.0 * np.cos(np.linspace(0, 2 * np.pi, n)),
                   np.sin(np.linspace(0, 6 * np.pi, n))]).T
noise = np.random.normal(0, sigma, (n, 2))
noisy_points = points + noise

# Apply ccma
ccma_smoothed_points = ccma.filter(points)

# Apply ma
ma_smoothed_points = ccma.filter(points, cc_mode=False)

# Visualize
plt.plot(*points.T, 'b-', linewidth=5, alpha=0.25, label="original points")
plt.plot(*noisy_points.T, 'k-o', alpha=0.25, markersize=8, label="noisy points")
plt.plot(*ma_smoothed_points.T, 'g-o', label="ma-smoothed")
plt.plot(*ccma_smoothed_points.T, 'm-o', label="ccma-smoothed")
plt.grid(True)
plt.legend()
plt.show()
