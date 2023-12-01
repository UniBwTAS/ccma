import matplotlib.pyplot as plt
import numpy as np
from ccma import CCMA

# Create a noisy 2d-path
n = 100
sigma = 0.05
points = np.array([2.0 * np.cos(np.linspace(0, 2 * np.pi, n)),
                   np.sin(np.linspace(0, 6 * np.pi, n))]).T + np.random.normal(0, sigma, (n, 2))

# Create the CCMA-filter object
w_ma = 10
w_cc = 3
ccma = CCMA(w_ma, w_cc)


# Filter points with and w/o boundaries
ccma_points = ccma.filter(points)
ccma_points_wo_padding = ccma.filter(points, mode="none")
ma_points = ccma.filter(points, cc_mode=False)

# Visualize results
plt.plot(*points.T, "r-o", linewidth=3, alpha=0.3, markersize=10, label="original")
plt.plot(*ccma_points.T, linewidth=6, alpha=1.0, color="orange", label=f"ccma-smoothed with padding ({w_ma}, {w_cc})")
plt.plot(*ccma_points_wo_padding.T, linewidth=3, alpha=0.5, color="b", label=f"ccma-smoothed ({w_ma}, {w_cc})")
plt.plot(*ma_points.T, linewidth=2, alpha=0.5, color="green", label=f"ma-smoothed ({w_ma})")

# General settings
plt.grid(True)
plt.gca().set_aspect('equal')
plt.legend()
plt.tight_layout()
plt.gcf().set_size_inches(12, 6)
plt.xlabel("x")
plt.ylabel("y")
plt.title("CCMA - Example 1 (2d)")

plt.show()
