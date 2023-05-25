import matplotlib.pyplot as plt
import numpy as np
from ccma.ccma import CCMA

# Create a noisy 2d-path
n = 100
points = np.array([2.0 * np.cos(np.linspace(0, 2 * np.pi, n)),
                   np.sin(np.linspace(0, 4 * np.pi, n))]).T + np.random.normal(0, 0.05, (n, 2))

# Create the CCMA-filter object
w_ma = 6
w_cc = 3
ccma = CCMA(w_ma, w_cc, distrib="normal")

# Filter points with and w/o boundaries
ccma_points = ccma.filter(points, fill_boundary=False)
ccma_points_with_boundary = ccma.filter(points, fill_boundary=True)
ma_points = ccma.filter(points, fill_boundary=True, cc_mode=False)

# Visualize results
plt.plot(*points.T, linewidth=3, alpha=0.6, color="r", label="original")
plt.plot(*ccma_points_with_boundary.T, linewidth=3, alpha=0.5, color="b", label=f"ccma-filtered with boundary ({w_ma}, {w_cc})")
plt.plot(*ccma_points.T, linewidth=6, alpha=0.5, color="orange", label=f"ccma-filtered ({w_ma}, {w_cc})")
plt.plot(*ma_points.T, linewidth=2, alpha=0.5, color="green", label=f"ma-filtered ({w_ma})")

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
