import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from ccma import CCMA

# Create a noisy 2d-path
n = 100
sigma = 0.075
points = np.array([2.0 * np.cos(np.linspace(0, 2 * np.pi, n)),
                   np.sin(np.linspace(0, 6 * np.pi, n)),
                   np.linspace(0, 4, n)]).T + np.random.normal(0, sigma, (n, 3))

# Create the CCMA-filter object
w_ma = 6
w_cc = 3
ccma = CCMA(w_ma, w_cc, distrib="hanning")

# Filter points with and w/o boundaries
ccma_points = ccma.filter(points)
ccma_points_wo_padding = ccma.filter(points, mode="none")
ma_points = ccma.filter(points, cc_mode=False)

# Visualize results
# Create a 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the path
ax.plot(points[:, 0], points[:, 1], points[:, 2], "r-o", linewidth=1, alpha=0.25, markersize=5, label="original")
ax.plot(ma_points[:, 0], ma_points[:, 1], ma_points[:, 2], linewidth=1, alpha=0.5, color="green", label=f"ma-smoothed ({w_ma})")
ax.plot(ccma_points[:, 0], ccma_points[:, 1], ccma_points[:, 2], linewidth=5, alpha=0.5, color="orange",
        label=f"ccma-smoothed with padding ({w_ma}, {w_cc})")
ax.plot(ccma_points_wo_padding[:, 0],
        ccma_points_wo_padding[:, 1],
        ccma_points_wo_padding[:, 2],
        linewidth=2, alpha=0.5, color="b", label=f"ccma-smoothed ({w_ma}, {w_cc})")

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Path Visualization')

# General settings
plt.grid(True)
plt.gca().set_aspect('equal')
plt.legend()
plt.tight_layout()
plt.gcf().set_size_inches(12, 6)
plt.xlabel("x")
plt.ylabel("y")
plt.title("CCMA - Example 2 (3d)")

plt.show()