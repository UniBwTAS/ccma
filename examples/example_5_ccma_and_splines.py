import matplotlib.pyplot as plt
import numpy as np
from ccma import CCMA
from scipy.interpolate import splprep, splev


# Create straight path
n = 40
points = np.array([np.linspace(0, 1, n), np.linspace(0, 0, n)]).T

# Create outlier
points[n//2, 1] = 0.1

# TRY-IT-YOURSELF :: Uncomment to see the global influence of local changes for B-Splines
# points[n // 4, 1] = -0.05

# Create the CCMA-filter object
w_ma = 4
w_cc = 4
ccma = CCMA(w_ma, w_cc)

# Calc. CCMA-smoothed points
ccma_points = ccma.filter(points)

# Fit a 2D Spline to the CCMA-smoothed path
tck, u = splprep(x=ccma_points.T, s=0, k=3)
spline_samples = splev(np.linspace(0, 1, 1000), tck)
plt.plot(spline_samples[0], spline_samples[1], '-', color="magenta", linewidth=4, alpha=1.0, label="CCMA + B-Spline")

# Fit a 2D B-Spline to the original data
tck, u = splprep(x=points.T, s=0.0088, k=3)
bspline_samples = splev(x=np.linspace(0, 1, 1000), tck=tck)
plt.plot(bspline_samples[0], bspline_samples[1], 'b-', linewidth=4, alpha=0.75, label="P-Spline")

# Visualize original points
plt.plot(*points.T, "k-o", linewidth=2, alpha=0.15, markersize=10, label="Original points")

# General settings
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.gcf().set_size_inches(12, 6)
plt.title("CCMA+Splines vs. Splines")

plt.show()
