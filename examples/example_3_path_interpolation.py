"""
Example :: Path Interpolation Between a Sequence of Waypoints

Steps ::
    - Make waypoints dense (fill with additional points)
    - Replicate waypoints to artificially increase weighing
    - Apply CCMA
"""

import matplotlib.pyplot as plt
import numpy as np
from ccma import CCMA
import math


# Define waypoints and density between two consecutive points
waypoints_ = np.array([[0, 0], [10, 0], [10, 5], [5, 5], [10, 10], [0, 10]])
point_density = 0.25


def densify_path(waypoints, density, replicate_factor=1):
    """
    Densify a path represented by waypoints.

    Parameters:
    - waypoints (numpy.ndarray): Array of waypoints (n x m), where n is the number of waypoints,
                                 and m is the dimensionality of each waypoint.
    - density (float): Desired density of points along the path.
    - replicate_factor (int, optional): Factor by which to replicate each waypoint.

    Returns:
    - numpy.ndarray: Densified path with the specified density and replicated waypoints.
    """

    path = [np.tile(waypoints[0], (replicate_factor, 1))]

    for idx in range(waypoints.shape[0] - 1):
        distance = np.linalg.norm(waypoints[idx] - waypoints[idx + 1])
        num_points = max(2, math.ceil(distance / density))
        interpolated_points = np.linspace(waypoints[idx], waypoints[idx + 1], num_points, axis=0)[1:]
        path.append(interpolated_points)
        path.append(np.tile(waypoints[idx + 1], (replicate_factor - 1, 1)))

    return np.row_stack(path)


path_repl1 = densify_path(waypoints_, point_density, 1)
path_repl7 = densify_path(waypoints_, point_density, 7)

ccma = CCMA(w_ma=15, w_cc=3, distrib="normal")
smoothed_path_repl1 = ccma.filter(path_repl1)
smoothed_path_repl7 = ccma.filter(path_repl7)


plt.plot(*path_repl1.T, "r-o", alpha=0.35, label="Densified path")
plt.plot(*smoothed_path_repl1.T, "m-", alpha=0.35, linewidth=4, label="CCMA-smoothed (repl. 1)")
plt.plot(*smoothed_path_repl7.T, "b-", alpha=0.35, linewidth=4, label="CCMA-smoothed (repl. 7)")
plt.plot(*waypoints_.T, "ko", markersize=10, label="Waypoints")
plt.grid(True)
plt.legend()
plt.gca().set_aspect('equal')
plt.show()