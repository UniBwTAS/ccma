"""
CURVATURE CORRECTED MOVING AVERAGE (CCMA)

The CCMA is a model-free general-purpose smoothing algorithm designed for 2D/3D paths. It
addresses the issue of the inwards bending phenomenon in curves that commonly occurs with
conventional moving average filters. The CCMA method employs a symmetric filtering
approach, avoiding a delay as it is common for state-estimation approaches, e.g. alpha-beta filter.

FEATURES:

-----> 1. Versatile Kernel Options:

CCMA provides flexibility in choosing from a variety of kernels, allowing users to tailor
the smoothing process to their specific needs:

   - **Uniform Kernel:**
     - A uniform distribution of weights ensures a straightforward and evenly distributed
     smoothing effect.

   - **Truncated Normal Kernel:**
     - Utilizes a truncated normal distribution, offering a controllable truncation area
     with adjustable parameters for precision smoothing.

   - **Pascal's Triangle Kernel:**
     - Based on the rows of Pascal's triangle, this kernel serves as a discrete version
     of the normal distribution, providing a good compromise between accuracy and smoothness.

   - **Hanning Kernel:**
     - Is a popular choice in signal processing and produces supersmooth results, but is less accurate.
     However, combined with curvature correction it can produce outstanding results.
     - Also known as raised cosine window.

-----> 2. Boundary Behavior Options:

Choose from different boundary behaviors to customize the filtering process according to
your application requirements:

   - **Padding:**
     - Pads the first and last points to preserve the length of the path during filtering.

   - **Wrapping:**
     - Treats the points as cyclic, creating a seamless transition between the end and
     the beginning of the path.

   - **Decreased Filtering Width (Fill Boundary):**
     - Implements filtering with decreased width parameters, preserving the length of
     the path while smoothing.

   - **None:**
     - Skips boundary processing, allowing the algorithm to filter points without any
     length-preserving adjustments.

While the code itself serves as a robust tool, a comprehensive understanding of CCMA,
its intricacies, and potential applications can be gained by referring to the accompanying paper:

T. Steinecker and H.-J. Wuensche, "A Simple and Model-Free Path Filtering
Algorithm for Smoothing and Accuracy", in Proc. IEEE Intelligent Vehicles
Symposium (IV), 2023
"""

# =================================================================================================
# -----   Imports   -------------------------------------------------------------------------------
# =================================================================================================

from typing import Optional, Union, Callable, List, Dict
import numpy as np
from scipy.stats import norm

# =================================================================================================
# -----   Auxiliary Functions   -------------------------------------------------------------------
# =================================================================================================

def get_unit_vector(vector: np.ndarray) -> np.ndarray:
    """
    Compute the unit vector for a given vector.

    Parameters:
    - vector (np.ndarray): Input vector.

    Returns:
    - np.ndarray: Unit vector of the input vector.
    """
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0.0:
        return vec
    else:
        return vec / vec_norm

# =================================================================================================
# -----   CCMA   ----------------------------------------------------------------------------------
# =================================================================================================

class CCMA:
    def __init__(self, w_ma: float = 5, w_cc: float = 3, distrib: str = "pascal",
                 distrib_ma: str = None, distrib_cc: str = None, rho_ma: float = 0.95, rho_cc: float = 0.95):
        """
        Initialize the CCMA object with specified parameters.

        Parameters:
        - w_ma (float): Width parameter for the moving average.
        - w_cc (float): Width parameter for the curvature correction.
        - distrib (str, optional): Type of kernel used for filtering. Options include:
            - "uniform": Uniform distribution of weights.
            - "normal": Truncated normal distribution with specified truncation area (see rho_ma and rho_cc).
            - "pascal": Kernel based on rows of Pascal's triangle, a discretized version of the normal distribution.
              (Default is "pascal")
            - "hanning": The famous hanning kernel, which is most often used in signal processing. Less accurate,
              but best smoothing characteristics.
        - rho_ma (float, optional): Truncation area for the normal distribution in the moving average.
            (Default is 0.95)
        - rho_cc (float, optional): Truncation area for the normal distribution in the curvature correction.
            (Default is 0.95)

        Note:
        - The 'distrib' parameter specifies the type of kernel used for filtering.
        - 'rho_ma' and 'rho_cc' are relevant only for the "normal" kernel and represent the truncation areas for
          the respective distributions.
        - If 'rho' approximates 0, the 'normal' kernel approximates the 'uniform' kernel.

        Example:
        ```python
        ccma = CCMA(w_ma=5, w_cc=3, distrib="normal", rho_ma=0.99, rho_cc=0.95)
        ```

        This example initializes a CCMA object with a normal kernel, specified widths, and truncation areas.
        """

        # Width for moving averages of points and shifts
        self.w_ma = w_ma
        self.w_cc = w_cc
        # Overall width (+1 -> see paper)
        self.w_ccma = w_ma + w_cc + 1

        # Distribution of weights for smoothing (ma) and curvature correction (cc).
        # If neither distrib_ma nor distrib_cc is set, general distrib is used.
        self.distrib_ma = distrib_ma if distrib_ma else distrib
        self.distrib_cc = distrib_cc if distrib_cc else distrib

        # Truncation value. The larger the width, the larger rho should be chosen.
        self.rho_ma = rho_ma
        self.rho_cc = rho_cc

        # Calculate the weights
        self.weights_ma = self._get_weights(w_ma, self.distrib_ma, rho_ma)
        self.weights_cc = self._get_weights(w_cc, self.distrib_cc, rho_cc)

    @staticmethod
    def _get_weights(w: float, distrib: str, rho: float):
        """
        Generate weights based on the specified distribution and parameters.

        Parameters:
        - w (float): Width parameter for the weights.
        - distrib (str): Type of distribution for generating weights.
        - rho (float): Truncation area parameter.

        Returns:
        - list: List of weight arrays.

        Raises:
        - ValueError: If an invalid distribution type is provided.
        """

        weight_list = []

        if distrib == "normal":
            # Get start/end of truncated normal distribution
            x_start = norm.ppf((1 - rho) / 2)
            x_end = norm.ppf(1 - ((1 - rho) / 2))

            for w_i in range(int(w) + 1):
                x_values = np.linspace(x_start, x_end, 2 * w_i + 1 + 1)
                weights = np.zeros((2 * w_i + 1))

                for idx in range(2 * w_i + 1):
                    weights[idx] = norm.cdf(x_values[idx + 1]) - norm.cdf(x_values[idx])

                # Adjust weights by rho to make it a meaningful distribution
                weights = (1 / rho) * weights

                weight_list.append(weights)

        elif distrib == "uniform":
            for w_i in range(int(w) + 1):
                weights = np.ones(2 * w_i + 1) * (1 / (2 * w_i + 1))
                weight_list.append(weights)

        elif distrib == "pascal":

            def get_pascal_row(row_index):
                cur_row = [1]

                if row_index == 0:
                    return cur_row

                prev = get_pascal_row(row_index - 1)

                for idx_ in range(1, len(prev)):
                    cur = prev[idx_ - 1] + prev[idx_]
                    cur_row.append(cur)

                cur_row.append(1)
                return cur_row

            for w_i in range(int(w) + 1):
                pascal_row_index = w_i * 2
                row = np.array(get_pascal_row(pascal_row_index))
                weight_list.append(row / np.sum(row))

        elif distrib == "hanning":
            def get_hanning_kernel(window_size):
                # Add two as the first and last element of the hanning kernel is 0.
                window_size += 2
                hanning_kernel = (0.5 * (1 - np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1))))[1:-1]
                return hanning_kernel / np.sum(hanning_kernel)

            for w_i in range(int(w) + 1):
                weight_list.append(get_hanning_kernel(w_i * 2 + 1))

        elif callable(distrib):
            for w_i in range(int(w) + 1):
                weight_list.append(distrib(w_i * 2 + 1))

        else:
            raise ValueError("Distribution must either be 'uniform', 'pascal', 'hanning, or 'normal'.")

        return weight_list

    @staticmethod
    def _get_3d_from_2d(points: np.ndarray) -> np.ndarray:
        """
        Convert 2D points array to 3D by adding a zero-filled dimension.

        Parameters:
        - points (np.ndarray): Input array of points with dimensions nx2.

        Returns:
        - np.ndarray: Converted array of points with dimensions nx3.
        """
        return np.column_stack([points, np.zeros(points.shape[0])])

    @staticmethod
    def _get_ma_points(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Calculate moving average points using convolution for each dimension.

        Parameters:
        - points (np.ndarray): Input array of points with dimensions nx3.
        - weights (np.ndarray): Weights for convolution.

        Returns:
        - np.ndarray: Convolved array of points with dimensions nx3.
        """
        return np.column_stack([np.convolve(points[:, 0], weights, mode='valid'),
                                np.convolve(points[:, 1], weights, mode='valid'),
                                np.convolve(points[:, 2], weights, mode='valid')])

    @staticmethod
    def _get_curvature_vectors(points: np.ndarray) -> np.ndarray:
        """
        Calculate curvature vectors for given points.

        Parameters:
        - points (np.ndarray): Input array of points with dimensions nx3.

        Returns:
        - np.ndarray: Array of curvature vectors with dimensions nx3.
        """
        curvature_vectors = np.zeros_like(points)

        for i in range(1, points.shape[0] - 1):
            p0 = points[i - 1]
            p1 = points[i]
            p2 = points[i + 1]

            v1 = p1 - p0
            v2 = p2 - p1
            cross = np.cross(v1, v2)
            if np.linalg.norm(cross) != 0.0:
                radius = np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(p2 - p0) / (2 * np.linalg.norm(cross))
                curvature = 1 / radius
            else:
                curvature = 0.0

            curvature_vectors[i] = curvature * get_unit_vector(cross)

        return curvature_vectors

    @staticmethod
    def _get_alphas(points: np.ndarray, curvatures: np.ndarray) -> np.ndarray:
        """
        Calculate alpha values based on points and curvatures.

        Parameters:
        - points (np.ndarray): Input array of points with dimensions nx3.
        - curvatures (np.ndarray): Array of curvature values.

        Returns:
        - np.ndarray: Array of alpha values.
        """
        alphas = np.zeros(points.shape[0])

        for idx in range(1, points.shape[0] - 1):
            if curvatures[idx] != 0.0:
                radius = 1 / curvatures[idx]
                dist_neighbors = np.linalg.norm(points[idx + 1] - points[idx - 1])
                alphas[idx] = np.sin((dist_neighbors / 2) / radius)
            else:
                alphas[idx] = 0.0

        return alphas

    @staticmethod
    def _get_radii_ma(alphas: np.ndarray, w: float, weights: np.ndarray) -> np.ndarray:
        """
        Calculate radii for moving average based on alphas and weights.

        Parameters:
        - alphas (np.ndarray): Array of alpha values.
        - w (float): Width parameter.
        - weights (np.ndarray): Weights for convolution.

        Returns:
        - np.ndarray: Array of radii for moving average.
        """
        # Offset for convolution
        pad = np.zeros(int(w))
        alphas = np.concatenate([pad, alphas, pad])

        return 1 / np.convolve(alphas, weights, mode='valid')

    @staticmethod
    def _shift_points(points: np.ndarray, curvatures: np.ndarray, radii_ma: np.ndarray) -> np.ndarray:
        """
        Apply shifts to points based on curvature and moving average radii.

        Parameters:
        - points (np.ndarray): Input array of points with dimensions nx3.
        - curvatures (np.ndarray): Array of curvature values.
        - radii_ma (np.ndarray): Array of radii for moving average.

        Returns:
        - np.ndarray: Shifted points array with dimensions nx3.
        """
        shifted_points = np.zeros_like(points)

        for i in range(points.shape[0]):
            shifted_points[i] = points[i] + curvatures[i] * radii_ma[i]

        return shifted_points

    def get_smooth_path(self, points: np.ndarray) -> np.ndarray:
        """
        Generate a smooth path based on input points using the CCMA algorithm.

        Parameters:
        - points (np.ndarray): Input array of points with dimensions nx2 or nx3.

        Returns:
        - np.ndarray: Smoothed path array with dimensions nx2 or nx3.
        """
        # Move from 2D to 3D
        points = self._get_3d_from_2d(points) if points.shape[1] == 2 else points

        for w_i in range(int(self.w_ccma)):
            ma_points = self._get_ma_points(points, self.weights_ma[w_i])
            curvatures = self._get_curvature_vectors(ma_points)
            alphas = self._get_alphas(ma_points, curvatures)
            radii_ma = self._get_radii_ma(alphas, self.w_cc, self.weights_cc[w_i])
            points = self._shift_points(points, curvatures, radii_ma)

        return points
