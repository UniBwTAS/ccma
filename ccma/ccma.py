"""
Curvature Corrected Moving Average (CCMA)

The CCMA is a model-free filtering technique designed for 2D and 3D paths. It
addresses the issue of the inwards bending phenomenon in curves that commonly occurs with
conventional moving average filters. The CCMA method employs a symmetric filtering
approach, avoiding a delay as it is common for state-estimation approaches, e.g. alpha-beta filter.

The implementation includes the ability to filter given points represented as a
numpy array. Additionally, the method supports filtering the boundaries with
decreasing filtering width.

While the code itself may not provide a complete understanding, further details
and insights can be found in the paper:

T. Steinecker and H.-J. Wuensche, "A Simple and Model-Free Path Filtering
Algorithm for Smoothing and Accuracy", in Proc. IEEE Intelligent Vehicles
Symposium (IV), 2023
"""

# =================================================================================================
# -----   Imports   -------------------------------------------------------------------------------
# =================================================================================================


import numpy as np
from scipy.stats import norm


# =================================================================================================
# -----   Auxiliary Functions   -------------------------------------------------------------------
# =================================================================================================


def get_unit_vector(vec):
    if (vec_norm := np.linalg.norm(vec)) == 0.0:
        return vec
    else:
        return vec / np.linalg.norm(vec)


# =================================================================================================
# -----   CCMA   ----------------------------------------------------------------------------------
# =================================================================================================


class CCMA:
    def __init__(self, w_ma, w_cc, distrib="normal", rho_ma=0.95, rho_cc=0.95):
        # Width for moving averages of points and shifts
        self.w_ma = w_ma
        self.w_cc = w_cc
        # Overall width (+1 -> see paper)
        self.w_ccma = w_ma + w_cc + 1

        # Distribution of weights. Either truncated normal distribution or uniform distribution.
        self.distrib = distrib
        # Truncation value. The larger the width, the larger rho should be chosen.
        self.rho_ma = rho_ma
        self.rho_cc = rho_cc

        # Calculate the weights
        self.weights_ma = self._get_weights(w_ma, rho_ma)
        self.weights_cc = self._get_weights(w_cc, rho_cc)

    def _get_weights(self, w, rho):
        weight_list = []

        if self.distrib == "normal":
            # Get start/end of truncated normal distribution
            x_start = norm.ppf((1 - rho) / 2)
            x_end = norm.ppf(1 - ((1 - rho) / 2))

            for w_i in range(w + 1):
                x_values = np.linspace(x_start, x_end, 2 * w_i + 1 + 1)
                weights = np.zeros((2 * w_i + 1))

                for idx in range(2 * w_i + 1):
                    weights[idx] = norm.cdf(x_values[idx + 1]) - norm.cdf(x_values[idx])

                # Adjust weights by rho to make it a meaningful distribution
                weights = (1 / rho) * weights

                weight_list.append(weights)

            return weight_list

        elif self.distrib == "uniform":
            for w_i in range(w + 1):
                weights = np.ones(2 * w_i + 1) * (1 / (2 * w_i + 1))

                weight_list.append(weights)

            return weight_list

        else:
            raise ValueError("Distribution must be either be 'uniform' or 'normal'.")

    @staticmethod
    def _get_3d_from_2d(points):
        # Add z-dimension by filling up with 0s
        return np.column_stack([points, np.zeros(points.shape[0])])

    @staticmethod
    def _get_ma_points(points, weights):
        # Perform convolution for each dimension
        return np.column_stack([np.convolve(points[:, 0], weights, mode='valid'),
                                np.convolve(points[:, 1], weights, mode='valid'),
                                np.convolve(points[:, 2], weights, mode='valid')])

    @staticmethod
    def _get_curvature_vectors(points):
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
    def _get_alphas(points, curvatures):
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
    def _get_radii_ma(alphas, w, weights):
        radii_ma = np.zeros_like(alphas)

        for idx in range(1, len(alphas) - 1):
            radii_ma[idx] = weights[w]
            for k in range(1, w + 1):
                radii_ma[idx] += 2 * np.cos(alphas[idx] * k) * weights[w + k]

            # Apply threshold to MA-estimated radius to avoid unstable correction (-> limited correction)(see paper)
            radii_ma[idx] = max(0.5, radii_ma[idx])

        return radii_ma

    def _get_descending_width(self):
        """
        Reduces the width parameters w_ma and w_cc until both parameters are 0.

        Returns:
            list: A list of dictionaries, where each dictionary holds the w_ma and w_cc values.

        Notes:
            - The width parameters w_ma and w_cc are reduced equally by always reducing the larger of the two.
        """

        # Allocate & Initialize
        descending_width_list = []
        w_ma_cur = self.w_ma
        w_cc_cur = self.w_cc

        while not (w_ma_cur == 0 and w_cc_cur == 0):
            if w_cc_cur >= w_ma_cur:
                w_cc_cur -= 1
            else:
                w_ma_cur -= 1
            descending_width_list.append({"w_ma": w_ma_cur, "w_cc": w_cc_cur})

        return descending_width_list

    def _filter(self, points, w_ma, w_cc, cc_mode):
        w_ccma = w_ma + w_cc + 1

        # Calculate moving-average points
        points_ma = self._get_ma_points(points, self.weights_ma[w_ma])

        if not cc_mode:
            return points_ma[w_cc + 1:-w_cc - 1]

        # Calculate curvature vectors & curvatures
        curvature_vectors = self._get_curvature_vectors(points_ma)
        curvatures = np.linalg.norm(curvature_vectors, axis=1)

        # Calculate alphas (angles defined two consecutive points defined by the assumption of const. curvature)
        alphas = self._get_alphas(points_ma, curvatures)#

        # Calculate radii
        radii_ma = self._get_radii_ma(alphas, w_ma, self.weights_ma[w_ma])

        # Allocate
        points_ccma = np.zeros((points.shape[0] - 2 * w_ccma, 3))

        for idx in range(points.shape[0] - 2 * w_ccma):
            # Get tangent vector for the shifting point
            unit_tangent = get_unit_vector(points_ma[w_cc + idx + 1 + 1] - points_ma[w_cc + idx - 1 + 1])

            # Calculate the weighted shift
            shift = np.zeros(3)
            for idx_cc in range(2 * w_cc + 1):
                if curvatures[idx + idx_cc + 1] != 0.0:
                    u = get_unit_vector(curvature_vectors[idx + idx_cc + 1 - w_cc])
                    weight = self.weights_cc[w_cc][idx_cc]
                    shift_magnitude = (1 / curvatures[idx + idx_cc + 1]) * (1 / radii_ma[idx + idx_cc + 1] - 1)
                    shift += u * weight * shift_magnitude

            # Reconstruction
            points_ccma[idx] = points_ma[idx + w_cc + 1] + np.cross(unit_tangent, shift)

        return points_ccma

    def filter(self, points: np.ndarray, fill_boundary: bool = False, cc_mode: bool = True):
        """
        Filters points based on the CCMA-filtering algorithm.

        Parameters:
            - points (numpy.ndarray): An array of shape (n, 2) or (n, 3) containing the coordinates of points.
            - fill_boundary (bool, optional): Specifies if the boundaries should be filtered with descending width parameters.
                                             Defaults to False.
            - cc_mode (bool, optional): Specifies if curvature-correction is applied.

        Returns:
            numpy.ndarray: CCMA-filtered point array.
        """

        if points.shape[0] < 3:
            raise RuntimeError("At least 3 points are necessary for the CCMA-filtering")

        if points.shape[0] < self.w_ccma * 2 + 1:
            raise RuntimeError("Not enough points are given for complete filtering!")

        # Convert 2d points to 3d points (if given as 2d)
        if is_2d := (points.shape[1] == 2):
            points = self._get_3d_from_2d(points)

        if not fill_boundary:
            if is_2d:
                return self._filter(points, w_ma=self.w_ma, w_cc=self.w_cc, cc_mode=cc_mode)[:, 0:2]
            return self._filter(points, w_ma=self.w_ma, w_cc=self.w_cc, cc_mode=cc_mode)
        else:
            # Define dimension for fast access of relevant dimensions
            dim = 2 if is_2d else 3

            points_ccma = np.zeros((points.shape[0], dim))
            descending_width_list = self._get_descending_width()

            # First and last point
            points_ccma[0] = points[0, 0:dim]
            points_ccma[-1] = points[-1, 0:dim]

            # Ascending points
            for width_set, idx_w in zip(descending_width_list[::-1], range(len(descending_width_list))):
                w_ccma = width_set["w_ma"] + width_set["w_cc"] + 1
                points_ccma[idx_w + 1] = self._filter(
                    points[:idx_w + 1 + w_ccma + 1],
                    width_set["w_ma"],
                    width_set["w_cc"],
                    False if width_set["w_ma"] <= 1 or not cc_mode else True)[:, :dim]

            # Full-filtered points
            points_ccma[self.w_ccma: points.shape[0] - self.w_ccma] = self._filter(points, self.w_ma, self.w_cc, cc_mode)[:, :dim]

            # Descending points
            for width_set, idx_w in zip(descending_width_list[::-1], range(len(descending_width_list))):
                w_ccma = width_set["w_ma"] + width_set["w_cc"] + 1
                points_ccma[-idx_w - 2] = self._filter(
                    points[-idx_w - 2 - w_ccma:],
                    width_set["w_ma"],
                    width_set["w_cc"],
                    False if width_set["w_ma"] <= 1 or not cc_mode else True)[:, :dim]

            return points_ccma

