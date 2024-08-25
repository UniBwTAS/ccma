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


import numpy as np
from scipy.stats import norm


# =================================================================================================
# -----   Auxiliary Functions   -------------------------------------------------------------------
# =================================================================================================


def get_unit_vector(vec):
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0.0:
        return vec
    else:
        return vec / vec_norm


# =================================================================================================
# -----   CCMA   ----------------------------------------------------------------------------------
# =================================================================================================


class CCMA:
    def __init__(self, w_ma=5, w_cc=3, distrib="pascal", distrib_ma=None, distrib_cc=None, rho_ma=0.95, rho_cc=0.95):
        """
        Initialize the SmoothingFilter object with specified parameters.

        Parameters:
        - w_ma (float): Width parameter for the moving average.
        - w_cc (float): Width parameter for the curvature correction.
        - distrib (str, optional): Type of kernel used for filtering. Options include:
            - "uniform": Uniform distribution of weights.
            - "normal": Truncated normal distribution with specified truncation area (see rho_ma and rho_cc).
            - "pascal": Kernel based on rows of Pascal's triangle, a discretized version of the normal distribution.
              (Default is "pascal")
            - "hanning": The famous hanning kernel, which is most often used in signal processing. Less accurate, but best smoothing characteristics.
        - rho_ma (float, optional): Truncation area for the normal distribution in the moving average.
            (Default is 0.95)
        - rho_cc (float, optional): Truncation area for the normal distribution in the curvature correction.
            (Default is 0.95)

        Note:
        - The 'kernel' parameter specifies the type of kernel used for filtering -- The kernel is the shape of weights that is used for convolution.
        - 'rho_ma' and 'rho_cc' are relevant only for the "normal" kernel and represent the truncation areas for the respective distributions.
        - If 'rho' approximates 0, the 'normal' kernel approximates the 'uniform' kernel.

        Example:
        ```python
        ccma = CCMA(w_ma=5, w_cc=3, kernel="normal", rho_ma=0.99, rho_cc=0.95)
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
    def _get_weights(w, distrib, rho):
        weight_list = []

        if distrib == "normal":
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

        elif distrib == "uniform":
            for w_i in range(w + 1):
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

            for w_i in range(w + 1):
                pascal_row_index = w_i * 2
                row = np.array(get_pascal_row(pascal_row_index))
                weight_list.append(row / np.sum(row))

        elif distrib == "hanning":
            def get_hanning_kernel(window_size):
                # Add two as the first and last element of the hanning kernel is 0.
                window_size += 2
                hanning_kernel = (0.5 * (1 - np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1))))[1:-1]
                return hanning_kernel / np.sum(hanning_kernel)

            for w_i in range(w + 1):
                weight_list.append(get_hanning_kernel(w_i * 2 + 1))

        elif callable(distrib):
            for w_i in range(w + 1):
                weight_list.append(distrib(w_i * 2 + 1))

        else:
            raise ValueError("Distribution must either be 'uniform', 'pascal', 'hanning, or 'normal'.")

        return weight_list

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
                alphas[idx] = np.arcsin((dist_neighbors / 2) / radius)
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

            # TODO :: Maybe add a warning if the threshold gets active.
            # Apply threshold to MA-estimated radius to avoid unstable correction (-> limited correction)(see paper)
            radii_ma[idx] = max(0.35, radii_ma[idx])

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
            return points_ma

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
                    u = get_unit_vector(curvature_vectors[idx + idx_cc + 1])
                    weight = self.weights_cc[w_cc][idx_cc]
                    shift_magnitude = (1 / curvatures[idx + idx_cc + 1]) * (1 / radii_ma[idx + idx_cc + 1] - 1)
                    shift += u * weight * shift_magnitude

            # Reconstruction
            points_ccma[idx] = points_ma[idx + w_cc + 1] + np.cross(unit_tangent, shift)

        return points_ccma

    def filter(self, points: np.ndarray, mode: str = "padding", cc_mode: bool = True):
        """
        Apply filtering to a set of points using a specified mode.

        Parameters:
        - points (np.ndarray): Input array of points with dimensions nx2 or nx3.
        - mode (str, optional): Filtering mode. Options include:
            - "padding": Pad the first and last points to preserve path length.
            - "wrapping": Treat points as cyclic (wrapping around).
            - "fill_boundary": Filter with decreased width parameters to preserve length.
            - "none": Do not preserve length; no pre-processing.
        - cc_mode (bool, optional): Specifies if curvature correction is active.

        Returns:
        - np.ndarray: Filtered points.

        Raises:
        - ValueError: If the 'mode' parameter is not one of "padding", "wrapping", "none", or "fill_boundary".
        - RuntimeError: If the number of 'points' is insufficient.
        """

        if mode not in ["none", "padding", "wrapping", "fill_boundary"]:
            raise ValueError("Invalid mode! Got :: {mode}. Expected :: none | padding | wrapping | fill_boundary.")

        if points.shape[0] < 3:
            raise RuntimeError("At least 3 points are necessary for the CCMA-filtering")

        if mode == "padding":
            n_padding = self.w_ccma if cc_mode else self.w_ma
            points = np.row_stack((np.tile(points[0], (n_padding, 1)),
                                   points,
                                   np.tile(points[-1], (n_padding, 1))))

        if points.shape[0] < self.w_ccma * 2 + 1:
            raise RuntimeError("Not enough points are given for complete filtering!")

        if mode == "wrapping":
            n_padding = self.w_ccma if cc_mode else self.w_ma
            points = np.row_stack((points[-n_padding:],
                                   points,
                                   points[:n_padding]))

        # Convert 2d points to 3d points (if given as 2d)
        is_2d = points.shape[1] == 2
        if is_2d:
            points = self._get_3d_from_2d(points)

        if not (mode == "fill_boundary"):
            if is_2d:
                return self._filter(points, w_ma=self.w_ma, w_cc=self.w_cc, cc_mode=cc_mode)[:, 0:2]
            return self._filter(points, w_ma=self.w_ma, w_cc=self.w_cc, cc_mode=cc_mode)
        else:
            # Define dimension for fast access of relevant dimensions
            dim = 2 if is_2d else 3

            # Descending filtering for CCMA
            if cc_mode:
                points_ccma = np.zeros((points.shape[0], dim))
                descending_width_list = self._get_descending_width()[::-1]

                # First and last point
                points_ccma[0] = points[0, 0:dim]
                points_ccma[-1] = points[-1, 0:dim]

                # Full-filtered points
                points_ccma[self.w_ccma: points.shape[0] - self.w_ccma] = self._filter(points, self.w_ma, self.w_cc, cc_mode)[:, :dim]

                # Ascending/Descending points
                for width_set, idx_w in zip(descending_width_list, range(len(descending_width_list))):
                    w_ccma = width_set["w_ma"] + width_set["w_cc"] + 1

                    # The following is a design choice!
                    # The second last points are smoothed via MA but are not curvature corrected (you cannot apply both)
                    # The reason is that curvature correction has no effect without MA>0.
                    # Consequently, it is better to smooth the second last point instead of doing nothing.
                    # If, however, global w_ma was set to zero, no MA should be applied (consistency)!
                    use_ma_1 = True if width_set["w_ma"] == 0 and self.w_ma != 0 else False

                    # Ascending points
                    points_ccma[idx_w + 1] = self._filter(
                        points[:idx_w + 1 + w_ccma + 1],
                        # In case of w_ma==0, do not make curvature correction, but use w_ma==1 instead (this is a design choice)
                        width_set["w_ma"] if not use_ma_1 else 1,
                        width_set["w_cc"],
                        False if use_ma_1 else True)[:, :dim]

                    # Descending points
                    points_ccma[-idx_w - 2] = self._filter(
                        points[-idx_w - 2 - w_ccma:],
                        # In case of w_ma==0, do not make curvature correction, but use w_ma==1 instead (this is a design choice)
                        width_set["w_ma"] if not use_ma_1 else 1,
                        width_set["w_cc"],
                        False if use_ma_1 else True)[:, :dim]

                return points_ccma

            # Descending filtering for MA without curvature correction
            else:
                points_ma = np.zeros((points.shape[0], dim))
                descending_width_list = list(range(self.w_ma))

                # Full-filtered points
                points_ma[self.w_ma: points.shape[0] - self.w_ma] = self._filter(points, self.w_ma, 0, False)[:, :dim]

                # Ascending/Descending points
                for idx, width in zip(descending_width_list, descending_width_list):

                    # Ascending points
                    points_ma[idx] = self._filter(
                        points[:2 * width + 1],
                        width,
                        0,
                        False)[:, :dim]

                    # Descending points
                    points_ma[- idx - 1] = self._filter(
                        points[-2 * width - 1:],
                        width,
                        0,
                        False)[:, :dim]

                return points_ma

