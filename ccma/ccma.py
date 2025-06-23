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
    return vec if vec_norm == 0.0 else vec / vec_norm


# =================================================================================================
# -----   CCMA   ----------------------------------------------------------------------------------
# =================================================================================================


class CCMA:
    def __init__(self, w_ma=5, w_cc=3, distrib="pascal", distrib_ma=None, distrib_cc=None, rho_ma=0.95, rho_cc=0.95):
        """
        Initialize the SmoothingFilter object with specified parameters for
        smoothing and curvature correction.

        Parameters:
        - w_ma (float): Width parameter for the moving average.
        - w_cc (float): Width parameter for the curvature correction.
        - distrib (str, optional): Type of kernel used for filtering. Options
          include:
            - "uniform": Uniform distribution of weights.
            - "normal": Truncated normal distribution with specified truncation
              area (see `rho_ma` and `rho_cc`).
            - "pascal": Kernel based on rows of Pascal's triangle, a discretized
              version of the normal distribution. (Default is "pascal")
            - "hanning": A common kernel in signal processing, less accurate but
              offers good smoothing characteristics.
        - distrib_ma (str, optional): Type of kernel used only for moving average
          (ma). Defaults to `distrib` if not specified.
        - distrib_cc (str, optional): Type of kernel used only for curvature
          correction (cc). Defaults to `distrib` if not specified.
        - rho_ma (float, optional): Truncation area for the normal distribution in
          the moving average. Relevant only if `distrib` or `distrib_ma` is
          "normal". (Default is 0.95)
        - rho_cc (float, optional): Truncation area for the normal distribution in
          the curvature correction. Relevant only if `distrib` or `distrib_cc` is
          "normal". (Default is 0.95)

        Notes:
        - The `distrib` parameter specifies the type of kernel used for filtering;
          the kernel is the shape of weights used in the convolution process.
        - `rho_ma` and `rho_cc` apply only when the "normal" kernel is selected and
          control the truncation of the normal distribution.
        - If `rho_ma` or `rho_cc` approximates 0, the "normal" kernel approximates
          the "uniform" kernel.

        Example:
        ```python
        ccma_instance = CCMA(w_ma=5, w_cc=3, distrib="normal",
                             rho_ma=0.99, rho_cc=0.95)
        ```
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
        """
        Generate a list of weight arrays (kernels) for smoothing and curvature
        correction based on the specified distribution.

        This method creates kernels of increasing size, which are used in the
        smoothing and curvature correction processes. The type of kernel is
        determined by the `distrib` parameter, and the size of each kernel is
        determined by the `w` parameter. If the normal distribution is used, the
        `rho` parameter controls the truncation of the distribution.

        Parameters:
        - w (int): The base width for the kernels. The method generates `w+1`
          kernels, starting from size 1 up to size `2*w + 1`.
        - distrib (str or callable): The type of distribution used to generate the
          kernel weights.
        - rho (float): The truncation area for the normal distribution. This is
          only relevant when `distrib` is "normal". It represents the area under
          the curve that is included in the kernel (i.e., `rho = 0.95` includes
          95% of the distribution).

        Returns:
        - weight_list (list of numpy.ndarray): A list of 1D numpy arrays, where
          each array represents the weights of a kernel for a specific window size.

        Raises:
        - ValueError: If `distrib` is not one of the recognized strings ("normal",
          "uniform", "pascal", "hanning") or a callable function.

        Notes:
        - For the "normal" distribution, the weights are calculated by integrating
          the probability density function (PDF) over small intervals determined
          by `rho`.
        - The "pascal" distribution uses rows from Pascal's triangle, normalized
          to sum to 1, creating a set of weights that resemble the binomial
          distribution.
        - The "hanning" distribution generates weights using the Hanning window
          function, which is commonly used for smoothing in signal processing.

        Example:
        ```python
        weights = ccma_instance._get_weights(w=3, distrib="normal", rho=0.95)
        ```
        This example generates a list of kernels using a normal distribution with
        a base width of 3 and a truncation area of 95%.
        """

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
                cur_row = [1.0]

                if row_index == 0:
                    return cur_row

                prev = get_pascal_row(row_index - 1)

                for idx_ in range(1, len(prev)):
                    cur_row.append(prev[idx_ - 1] + prev[idx_])

                cur_row.append(1.0)

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
            raise ValueError("Distribution must either be 'uniform', 'pascal', 'hanning, 'normal' or a callable function, that generates a valid distribution.")

        return weight_list

    @staticmethod
    def _get_3d_from_2d(points):
        """
        Convert 2D points to 3D by adding a zero-filled z-dimension.

        This function takes a set of 2D points and adds a third dimension (z-axis)
        by filling it with zeros, allowing the same calculations to be used for
        both 2D and 3D points.

        Parameters:
        - points (numpy.ndarray): A 2D array of shape (n, 2), where `n` is the
          number of points. Each row represents a point in 2D space.

        Returns:
        - numpy.ndarray: A 2D array of shape (n, 3), where each 2D point is
          augmented with a zero in the z-dimension, effectively converting it to
          3D.

        Example:
        ```python
        points_2d = np.array([[1, 2], [3, 4], [5, 6]])
        points_3d = _get_3d_from_2d(points_2d)
        # points_3d is now array([[1., 2., 0.],
        #                        [3., 4., 0.],
        #                        [5., 6., 0.]])
        ```
        """

        return np.column_stack([points, np.zeros(points.shape[0])])

    @staticmethod
    def _get_ma_points(points, weights):
        """
        Apply convolution to each dimension of a set of points and combine the
        results.

        This function performs a moving average (MA) calculation on each dimension
        (x, y, z) of a set of 3D points using the specified weights. The
        convolution is applied separately to each dimension, and the results are
        then combined into a single array.

        Parameters:
        - points (numpy.ndarray): A 2D array of shape (n, 3), where `n` is the
          number of points. Each row represents a point in 3D space, with columns
          corresponding to the x, y, and z coordinates.
        - weights (numpy.ndarray): A 1D array of weights to be used in the
          convolution. These weights define the moving average filter applied to
          each dimension.

        Returns:
        - numpy.ndarray: A 2D array of shape (m, 3), where `m` is the reduced
          number of points after applying the convolution (dependent on the
          `weights` array length). Each row contains the convolved x, y, and z
          coordinates.

        Example:
        ```python
        points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        weights = np.array([0.25, 0.5, 0.25])
        smoothed_points = ccma_instance._get_ma_points(points, weights)
        # smoothed_points is now array([[4., 5., 6.],
        #                               [7., 8., 9.]])
        ```
        """

        return np.column_stack([np.convolve(points[:, 0], weights, mode='valid'),
                                np.convolve(points[:, 1], weights, mode='valid'),
                                np.convolve(points[:, 2], weights, mode='valid')])

    @staticmethod
    def _get_curvature_vectors(points):
        """
        Calculate curvature vectors for a sequence of 3D points.

        This function computes the curvature at each point in a sequence of 3D
        points by evaluating the circumcircle formed by each triplet of consecutive
        points. The curvature vector is directed along the normal to the plane
        defined by these points and its magnitude is inversely proportional to the
        radius of the circumcircle.

        Parameters:
        - points (numpy.ndarray): A 2D array of shape (n, 3), where `n` is the
          number of points. Each row represents a point in 3D space with x, y, and
          z coordinates.

        Returns:
        - numpy.ndarray: A 2D array of shape (n, 3) containing the curvature
          vectors for each point. The curvature vector at the first and last points
          will be zero since curvature cannot be calculated at the endpoints.

        Example:
        ```python
        points = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, -1, 0]])
        curvature_vectors = ccma_instance._get_curvature_vectors(points)
        # curvature_vectors is now array([[0., 0., 0.],
        #                                 [0., 0., 1.],
        #                                 [0., 0., -1.],
        #                                 [0., 0., 0.]])
        ```
        """

        curvature_vectors = np.zeros_like(points)

        for i in range(1, points.shape[0] - 1):
            # Extract points of circumcircle
            p0, p1, p2 = points[i - 1], points[i], points[i + 1]

            # Calculate vectors, cross product and the norm of the cross product
            v1 = p1 - p0
            v2 = p2 - p1
            cross = np.cross(v1, v2)
            cross_norm = np.linalg.norm(cross)

            if cross_norm != 0.0:
                radius = np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(p2 - p0) / (2 * cross_norm)
                curvature = 1.0 / radius
            else:
                # If vectors v1 and v2 are aligned the cross product is 0 and consequently, the form a straight line
                curvature = 0.0

            curvature_vectors[i] = curvature * get_unit_vector(cross)

        return curvature_vectors

    @staticmethod
    def _get_alphas(points, curvatures):
        """
        Calculate the angles (alphas) between consecutive points assuming that
        three points form a circle.

        This function computes the angle (alpha) at each point in a sequence of 3D
        points. The angle is derived under the assumption that three consecutive
        points lie on the circumference of a circle, and is based on the
        curvature at each point.

        Parameters:
        - points (numpy.ndarray): A 2D array of shape (n, 3), where `n` is the
          number of points. Each row represents a point in 3D space with x, y, and
          z coordinates.
        - curvatures (numpy.ndarray): A 1D array of shape (n,) containing the
          curvature at each point. The curvature should be precomputed and provided
          to the function.

        Returns:
        - numpy.ndarray: A 1D array of shape (n,) containing the angle (alpha) at
          each point. The angles at the first and last points will be zero since
          they do not have two neighbors to form a circle and do not matter anyway.
        """

        alphas = np.zeros(points.shape[0])

        # Loop through each triplet of points (excluding the first and last points)
        for idx in range(1, points.shape[0] - 1):
            curvature = curvatures[idx]
            if curvature != 0.0:
                radius = 1 / curvature
                dist_neighbors = np.linalg.norm(points[idx + 1] - points[idx - 1])

                # If this calculation is not clear please refer to ::
                #   https://github.com/UniBwTAS/ccma/issues/2
                alphas[idx] = np.arcsin((dist_neighbors / 2) / radius)
            else:
                alphas[idx] = 0.0

        return alphas

    @staticmethod
    def _get_normalized_ma_radii(alphas, w_ma, weights):
        """
        Calculate normalized radii for each point based on moving average filter
        and curvature angles.

        This function computes the normalized radius for each point by applying
        a moving average filter to the angles (alphas) between consecutive points.
        It is assumed that the original radius is 1, so the computed radii are
        normalized accordingly (<=1). The result is used to correct the curvature
        by adjusting the points based on the filtered radii.

        This function is crucial to understand the CCMA. We recommend to have
        a look at the paper (figure 2) ::
        https://www.researchgate.net/publication/372692752_A_Simple_and_Model-Free_Path_Filtering_Algorithm_for_Smoothing_and_Accuracy

        Parameters:
        - alphas (numpy.ndarray): A 1D array of angles (alphas) derived from
          consecutive points.
        - w_ma (int): The width parameter for the moving average filter, which
          determines the number of neighboring points considered in the radius
          calculation.
        - weights (numpy.ndarray): A 1D array of weights for the moving average
          filter. The length of this array should be consistent with the value of
          `w_ma`.

        Returns:
        - numpy.ndarray: A 1D array of normalized radii corresponding to each point.
          The radii are adjusted with a minimum threshold to avoid unstable corrections.

        Example:
        ```python
        alphas = np.array([0.1, 1.0, 1.0, 0.5, 0.1, 0.2])
        w_ma = 1
        weights = np.array([0.25, 0.5, 0.25])
        radii_ma = ccma_instance._get_normalized_ma_radii(alphas, w_ma, weights)
        # radii_ma is now array([0.   0.77015115 0.77015115 0.93879128 0.99750208 0. ])
        ```
        """

        radii_ma = np.zeros_like(alphas)

        # Compute the normalized ma-radii for each point (excluding the first and last points)
        for idx in range(1, len(alphas) - 1):
            # Start with the central weight
            radius = 1.0 * weights[w_ma]

            for k in range(1, w_ma + 1):
                radius += 2 * np.cos(alphas[idx] * k) * weights[w_ma + k]

            # Apply a threshold to prevent unstable corrections
            radii_ma[idx] = max(0.35, radius)

        return radii_ma

    def _get_descending_width(self):
        """
        Generate a sequence of width parameters (`w_ma` and `w_cc`) with decreasing sizes.

        This function reduces the width parameters `w_ma` and `w_cc` iteratively,
        ensuring that at each step, the larger of the two parameters is reduced by 1.
        The process continues until both parameters are reduced to 0. The result
        is a list of dictionaries, each containing the current values of `w_ma` and `w_cc`.

        Returns:
            list of dict: A list of dictionaries, where each dictionary has two keys:
                - "w_ma": The current width parameter for the moving average.
                - "w_cc": The current width parameter for curvature correction.

        Notes:
            - The reduction ensures that the widths decrease in a controlled manner,
              with priority given to the larger width.
            - This approach helps in handling boundaries effectively by gradually
              reducing the size of the kernel applied to the path.

        Example:
            ```python
            descending_widths = ccma_instance._get_descending_width()
            # descending_widths is now [{'w_ma': 5, 'w_cc': 2}, {'w_ma': 4, 'w_cc': 2}, ...]
            ```
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
        """
        Apply a curvature-corrected moving average (CCMA) filter to a set of 3D points.

        This function performs filtering on the input 3D points using a moving average
        filter. If curvature correction (`cc_mode`) is enabled, it also computes and
        applies curvature correction to the filtered points.

        Parameters:
            points (numpy.ndarray): A 2D array of shape (n, 3), where n is the number
                                    of points, each with 3 coordinates (x, y, z).
            w_ma (int): The width parameter for the moving average filter, which determines
                        the number of neighboring points considered for averaging over the
                        original points.
            w_cc (int): The width parameter for curvature correction, which determines
                        the range of points used for calculating the relationship between
                        the original and filtered curvature.
            cc_mode (bool): Flag indicating whether curvature correction should be applied
                            (True) or only the moving average filter should be used (False).

        Returns:
            numpy.ndarray: A 2D array of filtered points with shape (n - 2 * (w_ma + w_cc + 1), 3)
                           if `cc_mode` is True. Otherwise, the output is (n - 2 * w_ma, 3)

        Notes:
            - When `cc_mode` is False, the function only applies the moving average filter and
              returns the result.
            - When `cc_mode` is True, the function applies curvature correction by calculating
              curvature vectors, angles between points, and adjusting the points based on
              curvature and radii information.

        Example:
            ```python
            filtered_points = filter_instance._filter(points, w_ma=5, w_cc=3, cc_mode=True)
            ```
        """

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
        radii_ma = self._get_normalized_ma_radii(alphas, w_ma, self.weights_ma[w_ma])

        # Allocate
        points_ccma = np.zeros((points.shape[0] - 2 * w_ccma, 3))

        for idx in range(points.shape[0] - 2 * w_ccma):
            # Get tangent vector for the calculation of the shifting
            unit_tangent = get_unit_vector(points_ma[w_cc + idx + 1 + 1] - points_ma[w_cc + idx - 1 + 1])

            # Calculate the weighted shift
            shift = np.zeros(3)
            for idx_cc in range(2 * w_cc + 1):
                # In case the path is straight, no curvature correction is necessary
                if curvatures[idx + idx_cc + 1] == 0.0:
                    continue

                u_vec = get_unit_vector(curvature_vectors[idx + idx_cc + 1])
                weight = self.weights_cc[w_cc][idx_cc]
                shift_magnitude = (1 / curvatures[idx + idx_cc + 1]) * (1 / radii_ma[idx + idx_cc + 1] - 1)
                shift += u_vec * weight * shift_magnitude

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

