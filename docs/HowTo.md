# How to Use the CCMA Library

This guide provides a comprehensive overview of the core functionalities and best practices for the CCMA library.

## General Usage

The CCMA library is designed in an object-oriented style. To begin, initialize the CCMA object:

```python
from ccma import CCMA

ccma = CCMA()
```

During initialization, the CCMA object prepares kernels and sets filtering width parameters. You can customize the filtering for moving average over points (`w_ma`) and moving average for curvature correction (`w_cc`) during initialization:

```python
ccma = CCMA(w_ma=3, w_cc=2)
```

Once the CCMA object is initialized, you can smooth a sequence of N points, provided as a NumPy array with dimensions Nx2 or Nx3:

```python
smoothed_points = ccma.filter(noisy_points)
```

### Kernels

The CCMA utilizes kernels to determine weights for convolution operations. Several predefined kernels are available:

- **Uniform (Rectangular)** [`uniform`]—Assigns equal weights to all points.
- **Pascal's Triangle (Default)** [`pascal`]—A discrete Gaussian distribution, balancing accuracy and smoothness.
- **Hanning (Raised Cosine)** [`hanning`]—Ideal for signal processing, particularly effective for smoothing.
- **Truncated Gaussian** [`normal`]—Versatile kernel with adjustable cutoff areas (`rho_ma` and `rho_cc`).

Specify a kernel during initialization:

```python
ccma = CCMA(distrib="pascal")
```

For the truncated Gaussian kernel, you can set the cutoff area:

```python
ccma = CCMA(distrib="normal", rho_ma=0.95, rho_cc=0.9)
```

#### Custom Kernels

If the provided kernels don't meet your requirements, define your own [kernel](https://en.wikipedia.org/wiki/Window_function) function:

```python
ccma = CCMA(distrib=get_triangle_kernel)
```

Here's an example of a custom triangle kernel:

```python
import numpy as np

def get_triangle_kernel(width):
    ramp = np.array(range(1, width + 1))
    half_width = width // 2 + 1
    ramp[-half_width:] = ramp[:half_width][::-1]
    triangle_kernel = ramp / np.sum(ramp)
    return triangle_kernel
```

Note: Kernel width is always an odd value (e.g., 1, 3, 5, ...).

#### Hints

- Start with `pascal` or `hanning` for general use. Avoid `uniform` unless you have specific requirements.

### Boundary Strategies

Similar to convolution, boundaries require additional handling. Different strategies are available:

- **None** [`none`]—No boundary strategy is applied.
- **Padding** `padding`]—Replicate the last points on each end for `w_ccma` times.
- **Wrapping** [`wrapping`]—Treat the sequence as closed/periodic, adding the first and last `w_ccma` points to the opposite end.
- **Decreasing Filtering Width** [`fill_boundary`]—Filter the ends with decreasing width, trying to maintain equality between `w_ma` and `w_cc`. Furthermore, the objective is `w_ma` >= `w_cc`

Example:

```python
ccma = CCMA(mode="fill_boundary", w_ma=1, w_cc=2)
```

### Apply Moving Average Without Curvature Correction

To apply only the moving average without curvature correction, set `cc_mode` to `False`:

```python
ccma = CCMA()
# Apply the moving average without curvature correction
ccma.filter(noisy_points, cc_mode=False)
```

## Further Improvements?

If you have additional insights or suggestions to enhance the overall quality and utility of the code, please feel free to share them. Your input is highly valued, and any ideas for improvement that contribute to the code's clarity, efficiency, or user-friendliness will be greatly appreciated. 

Thank you in advance for your constructive feedback!