"""
Interactive Example:

- Start the script and enhance your comprehension of CCMA
  by dynamically adjusting the parameters in real-time.
- Observe that the uniform distribution performs poorly for both MA and CCMA,
  highlighting the importance of choosing suitable kernels.
- Recognize the superior accuracy of the Pascal triangle kernel compared to Hanning.
  However, be mindful that the Pascal kernel tends to be more accurate while sacrificing some smoothness.

Path Description
- The path consists of multiple straight lines, an arc, a sinus curve and a discontinuous jump.
- Consequently, the path provides C0-, C1- and C2-discontinuities.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import numpy as np
from ccma import CCMA


class InteractiveUpdater:
    def __init__(self, sigma_init=.025, w_ma_init=4, w_cc_init=2, rho_init=0.1, distrib_init="hanning"):
        self.w_ma = w_ma_init
        self.w_cc = w_cc_init
        self.sigma = sigma_init
        self.rho = rho_init
        self.distrib = distrib_init
        self.x_max, self.x_min, self.y_max, self.y_min = None, None, None, None
        self.points, self.noisy_points = None, None
        self.update_data()

        # Set up the initial figure and axis
        fig, self.ax = plt.subplots(figsize=(11, 7))

        # Create sliders
        ax_w_ma = plt.axes([0.1, 0.25, 0.8, 0.03])
        ax_w_cc = plt.axes([0.1, 0.21, 0.8, 0.03])
        ax_sigma = plt.axes([0.1, 0.17, 0.8, 0.03])
        ax_rho = plt.axes([0.1, 0.13, 0.8, 0.03])
        ax_zoom = plt.axes([0.1, 0.09, 0.8, 0.03])
        ax_x = plt.axes([0.1, 0.05, 0.8, 0.03])
        ax_y = plt.axes([0.1, 0.01, 0.8, 0.03])

        ax_checkboxes = plt.axes([0.05, 0.35, 0.15, 0.2])

        self.slider_w_ma = Slider(ax_w_ma, 'w_ma', 0, 15, valinit=self.w_ma, valstep=1)
        self.slider_w_cc = Slider(ax_w_cc, 'w_cc', 0, 15, valinit=self.w_cc, valstep=1)
        self.slider_sigma = Slider(ax_sigma, 'sigma', 0, 0.075, valinit=0.025)
        self.slider_rho = Slider(ax_rho, 'rho', 0.05, 0.2, valinit=self.rho)
        self.slider_zoom = Slider(ax_zoom, 'zoom', 1, 20, valinit=1)
        self.slider_x = Slider(ax_x, 'shift_x', -1, 1, valinit=0)
        self.slider_y = Slider(ax_y, 'shift_y', -1, 1, valinit=0)

        # Attach the update function to the sliders
        self.slider_w_ma.on_changed(self.update)
        self.slider_w_cc.on_changed(self.update)
        self.slider_sigma.on_changed(self.update)
        self.slider_rho.on_changed(self.update)
        self.slider_zoom.on_changed(self.update)
        self.slider_x.on_changed(self.update)
        self.slider_y.on_changed(self.update)

        self.checkbox_labels = ['hanning', 'pascal', 'uniform']
        self.checkboxes = CheckButtons(
            ax=ax_checkboxes,
            labels=self.checkbox_labels,
            actives=[ele for ele in [True, False, False]],
        )
        self.checkboxes.on_clicked(self.update)

        # General settings
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        self.ax.legend()
        self.ax.set_xlim([-2.25, 2.25])
        self.ax.set_ylim([-1.25, 1.25])
        self.ax.set_facecolor((0.95, 0.95, 0.95))

        box = self.ax.get_position()
        self.ax.set_position([box.x0 + 0.1, box.y0 + 0.2, box.width, box.height * 0.8])

        self.update(0.0)
        plt.show()

    def update_data(self):
        # Build path
        points = []

        # Straight line
        start = np.array([-2, 1])
        end = np.array([1, 1])
        dist = np.linalg.norm(start - end)
        points.append(np.linspace(start, end, int(dist / self.rho)))

        # Circle
        n_circle = int(np.pi / self.rho)
        points.append(np.array([1 + np.sin(np.linspace(0, np.pi, n_circle)),
                             np.cos(-np.linspace(0, np.pi, n_circle))]).T[1:])

        # Sinus wave
        n_sinus = int(2 / self.rho)
        points.append(np.array([np.linspace(1, -1, n_sinus),
                              -0.75 - 0.25 * np.cos(np.linspace(0, 2*np.pi, n_sinus))]).T[1:])

        # Straight line 2
        start = np.array([-1, -1])
        end = np.array([-2.0, -1])
        dist = np.linalg.norm(start - end)
        points.append(np.linspace(start, end, int(dist / self.rho))[1:])

        # Straight line 3
        start = np.array([-2.0, -1])
        end = np.array([-1.5, 0])
        dist = np.linalg.norm(start - end)
        points.append(np.linspace(start, end, int(dist / self.rho))[1:])

        # Straight line 4
        start = np.array([-2.0, 0])
        end = np.array([-2.0, 1])
        dist = np.linalg.norm(start - end)
        points.append(np.linspace(start, end, int(dist / self.rho))[1:-1])

        self.points = np.row_stack(points)
        noise = np.random.normal(0, self.sigma, self.points.shape)
        self.noisy_points = self.points + noise

        self.x_max = max(self.points[:, 0]) + 0.25
        self.x_min = min(self.points[:, 0]) - 0.25

        self.y_max = max(self.points[:, 1]) + 0.25
        self.y_min = min(self.points[:, 1]) - 0.25

    def update(self, val):

        # Handle checkbox clicks
        self.checkboxes.eventson = False
        if val in self.checkbox_labels:
            for idx, option_label in enumerate(self.checkbox_labels):
                if option_label == self.distrib:
                    self.checkboxes.set_active(idx)
            self.distrib = val
        self.checkboxes.eventson = True

        # Reload data only if sigma or rho was changed.
        if self.sigma != self.slider_sigma.val:
            self.sigma = self.slider_sigma.val
            self.update_data()

        if self.rho != self.slider_rho.val:
            self.rho = self.slider_rho.val
            self.update_data()

        self.w_ma = int(self.slider_w_ma.val)
        self.w_cc = int(self.slider_w_cc.val)

        # Create the CCMA-filter object and smooth
        ccma = CCMA(self.w_ma, self.w_cc, distrib=self.distrib)
        ccma_points = ccma.filter(self.noisy_points, mode="wrapping")
        ma_points = ccma.filter(self.noisy_points, cc_mode=False, mode="wrapping")

        # Visualize results
        self.ax.clear()
        self.ax.plot(*self.points.T, 'ro', markersize=4, alpha=0.25, label=f"original points")
        self.ax.plot(*self.noisy_points.T, "k-o", linewidth=1, alpha=0.15, markersize=6, label="noisy points")
        self.ax.plot(*ccma_points.T, linewidth=3, alpha=0.53, color="b", label=f"ccma-smoothed")
        self.ax.plot(*ma_points.T, linewidth=3, alpha=0.35, color="green", label=f"ma-smoothed")

        # General settings
        self.ax.grid(True, color="white", linewidth=2)
        self.ax.set_aspect('equal')
        self.ax.legend()

        # Handle zoom and shift
        x_diff = (self.x_max - self.x_min) / self.slider_zoom.val
        y_diff = (self.y_max - self.y_min) / self.slider_zoom.val
        x_center = self.x_min + (self.x_max - self.x_min) * (self.slider_x.val + 1) / 2
        y_center = self.y_min + (self.y_max - self.y_min) * (self.slider_y.val + 1) / 2
        x_min_cur = x_center - x_diff/2
        x_max_cur = x_center + x_diff/2
        y_min_cur = y_center - y_diff/2
        y_max_cur = y_center + y_diff/2
        self.ax.set_xlim([x_min_cur, x_max_cur])
        self.ax.set_ylim([y_min_cur, y_max_cur])

        plt.draw()


# Starts CCMA interaction window
InteractiveUpdater()
