from mesh_gen import MeshGen
from greeks import ComputeGreeks
import matplotlib.pyplot as plt
import numpy as np


class VisualizeResult:
    """
    Produces 3-D surface plots of the option price and its Greeks.
    """

    def __init__(self, greeks: ComputeGreeks):
        self.mesh_gen = greeks.mesh_gen
        self.cfg = self.mesh_gen.cfg
        self.S = self.mesh_gen.S  # shape (m_1 + 1,)
        self.V = self.mesh_gen.V  # shape (m_2 + 1,)
        self.m_1 = self.cfg.m_1
        self.m_2 = self.cfg.m_2
        self.K = self.cfg.K

        # Clip to financially relevant region: S in [0, 2K], V in [0, 1]
        i1 = np.searchsorted(self.S, 2 * self.K, side="right")
        j2 = np.searchsorted(self.V, 1.5, side="right")
        sl, vl = slice(0, i1), slice(0, j2)
        self.Sgrid, self.Vgrid = np.meshgrid(self.S[:i1], self.V[:j2], indexing="ij")

        self.U_out = greeks.U[sl, vl]
        self.Delta_out = greeks.Delta[sl, vl]
        self.Gamma_out = greeks.Gamma[sl, vl]
        self.Vega_out = greeks.Vega[sl, vl]

    def plot_option_value(self):
        """Surface plot of the option price U(S, V)."""
        self._plot_surface(self.U_out, "Option value under Heston")

    def plot_delta(self):
        """Surface plot of Delta = dU/dS."""
        self._plot_surface(self.Delta_out, "Delta  (dU/dS)")

    def plot_gamma(self):
        """Surface plot of Gamma = d2U/dS2."""
        self._plot_surface(self.Gamma_out, "Gamma  (d2U/dS2)")

    def plot_vega(self):
        """Surface plot of Vega = dU/dV."""
        self._plot_surface(self.Vega_out, "Vega  (dU/dV)")

    def plot_all(self):
        """Plot all four surfaces and display."""
        self.plot_option_value()
        self.plot_delta()
        self.plot_gamma()
        self.plot_vega()
        plt.show()

    # ------------------------------------------------------------------
    # Internal surface plotter
    # ------------------------------------------------------------------
    def _plot_surface(self, Z, title):
        """
        3-D surface in MATLAB-slide orientation:
          - V axis: front (1) -> back (0), inverted
          - S axis: 0 -> 2K
          - jet colormap, thin grid lines, elev=25 azim=-50
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            self.Vgrid,
            self.Sgrid,
            Z,
            cmap="jet",
            edgecolor="k",
            linewidths=0.15,
            antialiased=True,
        )
        ax.set_xlim(self.Vgrid.max(), 0)  # invert V: high V at back, 0 at front
        ax.set_ylim(0, 2 * self.K)
        ax.set_xlabel("V", fontsize=12, labelpad=8)
        ax.set_ylabel("S", fontsize=12, labelpad=8)
        ax.set_zlabel(title.split()[0], fontsize=12, labelpad=8)
        ax.view_init(elev=25, azim=-50)
        ax.set_title(title, fontsize=13)
        plt.tight_layout()
