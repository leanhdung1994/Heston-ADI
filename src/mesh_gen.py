from config import Config
import numpy as np


class MeshGen:
    """
    Generates the non-uniform spatial grids in the s- and v-directions.
    Both grids concentrate mesh points near financially important regions:
      - s-grid: dense near the strike K         (Section 2.1)
      - v-grid: dense near v = 0                (Section 2.2)
    The resulting arrays self.S and self.V are used by all downstream classes.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Heuristic constants (Section 2)
        self.constant_c = self.cfg.K / 10
        self.constant_r = 1 / 10
        self.constant_d = self.cfg.V_max / 500

        # Auxiliary objects to construct s-mesh points
        self.S_left = self.cfg.K * max(1 / 2, np.exp(-self.constant_r * self.cfg.T))
        self.S_right = self.cfg.K * min(3 / 2, np.exp(self.constant_r * self.cfg.T))
        self.xi_min = np.arcsinh(-self.S_left / self.constant_c)
        self.xi_int = (self.S_right - self.S_left) / self.constant_c
        self.xi_max = self.xi_int + np.arcsinh(
            (self.cfg.S_max - self.S_right) / self.constant_c
        )

        self.varphi = np.vectorize(self._varphi)
        self._gen_mesh_s()
        self._gen_mesh_v()

    def _varphi(self, xi):
        """
        Piecewise transformation φ (Section 2.1).
        """
        if self.xi_min <= xi <= 0:
            return self.S_left + self.constant_c * np.sinh(xi)
        elif 0 < xi < self.xi_int:
            return self.S_left + self.constant_c * xi
        elif self.xi_int <= xi <= self.xi_max:
            return self.S_right + self.constant_c * np.sinh(xi - self.xi_int)
        else:
            raise ValueError(
                f"{xi} is out of valid range [{self.xi_min}, {self.xi_max}]"
            )

    def _gen_mesh_s(self):
        """
        Build the non-uniform s-grid 0 = s_0 < s_1 < ... < s_{m_1} = S_max.
        """
        tmp = np.linspace(self.xi_min, self.xi_max, self.cfg.m_1 + 1)
        self.S = self.varphi(tmp)

    def _gen_mesh_v(self):
        """
        Build the non-uniform v-grid 0 = v_0 < v_1 < ... < v_{m_2} = V_max.
        """
        Delta_psi = np.arcsinh(self.cfg.V_max / self.constant_d) / self.cfg.m_2
        tmp = [j * Delta_psi for j in range(self.cfg.m_2 + 1)]
        tmp = [self.constant_d * np.sinh(psi) for psi in tmp]
        self.V = np.array(tmp)
