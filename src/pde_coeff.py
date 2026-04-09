from config import Config
import numpy as np


class PdeCoeff:
    """
    Evaluates the PDE coefficients of the Heston pricing PDE at a given grid point (s, v).
    Written in the form used for semi-discretization (Section 5, eq. 5.1):
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _drift(self, s, v):
        """Drift column vector (Section 1)"""
        res = np.array(
            [[(self.cfg.r_d - self.cfg.r_f) * s], [self.cfg.kappa * (self.cfg.eta - v)]]
        )
        return res

    def _diffusion(self, s, v):
        """
        Half-Diffusion matrix (Section 1)
        The (1/2) prefactor is included here so that eq. 1.1 can be treated directly as eq. 5.1.
        """
        res = (1 / 2) * np.array(
            [
                [v * pow(s, 2), self.cfg.rho * self.cfg.sigma * v * s],
                [self.cfg.rho * self.cfg.sigma * v * s, pow(self.cfg.sigma, 2) * v],
            ]
        )
        return res

    def alpha_0(self, s, v):
        return -self.cfg.r_d

    def alpha_1(self, s, v):
        return self._drift(s, v)[0, 0]

    def alpha_2(self, s, v):
        return self._drift(s, v)[1, 0]

    def beta_11(self, s, v):
        return self._diffusion(s, v)[0, 0]

    def beta_12(self, s, v):
        return self._diffusion(s, v)[0, 1]

    def beta_22(self, s, v):
        return self._diffusion(s, v)[1, 1]
