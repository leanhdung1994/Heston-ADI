from derivative_matrices import DerivativeMatrices
import numpy as np


class ForcingFactor:
    """
    Builds the time-dependent forcing matrices G¹, E¹, E¹¹ (Sections 5.1, 5.2).

    All matrices have shape (m_1+1) × (m_2+1); only specific entries are non-zero.
    """

    def __init__(self, derivative_matrices: DerivativeMatrices):
        self.derivative_matrices = derivative_matrices
        self.mesh_gen = derivative_matrices.mesh_gen
        self.cfg = derivative_matrices.mesh_gen.cfg
        self.m_1 = self.cfg.m_1
        self.m_2 = self.cfg.m_2
        self.S = self.mesh_gen.S

    def G_1(self, t):
        """
        Forcing matrix G¹(t) for the condition at v = V_max (eq. 5.7).
        """
        res = np.zeros(shape=(self.m_1 + 1, self.m_2 + 1))
        for i in range(self.m_1 + 1):
            res[i, self.m_2] = (
                -self.cfg.r_f * self.mesh_gen.S[i] * np.exp(-self.cfg.r_f * t)
            )
        return res

    def E_1(self, t):
        """
        Forcing matrix E¹(t) for the condition eq. 5.5.
        """
        res = np.zeros(shape=(self.m_1 + 1, self.m_2 + 1))
        for j in range(self.m_2 + 1):
            res[self.m_1, j] = np.exp(-self.cfg.r_f * t)
        return res

    def E_11(self, t):
        """
        Forcing matrix E¹¹(t) for the ghost-point correction in
        D_ss at s = S_max.
        """
        m_1 = self.m_1
        _list = self.S
        res = np.zeros(shape=(self.m_1 + 1, self.m_2 + 1))
        Delta_z = _list[m_1] - _list[m_1 - 1]  # Δ_{m_1}
        Delta_p = Delta_z  # ghost spacing ≈ Δ_{m_1}
        gamma_2 = 2 / (Delta_p * (Delta_z + Delta_p))  # δ₊ coefficient
        for j in range(self.m_2 + 1):
            res[self.m_1, j] = gamma_2 * (Delta_z + Delta_p) * np.exp(-self.cfg.r_f * t)
        return res
