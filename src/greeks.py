from derivative_matrices import DerivativeMatrices
import numpy as np


class ComputeGreeks:
    """
    Computes option Greeks from the solved price vector U(T).
    The original (unmodified) derivative matrices are intentional here.
    """

    def __init__(self, derivative_matrices: DerivativeMatrices, res: np.ndarray):
        self.mesh_gen = derivative_matrices.mesh_gen
        self.cfg = self.mesh_gen.cfg
        self.m_1 = self.cfg.m_1
        self.m_2 = self.cfg.m_2

        # Reshape the flat solution vector back to a (m_1+1) × (m_2+1) price matrix
        self.U = res.reshape((self.m_1 + 1, self.m_2 + 1), order="F")

        # Delta = ∂U/∂S
        self.Delta = derivative_matrices.D_s @ self.U

        # Gamma = ∂²U/∂S²
        self.Gamma = derivative_matrices.D_ss @ self.U

        # Vega = ∂U/∂V
        self.Vega = self.U @ derivative_matrices.D_v.T
