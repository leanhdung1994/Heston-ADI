from boundary_condition import ModifyCoeffMatrices
from boundary_condition import ModifyDerivativeMatrices
from forcing_factor import ForcingFactor
from scipy import sparse
import numpy as np


class SplitMatrices:
    """
    Assembles the ADI splitting A = A₀ + A₁ + A₂ and the corresponding
    forcing split g = g₀ + g₁ + g₂ required by the MCS schem.
    The full semi-discrete ODE is (Section 5.4):
        U'(t) = [A₀(t) + A₁(t) + A₂(t)] U(t) + [g₀(t) + g₁(t) + g₂(t)]
    Splitting convention:
        A₀  ← mixed derivative term
        A₁  ← s-direction terms
        A₂  ← v-direction terms
    """

    def __init__(
        self,
        mod_coeff_matrices: ModifyCoeffMatrices,
        mod_derivative_matrices: ModifyDerivativeMatrices,
        forcing_factor: ForcingFactor,
    ):
        self.mod_coeff_matrices = mod_coeff_matrices
        self.mod_derivative_matrices = mod_derivative_matrices
        self.forcing_factor = forcing_factor
        self.m_1 = forcing_factor.m_1
        self.m_2 = forcing_factor.m_2

    # ------------------------------------------------------------------
    # Spatial operator matrices (Section 5.4)
    # ------------------------------------------------------------------

    def A_0(self):
        # sv-mixed diffusion term
        tmp_0 = 2 * sparse.diags(self.mod_coeff_matrices.Omega_12().flatten("F"))
        tmp_1 = sparse.kron(
            self.mod_derivative_matrices.D_v, self.mod_derivative_matrices.D_s
        )
        res = tmp_0 @ tmp_1
        return res

    def A_1(self):
        # s-advection term
        tmp_01 = sparse.diags(self.mod_coeff_matrices.Omega_1().flatten("F"))
        tmp_02 = sparse.kron(sparse.eye(self.m_2 + 1), self.mod_derivative_matrices.D_s)
        tmp_0 = tmp_01 @ tmp_02

        # s-diffusion term
        tmp_11 = sparse.diags(self.mod_coeff_matrices.Omega_11().flatten("F"))
        tmp_12 = sparse.kron(
            sparse.eye(self.m_2 + 1), self.mod_derivative_matrices.D_ss
        )
        tmp_1 = tmp_11 @ tmp_12

        # Half the discount term
        tmp_21 = (1 / 2) * sparse.diags(self.mod_coeff_matrices.Omega_0().flatten("F"))
        tmp_22 = sparse.kron(sparse.eye(self.m_2 + 1), sparse.eye(self.m_1 + 1))
        tmp_2 = tmp_21 @ tmp_22

        res = tmp_0 + tmp_1 + tmp_2
        return res

    def A_2(self):
        # v-advection term
        tmp_01 = sparse.diags(self.mod_coeff_matrices.Omega_2().flatten("F"))
        tmp_02 = sparse.kron(self.mod_derivative_matrices.D_v, sparse.eye(self.m_1 + 1))
        tmp_0 = tmp_01 @ tmp_02

        # v-diffusion term
        tmp_11 = sparse.diags(self.mod_coeff_matrices.Omega_22().flatten("F"))
        tmp_12 = sparse.kron(
            self.mod_derivative_matrices.D_vv, sparse.eye(self.m_1 + 1)
        )
        tmp_1 = tmp_11 @ tmp_12

        # Half the discount term
        tmp_21 = (1 / 2) * sparse.diags(self.mod_coeff_matrices.Omega_0().flatten("F"))
        tmp_22 = sparse.kron(sparse.eye(self.m_2 + 1), sparse.eye(self.m_1 + 1))
        tmp_2 = tmp_21 @ tmp_22

        res = tmp_0 + tmp_1 + tmp_2
        return res

    # ------------------------------------------------------------------
    # Forcing vectors (Section 5.4)
    # ------------------------------------------------------------------

    def g_0(self, t):
        """
        No correction is needed for the A₀ (mixed-derivative) part.
        The forcing correction is distributed entirely into g₁ and g₂.
        """
        tmp = np.zeros(shape=(self.m_1 + 1, self.m_2 + 1))
        return tmp.flatten("F")

    def g_1(self, t):
        """
        Three contributions (all in matrix form, then vectorised column-major):
          - correction for the zeroed D_s bottom row (Neumann at S_max)
          - correction for the ghost-point elimination in D_ss
          - half the G¹ forcing term (for the v=V_max time-derivative condition)
        """
        tmp_0 = np.multiply(
            self.mod_coeff_matrices.Omega_1(), self.forcing_factor.E_1(t)
        )
        tmp_1 = np.multiply(
            self.mod_coeff_matrices.Omega_11(), self.forcing_factor.E_11(t)
        )
        tmp_2 = (1 / 2) * self.forcing_factor.G_1(t)
        tmp = tmp_0 + tmp_1 + tmp_2
        return tmp.flatten("F")

    def g_2(self, t):
        """
        The other half of the G¹ forcing term (for the v=V_max time-derivative condition).
        """
        tmp = (1 / 2) * self.forcing_factor.G_1(t)
        return tmp.flatten("F")
