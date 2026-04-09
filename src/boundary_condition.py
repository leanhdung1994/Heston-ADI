from coeff_matrices import CoeffMatrices
from derivative_matrices import DerivativeMatrices
import copy


class ModifyCoeffMatrices(CoeffMatrices):
    """
    Modifies the PDE coefficient matrices to enforce the boundary
    conditions (5.6) and (5.7) at s = 0 and v = V_max.
    """

    def __init__(self, original: CoeffMatrices):
        # Deep-copy all attributes from the base CoeffMatrices instance
        self.__dict__ = copy.deepcopy(original.__dict__)

    def _build_matrix(self, func):
        """
        Build the coefficient matrix and then zero out first row and last column
        """
        res = super()._build_matrix(func)
        res[0, :] = 0  # Enforce eq. 5.6
        res[:, -1] = 0  # Enforce eq. 5.7
        return res


class ModifyDerivativeMatrices(DerivativeMatrices):
    """
    Modifies the derivative matrices D_s, D_v, D_ss to incorporate boundary
    conditions at s = S_max and v = V_max (Sections 5.1 and 5.2).
    """

    def __init__(self, original: DerivativeMatrices):
        # Deep-copy all attributes from the base DerivativeMatrices instance
        self.__dict__ = copy.deepcopy(original.__dict__)
        self.m_1 = self.D_s.shape[0] - 1
        self.m_2 = self.D_v.shape[0] - 1
        self.D_s = self._apply_boundary_to_D_s(self.D_s)
        self.D_v = self._apply_boundary_to_D_v(self.D_v)
        self.D_ss = self._apply_boundary_to_D_ss(self.D_ss)

    def _apply_boundary_to_D_s(self, matrix):
        """
        Zero the bottom row of D_s (row index m_1, corresponding to s = S_max).
        """
        res = matrix.tolil()
        res[-1, :] = 0  # Zero row m_1 (s = S_max)
        return res.tocsr()

    def _apply_boundary_to_D_v(self, matrix):
        """
        Replace central-difference rows with backward-difference for vⱼ ≥ 1.
        """
        res = matrix.tolil()
        _list = self.V
        for j in range(self.m_2 + 1):
            if self.V[j] >= 1:
                Delta_z = _list[j] - _list[j - 1]  # Δⱼ
                Delta_m = _list[j - 1] - _list[j - 2]  # Δⱼ₋₁
                alpha_0 = Delta_z / (Delta_m * (Delta_m + Delta_z))
                alpha_1 = -(Delta_m + Delta_z) / (Delta_m * Delta_z)
                alpha_2 = (Delta_m + 2 * Delta_z) / (Delta_z * (Delta_m + Delta_z))
                res[j, :] = 0
                res[j, [j - 2, j - 1, j]] = [alpha_0, alpha_1, alpha_2]
        return res.tocsr()

    def _apply_boundary_to_D_ss(self, matrix):
        """
        Modify the bottom row of D_ss (i=m_1) to eliminate the ghost point.
        """
        m_1 = self.m_1
        _list = self.S
        res = matrix.tolil()
        res[m_1, :] = 0  # Clear existing row m_1
        Delta_z = _list[m_1] - _list[m_1 - 1]  # Δ_{m_1}
        Delta_p = Delta_z  # Δ_{m_1+1} ≈ Δ_{m_1} (ghost spacing)
        gamma_0 = 2 / (Delta_z * (Delta_z + Delta_p))  # δ₋
        gamma_1 = -2 / (Delta_z * Delta_p)  # δ₀
        gamma_2 = 2 / (Delta_p * (Delta_z + Delta_p))  # δ₊
        # After ghost-point elimination: coefficient of f_{m_1-1} becomes δ₋ + δ₊
        res[m_1, [m_1 - 1, m_1]] = [gamma_0 + gamma_2, gamma_1]
        return res.tocsr()


class BoundaryCondition:
    """
    Combines the two boundary-modification steps into a single object. Produces:
      - ModifyCoeffMatrices  : Ω-matrices with zeroed boundary rows/cols (Section 5.1)
      - ModifyDerivativeMatrices : D_s, D_v, D_ss with boundary modifications (Section 5.2)
    """

    def __init__(
        self, coeff_matrices: CoeffMatrices, derivative_matrices: DerivativeMatrices
    ):
        self._coeff = ModifyCoeffMatrices(coeff_matrices)
        self._deriv = ModifyDerivativeMatrices(derivative_matrices)

    def __iter__(self):
        return iter([self._coeff, self._deriv])
