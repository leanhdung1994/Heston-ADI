from mesh_gen import MeshGen
from scipy import sparse
import numpy as np


class DerivativeMatrices:
    """
    Builds the sparse finite-difference matrices D_s, D_v, D_ss, D_vv in Section 5.

    The coefficients are derived from the non-uniform divided-difference formulas
    in Section 3.

    Note: boundary modifications (zeroing certain rows, ghost-point elimination)
    are applied later in boundary_condition.py.
    """

    def __init__(self, mesh_gen: MeshGen):
        self.mesh_gen = mesh_gen
        self.S = self.mesh_gen.S
        self.V = self.mesh_gen.V
        self.D_s = self._gen_first_derivative("s")
        self.D_v = self._gen_first_derivative("v")
        self.D_ss = self._gen_second_derivative("s")
        self.D_vv = self._gen_second_derivative("v")

    def _gen_first_derivative(self, direction):
        """
        Build the (m+1)×(m+1) first-derivative matrix for the given direction:
        - Row 0   (i=0):   forward  difference (3-point, one-sided)
        - Rows 1..m-1:     central  difference (3-point, centred)
        - Row m   (i=m):   backward difference (3-point, one-sided)
        """
        if direction == "s":
            _list = self.S
        elif direction == "v":
            _list = self.V
        else:
            raise ValueError(f"Unknown direction: {direction}")
        m = len(_list) - 1
        res = np.zeros(shape=(m + 1, m + 1))
        for i in range(m + 1):
            if i == 0:
                # Forward difference stencil at the left boundary
                Delta_pp = _list[2] - _list[1]
                Delta_p = _list[1] - _list[0]
                gamma_0 = (-2 * Delta_p - Delta_pp) / (Delta_p * (Delta_p + Delta_pp))
                gamma_1 = (Delta_p + Delta_pp) / ((Delta_p * Delta_pp))
                gamma_2 = -Delta_p / (Delta_pp * (Delta_p + Delta_pp))
                res[0, [0, 1, 2]] = [gamma_0, gamma_1, gamma_2]
            elif i == m:
                # Backward difference stencil at the right boundary
                Delta_z = _list[m] - _list[m - 1]
                Delta_m = _list[m - 1] - _list[m - 2]
                alpha_0 = Delta_z / (Delta_m * (Delta_m + Delta_z))
                alpha_1 = -(Delta_m + Delta_z) / (Delta_m * Delta_z)
                alpha_2 = (Delta_m + 2 * Delta_z) / (Delta_z * (Delta_m + Delta_z))
                res[m, [m - 2, m - 1, m]] = [alpha_0, alpha_1, alpha_2]
            else:
                # Central difference stencil for interior rows
                Delta_z = _list[i] - _list[i - 1]
                Delta_p = _list[i + 1] - _list[i]
                h = Delta_z + Delta_p
                beta_m = -Delta_p / (Delta_z * h)
                beta_z = (Delta_p - Delta_z) / (Delta_z * Delta_p)
                beta_p = Delta_z / (Delta_p * h)
                res[i, [i - 1, i, i + 1]] = [beta_m, beta_z, beta_p]
        res = sparse.csr_matrix(res)
        return res

    def _gen_second_derivative(self, direction):
        """
        Build the (m+1)×(m+1) second-derivative matrix for the given direction.
        """
        if direction == "s":
            _list = self.S
        elif direction == "v":
            _list = self.V
        else:
            raise ValueError(f"Unknown direction: {direction}")
        m = len(_list) - 1
        res = np.zeros(shape=(m + 1, m + 1))
        for i in range(m + 1):
            if i == 0:
                # Forward difference stencil at the left boundary
                Delta_pp = _list[2] - _list[1]
                Delta_p = _list[1] - _list[0]
                gamma_0 = 2 / (Delta_p * (Delta_p + Delta_pp))
                gamma_1 = -2 / (Delta_p * Delta_pp)
                gamma_2 = 2 / (Delta_pp * (Delta_p + Delta_pp))
                res[0, [0, 1, 2]] = [gamma_0, gamma_1, gamma_2]
            elif i == m:
                # Backward difference stencil at the right boundary
                Delta_z = _list[m] - _list[m - 1]
                Delta_m = _list[m - 1] - _list[m - 2]
                alpha_0 = 2 / (Delta_m * (Delta_m + Delta_z))
                alpha_1 = -2 / (Delta_m * Delta_z)
                alpha_2 = 2 / (Delta_z * (Delta_m + Delta_z))
                res[m, [m - 2, m - 1, m]] = [alpha_0, alpha_1, alpha_2]
            else:
                # Central difference stencil for interior rows
                Delta_z = _list[i] - _list[i - 1]
                Delta_p = _list[i + 1] - _list[i]
                h = Delta_z + Delta_p
                beta_m = 2 / (Delta_z * h)
                beta_z = -2 / (Delta_z * Delta_p)
                beta_p = 2 / (Delta_p * h)
                res[i, [i - 1, i, i + 1]] = [beta_m, beta_z, beta_p]
        res = sparse.csr_matrix(res)
        return res
