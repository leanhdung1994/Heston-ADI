from split_matrices import SplitMatrices
import scipy.integrate as integrate
from scipy.sparse.linalg import splu
from scipy import sparse
import numpy as np


class HestonADI:
    """
    Solves the Heston pricing PDE using the Modified Craig-Sneyd (MCS) ADI scheme
    (Section 6, eq. 6.1).

    The MCS scheme advances from U^{n-1} to U^n over one time step Δt = T/N via
    seven sub-stages (eq. 6.1a–6.1e).

    θ = 1/3 gives the MCS scheme (second-order accurate; Section 6).
    """

    def __init__(self, split_matrices: SplitMatrices):
        self.mesh_gen = split_matrices.mod_derivative_matrices.mesh_gen
        self.cfg = self.mesh_gen.cfg
        self.T = self.cfg.T
        self.N = self.cfg.N
        self.theta = self.cfg.theta  # θ = 1/3 for MCS

        # Assemble the three split operators (Sections 5.4 and 6)
        self.A_0 = split_matrices.A_0()  # Mixed-derivative part
        self.A_1 = split_matrices.A_1()  # s-direction part
        self.A_2 = split_matrices.A_2()  # v-direction part
        self.A = self.A_0 + self.A_1 + self.A_2  # Full spatial operator

        # Forcing term g₀, g₁, g₂ (Section 5.4)
        self.g_0 = split_matrices.g_0
        self.g_1 = split_matrices.g_1
        self.g_2 = split_matrices.g_2

        self.m = self.A_0.shape[0]  # Total unknowns = (m_1+1)(m_2+1)
        self.Delta_t = self.cfg.T / self.cfg.N  # Time discretization step size

        # Pre-factor the two implicit systems (reused every time step)
        self.LU_1 = self._gen_LU_1()  # LU factorisation of (I − θΔt·A₁)
        self.LU_2 = self._gen_LU_2()  # LU factorisation of (I − θΔt·A₂)

    # ------------------------------------------------------------------
    # Right-hand side helpers
    # ------------------------------------------------------------------

    def g(self, t):
        """Total forcing vector g(t) = g₀(t) + g₁(t) + g₂(t)."""
        res = self.g_0(t) + self.g_1(t) + self.g_2(t)
        return res

    def F(self, t, U):
        """Full RHS: F(t, U) = A·U + g(t)  (used in explicit predictor stages)."""
        res = self.A @ U + self.g(t)
        return res

    def F_0(self, t, U):
        """Mixed-derivative part: F₀(t, U) = A₀·U + g₀(t)."""
        res = self.A_0 @ U + self.g_0(t)
        return res

    def F_1(self, t, U):
        """s-direction part: F₁(t, U) = A₁·U + g₁(t)."""
        res = self.A_1 @ U + self.g_1(t)
        return res

    def F_2(self, t, U):
        """v-direction part: F₂(t, U) = A₂·U + g₂(t)."""
        res = self.A_2 @ U + self.g_2(t)
        return res

    # ------------------------------------------------------------------
    # Initial condition
    # ------------------------------------------------------------------

    def init_func(self, s):
        """Payoff function φ(s) = max(s − K, 0) for a European call (Section 1)."""
        res = max(s - self.cfg.K, 0)
        return res

    def U_0(self):
        """
        Build the initial condition vector U(0) = φ(s) evaluated on the full grid.

        At the grid point sᵢ nearest to K, use the cell-average of the payoff
        (Section 4) to reduce the kink error.
        """
        S = self.mesh_gen.S
        V = self.mesh_gen.V
        res = np.zeros(shape=(len(S), len(V)))
        for i in range(len(S)):
            for j in range(len(V)):
                res[i, j] = self.init_func(S[i])

        # Replace the s-index nearest K with the cell-average value
        diff = np.abs(S - self.cfg.K)
        s_idx = np.where(diff == diff.min())[0]
        for i in s_idx:
            s_m = (S[i - 1] + S[i]) / 2
            s_p = (S[i] + S[i + 1]) / 2
            h = s_p - s_m
            for j in range(len(V)):
                tmp = integrate.quad(self.init_func, s_m, s_p)[0] / h
                res[i, j] = tmp

        return res.flatten("F")

    # ------------------------------------------------------------------
    # LU factorisations for implicit stages
    # ------------------------------------------------------------------

    def _gen_LU_1(self):
        """
        Sparse LU factorisation of (I − θΔt·A₁).
        This system is solved twice per time step (steps j=1 in eq. 6.1b and 6.1e).
        Factoring once and reusing is valid because A₁ is time-independent.
        """
        I = sparse.eye(self.m)
        tmp = I - self.theta * self.Delta_t * self.A_1
        res = splu(tmp.tocsc())
        return res

    def _gen_LU_2(self):
        """
        Sparse LU factorisation of (I − θΔt·A₂).
        This system is solved twice per time step (steps j=2 in eq. 6.1b and 6.1e).
        """
        I = sparse.eye(self.m)
        tmp = I - self.theta * self.Delta_t * self.A_2
        res = splu(tmp.tocsc())
        return res

    # ------------------------------------------------------------------
    # MCS time-stepping
    # ------------------------------------------------------------------

    def ADI_iteration(self, n, U):
        """
        Advance one time step using the MCS scheme (Section 6, eq. 6.1).
        """
        t_m = self.Delta_t * n
        t_z = self.Delta_t * (n + 1)
        U_m = U
        Y_0 = U_m + self.Delta_t * self.F(t_m, U_m)
        # Solve for Y_1
        b = Y_0 + self.theta * self.Delta_t * (self.g_1(t_z) - self.F_1(t_m, U_m))
        Y_1 = self.LU_1.solve(b)
        # Solve for Y_2
        b = Y_1 + self.theta * self.Delta_t * (self.g_2(t_z) - self.F_2(t_m, U_m))
        Y_2 = self.LU_2.solve(b)
        # Solve for hat_Y_0
        hat_Y_0 = Y_0 + self.theta * self.Delta_t * (
            self.F_0(t_z, Y_2) - self.F_0(t_m, U_m)
        )
        # Solve for tilde_Y_0
        tilde_Y_0 = hat_Y_0 + ((1 / 2) - self.theta) * self.Delta_t * (
            self.F(t_z, Y_2) - self.F(t_m, U_m)
        )
        # Solve for tilde_Y_1
        b = tilde_Y_0 + self.theta * self.Delta_t * (self.g_1(t_z) - self.F_1(t_m, U_m))
        tilde_Y_1 = self.LU_1.solve(b)
        # Solve for tilde_Y_2
        b = tilde_Y_1 + self.theta * self.Delta_t * (self.g_2(t_z) - self.F_2(t_m, U_m))
        tilde_Y_2 = self.LU_2.solve(b)
        return [n + 1, tilde_Y_2]

    def solver(self):
        """
        Run the full MCS time-march from t=0 to t=T over N steps.
        Returns the final solution vector U(T) of length (m_1+1)(m_2+1).
        """
        sol = [0, self.U_0()]
        for i in range(self.N):
            sol = self.ADI_iteration(*sol)
        res = sol[1]
        return res
