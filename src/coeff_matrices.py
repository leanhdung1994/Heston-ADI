from mesh_gen import MeshGen
from pde_coeff import PdeCoeff
import numpy as np


class CoeffMatrices:
    """
    Evaluates the PDE coefficient functions on the full (s, v) grid and stores
    them as (m_1+1) × (m_2+1) matrices.

    These correspond to the coefficient matrices Ω⁰, Ω¹, Ω², Ω¹¹, Ω²², Ω¹²
    introduced in Section 5:
        Ω⁰ᵢⱼ  = α₀(sᵢ, vⱼ)    (zeroth-order / discount term)
        Ω¹ᵢⱼ  = α₁(sᵢ, vⱼ)    (first-order s-coefficient)
        Ω²ᵢⱼ  = α₂(sᵢ, vⱼ)    (first-order v-coefficient)
        Ω¹¹ᵢⱼ = β₁₁(sᵢ, vⱼ)   (second-order ss-coefficient)
        Ω²²ᵢⱼ = β₂₂(sᵢ, vⱼ)   (second-order vv-coefficient)
        Ω¹²ᵢⱼ = β₁₂(sᵢ, vⱼ)   (mixed sv-coefficient)
    """

    def __init__(self, mesh_gen: MeshGen, pde_coeff: PdeCoeff):
        self.mesh_gen = mesh_gen
        self.pde_coeff = pde_coeff

    def _build_matrix(self, func):
        """
        Evaluate scalar function func(s, v) at every grid point (sᵢ, vⱼ).
        """
        S = self.mesh_gen.S
        V = self.mesh_gen.V
        res = np.zeros(shape=(len(S), len(V)))
        for i in range(len(S)):
            for j in range(len(V)):
                res[i, j] = func(self.mesh_gen.S[i], self.mesh_gen.V[j])
        return res

    def Omega_0(self):
        return self._build_matrix(self.pde_coeff.alpha_0)

    def Omega_1(self):
        return self._build_matrix(self.pde_coeff.alpha_1)

    def Omega_2(self):
        return self._build_matrix(self.pde_coeff.alpha_2)

    def Omega_11(self):
        return self._build_matrix(self.pde_coeff.beta_11)

    def Omega_22(self):
        return self._build_matrix(self.pde_coeff.beta_22)

    def Omega_12(self):
        return self._build_matrix(self.pde_coeff.beta_12)
