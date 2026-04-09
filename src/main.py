from config import Config
from heston_adi import HestonADI
from mesh_gen import MeshGen
from coeff_matrices import CoeffMatrices
from derivative_matrices import DerivativeMatrices
from pde_coeff import PdeCoeff
from boundary_condition import BoundaryCondition
from split_matrices import SplitMatrices
from forcing_factor import ForcingFactor
from greeks import ComputeGreeks
from visualize import VisualizeResult

# ---------------------------------------------------------------------------
# Model and numerical parameters (heuristic choices from Section 2)
# ---------------------------------------------------------------------------
# S_max = 30·K  (Section 2 heuristic)
# V_max = 15    (Section 2 heuristic)
# m_2 = m_1/2, N = m_1/2  (balanced spatial / temporal resolution)
# theta = 1/3  gives the MCS scheme (Section 6)
cfg = Config(
    kappa=0.38,
    eta=0.09,
    sigma=1.26,
    rho=-0.55,
    r_d=0.01,
    r_f=0.06,
    K=100.0,
    m_1=100,
    m_2=round(100 / 2),
    S_max=30 * 100.0,
    V_max=15.0,
    T=4.0,
    N=round(100 / 2),
    theta=1 / 3,
)


def main():
    # --- Step 1: Build non-uniform grids in s and v (Section 2)
    mesh_gen = MeshGen(cfg)

    # --- Step 2: PDE coefficients α₀, α₁, α₂, β₁₁, β₁₂, β₂₂ (Sections 1, 5)
    pde_coeff = PdeCoeff(cfg)

    # --- Step 3: Evaluate coefficient matrices on the full grid (Section 5)
    coeff_matrices = CoeffMatrices(mesh_gen, pde_coeff)

    # --- Step 4: Build finite-difference matrices D_s, D_v, D_ss, D_vv (Sections 3, 5)
    derivative_matrices = DerivativeMatrices(mesh_gen)

    # --- Step 5: Apply boundary conditions (Sections 5.1 and 5.2)
    # mod_coeff_matrices  : Ω-matrices with zeroed boundary rows/cols
    # mod_derivative_matrices : D_s, D_v, D_ss with boundary modifications
    [mod_coeff_matrices, mod_derivative_matrices] = BoundaryCondition(
        coeff_matrices, derivative_matrices
    )

    # --- Step 6: Build forcing corrections G¹, E¹, E¹¹ (Sections 5.1, 5.2)
    forcing_factor = ForcingFactor(derivative_matrices)

    # --- Step 7: Assemble the ADI split A₀, A₁, A₂ and g₀, g₁, g₂ (Section 5.4)
    split_matrices = SplitMatrices(
        mod_coeff_matrices, mod_derivative_matrices, forcing_factor
    )

    # --- Step 8: Run the MCS algorithm (Section 6)
    heston_adi = HestonADI(split_matrices)
    res = heston_adi.solver()

    # --- Step 9: Compute Greeks from the solution
    greeks = ComputeGreeks(derivative_matrices, res)

    # --- Step 10: Visualise the option price and Greeks
    visual = VisualizeResult(greeks)
    visual.plot_all()


if __name__ == "__main__":
    main()
