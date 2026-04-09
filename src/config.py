from dataclasses import dataclass


@dataclass()
class Config:
    """
    Heston model parameters (Section 1):
        kappa   : mean-reversion rate κ ≥ 0 of the variance process dV
        eta     : long-run variance η (mean-reversion level of V)
        sigma   : volatility-of-variance σ ≥ 0
        rho     : correlation ρ ∈ [-1, 1] between the two Brownian motions W¹, W²
        r_d     : domestic interest rate r_d ≥ 0
        r_f     : foreign interest rate r_f ≥ 0
        K       : strike price of the European call option; payoff = max(S_T - K, 0)

    Spatial grid parameters (Sections 2.1 and 2.2):
        m_1     : number of subintervals in the s-direction (grid has m_1+1 points)
        m_2     : number of subintervals in the v-direction (grid has m_2+1 points)
        S_max   : upper truncation of the s-domain; heuristically 30·K (Section 2)
        V_max   : upper truncation of the v-domain; heuristically 15  (Section 2)

    Time integration parameters (Section 6):
        T       : maturity / time till expiry
        N       : number of uniform time steps; step size Δt = T/N
        theta   : implicitness parameter θ; set to 1/3 for the MCS scheme (eq. 6.1)
    """

    kappa: float
    eta: float
    sigma: float
    rho: float
    r_d: float
    r_f: float
    K: float
    m_1: int
    m_2: int
    S_max: float
    V_max: float
    T: float
    N: int
    theta: float
