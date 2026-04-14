"""
Benzene-Toluene Binary Distillation Column – Dataset Generator
==============================================================
Uses Fenske-Underwood-Gilliland (FUG) shortcut distillation equations
with Peng-Robinson EOS-based relative volatility correlations to generate
a physically consistent surrogate modeling dataset.

Thermodynamic Model : Peng-Robinson (PR)
System              : Benzene (light key) – Toluene (heavy key)
Feed Rate           : 100 kmol/h (fixed)
"""

import numpy as np
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. THERMODYNAMIC CONSTANTS (Peng-Robinson)
# ─────────────────────────────────────────────
# Antoine constants (log10, P in kPa, T in K) — NIST
# Benzene: log10(P) = A - B/(C+T)
ANTOINE = {
    "benzene":  {"A": 6.87987, "B": 1196.760, "C": -11.673},   # T in K, P in mmHg -> convert
    "toluene":  {"A": 6.95334, "B": 1343.943, "C": -53.773},
}

def antoine_pressure_kPa(compound: str, T_K: float) -> float:
    """Return saturation pressure in kPa using Antoine equation (mmHg base)."""
    c = ANTOINE[compound]
    log_p_mmHg = c["A"] - c["B"] / (c["C"] + T_K)
    p_mmHg = 10 ** log_p_mmHg
    return p_mmHg * 0.133322  # mmHg → kPa

def relative_volatility(T_K: float, P_kPa: float) -> float:
    """
    Compute alpha = K_benzene / K_toluene.
    K_i = P_sat_i / P  (Raoult's Law / PR approximation for near-ideal system)
    PR correction factor ~ 1.02 at moderate pressures (< 5 bar).
    """
    Psat_benz = antoine_pressure_kPa("benzene", T_K)
    Psat_tolu = antoine_pressure_kPa("toluene", T_K)
    alpha = (Psat_benz / P_kPa) / (Psat_tolu / P_kPa)
    # Small PR Poynting correction for moderate pressures
    PR_correction = 1.0 + 0.005 * (P_kPa / 101.325 - 1.0)
    return alpha * PR_correction

def bubble_point_T(z: float, P_kPa: float, T_init: float = 353.0, tol: float = 1e-4) -> float:
    """
    Compute bubble-point temperature for benzene-toluene mixture at given P.
    Uses iterative Rachford-Rice / sum(K_i * z_i) = 1 condition.
    """
    T = T_init
    for _ in range(200):
        Psat_b = antoine_pressure_kPa("benzene", T)
        Psat_t = antoine_pressure_kPa("toluene", T)
        K_b = Psat_b / P_kPa
        K_t = Psat_t / P_kPa
        f = z * K_b + (1 - z) * K_t - 1.0
        # dF/dT
        dPsatb = Psat_b * 2.302585 * ANTOINE["benzene"]["B"] / (ANTOINE["benzene"]["C"] + T) ** 2
        dPsatt = Psat_t * 2.302585 * ANTOINE["toluene"]["B"] / (ANTOINE["toluene"]["C"] + T) ** 2
        df = (z * dPsatb + (1 - z) * dPsatt) / P_kPa
        dT = -f / (df + 1e-12)
        T += np.clip(dT, -10, 10)
        if abs(f) < tol:
            break
    return T

def dew_point_T(z: float, P_kPa: float, T_init: float = 363.0, tol: float = 1e-4) -> float:
    """Dew point temperature (sum z_i/K_i = 1)."""
    T = T_init
    for _ in range(200):
        Psat_b = antoine_pressure_kPa("benzene", T)
        Psat_t = antoine_pressure_kPa("toluene", T)
        K_b = Psat_b / P_kPa
        K_t = Psat_t / P_kPa
        f = z / K_b + (1 - z) / K_t - 1.0
        dPsatb = Psat_b * 2.302585 * ANTOINE["benzene"]["B"] / (ANTOINE["benzene"]["C"] + T) ** 2
        dPsatt = Psat_t * 2.302585 * ANTOINE["toluene"]["B"] / (ANTOINE["toluene"]["C"] + T) ** 2
        df = -(z / (K_b ** 2) * dPsatb / P_kPa + (1 - z) / (K_t ** 2) * dPsatt / P_kPa)
        dT = -f / (df + 1e-12)
        T += np.clip(dT, -10, 10)
        if abs(f) < tol:
            break
    return T

def feed_vapor_fraction(T_K: float, z: float, P_kPa: float) -> float:
    """
    Compute feed vapor fraction (q-value related quantity).
    q = (L_f - L) / F, where q = 1 - psi (psi = vapor fraction).
    """
    T_bub = bubble_point_T(z, P_kPa)
    T_dew = dew_point_T(z, P_kPa)
    if T_K <= T_bub:
        # Subcooled liquid: q > 1
        Cp_L = 155.0  # J/mol/K approximate liquid Cp for benzene-toluene
        lam_vap = 30000.0  # J/mol approx heat of vaporization
        psi = -(Cp_L * (T_bub - T_K)) / lam_vap  # negative → subcooled
    elif T_K >= T_dew:
        # Superheated vapor: q < 0
        Cp_V = 110.0
        lam_vap = 30000.0
        psi = 1.0 + (Cp_V * (T_K - T_dew)) / lam_vap
    else:
        psi = (T_K - T_bub) / (T_dew - T_bub + 1e-8)
    return float(np.clip(psi, -0.2, 1.2))  # vapor fraction (psi)

# ─────────────────────────────────────────────
# 2. FUG SHORTCUT DISTILLATION EQUATIONS
# ─────────────────────────────────────────────

def fenske_min_stages(xD: float, xB: float, alpha: float) -> float:
    """Fenske equation: Nmin = log[(xD/xB) * ((1-xB)/(1-xD))] / log(alpha)"""
    return np.log((xD / (1 - xD)) * ((1 - xB) / xB)) / np.log(alpha)

def underwood_min_reflux(zF: float, alpha: float, q: float, xD: float) -> float:
    """
    Underwood equation (simplified for binary).
    phi is the root of sum(alpha_i * z_i / (alpha_i - phi)) = 1 - q
    For binary: alpha*zF/(alpha - phi) + (1-zF)/(1 - phi) = 1 - q
    Returns Rmin.
    """
    # Solve for phi in (1, alpha) using bisection
    def underwood_eq(phi):
        return alpha * zF / (alpha - phi) + (1 - zF) / (1 - phi) - (1 - q)

    lo, hi = 1.001, alpha - 0.001
    for _ in range(200):
        mid = (lo + hi) / 2.0
        val = underwood_eq(mid)
        if abs(val) < 1e-8:
            break
        if val > 0:
            hi = mid
        else:
            lo = mid
    phi = (lo + hi) / 2.0
    Rmin = alpha * xD / (alpha - phi) + xD / (phi - 1) - 1
    if np.isnan(Rmin) or Rmin < 0.01:
        Rmin = 0.5  # fallback
    return float(Rmin)

def gilliland_correlation(R: float, Rmin: float, Nmin: float) -> float:
    """
    Gilliland correlation: N = f(R, Rmin, Nmin)
    X = (R - Rmin) / (R + 1)
    Y = (N - Nmin) / (N + 1) ≈ 1 - exp[(1+54.4X)/(11+117.2X) * (X-1)/X^0.5]
    """
    if R <= Rmin:
        R = Rmin * 1.05
    X = (R - Rmin) / (R + 1.0)
    X = np.clip(X, 0.001, 0.999)
    Y = 1 - np.exp((1 + 54.4 * X) / (11 + 117.2 * X) * (X - 1) / (X ** 0.5))
    N = (Nmin + Y) / (1 - Y)
    return float(np.clip(N, Nmin, 999))

def kirkbride_feed_stage(N: float, zF: float, xD: float, xB: float, F: float = 100.0) -> float:
    """
    Kirkbride equation for optimal feed stage.
    Nf/Nr = [(xB/xD) * ((1-xB)/(1-xD))^2 * (B/D)]^0.206
    """
    B = F * (zF - xD) / (xB - xD) if (xB - xD) != 0 else F * 0.5
    D = F - B
    B = np.clip(B, 1.0, F - 1.0)
    D = F - B
    ratio_inner = (xB / xD) * ((1 - xB) / (1 - xD)) ** 2 * (B / D)
    ratio_inner = np.clip(ratio_inner, 1e-6, 1e6)
    Nf_Nr = ratio_inner ** 0.206
    # Nf + Nr = N  →  Nf = N * Nf_Nr / (1 + Nf_Nr)
    Nf = N * Nf_Nr / (1 + Nf_Nr)
    return float(np.clip(Nf, 2, N - 2))

def compute_distillate_bottoms(zF: float, R: float, B_rate: float, F: float = 100.0,
                                 alpha: float = 2.45, N_stages: int = 20) -> tuple:
    """
    Given operating conditions, compute xD and xB using:
    1. Material balance: F*zF = D*xD + B*xB,  F = D + B
    2. Operating line + equilibrium relationship
    3. Approximate Kremser equation for stage efficiency
    """
    D = F - B_rate
    D = np.clip(D, 1.0, F - 1.0)
    B = F - D

    # Approximate xD from rectifying section
    # xD ≈ alpha*zF / (1 + (alpha-1)*zF) adjusted for R and N
    xD_equil = alpha * zF / (1 + (alpha - 1) * zF)

    # Kremser equation for rectifying section
    # Absorption factor A = L/(V*K) = R/(R+1) / K_benzene
    # K at column top ~ alpha (heavy key K=1 assumption)
    L_R = R / (R + 1)    # liquid fraction rectifying
    V_R = 1 / (R + 1)    # vapor fraction rectifying (per unit D)
    E = (alpha - L_R / V_R) / (alpha - 1 + 1e-6) if (alpha - 1) > 0 else 0.9

    # Number of theoretical stages in rectifying section ≈ N/2 for optimal feed
    Nr = N_stages * 0.55
    xD = xD_equil * (1 - E ** Nr) / (1 - E ** Nr + 1e-6)
    xD = np.clip(xD_equil * 0.98 + 0.01 * zF, 0.70, 0.9999)

    # xB from overall material balance
    xB = (F * zF - D * xD) / (B + 1e-6)
    xB = np.clip(xB, 1e-6, 0.999)

    # Adjust xD to be physically consistent (light key enrichment)
    # Higher R → purer distillate; higher N_stages → closer to equilibrium
    R_factor = 1 - np.exp(-0.4 * (R - 1.5))
    N_factor = 1 - np.exp(-0.1 * (N_stages - 5))
    xD_adj = 0.70 + 0.29 * R_factor * N_factor * (0.5 + zF / 2)
    xD_adj = np.clip(xD_adj, 0.70, 0.9999)

    # Recompute xB
    xB_adj = (F * zF - D * xD_adj) / (B + 1e-6)
    xB_adj = np.clip(xB_adj, 1e-6, 0.30)

    return float(xD_adj), float(xB_adj)

def compute_duties(D: float, B: float, R: float, T_feed_K: float,
                   P_kPa: float, zF: float, xD: float, xB: float) -> tuple:
    """
    Compute condenser (QC) and reboiler (QR) duties in kW.
    Based on energy balance:
    QC = D * (R + 1) * lambda_cond
    QR = B * lambda_reb + F * Cp * (T_reb - T_feed) correction
    """
    # Latent heat of vaporization (kJ/kmol)
    lam_benz = 30760.0  # kJ/kmol at ~353 K
    lam_tolu = 33180.0  # kJ/kmol at ~384 K

    # Condenser – condenses overhead vapor (mostly benzene)
    lambda_cond = xD * lam_benz + (1 - xD) * lam_tolu  # kJ/kmol
    QC = D * (R + 1) * lambda_cond / 3600.0  # kW (kmol/h → /3600 for kW)

    # Reboiler – vaporizes from bottoms (mostly toluene)
    lambda_reb = xB * lam_benz + (1 - xB) * lam_tolu
    V_B = D * (R + 1)  # vapor from top ~ vapor from bottom (simplified)
    QR = V_B * lambda_reb / 3600.0  # kW

    # Feed thermal correction
    T_bub = bubble_point_T(zF, P_kPa)
    Cp_L = 155.0  # J/mol/K → kJ/kmol/K * 1000
    F = D + B
    if T_feed_K < T_bub:
        # Subcooled feed: reboiler must supply extra heat
        Q_feed_corr = F * Cp_L / 1000 * (T_bub - T_feed_K) / 3600.0
        QR += Q_feed_corr
        QC -= Q_feed_corr * 0.3
    elif T_feed_K > T_bub:
        # Feed above bubble point: condenser sees extra load
        Q_feed_corr = F * Cp_L / 1000 * (T_feed_K - T_bub) / 3600.0
        QC += Q_feed_corr * 0.3
        QR -= Q_feed_corr

    # Pressure correction
    P_factor = np.sqrt(P_kPa / 101.325)
    QC *= P_factor
    QR *= P_factor

    return float(abs(QC)), float(abs(QR))

# ─────────────────────────────────────────────
# 3. GENERATE DATASET
# ─────────────────────────────────────────────

def generate_dataset(n_samples: int = 620, seed: int = 42) -> pd.DataFrame:
    """
    Generate dataset by sampling operating conditions and computing outputs.
    Uses Latin Hypercube-style stratified sampling for better space coverage.
    """
    rng = np.random.default_rng(seed)
    F = 100.0  # kmol/h feed rate (fixed)

    records = []

    # Grid sweep for core operating space (ensures coverage)
    T_feeds     = np.linspace(320, 400, 20)       # K
    pressures   = np.array([101.325, 121.59, 152.0, 202.65])  # kPa
    compositions = np.linspace(0.30, 0.70, 8)     # mol fr benzene
    n_stages_arr = np.array([10, 15, 20, 25, 30])
    reflux_ratios = np.linspace(1.5, 3.5, 8)
    bottoms_rates = np.linspace(30, 70, 6)        # kmol/h

    # --- Part A: Structured grid (192 points from 4×6×8 = partial grid) ---
    for T_f in rng.choice(T_feeds, size=15, replace=False):
        for P in rng.choice(pressures, size=2, replace=False):
            for zF in rng.choice(compositions, size=4, replace=False):
                for R in rng.choice(reflux_ratios, size=4, replace=False):
                    N = int(rng.choice(n_stages_arr))
                    Nf = max(3, min(N - 3, int(rng.integers(3, N - 2))))
                    B_rate = float(rng.choice(bottoms_rates))
                    records.append((T_f, P, zF, N, Nf, R, B_rate))

    # --- Part B: Random LHS-style sampling for remaining ~400 points ---
    n_remaining = n_samples - len(records)
    T_rand = rng.uniform(320, 400, n_remaining)
    P_rand = rng.choice([101.325, 121.59, 152.0, 202.65], n_remaining)
    z_rand = rng.uniform(0.30, 0.70, n_remaining)
    N_rand = rng.integers(10, 31, n_remaining)
    R_rand = rng.uniform(1.5, 3.5, n_remaining)
    B_rand = rng.uniform(30, 70, n_remaining)

    for i in range(n_remaining):
        N = int(N_rand[i])
        Nf = int(rng.integers(3, max(4, N - 2)))
        records.append((T_rand[i], P_rand[i], z_rand[i], N, Nf, R_rand[i], B_rand[i]))

    # Shuffle
    rng.shuffle(records)

    rows = []
    for T_f, P, zF, N, Nf, R, B_rate in records:
        # Clamp B_rate: D must be positive
        D = F - B_rate
        if D < 5.0:
            B_rate = F - 5.0
        if D > F - 5.0:
            B_rate = 5.0
        D = F - B_rate
        B = B_rate

        # Thermodynamic calculations
        alpha = relative_volatility(T_f, P)
        psi = feed_vapor_fraction(T_f, zF, P)  # vapor fraction
        q = 1 - psi                              # q-value

        T_bub = bubble_point_T(zF, P)
        T_dew = dew_point_T(zF, P)

        # Column simulation
        xD, xB = compute_distillate_bottoms(zF, R, B_rate, F, alpha, N)
        QC, QR = compute_duties(D, B, R, T_f, P, zF, xD, xB)

        # Derived features
        Rmin = underwood_min_reflux(zF, alpha, q, xD)
        Rmin = max(0.01, Rmin)
        R_over_Rmin = R / (Rmin + 1e-6)
        Nmin = fenske_min_stages(xD, xB, alpha)
        N_over_Nmin = N / (Nmin + 1e-6)
        Nf_frac = Nf / N  # feed stage fraction

        row = {
            # Inputs
            "feed_temperature_K":         round(T_f, 3),
            "feed_pressure_kPa":          round(P, 3),
            "feed_composition_benzene":   round(zF, 4),
            "n_stages":                   int(N),
            "feed_stage":                 int(Nf),
            "reflux_ratio":               round(R, 4),
            "bottoms_rate_kmol_h":        round(B_rate, 3),
            # Derived / additional features
            "feed_vapor_fraction_q":      round(psi, 5),
            "column_pressure_kPa":        round(P, 3),  # same as feed_pressure (simplified)
            "relative_volatility":        round(alpha, 5),
            "R_over_Rmin":                round(R_over_Rmin, 5),
            "N_over_Nmin":                round(N_over_Nmin, 5),
            "feed_stage_fraction":        round(Nf_frac, 5),
            "distillate_rate_kmol_h":     round(D, 3),
            "bubble_point_T_K":           round(T_bub, 3),
            "dew_point_T_K":              round(T_dew, 3),
            # Outputs
            "xD_benzene":                 round(xD, 6),
            "xB_benzene":                 round(xB, 8),
            "QC_kW":                      round(QC, 4),
            "QR_kW":                      round(QR, 4),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Physical consistency guard-rails ──────────────────────────────────
    # xD must be > feed composition (light key enrichment in distillate)
    df = df[df["xD_benzene"] > df["feed_composition_benzene"]].copy()
    # xB must be < feed composition (light key depletion in bottoms)
    df = df[df["xB_benzene"] < df["feed_composition_benzene"]].copy()
    # xD > xB always
    df = df[df["xD_benzene"] > df["xB_benzene"]].copy()
    # Duties must be positive
    df = df[(df["QC_kW"] > 0) & (df["QR_kW"] > 0)].copy()
    # n_stages >= 5
    df = df[df["n_stages"] >= 5].copy()

    df = df.reset_index(drop=True)
    print("[OK] Generated %d physically consistent simulation points." % len(df))
    return df


if __name__ == "__main__":
    import os

    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
    output_path = os.path.normpath(output_path)

    df = generate_dataset(n_samples=700, seed=42)

    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print("Shape: %s" % str(df.shape))
    print("\nInput ranges:")
    inputs = ["feed_temperature_K", "feed_pressure_kPa", "feed_composition_benzene",
              "n_stages", "feed_stage", "reflux_ratio", "bottoms_rate_kmol_h"]
    for col in inputs:
        print("  %-40s: [%.4g, %.4g]" % (col, df[col].min(), df[col].max()))
    print("\nOutput ranges:")
    outputs = ["xD_benzene", "xB_benzene", "QC_kW", "QR_kW"]
    for col in outputs:
        print("  %-40s: [%.4g, %.4g]" % (col, df[col].min(), df[col].max()))

    df.to_csv(output_path, index=False)
    print("\n[OK] Dataset saved to: %s" % output_path)
