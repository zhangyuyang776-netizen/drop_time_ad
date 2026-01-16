"""
Gas-phase diffusive fluxes.

Responsibilities
----------------
- Compute conductive heat flux on gas faces using props + grid + Tg.
- Compute mixture-averaged species diffusive flux on gas faces, with a
  per-face correction so that sum_k J_k,face approx 0 on internal gas faces.
- Respect CaseConventions: radial_normal="+er", flux_sign="outward_positive",
  heat_flux_def="q=-k*dTdr".
- Do not mutate state or layout; no property evaluation here.

Scope / non-responsibilities
----------------------------
- Interface (inner) face species and energy fluxes are handled explicitly in
  physics/interface_bc.py (Stefan condition + phase-equilibrium at the droplet
  surface).
- Outer boundary (far-field) species and temperature are handled in the system
  assembly modules (build_system_SciPy, build_species_system_SciPy) via
  Dirichlet / convective BC.
- Enthalpy diffusion q_diff = sum_k(h_k * J_k) is assembled in
  physics/energy_flux.py using the J_diff_g returned here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.types import CaseConfig, Grid1D, Props

FloatArray = np.ndarray


def _check_conventions(cfg: CaseConfig) -> None:
    conv = cfg.conventions
    if conv.radial_normal != "+er":
        raise ValueError("flux_gas assumes radial_normal='+er'")
    if conv.flux_sign != "outward_positive":
        raise ValueError("flux_gas assumes flux_sign='outward_positive'")
    if conv.heat_flux_def != "q=-k*dTdr":
        raise ValueError("flux_gas assumes heat_flux_def='q=-k*dTdr'")


def compute_gas_diffusive_flux_T(
    cfg: CaseConfig,
    grid: Grid1D,
    props: Props,
    Tg: FloatArray,
) -> FloatArray:
    """
    Compute conductive heat flux on gas faces.

    This is the conductive component q_cond only; enthalpy diffusion q_diff
    = sum_k(h_k * J_k) is assembled in physics/energy_flux.py using the
    species diffusive fluxes J_diff_g from compute_gas_diffusive_flux_Y.

    Parameters
    ----------
    cfg : CaseConfig
    grid : Grid1D
    props : Props
        Must contain k_g shape (Ng,).
    Tg : (Ng,) ndarray
        Gas temperatures.

    Returns
    -------
    q_cond : (Nc+1,) ndarray
        Conductive heat flux component on faces, outward positive, consistent with q = -k dT/dr.
    """
    _check_conventions(cfg)
    Nl, Ng, Nc = grid.Nl, grid.Ng, grid.Nc
    gas_start = grid.gas_slice.start if grid.gas_slice is not None else Nl

    if Tg.shape != (Ng,):
        raise ValueError(f"Tg shape {Tg.shape} != ({Ng},)")
    if props.k_g.shape != (Ng,):
        raise ValueError(f"k_g shape {props.k_g.shape} != ({Ng},)")
    if grid.r_c.shape != (Nc,):
        raise ValueError(f"r_c shape {grid.r_c.shape} != ({Nc},)")
    if grid.r_f.shape != (Nc + 1,):
        raise ValueError(f"r_f shape {grid.r_f.shape} != ({Nc+1},)")
    if grid.A_f.shape != (Nc + 1,):
        raise ValueError(f"A_f shape {grid.A_f.shape} != ({Nc+1},)")

    q_cond = np.zeros_like(grid.A_f, dtype=np.float64)  # (Nc+1,)

    # internal gas faces
    for ig in range(Ng - 1):
        iL = gas_start + ig
        iR = gas_start + ig + 1
        f = iR  # shared face index
        rL = grid.r_c[iL]
        rR = grid.r_c[iR]
        dr = rR - rL
        if dr <= 0.0:
            raise ValueError("Non-positive dr in gas region")
        k_face = 0.5 * (props.k_g[ig] + props.k_g[ig + 1])
        dTdr = (Tg[ig + 1] - Tg[ig]) / dr
        q_cond[f] = -k_face * dTdr

    # Inner gas boundary (interface): keep zero here.
    # The actual interfacial energy balance is handled in physics/interface_bc.py,
    # not via this conductive-only helper.
    f_if = grid.iface_f
    q_cond[f_if] = 0.0

    # outer boundary Dirichlet with ghost T = T_inf
    T_inf = float(cfg.initial.T_inf)
    ig_last = Ng - 1
    i_last = gas_start + ig_last
    f_out = grid.Nc
    r_out = grid.r_f[-1]
    r_last = grid.r_c[i_last]
    dr_out = r_out - r_last
    if dr_out <= 0.0:
        raise ValueError("Non-positive dr at outer boundary")
    k_out = props.k_g[ig_last]
    dTdr_out = (T_inf - Tg[ig_last]) / dr_out
    q_cond[f_out] = -k_out * dTdr_out

    return q_cond


def compute_gas_diffusive_flux_Y(
    cfg: CaseConfig,
    grid: Grid1D,
    props: Props,
    Yg: FloatArray,
) -> FloatArray:
    """
    Compute gas-phase species diffusive flux on faces
    (mixture-averaged Fick + per-face correction).

    Raw definition (per species k)
    ------------------------------
    J_k,raw = - rho * D_k * dY_k/dr   [kg/(m^2·s)],
    outward (+er) positive if flux is outward.

    On each internal gas face, a Coffee-Heimerl-type correction is applied
    so that the corrected flux J_k satisfies a per-face mass-conservation
    constraint:

        sum_k J_k,face approx 0.

    Parameters
    ----------
    cfg : CaseConfig
    grid : Grid1D
        Provides r_c, r_f, iface_f, gas_slice.
    props : Props
        Must provide:
        - rho_g : (Ng,)
        - D_g   : (Ns_g, Ng) mixture-averaged diffusion coeffs [m^2/s].
    Yg : (Ns_g, Ng) ndarray
        Gas species mass fractions in mechanism order, full-length, normalized per cell.

    Returns
    -------
    J_diff_g : (Ns_g, Nc+1) ndarray
        Species diffusive flux on faces, outward positive.
        On internal gas faces, J_diff_g is the corrected flux with
        sum_k J_k approx 0 per face.
        Interface face (grid.iface_f) and outer boundary face (Nc) remain
        zero placeholders; their fluxes are handled explicitly by
        physics/interface_bc.py (inner BC) and the system assembly (outer BC).
    """
    _check_conventions(cfg)

    Nl, Ng, Nc = grid.Nl, grid.Ng, grid.Nc
    gas_start = grid.gas_slice.start if grid.gas_slice is not None else Nl
    iface_f = grid.iface_f

    if Yg.ndim != 2:
        raise ValueError(f"Yg must be 2D (Ns_g, Ng), got ndim={Yg.ndim}")
    Ns_g, Ng_y = Yg.shape
    if Ng_y != Ng:
        raise ValueError(f"Yg shape {Yg.shape} inconsistent with Ng={Ng}")

    if props.D_g is None:
        raise ValueError("Props.D_g is None; gas diffusion coeffs are required.")
    if props.D_g.shape != (Ns_g, Ng):
        raise ValueError(f"D_g shape {props.D_g.shape} != ({Ns_g}, {Ng})")
    if props.rho_g.shape != (Ng,):
        raise ValueError(f"rho_g shape {props.rho_g.shape} != ({Ng},)")
    if grid.r_c.shape != (Nc,):
        raise ValueError(f"r_c shape {grid.r_c.shape} != ({Nc},)")
    if grid.r_f.shape != (Nc + 1,):
        raise ValueError(f"r_f shape {grid.r_f.shape} != ({Nc+1},)")

    rho_g = props.rho_g
    D_g = props.D_g

    # Raw Fick fluxes and face mass fractions
    J_raw = np.zeros((Ns_g, Nc + 1), dtype=np.float64)
    Y_face = np.zeros((Ns_g, Nc + 1), dtype=np.float64)

    # Internal gas faces (between gas cells), raw Fick first
    for ig in range(Ng - 1):
        iL = gas_start + ig
        iR = gas_start + ig + 1
        f = iR  # shared face index
        rL = grid.r_c[iL]
        rR = grid.r_c[iR]
        dr = rR - rL
        if dr <= 0.0:
            raise ValueError("Non-positive dr in gas region for species flux")

        rho_f = 0.5 * (float(rho_g[ig]) + float(rho_g[ig + 1]))
        D_f = 0.5 * (D_g[:, ig] + D_g[:, ig + 1])  # (Ns_g,)
        dY_dr = (Yg[:, ig + 1] - Yg[:, ig]) / dr   # (Ns_g,)

        J_face_raw = -rho_f * D_f * dY_dr  # outward positive
        if not np.all(np.isfinite(J_face_raw)):
            raise ValueError(f"Non-finite species diffusive flux at face {f}")
        J_raw[:, f] = J_face_raw
        Y_face[:, f] = 0.5 * (Yg[:, ig] + Yg[:, ig + 1])

    # Now apply per-face correction so that sum_k J_k = 0 on internal gas faces
    J_diff = np.zeros_like(J_raw)
    tiny = 1e-30

    for ig in range(Ng - 1):
        f = gas_start + ig + 1
        Jf = J_raw[:, f]
        if not np.any(Jf):
            continue

        sum_Jf = float(np.sum(Jf))
        if abs(sum_Jf) < tiny:
            J_diff[:, f] = Jf
            continue

        Yf = Y_face[:, f]
        J_corr = Jf - Yf * sum_Jf
        if not np.all(np.isfinite(J_corr)):
            raise ValueError(f"Non-finite corrected species diffusive flux at face {f}")
        J_diff[:, f] = J_corr

    # Interface face: keep zero here. Inner BC (Stefan condition + Y_eq)
    # is handled explicitly in physics/interface_bc.py, not via this routine.
    J_diff[:, iface_f] = 0.0
    # Outer boundary face: keep zero here. Far-field species BC are imposed
    # in build_species_system_SciPy (Dirichlet / convective BC), not via
    # a diffusive flux condition.
    J_diff[:, Nc] = 0.0

    return J_diff
