"""
Liquid-phase conductive fluxes (Step 10 MVP).

Computes cell-face heat flux in the liquid region:
    q = -k_l * dT_l/dr   [W/m^2], outward (+er) positive.

Scope:
- Only liquid faces (f <= iface_f) are populated.
- Center face (r=0) and interface face use zero-flux placeholders.
- No interface coupling or state mutation here.
- This is only the conductive component q_cond; enthalpy diffusion q_diff will be handled via physics/energy_flux.py.
"""

from __future__ import annotations

import numpy as np

from core.types import CaseConfig, Grid1D, Props

FloatArray = np.ndarray


def _check_conventions(cfg: CaseConfig) -> None:
    conv = cfg.conventions
    if conv.radial_normal != "+er":
        raise ValueError("flux_liq assumes radial_normal='+er'")
    if conv.flux_sign != "outward_positive":
        raise ValueError("flux_liq assumes flux_sign='outward_positive'")
    if conv.heat_flux_def != "q=-k*dTdr":
        raise ValueError("flux_liq assumes heat_flux_def='q=-k*dTdr'")


def compute_liquid_diffusive_flux_T(
    cfg: CaseConfig,
    grid: Grid1D,
    props: Props,
    Tl: FloatArray,
) -> FloatArray:
    """
    Compute conductive heat flux in the liquid phase on faces.

    Definition:
        q = -k_l * dT_l/dr, outward (+er) positive.

    Parameters
    ----------
    cfg : CaseConfig
        Must satisfy standard conventions (+er, outward_positive, q=-k*dTdr).
    grid : Grid1D
        1D spherical grid. Liquid cells: 0..Nl-1, interface face = iface_f.
    props : Props
        Must provide k_l with shape (Nl,).
    Tl : (Nl,) ndarray
        Liquid temperatures at cell centers.

    Returns
    -------
    q_cond_l : (Nc+1,) ndarray
        Face-based conductive heat flux; non-zero only on liquid faces.
        - f=0            : center (r=0), zero-flux (symmetry)
        - f=1..iface_f-1 : internal liquid faces
        - f=iface_f      : interface face, zero placeholder in Step 10
        - f>iface_f      : remain zero (gas region)
    """
    _check_conventions(cfg)

    Nl, Ng, Nc = grid.Nl, grid.Ng, grid.Nc
    iface_f = grid.iface_f

    if Tl.shape != (Nl,):
        raise ValueError(f"Tl shape {Tl.shape} != ({Nl},)")
    if props.k_l.shape != (Nl,):
        raise ValueError(f"k_l shape {props.k_l.shape} != ({Nl},)")
    if grid.r_c.shape != (Nc,):
        raise ValueError(f"r_c shape {grid.r_c.shape} != ({Nc},)")
    if grid.r_f.shape != (Nc + 1,):
        raise ValueError(f"r_f shape {grid.r_f.shape} != ({Nc+1},)")
    if grid.A_f.shape != (Nc + 1,):
        raise ValueError(f"A_f shape {grid.A_f.shape} != ({Nc+1},)")
    if iface_f <= 0 or iface_f > Nc:
        raise ValueError(f"Invalid iface_f={iface_f} for Nc={Nc}")

    q_cond = np.zeros(Nc + 1, dtype=np.float64)

    # center symmetry: zero-flux at r=0
    q_cond[0] = 0.0

    # internal liquid faces (between liquid cells)
    for il in range(Nl - 1):
        iL = il
        iR = il + 1
        f = iR  # shared face index (right cell index as face index)
        if f >= iface_f:
            break
        rL = grid.r_c[iL]
        rR = grid.r_c[iR]
        dr = rR - rL
        if dr <= 0.0:
            raise ValueError("Non-positive dr in liquid region")
        kL = float(props.k_l[il])
        kR = float(props.k_l[il + 1])
        k_face = 0.5 * (kL + kR)
        dTdr = (float(Tl[il + 1]) - float(Tl[il])) / dr
        q_cond[f] = -k_face * dTdr  # outward positive

    # interface face: placeholder zero (interface coupling handled elsewhere)
    q_cond[iface_f] = 0.0

    # faces beyond interface (gas region) remain zero
    return q_cond


def compute_liq_diffusive_flux_Y(
    cfg: CaseConfig,
    grid: Grid1D,
    props: Props,
    Yl: FloatArray,
) -> FloatArray:
    """
    Compute liquid species diffusive flux on faces (mixture-averaged Fick).

    J_k = -rho_l * D_l * dY_k/dr, outward (+er) positive.
    Interface and center faces are zero placeholders in this MVP.
    """
    conv = cfg.conventions
    if conv.radial_normal != "+er" or conv.flux_sign != "outward_positive":
        raise ValueError("compute_liq_diffusive_flux_Y assumes +er/outward_positive conventions")

    Nl = grid.Nl
    Nc = grid.Nc
    iface_f = grid.iface_f

    if Yl.ndim != 2:
        raise ValueError(f"Yl must be 2D (Ns_l, Nl), got ndim={Yl.ndim}")
    Ns_l, Nl_y = Yl.shape
    if Nl_y != Nl:
        raise ValueError(f"Yl shape {Yl.shape} inconsistent with Nl={Nl}")

    if props.D_l is None:
        raise ValueError("Props.D_l is None; liquid diffusion coeffs required.")
    if props.D_l.shape != (Ns_l, Nl):
        raise ValueError(f"D_l shape {props.D_l.shape} != ({Ns_l}, {Nl})")
    if props.rho_l.shape != (Nl,):
        raise ValueError(f"rho_l shape {props.rho_l.shape} != ({Nl},)")

    J_diff = np.zeros((Ns_l, Nc + 1), dtype=np.float64)
    rho_l = props.rho_l
    D_l = props.D_l

    # Internal liquid faces (between liquid cells)
    for il in range(Nl - 1):
        iL = il
        iR = il + 1
        f = iR  # shared face index
        if f >= iface_f:
            break
        rL = grid.r_c[iL]
        rR = grid.r_c[iR]
        dr = rR - rL
        if dr <= 0.0:
            raise ValueError("Non-positive dr in liquid region for species flux")

        rho_f = 0.5 * (float(rho_l[iL]) + float(rho_l[iR]))
        D_f = 0.5 * (D_l[:, iL] + D_l[:, iR])  # (Ns_l,)
        dY_dr = (Yl[:, iR] - Yl[:, iL]) / dr   # (Ns_l,)

        J_face = -rho_f * D_f * dY_dr
        if not np.all(np.isfinite(J_face)):
            raise ValueError(f"Non-finite liquid species diffusive flux at face {f}")
        J_diff[:, f] = J_face

    # Center symmetry and interface placeholders
    J_diff[:, 0] = 0.0
    J_diff[:, iface_f] = 0.0
    return J_diff
