"""
Gas-phase convective fluxes (Step 8 MVP).

This module computes face-based energy convective flux for the gas phase:
    q_conv = rho * u * cp * T   [W/m^2], outward (+er) positive.

Scope:
- Only temperature equation convective term is provided here.
- Species convective flux is left as a placeholder.
- No residual assembly, no state mutation, no property evaluation.
- For future enthalpy-form energy equations, the convective term is rho * u * h; here MVP uses h ~= cp * T.
- Time discretization (explicit/implicit) is decided by the caller (assembly).

Assumptions/Conventions:
- radial_normal = "+er" (outward)
- flux_sign     = "outward_positive"
- heat_flux_def = "q=-k*dTdr" (kept for consistency with other modules)
"""

from __future__ import annotations

import numpy as np

from core.types import CaseConfig, Grid1D, Props

FloatArray = np.ndarray


def _check_conventions(cfg: CaseConfig) -> None:
    conv = cfg.conventions
    if conv.radial_normal != "+er":
        raise ValueError("flux_convective_gas assumes radial_normal='+er'")
    if conv.flux_sign != "outward_positive":
        raise ValueError("flux_convective_gas assumes flux_sign='outward_positive'")
    if conv.heat_flux_def != "q=-k*dTdr":
        raise ValueError("flux_convective_gas assumes heat_flux_def='q=-k*dTdr'")


def compute_gas_convective_flux_T(
    cfg: CaseConfig,
    grid: Grid1D,
    props: Props,
    Tg: FloatArray,
    u_face: FloatArray,
) -> FloatArray:
    """
    Compute gas-phase energy convective flux on faces.

    Parameters
    ----------
    cfg : CaseConfig
    grid : Grid1D
    props : Props
        Must provide rho_g, cp_g with shape (Ng,).
    Tg : (Ng,) ndarray
        Gas temperatures.
    u_face : (Nc+1,) ndarray
        Face-centered radial velocities, outward (+er) positive.

    Returns
    -------
    q_conv_T : (Nc+1,) ndarray
        Energy convective flux on faces (rho * u * cp * T), outward positive.
    """
    _check_conventions(cfg)

    Nl, Ng, Nc = grid.Nl, grid.Ng, grid.Nc
    gas_start = grid.gas_slice.start if grid.gas_slice is not None else Nl
    iface_f = grid.iface_f

    if Tg.shape != (Ng,):
        raise ValueError(f"Tg shape {Tg.shape} != ({Ng},)")
    if props.rho_g.shape != (Ng,):
        raise ValueError(f"rho_g shape {props.rho_g.shape} != ({Ng},)")
    if props.cp_g.shape != (Ng,):
        raise ValueError(f"cp_g shape {props.cp_g.shape} != ({Ng},)")
    if u_face.shape != (Nc + 1,):
        raise ValueError(f"u_face shape {u_face.shape} != ({Nc+1},)")
    if grid.r_c.shape != (Nc,):
        raise ValueError(f"r_c shape {grid.r_c.shape} != ({Nc},)")
    if grid.r_f.shape != (Nc + 1,):
        raise ValueError(f"r_f shape {grid.r_f.shape} != ({Nc+1},)")

    q_conv = np.zeros(Nc + 1, dtype=np.float64)

    # Interface face: placeholder zero (handled by interface BC elsewhere)
    q_conv[iface_f] = 0.0

    # Internal gas faces (between gas cells)
    for f in range(iface_f + 1, Nc):
        iL = f - 1
        iR = f
        igL = iL - gas_start
        igR = iR - gas_start
        rho_f = 0.5 * (float(props.rho_g[igL]) + float(props.rho_g[igR]))
        cp_f = 0.5 * (float(props.cp_g[igL]) + float(props.cp_g[igR]))
        u = float(u_face[f])
        T_up = float(Tg[igL] if u >= 0.0 else Tg[igR])
        q_conv[f] = rho_f * cp_f * u * T_up

    # Outer boundary face: upwind with T_inf if inflow
    T_inf = float(cfg.initial.T_inf)
    ig_last = Ng - 1
    rho_last = float(props.rho_g[ig_last])
    cp_last = float(props.cp_g[ig_last])
    u_out = float(u_face[Nc])
    T_up_out = float(Tg[ig_last] if u_out >= 0.0 else T_inf)
    q_conv[Nc] = rho_last * cp_last * u_out * T_up_out

    # Liquid-side faces (f < iface_f) remain zero
    return q_conv


def compute_gas_convective_flux_Y(
    cfg: CaseConfig,
    grid: Grid1D,
    props: Props,
    Yg: FloatArray,
    u_face: FloatArray,
) -> FloatArray:
    """
    Gas species convective flux on faces.

    Definition:
        J_i = rho * u * Y_i   [kg/(m^2*s)], outward (+er) positive.

    Parameters
    ----------
    cfg, grid, props : as usual
    Yg : (Ns_g, Ng)
        Mechanism-order mass fractions, full-length, normalized per cell.
    u_face : (Nc+1,)
        Face-centered radial velocities, outward positive.

    Returns
    -------
    J_conv_Y : (Ns_g, Nc+1)
        Species convective flux on faces, outward positive.
        Interface face set to 0. Outer boundary uses last cell composition (MVP).
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
    if props.rho_g.shape != (Ng,):
        raise ValueError(f"rho_g shape {props.rho_g.shape} != ({Ng},)")
    if u_face.shape != (Nc + 1,):
        raise ValueError(f"u_face shape {u_face.shape} != ({Nc+1},)")
    if grid.r_c.shape != (Nc,):
        raise ValueError(f"r_c shape {grid.r_c.shape} != ({Nc},)")
    if grid.r_f.shape != (Nc + 1,):
        raise ValueError(f"r_f shape {grid.r_f.shape} != ({Nc+1},)")

    J_conv = np.zeros((Ns_g, Nc + 1), dtype=np.float64)
    rho_g = props.rho_g

    # Interface face: placeholder zero (handled by interface BC later)
    J_conv[:, iface_f] = 0.0

    # Internal gas faces (between gas cells)
    for ig in range(Ng - 1):
        iL = gas_start + ig
        iR = gas_start + ig + 1
        f = iR  # shared face index

        rL = grid.r_c[iL]
        rR = grid.r_c[iR]
        dr = rR - rL
        if dr <= 0.0:
            raise ValueError("Non-positive dr in gas region for convective flux")

        rho_f = 0.5 * (float(rho_g[ig]) + float(rho_g[ig + 1]))
        u_f = float(u_face[f])
        Y_up = Yg[:, ig] if u_f >= 0.0 else Yg[:, ig + 1]

        J_face = rho_f * u_f * Y_up
        if not np.all(np.isfinite(J_face)):
            raise ValueError(f"Non-finite convective species flux at face {f}")
        J_conv[:, f] = J_face

    # Outer boundary face: use last cell composition as proxy for farfield (MVP)
    f_out = Nc
    u_out = float(u_face[f_out])
    ig_last = Ng - 1
    rho_last = float(rho_g[ig_last])
    Y_last = Yg[:, ig_last]
    Y_up_out = Y_last  # inflow uses farfield ~ last cell comp in this MVP
    J_out = rho_last * u_out * Y_up_out
    if not np.all(np.isfinite(J_out)):
        raise ValueError("Non-finite convective species flux at outer boundary")
    J_conv[:, f_out] = J_out

    return J_conv
