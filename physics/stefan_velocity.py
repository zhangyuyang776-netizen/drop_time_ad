"""
Stefan velocity (gas-phase radial velocity) under evaporation-only assumption.

Assumptions (Step 8 MVP):
- Spherical symmetry, no external flow; only evaporation drives gas motion.
- Mass conservation: rho_g * u_r * r^2 = const = mpp * Rd^2
- Sign conventions follow CaseConventions:
  radial_normal = "+er", flux_sign = "outward_positive",
  evap_sign = "mpp_positive_liq_to_gas".

Outputs:
- u_face: face-centered radial velocity, shape (Nc+1,), +er is positive.
- u_cell: cell-centered velocity, shape (Nc,), liquid cells remain zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.types import CaseConfig, Grid1D, Props, State

FloatArray = np.ndarray


@dataclass(slots=True)
class StefanVelocity:
    """Container for Stefan flow velocities."""

    u_face: FloatArray  # (Nc+1,) face-centered velocity [m/s], outward positive
    u_cell: FloatArray  # (Nc,)   cell-centered velocity [m/s], liquid cells zero


def _check_conventions(cfg: CaseConfig) -> None:
    conv = cfg.conventions
    if conv.radial_normal != "+er":
        raise ValueError("stefan_velocity assumes radial_normal='+er'")
    if conv.flux_sign != "outward_positive":
        raise ValueError("stefan_velocity assumes flux_sign='outward_positive'")
    # Accept both historical and explicit wording for evaporation sign
    if conv.evap_sign not in ("mpp_positive_liq_to_gas", "mpp_positive_evaporation"):
        raise ValueError("stefan_velocity assumes evap_sign indicates mpp>0 is liq->gas")


def compute_stefan_velocity(
    cfg: CaseConfig,
    grid: Grid1D,
    props: Props,
    state: State,
) -> StefanVelocity:
    """
    Compute Stefan flow velocities in the gas phase.

    Parameters
    ----------
    cfg : CaseConfig
        Must satisfy the standard conventions (+er, outward_positive).
    grid : Grid1D
        Geometry container; iface_f marks liquid/gas interface face.
    props : Props
        Must provide rho_g with shape (Ng,).
    state : State
        Provides mpp (kg/m^2/s) and Rd (m).

    Returns
    -------
    StefanVelocity
        u_face (Nc+1,), u_cell (Nc,), outward positive.
    """
    _check_conventions(cfg)

    Nl, Ng, Nc = grid.Nl, grid.Ng, grid.Nc
    gas_start = grid.gas_slice.start if grid.gas_slice is not None else Nl
    iface_f = grid.iface_f

    if props.rho_g.shape != (Ng,):
        raise ValueError(f"rho_g shape {props.rho_g.shape} != ({Ng},)")
    if grid.r_c.shape != (Nc,):
        raise ValueError(f"r_c shape {grid.r_c.shape} != ({Nc},)")
    if grid.r_f.shape != (Nc + 1,):
        raise ValueError(f"r_f shape {grid.r_f.shape} != ({Nc+1},)")

    mpp = float(state.mpp)
    Rd = float(state.Rd)
    if Rd <= 0.0:
        raise ValueError(f"Invalid droplet radius Rd={Rd}")

    r_if = float(grid.r_f[iface_f])
    if not np.isfinite(r_if) or r_if <= 0.0:
        raise ValueError("Invalid interface face radius.")

    # If no evaporation/condensation, return zeros.
    if abs(mpp) < 1e-16:
        return StefanVelocity(
            u_face=np.zeros(Nc + 1, dtype=np.float64),
            u_cell=np.zeros(Nc, dtype=np.float64),
        )

    u_cell = np.zeros(Nc, dtype=np.float64)
    # Gas cell-centered velocities
    for ig in range(Ng):
        i = gas_start + ig
        r = float(grid.r_c[i])
        rho = float(props.rho_g[ig])
        if r <= 0.0:
            raise ValueError("Non-positive radius in gas cell.")
        if rho <= 0.0:
            raise ValueError("Non-positive rho_g in gas cell.")
        u_cell[i] = mpp * (Rd ** 2) / (rho * (r ** 2))
    # liquid cells remain zero

    u_face = np.zeros(Nc + 1, dtype=np.float64)
    # liquid-side faces remain zero by default (f < iface_f)

    # Interface face: use first gas cell density
    rho_if = float(props.rho_g[0])
    if rho_if <= 0.0:
        raise ValueError("Non-positive rho_g at interface face.")
    u_face[iface_f] = mpp * (Rd ** 2) / (rho_if * (r_if ** 2))

    # Internal gas faces between gas cells
    for f in range(iface_f + 1, Nc):
        iL = f - 1
        iR = f
        if iL < gas_start or iR >= Nc:
            raise ValueError("Face index out of gas region range.")
        igL = iL - gas_start
        igR = iR - gas_start

        rho_L = float(props.rho_g[igL])
        rho_R = float(props.rho_g[igR])
        rho_f = 0.5 * (rho_L + rho_R)
        r_f = float(grid.r_f[f])
        if r_f <= 0.0:
            raise ValueError("Non-positive face radius in gas region.")
        if rho_f <= 0.0:
            raise ValueError("Non-positive rho_g at gas face.")
        u_face[f] = mpp * (Rd ** 2) / (rho_f * (r_f ** 2))

    # Outer boundary face: use last gas cell density
    ig_last = Ng - 1
    rho_out = float(props.rho_g[ig_last])
    r_out = float(grid.r_f[Nc])
    if r_out <= 0.0:
        raise ValueError("Non-positive outer face radius.")
    if rho_out <= 0.0:
        raise ValueError("Non-positive rho_g at outer face.")
    u_face[Nc] = mpp * (Rd ** 2) / (rho_out * (r_out ** 2))

    return StefanVelocity(u_face=u_face, u_cell=u_cell)
