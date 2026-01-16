"""
Droplet radius evolution equation (Step 11 coupling with mpp).

Responsibilities (current version):
- Build a single implicit time-discrete equation for Rd using mpp and liquid density.
- Form used: (Rd^{n+1} - Rd^{n}) / dt + mpp^{n+1} / rho_l_if = 0
  => (1/dt) * Rd^{n+1} + (1/rho_l_if) * mpp^{n+1} = Rd^{n} / dt

Direction and sign conventions (must match core/types.py and interface_bc.py):
- Radial coordinate r increases outward (droplet center -> far field).
- mpp > 0 means evaporation (liquid -> gas).
- Evaporation implies Rd decreases (dR/dt < 0).

This module MUST NOT:
- mutate grid/state/props,
- compute new properties,
- normalize mass fractions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from core.types import CaseConfig, Grid1D, State, Props
from core.layout import UnknownLayout

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RadiusCoeffs:
    """
    Matrix row definition for the droplet radius equation.

    row  : global row index (layout.idx_Rd())
    cols : global column indices (Rd, mpp, ... if extended later)
    vals : corresponding coefficients
    rhs  : right-hand side
    diag : diagnostics for logging / unit tests
    """

    row: int
    cols: List[int] = field(default_factory=list)
    vals: List[float] = field(default_factory=list)
    rhs: float = 0.0
    diag: Dict[str, Any] = field(default_factory=dict)


def build_radius_row(
    grid: Grid1D,
    state_old: State,
    state_guess: State,
    props: Props,
    layout: UnknownLayout,
    dt: float,
    cfg: CaseConfig,
) -> RadiusCoeffs:
    """
    Build the implicit time-discrete equation for droplet radius Rd.

    Backward-Euler form (MVP):

        (Rd^{n+1} - Rd^{n}) / dt + mpp^{n+1} / rho_l_if = 0

    where rho_l_if is a representative liquid density near the interface.

    This function:
    - uses state_old.* only on the RHS (known previous-time data),
    - uses state_guess.* only for diagnostics (no coefficients depend on it),
    - depends on layout.idx_Rd() and layout.idx_mpp().

    It does NOT:
    - update grid or state,
    - compute new properties.
    """
    if dt <= 0.0:
        raise ValueError(f"Time step dt must be positive, got dt={dt}")

    phys = cfg.physics
    if not phys.include_Rd:
        logger.error("build_radius_row called but cfg.physics.include_Rd is False.")
        raise ValueError("Radius equation requested while include_Rd=False.")

    if not layout.has_block("Rd"):
        raise ValueError("Layout missing 'Rd' block required by radius_eq.")

    if not phys.include_mpp or not layout.has_block("mpp"):
        raise ValueError("Radius equation requires mpp unknown (cfg.physics.include_mpp=True, layout block 'mpp').")

    if grid.Nl <= 0:
        raise ValueError("Radius equation requires at least one liquid cell (Nl>0).")

    il_if = grid.Nl - 1
    try:
        rho_l_if = float(props.rho_l[il_if])
    except Exception as exc:  # pragma: no cover - defensive access
        logger.error("Failed to access liquid density rho_l at interface cell %d: %s", il_if, exc)
        raise

    if rho_l_if <= 0.0:
        raise ValueError(f"Non-positive interface liquid density rho_l_if={rho_l_if}")

    idx_Rd = layout.idx_Rd()
    idx_mpp = layout.idx_mpp()

    Rd_old = float(state_old.Rd)
    Rd_guess = float(state_guess.Rd)
    mpp_guess = float(state_guess.mpp)

    coeff_Rd = 1.0 / dt
    coeff_mpp = 1.0 / rho_l_if
    rhs = Rd_old / dt

    cols = [idx_Rd, idx_mpp]
    vals = [coeff_Rd, coeff_mpp]

    dRdt_guess = (Rd_guess - Rd_old) / dt
    residual_guess = dRdt_guess + mpp_guess / rho_l_if

    rho_l = rho_l_if  # MVP: use same representative density
    mass_old = (4.0 * np.pi / 3.0) * (Rd_old ** 3) * rho_l
    mass_new = (4.0 * np.pi / 3.0) * (Rd_guess ** 3) * rho_l
    A_star = 4.0 * np.pi * (Rd_guess ** 2)
    mass_balance = mass_new - mass_old + A_star * mpp_guess * dt

    diag = {
        "radius_eq": {
            "dt": dt,
            "rho_l_if": rho_l_if,
            "idx_Rd": idx_Rd,
            "idx_mpp": idx_mpp,
            "Rd_old": Rd_old,
            "Rd_guess": Rd_guess,
            "mpp_guess": mpp_guess,
            "dRdt_guess": dRdt_guess,
            "residual_guess": residual_guess,
            "mass_old": mass_old,
            "mass_new": mass_new,
            "A_star": A_star,
            "mass_balance": mass_balance,
            "coeffs": {
                "Rd": coeff_Rd,
                "mpp": coeff_mpp,
            },
            "rhs": rhs,
            "interface_cell_liq": il_if,
        }
    }

    return RadiusCoeffs(
        row=idx_Rd,
        cols=cols,
        vals=vals,
        rhs=rhs,
        diag=diag,
    )
