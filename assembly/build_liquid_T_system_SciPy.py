"""
SciPy/Numpy assembly of liquid temperature equations (Tl block) in spherical 1D.

Modes:
- couple_interface=True: returns only liquid-domain contributions (no interface Dirichlet);
  interface coupling handled elsewhere.
- couple_interface=False: applies strong Dirichlet at the interface face using current Ts
  (or Ts_fixed if provided in state/config), matching Step 10 tests.
"""

from __future__ import annotations

import numpy as np

from core.layout import UnknownLayout
from core.types import CaseConfig, Grid1D, Props, State


def build_liquid_T_system(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props: Props,
    dt: float,
    *,
    couple_interface: bool = False,
):
    """Assemble liquid temperature system (Nl x Nl dense)."""
    Nl = grid.Nl
    if Nl <= 0 or not layout.has_block("Tl"):
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    theta = float(cfg.discretization.theta)
    if abs(theta - 1.0) > 1e-12:
        raise ValueError("Liquid T assembly assumes theta=1.0 (fully implicit).")

    if props.rho_l.shape != (Nl,):
        raise ValueError(f"rho_l shape {props.rho_l.shape} != ({Nl},)")
    if props.cp_l.shape != (Nl,):
        raise ValueError(f"cp_l shape {props.cp_l.shape} != ({Nl},)")
    if props.k_l.shape != (Nl,):
        raise ValueError(f"k_l shape {props.k_l.shape} != ({Nl},)")

    if grid.r_c.shape[0] < Nl or grid.V_c.shape[0] < Nl or grid.A_f.shape[0] < Nl + 1:
        raise ValueError("Grid shapes insufficient for liquid region.")

    A = np.zeros((Nl, Nl), dtype=np.float64)
    b = np.zeros(Nl, dtype=np.float64)

    rho_l = props.rho_l
    cp_l = props.cp_l
    k_l = props.k_l
    gas_start = grid.gas_slice.start if grid.gas_slice is not None else Nl

    # Interface Dirichlet value when decoupled
    Ts_bc = float(state_old.Ts)
    if hasattr(cfg.physics, "interface") and getattr(cfg.physics.interface, "bc_mode", "") == "Ts_fixed":
        Ts_bc = float(getattr(cfg.physics.interface, "Ts_fixed", Ts_bc))

    for il in range(Nl):
        row = il
        cell_idx = il  # liquid cells start at 0 globally

        rho_i = float(rho_l[il])
        cp_i = float(cp_l[il])
        k_i = float(k_l[il])
        V = float(grid.V_c[cell_idx])

        aP_time = rho_i * cp_i * V / dt
        aP = aP_time
        b_i = aP_time * float(state_old.Tl[il])

        # Left face (toward center r=0): symmetry (zero gradient) using reflected ghost
        if il == 0 and Nl > 1:
            rC = grid.r_c[cell_idx]
            rR = grid.r_c[cell_idx + 1]
            A_f = float(grid.A_f[cell_idx])
            dr = rR - rC
            if dr <= 0.0:
                raise ValueError("Non-positive dr at liquid center face.")
            k_face = 0.5 * (k_i + float(k_l[il + 1]))
            coeff = k_face * A_f / dr
            aP += coeff
            A[row, il + 1] += -coeff
        elif il > 0:
            rL = grid.r_c[cell_idx - 1]
            rC = grid.r_c[cell_idx]
            A_f = float(grid.A_f[cell_idx])
            dr = rC - rL
            if dr <= 0.0:
                raise ValueError("Non-positive dr on left liquid face.")
            k_face = 0.5 * (k_i + float(k_l[il - 1]))
            coeff = k_face * A_f / dr
            aP += coeff
            A[row, il - 1] += -coeff

        # Right face (toward interface)
        if il < Nl - 1:
            rC = grid.r_c[cell_idx]
            rR = grid.r_c[cell_idx + 1]
            A_f = float(grid.A_f[cell_idx + 1])
            dr = rR - rC
            if dr <= 0.0:
                raise ValueError("Non-positive dr on right liquid face.")
            k_face = 0.5 * (k_i + float(k_l[il + 1]))
            coeff = k_face * A_f / dr
            aP += coeff
            A[row, il + 1] += -coeff
        else:
            # Interface boundary at il == Nl-1
            if couple_interface:
                # Coupled mode: no Dirichlet; leave interface handled elsewhere
                pass
            else:
                # Strong Dirichlet to Ts (or Ts_fixed)
                row = Nl - 1
                A[row, :] = 0.0
                A[row, row] = 1.0
                b[row] = Ts_bc
                continue

        A[row, row] += aP
        b[row] += b_i

    return A, b
