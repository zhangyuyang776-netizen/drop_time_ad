"""
SciPy/Numpy assembly for liquid species mass-fraction equations (Yl block, reduced species).

Scope (Step 19.4.3):
- Time term implicit (theta = 1.0, backward Euler).
- Diffusion + interface evaporation flux evaluated from state_guess (implicit in Yl).
- No convection; interface diffusive flux placeholder zero (Neumann).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.layout import UnknownLayout
from core.types import CaseConfig, Grid1D, Props, State
from physics.flux_liq import compute_liq_diffusive_flux_Y


def build_liquid_species_system(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props: Props,
    dt: float,
    interface_evap: Dict[str, Any] | None = None,
    *,
    A_out: np.ndarray | None = None,
    b_out: np.ndarray | None = None,
    return_diag: bool = False,
    state_guess: State | None = None,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Assemble liquid-species mass-fraction equations (reduced species) into global system.

    Time + divergence of (diffusive + interface evaporation) fluxes; implicit time.
    """
    if not layout.has_block("Yl"):
        raise ValueError("layout missing 'Yl' block required for liquid species assembly.")
    theta = float(cfg.discretization.theta)
    if abs(theta - 1.0) > 1e-12:
        raise ValueError("Liquid species assembly assumes theta=1.0 (fully implicit time term).")

    if state_guess is None:
        state_guess = state_old

    Ns_full, Nl = state_old.Yl.shape
    if props.D_l is None:
        raise ValueError("Props.D_l is None; liquid diffusion coefficients required.")
    if props.D_l.shape != (Ns_full, Nl):
        raise ValueError(f"D_l shape {props.D_l.shape} != ({Ns_full}, {Nl})")
    if props.rho_l.shape != (Nl,):
        raise ValueError(f"rho_l shape {props.rho_l.shape} != ({Nl},)")

    N = layout.n_dof()
    if A_out is None:
        A = np.zeros((N, N), dtype=np.float64)
    else:
        A = A_out
    if b_out is None:
        b = np.zeros(N, dtype=np.float64)
    else:
        b = b_out

    diag: Dict[str, Any] = {"bc": {}, "evap": {}}

    gas_start = grid.Nl  # liquid cells start at 0, but keep symmetry with gas indexing
    iface_f = grid.iface_f

    # Fluxes evaluated from state_guess (implicit in Yl)
    J_diff = compute_liq_diffusive_flux_Y(cfg, grid, props, state_guess.Yl)
    J_tot = np.array(J_diff, copy=True)

    mpp = float(getattr(state_guess, "mpp", 0.0))
    if interface_evap is not None:
        diag["evap"]["mpp_eval"] = float(interface_evap.get("mpp_eval", 0.0))
        diag["evap"]["mpp_state"] = mpp
    if mpp != 0.0:
        # Use Yl at n+1 guess on the last liquid cell next to interface
        Yl_face = np.asarray(state_guess.Yl[:, Nl - 1], dtype=np.float64)
        J_evap = mpp * Yl_face
        J_tot[:, iface_f] += J_evap
        diag["evap"]["mpp"] = mpp
        diag["evap"]["Yl_face"] = Yl_face

    rho_l = props.rho_l
    V_c = grid.V_c
    A_f = grid.A_f

    for k_red in range(layout.Ns_l_eff):
        k_full = layout.liq_reduced_to_full_idx[k_red]
        for il in range(Nl):
            row = layout.idx_Yl(k_red, il)
            cell_idx = il  # liquid cells occupy the first Nl entries

            rho_i = float(rho_l[il])
            V = float(V_c[cell_idx])

            aP_time = rho_i * V / dt
            aP = aP_time
            b_i = aP_time * float(state_old.Yl[k_full, il])

            f_L = il
            f_R = il + 1
            A_L = float(A_f[f_L])
            A_R = float(A_f[f_R])
            J_L = float(J_tot[k_full, f_L])
            J_R = float(J_tot[k_full, f_R])
            div = A_R * J_R - A_L * J_L

            A[row, row] += aP
            b[row] += b_i - div

    if return_diag:
        return A, b, diag
    return A, b
