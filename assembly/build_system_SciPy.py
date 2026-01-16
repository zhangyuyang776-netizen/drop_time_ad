"""
SciPy/Numpy assembly of linear system A x = b for Tg + interface (Ts, mpp) + radius (Rd).

Differences vs PETSc build_system:
- Returns dense numpy arrays (A, b) for SciPy-based solvers.
- No PETSc dependency; intended for Windows / SciPy workflow.

Scope (Step 19.4.4, SciPy backend):
- Gas temperature Tg:
  - fully implicit time term (theta = 1.0),
  - implicit diffusion in gas cells,
  - convective term uses Stefan velocity and Tg from state_guess (implicit in Tg),
  - outer boundary: strong Dirichlet T = T_inf.
- Interface conditions (optional, controlled by cfg.physics.include_Ts / include_mpp):
  - Ts energy jump: gas/liquid conduction + latent heat (single-condensable MVP, no diffusion enthalpy yet),
  - mpp Stefan mass balance: diffusive flux of condensable species vs eq_result["Yg_eq"].
- Radius evolution (optional, cfg.physics.include_Rd):
  - backward-Euler dR/dt = -mpp / rho_l_if,
  - coupled to mpp via radius_eq.build_radius_row.

Inputs:
- cfg, grid, layout, state_old, props, dt,
- optional eq_result (interface equilibrium Yg_eq) and state_guess (current nonlinear guess).

Contract:
- layout must contain Tg block; Ts / mpp / Rd blocks only used if cfg.physics.include_* is True.
- if include_mpp is True, eq_result must contain 'Yg_eq' with full gas-species mass fractions.
"""


from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

from core.layout import UnknownLayout
from core.types import CaseConfig, Grid1D, Props, State
from physics.stefan_velocity import compute_stefan_velocity
from physics.flux_convective_gas import compute_gas_convective_flux_T
from physics.interface_bc import build_interface_coeffs, EqResultLike
from physics.radius_eq import build_radius_row
from physics.interface_bc import InterfaceCoeffs  # type hint only
from physics.radius_eq import RadiusCoeffs  # type hint only
from assembly.build_species_system_SciPy import build_gas_species_system_global
from assembly.build_liquid_species_system_SciPy import build_liquid_species_system

logger = logging.getLogger(__name__)


def _apply_center_bc_Tg(A: np.ndarray, row: int, coeff: float, col_neighbor: int) -> None:
    """Inner gas boundary Neumann (zero-gradient) using a reflected ghost cell."""
    A[row, col_neighbor] += -coeff


def _apply_outer_dirichlet_Tg(A: np.ndarray, b: np.ndarray, row: int, T_far: float) -> None:
    """Strong Dirichlet at outer boundary: T = T_far."""
    A[row, :] = 0.0
    A[row, row] = 1.0
    b[row] = T_far


def _scatter_interface_rows(
    A: np.ndarray,
    b: np.ndarray,
    iface_coeffs: "InterfaceCoeffs",
) -> None:
    """
    Scatter interface rows (Ts, mpp) into global matrix/vector.

    Each InterfaceRow.row is a global row index; cols/vals are global unknown indices/coefficients.
    """
    for row_def in iface_coeffs.rows:
        r = row_def.row
        if not row_def.cols:
            # Allow placeholder rows with no coefficients.
            continue
        for c, v in zip(row_def.cols, row_def.vals):
            A[r, c] += v
        b[r] += row_def.rhs


def _scatter_radius_row(
    A: np.ndarray,
    b: np.ndarray,
    rad_coeffs: "RadiusCoeffs",
) -> None:
    """Scatter radius equation row into global matrix/vector."""
    r = rad_coeffs.row
    for c, v in zip(rad_coeffs.cols, rad_coeffs.vals):
        A[r, c] += v
    b[r] += rad_coeffs.rhs


def build_transport_system(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props: Props,
    dt: float,
    state_guess: State | None = None,
    eq_model=None,
    eq_result: EqResultLike | None = None,
    return_diag: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Assemble the coupled linear system A x = b for one time step.

    Unknowns (depending on cfg/layout):
    - Tg: gas temperature in Ng cells;
    - Ts: interface temperature (single scalar);
    - mpp: interface mass flux (single scalar, positive for evaporation);
    - Rd: droplet radius (single scalar).

    Contents:
    - Tg block:
      time term + implicit diffusion + Stefan convection from state_guess;
      outer gas cell has strong Dirichlet T = cfg.initial.T_inf.
    - Interface block (via physics.interface_bc.build_interface_coeffs):
      Ts energy jump (q_g + q_l - q_lat = 0),
      mpp Stefan mass condition (single condensable species).
    - Radius block (via physics.radius_eq.build_radius_row):
      backward-Euler dR/dt = -mpp / rho_l_if.

    Parameters
    ----------
    cfg, grid, layout, state_old, props, dt :
        Core problem definition and previous time level state.
    eq_result : mapping, optional
        Must contain 'Yg_eq' (full gas mass-fraction vector) if include_mpp is True.
        No internal fallback is provided when include_mpp is enabled.
    state_guess : State, optional
        Current nonlinear guess; used for interface and radius assembly when provided.
    return_diag : bool, default False
        If True, returns (A, b, diag_sys) where diag_sys aggregates diagnostics
        from interface_bc and radius_eq; otherwise returns (A, b) only.
    """
    if not layout.has_block("Tg"):
        raise ValueError("layout missing Tg block required for Step 6 assembly.")
    if state_old.Tg.shape != (grid.Ng,):
        raise ValueError(f"Tg shape {state_old.Tg.shape} != ({grid.Ng},)")

    theta = float(cfg.discretization.theta)
    if abs(theta - 1.0) > 1e-12:
        raise ValueError("Step 6 MVP only supports theta=1.0 (fully implicit Tg diffusion).")

    if state_guess is None:
        state_guess = state_old

    phys = cfg.physics
    if phys.include_mpp and layout.has_block("mpp") and eq_result is None:
        raise ValueError("include_mpp=True requires eq_result with 'Yg_eq' (no internal fallback).")
    Ng = grid.Ng
    Nc = grid.Nc
    N = layout.n_dof()
    A = np.zeros((N, N), dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)
    diag_sys: Dict[str, Any] = {}

    gas_start = grid.gas_slice.start if grid.gas_slice is not None else grid.Nl

    for ig in range(Ng):
        row = layout.idx_Tg(ig)
        rho = float(props.rho_g[ig])
        cp = float(props.cp_g[ig])
        k_i = float(props.k_g[ig])

        cell_idx = gas_start + ig
        V = float(grid.V_c[cell_idx])
        aP_time = rho * cp * V / dt

        aP = aP_time
        b_i = aP_time * state_old.Tg[ig]

        # Left face (ig-1/2)
        if ig > 0:
            rL = grid.r_c[cell_idx - 1]
            rC = grid.r_c[cell_idx]
            A_f = float(grid.A_f[cell_idx])
            dr = rC - rL
            k_face = 0.5 * (k_i + float(props.k_g[ig - 1]))
            coeff = k_face * A_f / dr
            aP += coeff
            A[row, layout.idx_Tg(ig - 1)] += -coeff
        else:
            # Interface coupling (gas-side conduction to Ts or Ts_fixed)
            iface_f = grid.iface_f
            A_if = float(grid.A_f[iface_f])
            dr_if = float(grid.r_c[cell_idx] - grid.r_f[iface_f])
            if dr_if <= 0.0:
                raise ValueError("Non-positive gas-side spacing at interface for Tg coupling.")
            k_face = float(props.k_g[0])
            coeff_if = k_face * A_if / dr_if

            aP += coeff_if
            Ts_used = "unknown"
            Ts_val = float(state_old.Ts)
            if phys.include_Ts and layout.has_block("Ts"):
                A[row, layout.idx_Ts()] += -coeff_if
            else:
                Ts_bc = float(state_old.Ts)
                if getattr(cfg.physics, "interface", None) is not None:
                    if getattr(cfg.physics.interface, "bc_mode", "") == "Ts_fixed":
                        Ts_bc = float(getattr(cfg.physics.interface, "Ts_fixed", Ts_bc))
                b_i += coeff_if * Ts_bc
                Ts_used = "fixed"
                Ts_val = Ts_bc
            diag_sys.setdefault("gas", {})["Tg_interface_coupling"] = {
                "coeff_if": float(coeff_if),
                "dr_if": float(dr_if),
                "A_if": float(A_if),
                "k_face": float(k_face),
                "Tg0_old": float(state_old.Tg[0]),
                "Ts_used": Ts_used,
                "Ts_value": float(Ts_val),
            }

        # Right face (ig+1/2)
        if ig < Ng - 1:
            rC = grid.r_c[cell_idx]
            rR = grid.r_c[cell_idx + 1]
            A_f = float(grid.A_f[cell_idx + 1])
            dr = rR - rC
            k_face = 0.5 * (k_i + float(props.k_g[ig + 1]))
            coeff = k_face * A_f / dr
            aP += coeff
            A[row, layout.idx_Tg(ig + 1)] += -coeff

        A[row, row] += aP
        b[row] += b_i

    # --- Stefan velocity and convective flux (implicit in Tg via state_guess) ---
    # Step 19.4.1: use state_guess (~= T^{n+1}) for Stefan velocity and convective heat flux.
    stefan = compute_stefan_velocity(cfg, grid, props, state_guess)
    u_face = stefan.u_face

    q_conv = compute_gas_convective_flux_T(
        cfg=cfg,
        grid=grid,
        props=props,
        Tg=state_guess.Tg,  # implicit in Tg: depends on current state_guess
        u_face=u_face,
    )

    # Placeholder for interface diagnostics (filled later if interface BC built)
    iface_coeffs = None
    iface_evap = None

    # Add convective source to RHS: b[row] -= (A_R*q_R - A_L*q_L)
    for ig in range(Ng):
        row = layout.idx_Tg(ig)
        cell_idx = gas_start + ig
        f_L = cell_idx
        f_R = cell_idx + 1
        A_L = float(grid.A_f[f_L])
        A_R = float(grid.A_f[f_R])
        q_L = float(q_conv[f_L])
        q_R = float(q_conv[f_R])
        S_conv = A_R * q_R - A_L * q_L  # net outward convective power (W)
        b[row] -= S_conv

    # Outer boundary Dirichlet on last gas cell
    Tg_far = float(cfg.initial.T_inf)
    row_bc = layout.idx_Tg(Ng - 1)
    _apply_outer_dirichlet_Tg(A, b, row_bc, Tg_far)

    # --- Liquid temperature equations (Tl block) ---
    if layout.has_block("Tl"):
        from assembly.build_liquid_T_system_SciPy import build_liquid_T_system

        A_l, b_l = build_liquid_T_system(
            cfg=cfg,
            grid=grid,
            layout=layout,
            state_old=state_old,
            props=props,
            dt=dt,
            couple_interface=True,  # Coupled mode: no Dirichlet BC at interface
        )

        # Embed local Tl system (Nl-by-Nl) into global matrix (N-by-N)
        Nl = grid.Nl
        for il in range(Nl):
            row_global = layout.idx_Tl(il)
            for il2 in range(Nl):
                col_global = layout.idx_Tl(il2)
                A[row_global, col_global] += A_l[il, il2]
            b[row_global] += b_l[il]
        # Interface coupling: add conduction to Ts (or Ts_fixed if Ts not solved)
        il_last = Nl - 1
        row_last = layout.idx_Tl(il_last)
        iface_f = grid.iface_f
        r_if = float(grid.r_f[iface_f])
        r_last = float(grid.r_c[il_last])
        dr_if = r_if - r_last
        if dr_if <= 0.0:
            raise ValueError("Non-positive liquid-side spacing at interface for Tl coupling.")
        A_if = float(grid.A_f[iface_f])
        k_last = float(props.k_l[il_last])
        coeff_if = k_last * A_if / dr_if
        A[row_last, row_last] += coeff_if
        if phys.include_Ts and layout.has_block("Ts"):
            A[row_last, layout.idx_Ts()] += -coeff_if
        else:
            Ts_bc = float(state_old.Ts)
            if getattr(cfg.physics, "interface", None) is not None:
                if getattr(cfg.physics.interface, "bc_mode", "") == "Ts_fixed":
                    Ts_bc = float(getattr(cfg.physics.interface, "Ts_fixed", Ts_bc))
            b[row_last] += coeff_if * Ts_bc

    # --- Interface equations: Ts energy jump + mpp Stefan (single-condensable) ---
    phys = cfg.physics
    if (phys.include_Ts or phys.include_mpp) and (layout.has_block("Ts") or layout.has_block("mpp")):
        if phys.include_mpp and layout.has_block("mpp") and eq_result is None:
            raise ValueError("Step 11: mpp equation requires eq_result with 'Yg_eq'.")
        iface_coeffs = build_interface_coeffs(
            grid=grid,
            state=state_guess,
            props=props,
            layout=layout,
            cfg=cfg,
            eq_result=eq_result,
        )
        _scatter_interface_rows(A, b, iface_coeffs)
        diag_sys.update(iface_coeffs.diag)
        iface_evap = iface_coeffs.diag.get("evaporation")

        # Interface enthalpy flux contribution to gas energy (optional, MVP)
        if iface_evap is not None and state_old.Tg.size > 0:
            iface_f = grid.iface_f
            A_if = float(grid.A_f[iface_f])
            mpp_evap = float(iface_evap.get("mpp_state", iface_evap.get("mpp_eval", 0.0)))
            j_corr_full = np.asarray(iface_evap.get("j_corr_full", []), dtype=np.float64)
            Yg_eq_full = np.asarray(iface_evap.get("Yg_eq_full", []), dtype=np.float64)
            Ns_full = state_old.Yg.shape[0]
            if j_corr_full.shape[0] != Ns_full and j_corr_full.size != 0:
                raise ValueError("j_corr_full length mismatch for enthalpy flux injection.")
            if Yg_eq_full.shape[0] != Ns_full and Yg_eq_full.size != 0:
                raise ValueError("Yg_eq_full length mismatch for enthalpy flux injection.")

            # Use gas enthalpy at first gas cell as interface approximation
            h_mix_if = float(getattr(props, "h_g", np.array([0.0]))[0]) if getattr(props, "h_g", None) is not None else 0.0
            if getattr(props, "h_gk", None) is not None and np.asarray(props.h_gk).shape[0] >= Ns_full:
                h_k_if = np.asarray(props.h_gk[:Ns_full, 0], dtype=np.float64)
            else:
                h_k_if = np.zeros(Ns_full, dtype=np.float64)

            q_iface = mpp_evap * h_mix_if
            if j_corr_full.size:
                q_iface += float(np.dot(h_k_if, j_corr_full))

            row_Tg0 = layout.idx_Tg(0)
            b[row_Tg0] -= A_if * q_iface

    # --- Radius evolution equation (Rd-mpp coupling) ---
    if phys.include_Rd and layout.has_block("Rd"):
        rad_coeffs = build_radius_row(
            grid=grid,
            state_old=state_old,
            state_guess=state_guess,
            props=props,
            layout=layout,
            dt=dt,
            cfg=cfg,
        )
        _scatter_radius_row(A, b, rad_coeffs)
        diag_sys.update(rad_coeffs.diag)

    # --- Gas species equations: Yg (strongly coupled, Step 15.3) ---
    if getattr(cfg.physics, "solve_Yg", False) and layout.has_block("Yg") and layout.Ns_g_eff > 0:
        if phys.include_mpp and eq_result is None:
            raise ValueError("Species assembly with include_mpp=True requires eq_result with 'Yg_eq'.")

        if return_diag:
            _, _, diag_y = build_gas_species_system_global(
                cfg=cfg,
                grid=grid,
                layout=layout,
                state_old=state_old,
                props=props,
                dt=dt,
                eq_result=eq_result,
                interface_evap=iface_evap,
                A_out=A,
                b_out=b,
                return_diag=True,
                state_guess=state_guess,
            )
            diag_sys.setdefault("blocks", {})["Yg"] = diag_y
        else:
            build_gas_species_system_global(
                cfg=cfg,
                grid=grid,
                layout=layout,
                state_old=state_old,
                props=props,
                dt=dt,
                eq_result=eq_result,
                interface_evap=iface_evap,
                A_out=A,
                b_out=b,
                return_diag=False,
                state_guess=state_guess,
            )

        # Sanity checks on Yg rows
        test_row = layout.idx_Yg(0, 0)
        if np.allclose(A[test_row, :], 0.0):
            raise RuntimeError("Yg block present but species assembly produced an all-zero row (k=0, ig=0).")
        ig_bc = grid.Ng - 1
        row_bc = layout.idx_Yg(0, ig_bc)
        row_vals = A[row_bc, :]
        if not (np.isclose(row_vals[row_bc], 1.0) and np.allclose(np.delete(row_vals, row_bc), 0.0)):
            raise RuntimeError("Outer Dirichlet row for Yg (k=0) is not an identity row.")

    # --- Liquid species equations: Yl (implicit diffusion + evaporation flux, implicit time) ---
    if getattr(cfg.physics, "solve_Yl", False) and layout.has_block("Yl") and layout.Ns_l_eff > 0:
        if props.D_l is None:
            raise ValueError("solve_Yl=True requires props.D_l to be provided.")

        if return_diag:
            _, _, diag_yl = build_liquid_species_system(
                cfg=cfg,
                grid=grid,
                layout=layout,
                state_old=state_old,
                props=props,
                dt=dt,
                interface_evap=iface_evap,
                A_out=A,
                b_out=b,
                return_diag=True,
                state_guess=state_guess,
            )
            diag_sys.setdefault("blocks", {})["Yl"] = diag_yl
        else:
            build_liquid_species_system(
                cfg=cfg,
                grid=grid,
                layout=layout,
                state_old=state_old,
                props=props,
                dt=dt,
                interface_evap=iface_evap,
                A_out=A,
                b_out=b,
                return_diag=False,
                state_guess=state_guess,
            )

    if return_diag:
        return A, b, diag_sys
    return A, b


def build_transport_system_petsc(*args, **kwargs):
    raise RuntimeError(
        "PETSc backend not available in SciPy workflow. Use build_transport_system() with SciPy solvers."
    )
