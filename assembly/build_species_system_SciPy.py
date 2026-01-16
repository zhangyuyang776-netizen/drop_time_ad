"""
SciPy/Numpy assembly for gas species mass-fraction equations (Yg block, reduced species).

Scope (Step 19.4.2):
- Assemble all active gas species (layout.Ns_g_eff) into the global system index space.
- Implicit time + diffusion; optional convection (Stefan velocity) when enabled.
- Inner boundary (interface): Dirichlet for the single condensable species using Yg_eq; zero flux for others.
- Outer boundary: strong Dirichlet to farfield composition from cfg.initial.Yg for every solved species.

Notes
-----
- Rows/cols always use reduced-species indices via layout.idx_Yg(k_red, ig); full-species
  indices are only used to read State/Props (Yg, D_g).
- This assembly currently returns dense numpy arrays shaped to the full global system.
  Large systems should migrate to sparse to avoid memory blow-up.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.layout import UnknownLayout
from core.types import CaseConfig, Grid1D, Props, State
from physics.flux_convective_gas import compute_gas_convective_flux_Y
from physics.interface_bc import EqResultLike, _get_balance_species_indices
from physics.stefan_velocity import compute_stefan_velocity


def _build_species_system_toy_DO_NOT_USE(*args, **kwargs):
    """Legacy single-species toy builder (Step 9). Kept only to avoid silent imports."""
    raise NotImplementedError("Deprecated: use build_gas_species_system_global instead.")


def build_gas_species_system_global(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props: Props,
    dt: float,
    eq_result: EqResultLike = None,
    interface_evap: Dict[str, Any] | None = None,
    *,
    A_out: np.ndarray | None = None,
    b_out: np.ndarray | None = None,
    return_diag: bool = False,
    state_guess: State | None = None,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Assemble gas-species mass-fraction equations into the global system (dense).

    Returns
    -------
    A_y, b_y : np.ndarray, np.ndarray
        Dense arrays shaped (layout.size, layout.size) and (layout.size,) with only Yg rows filled.
        If A_out/b_out are provided, assembly is performed in-place on those arrays.
    diag_y : dict, optional
        Diagnostics (condensable info, BC previews, convection flag).

    Notes
    -----
    If convection is enabled, the convective term uses state_guess when provided.
    """
    if not layout.has_block("Yg"):
        raise ValueError("layout missing 'Yg' block required for gas species assembly.")

    theta = float(cfg.discretization.theta)
    if abs(theta - 1.0) > 1e-12:
        raise ValueError("Species assembly currently assumes theta=1.0 (fully implicit diffusion).")

    if state_guess is None:
        state_guess = state_old

    Ns_full, Ng = state_old.Yg.shape
    if props.D_g is None:
        raise ValueError("Props.D_g is None; gas diffusion coefficients required.")
    if props.D_g.shape != (Ns_full, Ng):
        raise ValueError(f"D_g shape {props.D_g.shape} != ({Ns_full}, {Ng})")
    if props.rho_g.shape != (Ng,):
        raise ValueError(f"rho_g shape {props.rho_g.shape} != ({Ng},)")

    N = layout.n_dof()
    if A_out is None:
        A = np.zeros((N, N), dtype=np.float64)
    else:
        if A_out.shape != (N, N):
            raise ValueError(f"A_out shape {A_out.shape} mismatch for global size {N}")
        A = A_out
    if b_out is None:
        b = np.zeros(N, dtype=np.float64)
    else:
        if b_out.shape != (N,):
            raise ValueError(f"b_out shape {b_out.shape} mismatch for global size {N}")
        b = b_out
    diag: Dict[str, Any] = {
        "species": {"Ns_g_eff": layout.Ns_g_eff},
        "bc": {},
        "convection": {},
    }

    gas_start = grid.Nl
    if grid.gas_slice is not None and grid.gas_slice.start is not None:
        gas_start = int(grid.gas_slice.start)
    iface_f = grid.iface_f

    # Condensable mapping + equilibrium value
    k_cond_full: int | None = None
    k_cond_red: int | None = None
    condensable_name: str | None = None
    Yg_eq_face: float | None = None
    J_iface_full: np.ndarray | None = None

    if cfg.physics.include_mpp:
        if eq_result is None or "Yg_eq" not in (eq_result or {}):
            raise ValueError("mpp requires eq_result with 'Yg_eq' for condensable Dirichlet BC.")
    if eq_result is not None and "Yg_eq" in eq_result:
        Yg_eq = np.asarray(eq_result["Yg_eq"], dtype=np.float64)
        if Yg_eq.shape[0] < layout.Ns_g_full:
            raise ValueError(
                f"eq_result['Yg_eq'] length {Yg_eq.shape[0]} shorter than Ns_g_full={layout.Ns_g_full}"
            )
    else:
        Yg_eq = None

    try:
        bal = _get_balance_species_indices(cfg, layout)
        condensable_name = bal["g_name"]
        k_cond_full = bal["k_g_full"]
        k_cond_red = bal["k_g_red"]
    except Exception:
        k_cond_full = k_cond_red = None
        condensable_name = None

    if k_cond_full is not None and Yg_eq is not None:
        Yg_eq_face = float(Yg_eq[k_cond_full])

    if cfg.physics.include_mpp:
        if k_cond_full is None or k_cond_red is None:
            raise ValueError(
                "include_mpp=True requires condensable gas species to be present (non-closure) in the mechanism. "
                "Check liq2gas_map and gas_balance_species."
            )
        if Yg_eq is None or Yg_eq_face is None:
            raise ValueError("include_mpp=True requires eq_result['Yg_eq'] with condensable entry.")

    if interface_evap is not None:
        if "J_full_state" in interface_evap:
            J_iface_full = np.asarray(interface_evap["J_full_state"], dtype=np.float64).reshape(-1)
        elif "J_full" in interface_evap:
            J_iface_full = np.asarray(interface_evap["J_full"], dtype=np.float64).reshape(-1)
        if J_iface_full.shape[0] != Ns_full:
            raise ValueError(f"interface_evap['J_full'] length {J_iface_full.shape[0]} != Ns_full={Ns_full}")

    diag["species"]["condensable"] = {
        "name": condensable_name,
        "k_full": k_cond_full,
        "k_red": k_cond_red,
        "Yg_eq": Yg_eq_face,
    }

    # Farfield composition map for solved species
    seed = float(getattr(cfg.initial, "Y_seed", 1e-12))
    Y_far_map: Dict[str, float] = {}
    for name in layout.gas_species_reduced:
        Y_far_map[name] = float(cfg.initial.Yg.get(name, seed))
    diag["bc"]["outer"] = {"type": "Dirichlet_all_solved", "Y_far_preview": Y_far_map}

    # Optional convection toggle
    convection_enabled = bool(getattr(cfg.physics, "stefan_velocity", False) and getattr(cfg.physics, "species_convection", False))
    diag["convection"]["enabled"] = convection_enabled

    rho_g = props.rho_g
    D_g = props.D_g
    Yg_old = state_old.Yg

    for k_red in range(layout.Ns_g_eff):
        k_full = layout.gas_reduced_to_full_idx[k_red]
        name = layout.gas_species_reduced[k_red]
        is_condensable = k_cond_red is not None and k_red == k_cond_red

        for ig in range(Ng):
            row = layout.idx_Yg(k_red, ig)
            cell_idx = gas_start + ig

            rho_i = float(rho_g[ig])
            Dk_i = float(D_g[k_full, ig])
            V = float(grid.V_c[cell_idx])

            aP_time = rho_i * V / dt
            aP = aP_time
            b_i = aP_time * float(Yg_old[k_full, ig])

            # Left face: interface flux override if provided; else fallback to Dirichlet (cond) or zero-flux
            if ig == 0:
                if J_iface_full is not None:
                    A_if = float(grid.A_f[iface_f])
                    J_L = float(J_iface_full[k_full])
                    b_i += A_if * J_L  # positive outward flux increases RHS for interface-adjacent gas cell
                    diag["bc"].setdefault("inner_flux", []).append({"species": name, "J_if": J_L})
                elif is_condensable and Yg_eq_face is not None:
                    dr_if = float(grid.r_c[cell_idx] - grid.r_f[iface_f])
                    if dr_if <= 0.0:
                        raise ValueError("Non-positive dr_if for interface Dirichlet.")
                    A_if = float(grid.A_f[iface_f])
                    coeff_if = rho_i * Dk_i * A_if / dr_if
                    aP += coeff_if
                    b_i += coeff_if * Yg_eq_face
                    if "inner" not in diag["bc"]:
                        diag["bc"]["inner"] = {
                            "type": "Dirichlet_cond_only",
                            "Yg_bc_if": Yg_eq_face,
                            "dr_if": dr_if,
                            "A_if": A_if,
                            "species": name,
                        }
                else:
                    diag["bc"].setdefault("inner_noncondensables", []).append(name)
            else:
                iL = cell_idx - 1
                rL = grid.r_c[iL]
                rC = grid.r_c[cell_idx]
                A_f_L = float(grid.A_f[cell_idx])
                dr_L = rC - rL
                if dr_L <= 0.0:
                    raise ValueError("Non-positive dr on left face in species diffusion assembly")

                rho_L = float(rho_g[ig - 1])
                Dk_L = float(D_g[k_full, ig - 1])

                rho_f_L = 0.5 * (rho_L + rho_i)
                Dk_f_L = 0.5 * (Dk_L + Dk_i)
                coeff_L = rho_f_L * Dk_f_L * A_f_L / dr_L

                aP += coeff_L
                A[row, layout.idx_Yg(k_red, ig - 1)] += -coeff_L

            # Right face diffusion (internal faces)
            if ig < Ng - 1:
                rC = grid.r_c[cell_idx]
                rR = grid.r_c[cell_idx + 1]
                A_f_R = float(grid.A_f[cell_idx + 1])
                dr = rR - rC
                if dr <= 0.0:
                    raise ValueError("Non-positive dr in species diffusion assembly")
                rho_R = float(rho_g[ig + 1])
                Dk_R = float(D_g[k_full, ig + 1])
                rho_f = 0.5 * (rho_i + rho_R)
                Dk_f = 0.5 * (Dk_i + Dk_R)
                coeff_R = rho_f * Dk_f * A_f_R / dr  # kg/s
                aP += coeff_R
                A[row, layout.idx_Yg(k_red, ig + 1)] += -coeff_R

            A[row, row] += aP
            b[row] += b_i

    # Convective flux (optional)
    if convection_enabled:
        # Step 19.4.2: Stefan velocity consistent with Yg^{n+1}
        stefan = compute_stefan_velocity(cfg, grid, props, state_guess)
        u_face = stefan.u_face
        J_conv_all = compute_gas_convective_flux_Y(cfg, grid, props, state_guess.Yg, u_face)

        # Optional farfield upwind for outer inflow
        u_out = float(u_face[grid.Nc])
        if u_out < 0.0:
            # Build full-length farfield vector in mechanism order
            seed = float(getattr(cfg.initial, "Y_seed", 1e-12))
            Y_far_full = np.zeros((layout.Ns_g_full,), dtype=np.float64)
            for idx_full, spec_name in enumerate(layout.gas_species_full):
                Y_far_full[idx_full] = float(cfg.initial.Yg.get(spec_name, seed))
            J_conv_all[:, grid.Nc] = float(props.rho_g[-1]) * u_out * Y_far_full

        for k_red in range(layout.Ns_g_eff):
            k_full = layout.gas_reduced_to_full_idx[k_red]
            for ig in range(Ng):
                row = layout.idx_Yg(k_red, ig)
                cell_idx = gas_start + ig
                f_L = cell_idx
                f_R = cell_idx + 1
                A_L = float(grid.A_f[f_L])
                A_R = float(grid.A_f[f_R])
                J_L = float(J_conv_all[k_full, f_L])
                J_R = float(J_conv_all[k_full, f_R])
                S_conv = A_R * J_R - A_L * J_L  # outward positive mass flow (kg/s)
                b[row] -= S_conv

    # Outer boundary Dirichlet for all solved species
    seed = float(getattr(cfg.initial, "Y_seed", 1e-12))
    ig_bc = Ng - 1
    for k_red in range(layout.Ns_g_eff):
        row_bc = layout.idx_Yg(k_red, ig_bc)
        name = layout.gas_species_reduced[k_red]
        Y_far = float(cfg.initial.Yg.get(name, seed))
        A[row_bc, :] = 0.0
        A[row_bc, row_bc] = 1.0
        b[row_bc] = Y_far

    diag["bc"].setdefault("inner", {"type": "zero_flux_noncondensable"})

    if return_diag:
        return A, b, diag
    return A, b
