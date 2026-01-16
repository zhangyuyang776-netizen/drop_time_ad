"""
PETSc bridge for numpy-based assembly (Phase B.1.2).

This module wraps the existing *_SciPy.py assemblers, converts dense numpy A,b
into PETSc Mat/Vec, and keeps a single source of assembly truth.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

from assembly.jacobian_pattern import JacobianPattern, build_jacobian_pattern
from assembly.petsc_prealloc import build_petsc_prealloc_from_pattern
from core.layout import UnknownLayout
from core.types import CaseConfig, Grid1D, Props, State
from physics.flux_convective_gas import compute_gas_convective_flux_T, compute_gas_convective_flux_Y
from physics.flux_liq import compute_liq_diffusive_flux_Y
from physics.interface_bc import EqResultLike, build_interface_coeffs, _get_balance_species_indices
from physics.radius_eq import build_radius_row
from physics.stefan_velocity import compute_stefan_velocity
from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

from assembly.build_system_SciPy import build_transport_system as build_transport_system_numpy
from assembly.build_liquid_T_system_SciPy import build_liquid_T_system as build_liquid_T_system_numpy
from assembly.build_species_system_SciPy import (
    build_gas_species_system_global as build_gas_species_system_global_numpy,
)
from assembly.build_liquid_species_system_SciPy import (
    build_liquid_species_system as build_liquid_species_system_numpy,
)

if TYPE_CHECKING:
    from petsc4py import PETSc


def _get_petsc():
    try:
        bootstrap_mpi_before_petsc()
        from petsc4py import PETSc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("petsc4py is required for PETSc assembly bridge.") from exc
    return PETSc


def _mat_add(A, row: int, col: int, val: float) -> None:
    A.setValue(int(row), int(col), float(val), addv=True)


def _vec_add(b, row: int, val: float) -> None:
    b.setValue(int(row), float(val), addv=True)


def _record_dirichlet(dirichlet_rows: list[tuple[int, float]], row: int, value: float) -> None:
    dirichlet_rows.append((int(row), float(value)))


def _apply_dirichlet_after_assembly(A, b, dirichlet_rows: list[tuple[int, float]]) -> None:
    if not dirichlet_rows:
        return

    row_to_val: dict[int, float] = {}
    for r, v in dirichlet_rows:
        row_to_val[int(r)] = float(v)

    rows = list(row_to_val.keys())
    vals = [row_to_val[r] for r in rows]

    A.zeroRows(rows, diag=1.0)
    b.setValues(rows, vals, addv=False)
    b.assemblyBegin()
    b.assemblyEnd()


def _scatter_interface_rows_petsc(
    A,
    b,
    iface_coeffs,
) -> None:
    for row_def in iface_coeffs.rows:
        r = row_def.row
        if not row_def.cols:
            continue
        A.setValues(r, row_def.cols, row_def.vals, addv=True)
        _vec_add(b, r, row_def.rhs)


def _scatter_radius_row_petsc(
    A,
    b,
    rad_coeffs,
) -> None:
    r = rad_coeffs.row
    if rad_coeffs.cols:
        A.setValues(r, rad_coeffs.cols, rad_coeffs.vals, addv=True)
    _vec_add(b, r, rad_coeffs.rhs)


def numpy_dense_to_petsc_aij(
    A: np.ndarray,
    b: np.ndarray,
    comm=None,
):
    """
    Convert dense numpy (A,b) to PETSc AIJ Mat and Vec.

    Bridge phase: SERIAL ONLY (comm size must be 1).
    """
    PETSc = _get_petsc()
    if comm is None:
        comm = PETSc.COMM_WORLD
    if comm.getSize() != 1:
        raise NotImplementedError("Bridge assembly is serial-only.")

    A = np.asarray(A, dtype=PETSc.ScalarType)
    if A.ndim != 2:
        raise TypeError(f"A must be 2D, got ndim={A.ndim}")
    m, n = A.shape
    if m != n:
        raise ValueError(f"A must be square, got shape {A.shape}")

    b = np.asarray(b, dtype=PETSc.ScalarType)
    if b.ndim != 1:
        raise TypeError(f"b must be 1D, got ndim={b.ndim}")
    if b.shape[0] != n:
        raise ValueError(f"b shape {b.shape} does not match A dimension {n}")

    if n == 0:
        A_p = PETSc.Mat().createAIJ(size=(0, 0), comm=PETSc.COMM_SELF)
        b_p = PETSc.Vec().createSeq(0, comm=PETSc.COMM_SELF)
        return A_p, b_p

    nnz_row = np.count_nonzero(A, axis=1).astype(PETSc.IntType, copy=False)
    A_p = PETSc.Mat().createAIJ(size=(n, n), comm=comm, nnz=nnz_row)
    for i in range(n):
        cols = np.nonzero(A[i])[0]
        if cols.size == 0:
            continue
        cols_i = cols.astype(PETSc.IntType, copy=False)
        vals = A[i, cols]
        A_p.setValues(i, cols_i, vals)

    A_p.assemblyBegin()
    A_p.assemblyEnd()

    b_p = PETSc.Vec().createSeq(n, comm=PETSc.COMM_SELF)
    idx = np.arange(n, dtype=PETSc.IntType)
    b_p.setValues(idx, b)
    b_p.assemblyBegin()
    b_p.assemblyEnd()

    return A_p, b_p


def create_empty_transport_mat_vec_petsc(
    pattern: JacobianPattern,
    comm: Optional["PETSc.Comm"] = None,
):
    """
    Create an empty PETSc AIJ matrix + RHS vector for the global transport system.

    Stage 3: serial-only preallocation from a global Jacobian sparsity pattern.
    """
    PETSc = _get_petsc()
    if comm is None:
        comm = PETSc.COMM_WORLD

    N, d_nz, o_nz = build_petsc_prealloc_from_pattern(pattern)

    A = PETSc.Mat().createAIJ(size=(N, N), comm=comm, nnz=(d_nz, o_nz))
    A.setUp()

    b = PETSc.Vec().createMPI(N, comm=comm)
    b.set(0.0)
    b.assemblyBegin()
    b.assemblyEnd()

    return A, b


def build_transport_system_petsc(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props: Props,
    dt: float,
    state_guess: Optional[State] = None,
    eq_model=None,
    eq_result=None,
    return_diag: bool = False,
    comm=None,
):
    """Bridge: numpy assembly -> PETSc Mat/Vec (global transport system)."""
    if return_diag:
        A_np, b_np, diag = build_transport_system_numpy(
            cfg=cfg,
            grid=grid,
            layout=layout,
            state_old=state_old,
            props=props,
            dt=dt,
            state_guess=state_guess,
            eq_model=eq_model,
            eq_result=eq_result,
            return_diag=True,
        )
        A_p, b_p = numpy_dense_to_petsc_aij(A_np, b_np, comm=comm)
        return A_p, b_p, diag

    A_np, b_np = build_transport_system_numpy(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state_old,
        props=props,
        dt=dt,
        state_guess=state_guess,
        eq_model=eq_model,
        eq_result=eq_result,
        return_diag=False,
    )
    return numpy_dense_to_petsc_aij(A_np, b_np, comm=comm)


def build_liquid_T_system_petsc(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props: Props,
    dt: float,
    *,
    couple_interface: bool = False,
    comm=None,
):
    """Bridge: numpy assembly -> PETSc Mat/Vec (local liquid temperature system)."""
    A_np, b_np = build_liquid_T_system_numpy(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state_old,
        props=props,
        dt=dt,
        couple_interface=couple_interface,
    )
    return numpy_dense_to_petsc_aij(A_np, b_np, comm=comm)


def build_gas_species_system_global_petsc(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props: Props,
    dt: float,
    eq_result=None,
    interface_evap: Optional[dict[str, Any]] = None,
    *,
    A_out: Optional[np.ndarray] = None,
    b_out: Optional[np.ndarray] = None,
    return_diag: bool = False,
    state_guess: Optional[State] = None,
    comm=None,
):
    """Bridge: numpy assembly -> PETSc Mat/Vec (global gas species system)."""
    if return_diag:
        A_np, b_np, diag = build_gas_species_system_global_numpy(
            cfg=cfg,
            grid=grid,
            layout=layout,
            state_old=state_old,
            props=props,
            dt=dt,
            eq_result=eq_result,
            interface_evap=interface_evap,
            A_out=A_out,
            b_out=b_out,
            return_diag=True,
            state_guess=state_guess,
        )
        A_p, b_p = numpy_dense_to_petsc_aij(A_np, b_np, comm=comm)
        return A_p, b_p, diag

    A_np, b_np = build_gas_species_system_global_numpy(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state_old,
        props=props,
        dt=dt,
        eq_result=eq_result,
        interface_evap=interface_evap,
        A_out=A_out,
        b_out=b_out,
        return_diag=False,
        state_guess=state_guess,
    )
    return numpy_dense_to_petsc_aij(A_np, b_np, comm=comm)


def build_liquid_species_system_petsc(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props: Props,
    dt: float,
    interface_evap: Optional[dict[str, Any]] = None,
    *,
    A_out: Optional[np.ndarray] = None,
    b_out: Optional[np.ndarray] = None,
    return_diag: bool = False,
    state_guess: Optional[State] = None,
    comm=None,
):
    """Bridge: numpy assembly -> PETSc Mat/Vec (global liquid species system)."""
    if return_diag:
        A_np, b_np, diag = build_liquid_species_system_numpy(
            cfg=cfg,
            grid=grid,
            layout=layout,
            state_old=state_old,
            props=props,
            dt=dt,
            interface_evap=interface_evap,
            A_out=A_out,
            b_out=b_out,
            return_diag=True,
            state_guess=state_guess,
        )
        A_p, b_p = numpy_dense_to_petsc_aij(A_np, b_np, comm=comm)
        return A_p, b_p, diag

    A_np, b_np = build_liquid_species_system_numpy(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state_old,
        props=props,
        dt=dt,
        interface_evap=interface_evap,
        A_out=A_out,
        b_out=b_out,
        return_diag=False,
        state_guess=state_guess,
    )
    return numpy_dense_to_petsc_aij(A_np, b_np, comm=comm)


# Explicit bridge alias for linear assembly mode switching.
build_transport_system_petsc_bridge = build_transport_system_petsc


def build_transport_system_petsc_native(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props: Props,
    dt: float,
    state_guess: Optional[State] = None,
    eq_model=None,
    eq_result=None,
    return_diag: bool = False,
    comm=None,
):
    """
    Placeholder for native PETSc AIJ assembly (Stage 3).
    """
    PETSc = _get_petsc()
    if comm is None:
        comm = PETSc.COMM_WORLD
    if comm.getSize() != 1:
        raise NotImplementedError("Native PETSc assembly is serial-only for now.")

    if not layout.has_block("Tg"):
        raise ValueError("layout missing Tg block required for transport system assembly.")
    if state_old.Tg.shape != (grid.Ng,):
        raise ValueError(f"Tg shape {state_old.Tg.shape} != ({grid.Ng},)")

    theta = float(cfg.discretization.theta)
    if abs(theta - 1.0) > 1e-12:
        raise ValueError("Transport assembly currently assumes theta=1.0 (fully implicit).")

    if state_guess is None:
        state_guess = state_old

    phys = cfg.physics
    if phys.include_mpp and layout.has_block("mpp") and eq_result is None:
        raise ValueError("include_mpp=True requires eq_result with 'Yg_eq' (no internal fallback).")

    Ng = grid.Ng
    N = layout.n_dof()

    pattern = build_jacobian_pattern(cfg, grid, layout)
    A, b = create_empty_transport_mat_vec_petsc(pattern, comm=comm)
    diag_sys: Dict[str, Any] = {}
    dirichlet_rows: list[tuple[int, float]] = []

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
        b_i = aP_time * float(state_old.Tg[ig])

        # Left face (ig-1/2)
        if ig > 0:
            rL = grid.r_c[cell_idx - 1]
            rC = grid.r_c[cell_idx]
            A_f = float(grid.A_f[cell_idx])
            dr = rC - rL
            k_face = 0.5 * (k_i + float(props.k_g[ig - 1]))
            coeff = k_face * A_f / dr
            aP += coeff
            _mat_add(A, row, layout.idx_Tg(ig - 1), -coeff)
        else:
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
                _mat_add(A, row, layout.idx_Ts(), -coeff_if)
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
            _mat_add(A, row, layout.idx_Tg(ig + 1), -coeff)

        _mat_add(A, row, row, aP)
        _vec_add(b, row, b_i)

    # Stefan velocity and convective flux (implicit in Tg via state_guess)
    stefan = compute_stefan_velocity(cfg, grid, props, state_guess)
    u_face = stefan.u_face

    q_conv = compute_gas_convective_flux_T(
        cfg=cfg,
        grid=grid,
        props=props,
        Tg=state_guess.Tg,
        u_face=u_face,
    )

    iface_coeffs = None
    iface_evap = None

    for ig in range(Ng):
        row = layout.idx_Tg(ig)
        cell_idx = gas_start + ig
        f_L = cell_idx
        f_R = cell_idx + 1
        A_L = float(grid.A_f[f_L])
        A_R = float(grid.A_f[f_R])
        q_L = float(q_conv[f_L])
        q_R = float(q_conv[f_R])
        S_conv = A_R * q_R - A_L * q_L
        _vec_add(b, row, -S_conv)

    Tg_far = float(cfg.initial.T_inf)
    row_bc = layout.idx_Tg(Ng - 1)
    _record_dirichlet(dirichlet_rows, row_bc, Tg_far)

    # --- Liquid temperature equations (Tl block) ---
    if layout.has_block("Tl"):
        A_l, b_l = build_liquid_T_system_numpy(
            cfg=cfg,
            grid=grid,
            layout=layout,
            state_old=state_old,
            props=props,
            dt=dt,
            couple_interface=True,
        )
        Nl = grid.Nl
        for il in range(Nl):
            row_global = layout.idx_Tl(il)
            row_vals = A_l[il]
            cols_local = np.nonzero(row_vals)[0]
            if cols_local.size:
                cols_global = [layout.idx_Tl(int(il2)) for il2 in cols_local]
                A.setValues(row_global, cols_global, row_vals[cols_local], addv=True)
            if b_l[il] != 0.0:
                _vec_add(b, row_global, b_l[il])

        il_last = Nl - 1
        if il_last >= 0:
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
            _mat_add(A, row_last, row_last, coeff_if)
            if phys.include_Ts and layout.has_block("Ts"):
                _mat_add(A, row_last, layout.idx_Ts(), -coeff_if)
            else:
                Ts_bc = float(state_old.Ts)
                if getattr(cfg.physics, "interface", None) is not None:
                    if getattr(cfg.physics.interface, "bc_mode", "") == "Ts_fixed":
                        Ts_bc = float(getattr(cfg.physics.interface, "Ts_fixed", Ts_bc))
                _vec_add(b, row_last, coeff_if * Ts_bc)

    # --- Interface equations: Ts energy jump + mpp Stefan (single-condensable) ---
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
        _scatter_interface_rows_petsc(A, b, iface_coeffs)
        diag_sys.update(iface_coeffs.diag)
        iface_evap = iface_coeffs.diag.get("evaporation")

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

            if getattr(props, "h_g", None) is not None:
                h_mix_if = float(np.asarray(props.h_g, dtype=np.float64)[0])
            else:
                h_mix_if = 0.0

            if getattr(props, "h_gk", None) is not None and np.asarray(props.h_gk).shape[0] >= Ns_full:
                h_k_if = np.asarray(props.h_gk[:Ns_full, 0], dtype=np.float64)
            else:
                h_k_if = np.zeros(Ns_full, dtype=np.float64)

            q_iface = mpp_evap * h_mix_if
            if j_corr_full.size:
                q_iface += float(np.dot(h_k_if, j_corr_full))

            row_Tg0 = layout.idx_Tg(0)
            _vec_add(b, row_Tg0, -A_if * q_iface)

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
        _scatter_radius_row_petsc(A, b, rad_coeffs)
        diag_sys.update(rad_coeffs.diag)

    # --- Gas species equations: Yg (strongly coupled) ---
    if getattr(cfg.physics, "solve_Yg", False) and layout.has_block("Yg") and layout.Ns_g_eff > 0:
        if phys.include_mpp and eq_result is None:
            raise ValueError("Species assembly with include_mpp=True requires eq_result with 'Yg_eq'.")

        Ns_full, Ng = state_old.Yg.shape
        if props.D_g is None:
            raise ValueError("Props.D_g is None; gas diffusion coefficients required.")
        if props.D_g.shape != (Ns_full, Ng):
            raise ValueError(f"D_g shape {props.D_g.shape} != ({Ns_full}, {Ng})")
        if props.rho_g.shape != (Ng,):
            raise ValueError(f"rho_g shape {props.rho_g.shape} != ({Ng},)")

        gas_start = grid.Nl
        if grid.gas_slice is not None and grid.gas_slice.start is not None:
            gas_start = int(grid.gas_slice.start)
        iface_f = grid.iface_f

        k_cond_full = None
        k_cond_red = None
        condensable_name = None
        Yg_eq_face = None
        J_iface_full = None

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

        if iface_evap is not None:
            if "J_full_state" in iface_evap:
                J_iface_full = np.asarray(iface_evap["J_full_state"], dtype=np.float64).reshape(-1)
            elif "J_full" in iface_evap:
                J_iface_full = np.asarray(iface_evap["J_full"], dtype=np.float64).reshape(-1)
            if J_iface_full is not None and J_iface_full.shape[0] != Ns_full:
                raise ValueError(f"interface_evap['J_full'] length {J_iface_full.shape[0]} != Ns_full={Ns_full}")

        diag_y: Dict[str, Any] = {
            "species": {"Ns_g_eff": layout.Ns_g_eff},
            "bc": {},
            "convection": {},
        }
        diag_y["species"]["condensable"] = {
            "name": condensable_name,
            "k_full": k_cond_full,
            "k_red": k_cond_red,
            "Yg_eq": Yg_eq_face,
        }

        seed = float(getattr(cfg.initial, "Y_seed", 1e-12))
        Y_far_map: Dict[str, float] = {}
        for name in layout.gas_species_reduced:
            Y_far_map[name] = float(cfg.initial.Yg.get(name, seed))
        diag_y["bc"]["outer"] = {"type": "Dirichlet_all_solved", "Y_far_preview": Y_far_map}

        convection_enabled = bool(
            getattr(cfg.physics, "stefan_velocity", False) and getattr(cfg.physics, "species_convection", False)
        )
        diag_y["convection"]["enabled"] = convection_enabled

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

                if ig == 0:
                    if J_iface_full is not None:
                        A_if = float(grid.A_f[iface_f])
                        J_L = float(J_iface_full[k_full])
                        b_i += A_if * J_L
                        diag_y["bc"].setdefault("inner_flux", []).append({"species": name, "J_if": J_L})
                    elif is_condensable and Yg_eq_face is not None:
                        dr_if = float(grid.r_c[cell_idx] - grid.r_f[iface_f])
                        if dr_if <= 0.0:
                            raise ValueError("Non-positive dr_if for interface Dirichlet.")
                        A_if = float(grid.A_f[iface_f])
                        coeff_if = rho_i * Dk_i * A_if / dr_if
                        aP += coeff_if
                        b_i += coeff_if * Yg_eq_face
                        if "inner" not in diag_y["bc"]:
                            diag_y["bc"]["inner"] = {
                                "type": "Dirichlet_cond_only",
                                "Yg_bc_if": Yg_eq_face,
                                "dr_if": dr_if,
                                "A_if": A_if,
                                "species": name,
                            }
                    else:
                        diag_y["bc"].setdefault("inner_noncondensables", []).append(name)
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
                    _mat_add(A, row, layout.idx_Yg(k_red, ig - 1), -coeff_L)

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
                    coeff_R = rho_f * Dk_f * A_f_R / dr
                    aP += coeff_R
                    _mat_add(A, row, layout.idx_Yg(k_red, ig + 1), -coeff_R)

                _mat_add(A, row, row, aP)
                _vec_add(b, row, b_i)

        if convection_enabled:
            stefan = compute_stefan_velocity(cfg, grid, props, state_guess)
            u_face = stefan.u_face
            J_conv_all = compute_gas_convective_flux_Y(cfg, grid, props, state_guess.Yg, u_face)

            u_out = float(u_face[grid.Nc])
            if u_out < 0.0:
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
                    S_conv = A_R * J_R - A_L * J_L
                    _vec_add(b, row, -S_conv)

        seed = float(getattr(cfg.initial, "Y_seed", 1e-12))
        ig_bc = Ng - 1
        for k_red in range(layout.Ns_g_eff):
            row_bc = layout.idx_Yg(k_red, ig_bc)
            name = layout.gas_species_reduced[k_red]
            Y_far = float(cfg.initial.Yg.get(name, seed))
            _record_dirichlet(dirichlet_rows, row_bc, Y_far)

        diag_y["bc"].setdefault("inner", {"type": "zero_flux_noncondensable"})

        if return_diag:
            diag_sys.setdefault("blocks", {})["Yg"] = diag_y

    # --- Liquid species equations: Yl ---
    if getattr(cfg.physics, "solve_Yl", False) and layout.has_block("Yl") and layout.Ns_l_eff > 0:
        if props.D_l is None:
            raise ValueError("solve_Yl=True requires props.D_l to be provided.")

        Ns_full, Nl = state_old.Yl.shape
        if props.D_l.shape != (Ns_full, Nl):
            raise ValueError(f"D_l shape {props.D_l.shape} != ({Ns_full}, {Nl})")
        if props.rho_l.shape != (Nl,):
            raise ValueError(f"rho_l shape {props.rho_l.shape} != ({Nl},)")

        diag_yl: Dict[str, Any] = {"bc": {}, "evap": {}}

        J_diff = compute_liq_diffusive_flux_Y(cfg, grid, props, state_guess.Yl)
        J_tot = np.array(J_diff, copy=True)

        mpp = float(getattr(state_guess, "mpp", 0.0))
        if iface_evap is not None:
            diag_yl["evap"]["mpp_eval"] = float(iface_evap.get("mpp_eval", 0.0))
            diag_yl["evap"]["mpp_state"] = mpp
        if mpp != 0.0:
            Yl_face = np.asarray(state_guess.Yl[:, Nl - 1], dtype=np.float64)
            J_evap = mpp * Yl_face
            J_tot[:, grid.iface_f] += J_evap
            diag_yl["evap"]["mpp"] = mpp
            diag_yl["evap"]["Yl_face"] = Yl_face

        rho_l = props.rho_l
        V_c = grid.V_c
        A_f = grid.A_f

        for k_red in range(layout.Ns_l_eff):
            k_full = layout.liq_reduced_to_full_idx[k_red]
            for il in range(Nl):
                row = layout.idx_Yl(k_red, il)
                cell_idx = il

                rho_i = float(rho_l[il])
                V = float(V_c[cell_idx])

                aP_time = rho_i * V / dt
                b_i = aP_time * float(state_old.Yl[k_full, il])

                f_L = il
                f_R = il + 1
                A_L = float(A_f[f_L])
                A_R = float(A_f[f_R])
                J_L = float(J_tot[k_full, f_L])
                J_R = float(J_tot[k_full, f_R])
                div = A_R * J_R - A_L * J_L

                _mat_add(A, row, row, aP_time)
                _vec_add(b, row, b_i - div)

        if return_diag:
            diag_sys.setdefault("blocks", {})["Yl"] = diag_yl

    A.assemblyBegin()
    A.assemblyEnd()
    b.assemblyBegin()
    b.assemblyEnd()
    _apply_dirichlet_after_assembly(A, b, dirichlet_rows)

    if return_diag:
        diag_sys.setdefault("meta", {})["backend"] = "petsc_native_aij"
        return A, b, diag_sys
    return A, b
