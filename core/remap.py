from __future__ import annotations

import logging
import os
import numpy as np

from core.types import CaseConfig, Grid1D, RemapPlan, State
from core.layout import UnknownLayout
from core.simplex import project_Y_cellwise

logger = logging.getLogger(__name__)


def _interp_cell_centered(r_old: np.ndarray, v_old: np.ndarray, r_new: np.ndarray) -> np.ndarray:
    """Simple 1D linear interpolation for cell-centered quantities."""
    return np.interp(r_new, r_old, v_old)


def _vol_shell(rL: float, rR: float) -> float:
    """Volume of spherical shell [rL, rR] in 3D."""
    rL = float(rL)
    rR = float(rR)
    if rR <= rL:
        return 0.0
    return (4.0 * np.pi / 3.0) * (max(rR, 0.0) ** 3 - max(rL, 0.0) ** 3)


def _vol_shell_3(rL: float, rR: float) -> float:
    """Spherical shell volume factor without constant 4*pi/3."""
    rL = float(rL)
    rR = float(rR)
    if rR <= rL:
        return 0.0
    rL = max(rL, 0.0)
    rR = max(rR, 0.0)
    return rR**3 - rL**3


def _faces_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a, b)


def _remap_cellavg_overlap_spherical(
    q_old: np.ndarray,
    r_f_old: np.ndarray,
    r_f_new: np.ndarray,
    *,
    j0: int = 0,
    j1: int | None = None,
) -> np.ndarray:
    """
    First-order conservative remap of cell-averages in spherical coordinates
    using cell-volume overlaps.
    """
    q_old = np.asarray(q_old, dtype=np.float64)
    r_f_old = np.asarray(r_f_old, dtype=np.float64)
    r_f_new = np.asarray(r_f_new, dtype=np.float64)

    N_old = q_old.size
    N_new = r_f_new.size - 1
    if j1 is None:
        j1 = N_new - 1
    if N_old <= 0 or N_new <= 0:
        return np.array([], dtype=np.float64)
    if r_f_old.size != N_old + 1:
        raise ValueError("r_f_old length must be N_old + 1.")
    if r_f_new.size != N_new + 1:
        raise ValueError("r_f_new length must be N_new + 1.")
    if j0 < 0 or j1 >= N_new or j0 > j1:
        raise ValueError("Invalid remap range for _remap_cellavg_overlap_spherical.")

    out = np.empty((j1 - j0 + 1,), dtype=np.float64)

    i = 0
    rL0 = float(r_f_new[j0])
    while i < N_old and r_f_old[i + 1] <= rL0:
        i += 1

    for jj, j in enumerate(range(j0, j1 + 1)):
        rL = float(r_f_new[j])
        rR = float(r_f_new[j + 1])
        Vj = _vol_shell(rL, rR)
        if Vj <= 0.0 or not np.isfinite(Vj):
            raise ValueError("Non-positive new cell volume in conservative remap.")

        while i < N_old and r_f_old[i + 1] <= rL:
            i += 1

        acc = 0.0
        k = i
        while k < N_old and r_f_old[k] < rR:
            a = max(rL, float(r_f_old[k]))
            b = min(rR, float(r_f_old[k + 1]))
            if b > a:
                acc += float(q_old[k]) * _vol_shell(a, b)
            if r_f_old[k + 1] <= rR:
                k += 1
            else:
                break

        out[jj] = acc / Vj
        i = k

    return out


def _remap_cellavg_overlap_spherical_with_overlap(
    q_old: np.ndarray,
    r_f_old: np.ndarray,
    r_f_new: np.ndarray,
    *,
    j0: int = 0,
    j1: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Conservative remap with overlap volume tracking.

    Returns:
      q_new[j0:j1+1]      : remapped cell averages
      V3_new[j0:j1+1]     : new cell V3 = rR^3 - rL^3
      V3_overlap[j0:j1+1] : overlapped V3 accumulated from old cells
    """
    q_old = np.asarray(q_old, dtype=np.float64)
    r_f_old = np.asarray(r_f_old, dtype=np.float64)
    r_f_new = np.asarray(r_f_new, dtype=np.float64)

    N_old = q_old.size
    N_new = r_f_new.size - 1
    if j1 is None:
        j1 = N_new - 1
    if N_old <= 0 or N_new <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if r_f_old.size != N_old + 1:
        raise ValueError("r_f_old length must be N_old + 1.")
    if r_f_new.size != N_new + 1:
        raise ValueError("r_f_new length must be N_new + 1.")
    if j0 < 0 or j1 >= N_new or j0 > j1:
        raise ValueError("Invalid remap range for _remap_cellavg_overlap_spherical_with_overlap.")

    out = np.empty((j1 - j0 + 1,), dtype=np.float64)
    V3_new = np.empty((j1 - j0 + 1,), dtype=np.float64)
    V3_ov = np.empty((j1 - j0 + 1,), dtype=np.float64)

    i = 0
    rL0 = float(r_f_new[j0])
    while i < N_old and r_f_old[i + 1] <= rL0:
        i += 1

    for jj, j in enumerate(range(j0, j1 + 1)):
        rL = float(r_f_new[j])
        rR = float(r_f_new[j + 1])
        V3j = _vol_shell_3(rL, rR)
        if V3j <= 0.0 or not np.isfinite(V3j):
            raise ValueError("Non-positive new cell volume in conservative remap.")

        while i < N_old and r_f_old[i + 1] <= rL:
            i += 1

        acc_qV3 = 0.0
        acc_V3 = 0.0
        k = i
        while k < N_old and r_f_old[k] < rR:
            a = max(rL, float(r_f_old[k]))
            b = min(rR, float(r_f_old[k + 1]))
            if b > a:
                dV3 = _vol_shell_3(a, b)
                acc_V3 += dV3
                acc_qV3 += float(q_old[k]) * dV3
            if r_f_old[k + 1] <= rR:
                k += 1
            else:
                break

        V3_new[jj] = V3j
        V3_ov[jj] = acc_V3
        out[jj] = acc_qV3 / V3j if V3j > 0.0 else 0.0
        i = k

    return out, V3_new, V3_ov


def _apply_uncovered_fill(
    q_new: np.ndarray,
    V3_new: np.ndarray,
    V3_ov: np.ndarray,
    *,
    q_fill: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply fill for uncovered volume using diluted remap result:
      q_new := q_remap + q_fill * (V3_un / V3_new)
    """
    q_new = np.asarray(q_new, dtype=np.float64)
    V3_new = np.asarray(V3_new, dtype=np.float64)
    V3_ov = np.asarray(V3_ov, dtype=np.float64)

    if q_new.shape != V3_new.shape or q_new.shape != V3_ov.shape:
        raise ValueError("q_new, V3_new, V3_ov must have the same shape.")

    if np.any(V3_new <= 0.0) or np.any(~np.isfinite(V3_new)):
        raise ValueError("Non-positive or non-finite V3_new in uncovered fill.")

    V3_ov = np.minimum(np.maximum(V3_ov, 0.0), V3_new)
    V3_un = np.maximum(V3_new - V3_ov, 0.0)

    if np.isscalar(q_fill):
        qf = float(q_fill)
    else:
        qf = np.asarray(q_fill, dtype=np.float64)
        if qf.shape != q_new.shape:
            raise ValueError("q_fill shape must match q_new for vector fill.")

    uncovered_frac = V3_un / V3_new
    q_out = q_new + qf * uncovered_frac
    return q_out, uncovered_frac
def _reconstruct_gas_closure(Yg_full: np.ndarray, layout: UnknownLayout) -> np.ndarray:
    """Recompute gas closure species so each column sums to one (includes unsolved background species)."""
    k_cl = getattr(layout, "gas_closure_index", None)
    if k_cl is None:
        return Yg_full
    if k_cl >= Yg_full.shape[0]:
        raise ValueError(f"Gas closure index {k_cl} out of bounds for Yg_full shape {Yg_full.shape}")

    # Sum over all species except closure itself
    sum_other = np.sum(Yg_full, axis=0) - Yg_full[k_cl, :]
    closure = 1.0 - sum_other

    # Soft clamp small numerical violations
    tol = 1e-10
    closure = np.where((closure < 0.0) & (closure >= -tol), 0.0, closure)
    closure = np.where((closure > 1.0) & (closure <= 1.0 + tol), 1.0, closure)
    closure = np.clip(closure, 0.0, 1.0)

    Yg_full[k_cl, :] = closure

    # Optional final renormalization to mitigate interpolation noise
    sums = np.sum(Yg_full, axis=0)
    mask = sums > 0.0
    Yg_full[:, mask] /= sums[mask]

    return Yg_full


def remap_state_to_new_grid(
    state_old: State,
    grid_old: Grid1D,
    grid_new: Grid1D,
    cfg: CaseConfig,
    layout: UnknownLayout,
    remap_plan: RemapPlan | None = None,
    iface_fill: dict | None = None,
    return_stats: bool = False,
) -> State | tuple[State, dict]:
    """
    Remap state arrays from grid_old to grid_new (first-order conservative overlap remap).

    Assumes Nl/Ng are unchanged; only geometry (r) moves with Rd.
    """
    state = state_old.copy()

    Nl, Ng = grid_old.Nl, grid_old.Ng
    r_f_old = grid_old.r_f
    r_f_new = grid_new.r_f

    if state.Yg.size and state.Yg.shape[0] != layout.Ns_g_full:
        raise ValueError(
            "remap_state_to_new_grid expects full gas species in State.Yg "
            f"(got Ns_g={state.Yg.shape[0]}, expected Ns_g_full={layout.Ns_g_full})."
        )
    if state.Yl.size and state.Yl.shape[0] != layout.Ns_l_full:
        raise ValueError(
            "remap_state_to_new_grid expects full liquid species in State.Yl "
            f"(got Ns_l={state.Yl.shape[0]}, expected Ns_l_full={layout.Ns_l_full})."
        )

    r_f_liq_old = r_f_old[: Nl + 1]
    r_f_gas_old = r_f_old[Nl : Nl + Ng + 1]
    r_f_liq_new = r_f_new[: Nl + 1]
    r_f_gas_new = r_f_new[Nl : Nl + Ng + 1]

    fill_Ts = None
    fill_Yg = None
    if iface_fill is not None:
        fill_Ts = iface_fill.get("Ts", None)
        fill_Yg = iface_fill.get("Yg_sat", None)
        if fill_Ts is not None and not np.isfinite(float(fill_Ts)):
            raise ValueError("iface_fill['Ts'] must be finite.")
        if fill_Yg is not None:
            fill_Yg = np.asarray(fill_Yg, dtype=np.float64)
            if fill_Yg.shape != (layout.Ns_g_full,):
                raise ValueError(
                    "iface_fill['Yg_sat'] must have shape "
                    f"({layout.Ns_g_full},), got {fill_Yg.shape}."
                )

    # Temperatures
    if state.Tl.size:
        if remap_plan is None or remap_plan.liq_remap:
            state.Tl = _remap_cellavg_overlap_spherical(state.Tl, r_f_liq_old, r_f_liq_new)
    if state.Tg.size:
        if remap_plan is None:
            if fill_Ts is None:
                state.Tg = _remap_cellavg_overlap_spherical(state.Tg, r_f_gas_old, r_f_gas_new)
            else:
                Tg_new, V3_new, V3_ov = _remap_cellavg_overlap_spherical_with_overlap(
                    state.Tg, r_f_gas_old, r_f_gas_new
                )
                state.Tg, _ = _apply_uncovered_fill(
                    Tg_new, V3_new, V3_ov, q_fill=float(fill_Ts)
                )
        else:
            state.Tg = _remap_gas_scalar(
                state.Tg,
                r_f_gas_old=r_f_gas_old,
                r_f_gas_new=r_f_gas_new,
                remap_plan=remap_plan,
                n_liq=Nl,
                iface_fill_value=float(fill_Ts) if fill_Ts is not None else None,
            )

    # Liquid species
    if state.Yl.size:
        Ns_l, _ = state.Yl.shape
        Yl_new = np.empty((Ns_l, Nl), dtype=np.float64)
        if remap_plan is None or remap_plan.liq_remap:
            for k in range(Ns_l):
                Yl_new[k, :] = _remap_cellavg_overlap_spherical(
                    state.Yl[k, :], r_f_liq_old, r_f_liq_new
                )
        else:
            Yl_new[:, :] = state.Yl[:, :Nl]
        # FIXED: Use simplex projection instead of clip to enforce sum(Y) = 1.0
        Yl_new = project_Y_cellwise(Yl_new, min_Y=1e-14, axis=0)
        state.Yl = Yl_new

    # Gas species
    if state.Yg.size:
        Ns_g, _ = state.Yg.shape
        Yg_new = np.empty((Ns_g, Ng), dtype=np.float64)
        if remap_plan is None:
            for k in range(Ns_g):
                if fill_Yg is None:
                    Yg_new[k, :] = _remap_cellavg_overlap_spherical(
                        state.Yg[k, :], r_f_gas_old, r_f_gas_new
                    )
                else:
                    y_new, V3_new, V3_ov = _remap_cellavg_overlap_spherical_with_overlap(
                        state.Yg[k, :], r_f_gas_old, r_f_gas_new
                    )
                    Yg_new[k, :], _ = _apply_uncovered_fill(
                        y_new, V3_new, V3_ov, q_fill=fill_Yg[k]
                    )
        else:
            for k in range(Ns_g):
                Yg_new[k, :] = _remap_gas_scalar(
                    state.Yg[k, :],
                    r_f_gas_old=r_f_gas_old,
                    r_f_gas_new=r_f_gas_new,
                    remap_plan=remap_plan,
                    n_liq=Nl,
                    iface_fill_value=fill_Yg[k] if fill_Yg is not None else None,
                )
        # FIXED: Use simplex projection instead of closure reconstruction + clip
        # This ensures sum(Y) = 1.0 with minimum perturbation, preventing
        # cumulative error accumulation from remapping (root cause of step 269 failure)
        Yg_new = project_Y_cellwise(Yg_new, min_Y=1e-14, axis=0)
        state.Yg = Yg_new

    # Keep Rd consistent with the new grid geometry.
    state.Rd = float(r_f_new[Nl])

    stats: dict | None = None
    want_stats = return_stats or os.getenv("DROPLET_REMAP_DEBUG", "0") == "1"
    if want_stats and Ng > 0:
        zeros = np.zeros(Ng, dtype=np.float64)
        _, V3_new, V3_ov = _remap_cellavg_overlap_spherical_with_overlap(
            zeros, r_f_gas_old, r_f_gas_new
        )
        V3_un = np.maximum(V3_new - V3_ov, 0.0)
        uncovered_frac = np.where(V3_new > 0.0, V3_un / V3_new, 0.0)
        uncovered_cells = int(np.sum(uncovered_frac > 0.0))
        uncovered_max = float(np.max(uncovered_frac)) if uncovered_frac.size else 0.0
        uncovered_p95 = float(np.percentile(uncovered_frac, 95)) if uncovered_frac.size else 0.0
        uncovered_band = None
        if remap_plan is not None and remap_plan.gas_remap_cells is not None:
            g0 = remap_plan.gas_remap_cells[0] - Nl
            g1 = remap_plan.gas_remap_cells[1] - Nl
            if 0 <= g0 <= g1 < Ng:
                uncovered_band = int(np.sum(uncovered_frac[g0 : g1 + 1] > 0.0))
        stats = {
            "uncovered_frac_gas": uncovered_frac,
            "V3_new_gas": V3_new,
            "V3_ov_gas": V3_ov,
            "uncovered_cells_gas": uncovered_cells,
            "uncovered_frac_max_gas": uncovered_max,
            "uncovered_frac_p95_gas": uncovered_p95,
            "uncovered_cells_band_gas": uncovered_band,
            "Rd_old": float(r_f_old[Nl]),
            "Rd_new": float(r_f_new[Nl]),
        }
        if os.getenv("DROPLET_REMAP_DEBUG", "0") == "1":
            logger.info(
                "remap uncovered gas: cells=%d max=%.3e p95=%.3e band=%s",
                uncovered_cells,
                uncovered_max,
                uncovered_p95,
                uncovered_band,
            )

    if return_stats:
        return state, (stats or {})
    return state


def _remap_gas_scalar(
    values_old: np.ndarray,
    *,
    r_f_gas_old: np.ndarray,
    r_f_gas_new: np.ndarray,
    remap_plan: RemapPlan,
    n_liq: int,
    iface_fill_value: float | None = None,
) -> np.ndarray:
    values_old = np.asarray(values_old, dtype=np.float64)
    r_f_gas_old = np.asarray(r_f_gas_old, dtype=np.float64)
    r_f_gas_new = np.asarray(r_f_gas_new, dtype=np.float64)
    Ng = values_old.size
    if r_f_gas_old.size != Ng + 1 or r_f_gas_new.size != Ng + 1:
        raise ValueError("r_f_gas arrays do not match gas segment sizes.")
    values_new = np.empty(Ng, dtype=np.float64)
    filled = np.zeros(Ng, dtype=bool)

    if remap_plan.gas_copy_cells is not None:
        g0 = remap_plan.gas_copy_cells[0] - n_liq
        g1 = remap_plan.gas_copy_cells[1] - n_liq
        if g0 < 0 or g1 >= Ng or g0 > g1:
            raise ValueError("gas_copy_cells out of bounds for gas segment.")
        if _faces_equal(r_f_gas_old[g0 : g1 + 2], r_f_gas_new[g0 : g1 + 2]):
            values_new[g0 : g1 + 1] = values_old[g0 : g1 + 1]
            filled[g0 : g1 + 1] = True

    if remap_plan.gas_remap_cells is not None:
        g0 = remap_plan.gas_remap_cells[0] - n_liq
        g1 = remap_plan.gas_remap_cells[1] - n_liq
        if g0 < 0 or g1 >= Ng or g0 > g1:
            raise ValueError("gas_remap_cells out of bounds for gas segment.")
        q_new, V3_new, V3_ov = _remap_cellavg_overlap_spherical_with_overlap(
            values_old, r_f_gas_old, r_f_gas_new, j0=g0, j1=g1
        )
        if iface_fill_value is not None:
            q_new, _ = _apply_uncovered_fill(q_new, V3_new, V3_ov, q_fill=iface_fill_value)
        values_new[g0 : g1 + 1] = q_new
        filled[g0 : g1 + 1] = True

    if not np.all(filled):
        idx = np.where(~filled)[0]
        q_full, V3_new, V3_ov = _remap_cellavg_overlap_spherical_with_overlap(
            values_old, r_f_gas_old, r_f_gas_new
        )
        if iface_fill_value is not None:
            q_full, _ = _apply_uncovered_fill(q_full, V3_new, V3_ov, q_fill=iface_fill_value)
        values_new[idx] = q_full[idx]

    return values_new
