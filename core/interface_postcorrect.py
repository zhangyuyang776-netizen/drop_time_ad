from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from core.types import CaseConfig, Grid1D, Props, State
from core.layout import UnknownLayout
from physics.interface_bc import build_interface_coeffs
from properties.equilibrium import build_equilibrium_model, compute_interface_equilibrium
from properties.compute_props import get_or_build_models

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InterfacePostCorrectStats:
    status: str
    accepted: bool
    res0: float
    res_best: float
    improved_ratio: float
    iters: int
    mode: str
    reason: str
    Ts_old: float
    Ts_new: float
    mpp_old: float
    mpp_new: float
    uncovered_cells_gas: int | None


def compute_interface_residual_on_grid(
    *,
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state: State,
    props: Props,
) -> tuple[np.ndarray, dict]:
    """
    Compute interface residuals on the given grid for the current state.

    Returns:
      r = [R_E, R_Y]
      aux: dict with eq_result and diagnostics
    """
    eq_result = _build_eq_result(cfg, grid, state)
    coeffs = build_interface_coeffs(
        grid=grid,
        state=state,
        props=props,
        layout=layout,
        cfg=cfg,
        eq_result=eq_result,
    )
    diag = coeffs.diag

    res_T = 0.0
    if "Ts_energy" in diag:
        res_T = float(diag["Ts_energy"]["balance_into_interface"]["balance_eq"])

    res_m = 0.0
    evap = diag.get("evaporation", {})
    deltaY = evap.get("DeltaY_eff", None)
    j_corr_full = evap.get("j_corr_full", None)
    k_b_full = evap.get("k_b_full", None)
    if deltaY is not None and j_corr_full is not None and k_b_full is not None:
        j_corr_full = np.asarray(j_corr_full, dtype=np.float64)
        if 0 <= int(k_b_full) < j_corr_full.size:
            res_m = float(deltaY) * float(state.mpp) - float(j_corr_full[int(k_b_full)])

    res = np.array([res_T, res_m], dtype=np.float64)
    aux = {"eq_result": eq_result, "diag": diag}
    return res, aux


def post_correct_interface_after_remap(
    *,
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state: State,
    iface_fill: Mapping[str, object] | None,
    evaporation: dict | None,
    props: Props | None,
    remap_stats: Mapping[str, object] | None = None,
    tol: float = 1e-10,
    rtol: float = 1e-3,
    tol_skip_abs: float = 1e-8,
    eps_rd_rel: float = 1e-6,
    skip_if_uncovered_zero: bool = True,
    max_iter: int = 12,
    damping: float = 0.7,
    fd_eps_T: float = 1e-3,
    fd_eps_m: float = 1e-6,
    fd_eps_T_rel: float = 1.0e-6,
    fd_eps_m_rel: float = 1.0e-6,
    line_search_eta: float = 1.0e-3,
    improve_min: float = 1.0e-3,
    debug: bool = False,
) -> tuple[State, dict, InterfacePostCorrectStats]:
    """
    Post-correct interface Ts/mpp on the new grid using a small Newton solve.
    """
    Ts_old = float(state.Ts)
    mpp_old = float(state.mpp)
    evap_out = dict(evaporation or {})

    def _stats(
        *,
        status: str,
        accepted: bool,
        res0: float,
        res_best: float,
        iters: int,
        reason: str,
        Ts_new: float,
        mpp_new: float,
        uncovered_cells_gas: int | None,
    ) -> InterfacePostCorrectStats:
        improved_ratio = float(res_best / res0) if res0 > 0.0 else float("nan")
        return InterfacePostCorrectStats(
            status=str(status),
            accepted=bool(accepted),
            res0=float(res0),
            res_best=float(res_best),
            improved_ratio=improved_ratio,
            iters=int(iters),
            mode="2x2",
            reason=str(reason),
            Ts_old=float(Ts_old),
            Ts_new=float(Ts_new),
            mpp_old=float(mpp_old),
            mpp_new=float(mpp_new),
            uncovered_cells_gas=uncovered_cells_gas,
        )

    if not (cfg.physics.include_Ts and cfg.physics.include_mpp):
        stats = _stats(
            status="SKIPPED",
            accepted=False,
            res0=0.0,
            res_best=0.0,
            iters=0,
            reason="post-correct skipped (Ts/mpp not enabled)",
            Ts_new=Ts_old,
            mpp_new=mpp_old,
            uncovered_cells_gas=None,
        )
        return state, evap_out, stats

    if not layout.has_block("Ts") or not layout.has_block("mpp"):
        stats = _stats(
            status="SKIPPED",
            accepted=False,
            res0=0.0,
            res_best=0.0,
            iters=0,
            reason="post-correct skipped (layout missing Ts/mpp)",
            Ts_new=Ts_old,
            mpp_new=mpp_old,
            uncovered_cells_gas=None,
        )
        return state, evap_out, stats

    if props is None:
        stats = _stats(
            status="ERROR",
            accepted=False,
            res0=0.0,
            res_best=0.0,
            iters=0,
            reason="post-correct requires props (got None)",
            Ts_new=Ts_old,
            mpp_new=mpp_old,
            uncovered_cells_gas=None,
        )
        return state, evap_out, stats

    Ts = Ts_old
    mpp = mpp_old

    Yg_sat_fallback = None
    if iface_fill is not None and "Yg_sat" in iface_fill:
        Yg_sat_fallback = np.asarray(iface_fill["Yg_sat"], dtype=np.float64)

    if bool(getattr(cfg.checks, "enforce_T_bounds", False)):
        T_min = float(cfg.checks.T_min)
        T_max = float(cfg.checks.T_max)
    else:
        Tl_if = float(state.Tl[-1]) if state.Tl.size else Ts_old
        Tg_if = float(state.Tg[0]) if state.Tg.size else Ts_old
        T_min = min(Tl_if, Tg_if) - 50.0
        T_max = max(Tl_if, Tg_if) + 200.0

    iface_type = getattr(cfg.physics.interface, "type", "no_condensation")

    def eval_res(Ts_val: float, mpp_val: float) -> tuple[np.ndarray, dict, Dict[str, Any]]:
        Ts_val = float(Ts_val)
        mpp_val = float(mpp_val)
        Ts_prev = state.Ts
        mpp_prev = state.mpp
        state.Ts = Ts_val
        state.mpp = mpp_val
        try:
            res_vec, aux = compute_interface_residual_on_grid(
                cfg=cfg,
                grid=grid,
                layout=layout,
                state=state,
                props=props,
            )
            eq_result = aux.get("eq_result", {})
            diag = aux.get("diag", {})
        finally:
            state.Ts = Ts_prev
            state.mpp = mpp_prev
        return res_vec, eq_result, diag

    try:
        res_vec, eq_result, diag = eval_res(Ts, mpp)
        res0 = float(np.max(np.abs(res_vec)))
        uncovered_cells_gas = None
        if remap_stats is not None:
            try:
                uncovered_cells_gas = int(remap_stats.get("uncovered_cells_gas", 0))
            except Exception:
                uncovered_cells_gas = None

        if skip_if_uncovered_zero and remap_stats is not None:
            Rd_old = remap_stats.get("Rd_old", None)
            Rd_new = remap_stats.get("Rd_new", None)
            try:
                Rd_old = float(Rd_old)
                Rd_new = float(Rd_new)
            except Exception:
                Rd_old = None
                Rd_new = None
            if Rd_old is not None and Rd_new is not None and Rd_old > 0.0:
                rel_change = abs(Rd_new - Rd_old) / Rd_old
                if (
                    rel_change <= float(eps_rd_rel)
                    and int(uncovered_cells_gas or 0) == 0
                    and tol_skip_abs is not None
                    and np.isfinite(res0)
                    and res0 <= float(tol_skip_abs)
                ):
                    return (
                        state,
                        evap_out,
                        _stats(
                            status="SKIPPED",
                            accepted=False,
                            res0=res0,
                            res_best=res0,
                            iters=0,
                            reason="post-correct skipped (policy)",
                            Ts_new=Ts_old,
                            mpp_new=mpp_old,
                            uncovered_cells_gas=uncovered_cells_gas,
                        ),
                    )

        if uncovered_cells_gas is not None and uncovered_cells_gas > 0:
            max_iter = max(int(max_iter), 20)

        best_Ts = Ts
        best_mpp = mpp
        best_res = res0
        best_eq_result = eq_result
        iters = 0
        res_curr = res0

        while iters < max_iter:
            iters += 1
            dT = max(float(fd_eps_T), float(fd_eps_T_rel) * max(1.0, abs(Ts)))
            dm = max(float(fd_eps_m), float(fd_eps_m_rel) * max(1.0, abs(mpp)))
            res_Ts, _, _ = eval_res(Ts + dT, mpp)
            res_mpp, _, _ = eval_res(Ts, mpp + dm)
            J = np.column_stack(((res_Ts - res_vec) / dT, (res_mpp - res_vec) / dm))
            try:
                dx = np.linalg.solve(J, -res_vec)
            except Exception:
                break

            best_trial = None
            best_trial_res = res_curr
            for alpha in (1.0, float(damping), 0.7, 0.5, 0.3, 0.1, 0.05):
                Ts_trial = Ts + alpha * float(dx[0])
                mpp_trial = mpp + alpha * float(dx[1])
                Ts_trial = float(np.clip(Ts_trial, T_min, T_max))
                if iface_type == "no_condensation" and mpp_trial < 0.0:
                    mpp_trial = 0.0
                res_trial_vec, eq_trial, _ = eval_res(Ts_trial, mpp_trial)
                res_trial = float(np.max(np.abs(res_trial_vec)))
                if res_trial < res_curr * (1.0 - float(line_search_eta)):
                    best_trial = (Ts_trial, mpp_trial, res_trial_vec, eq_trial, res_trial)
                    break
                if res_trial < best_trial_res:
                    best_trial_res = res_trial
                    best_trial = (Ts_trial, mpp_trial, res_trial_vec, eq_trial, res_trial)

            if best_trial is None or best_trial[4] >= res_curr:
                break

            Ts, mpp, res_vec, eq_result, res_curr = best_trial
            if res_curr < best_res:
                best_res = res_curr
                best_Ts = Ts
                best_mpp = mpp
                best_eq_result = eq_result

        accepted = bool((best_res < res0 * (1.0 - float(improve_min))) or (best_res <= float(tol)))
        if accepted:
            Yg_sat = None
            if best_eq_result is not None and "Yg_eq" in best_eq_result:
                Yg_sat = np.asarray(best_eq_result["Yg_eq"], dtype=np.float64)
            elif Yg_sat_fallback is not None:
                Yg_sat = Yg_sat_fallback
            if Yg_sat is None:
                Yg_sat = np.asarray(state.Yg[:, 0], dtype=np.float64)
            evap_out = _apply_iface_solution(
                state=state,
                evaporation=evap_out,
                Ts=float(best_Ts),
                mpp=float(best_mpp),
                Yg_sat=Yg_sat,
            )
            if debug:
                evap_out["postcorrect_diag"] = diag
            return (
                state,
                evap_out,
                _stats(
                    status="IMPROVED",
                    accepted=True,
                    res0=res0,
                    res_best=best_res,
                    iters=iters,
                    reason="residual reduced",
                    Ts_new=best_Ts,
                    mpp_new=best_mpp,
                    uncovered_cells_gas=uncovered_cells_gas,
                ),
            )

        return (
            state,
            evap_out,
            _stats(
                status="NO_IMPROVEMENT",
                accepted=False,
                res0=res0,
                res_best=best_res,
                iters=iters,
                reason="improvement below threshold",
                Ts_new=Ts_old,
                mpp_new=mpp_old,
                uncovered_cells_gas=uncovered_cells_gas,
            ),
        )
    except Exception as exc:
        return (
            state,
            evap_out,
            _stats(
                status="ERROR",
                accepted=False,
                res0=0.0,
                res_best=0.0,
                iters=0,
                reason=f"{type(exc).__name__}: {exc}",
                Ts_new=Ts_old,
                mpp_new=mpp_old,
                uncovered_cells_gas=None,
            ),
        )


def _build_eq_result(
    cfg: CaseConfig,
    grid: Grid1D,
    state: State,
) -> Dict[str, np.ndarray]:
    il_if = grid.Nl - 1
    ig_if = 0

    Ts = float(state.Ts)
    Yl_face = np.asarray(state.Yl[:, il_if], dtype=np.float64)
    Yg_face = np.asarray(state.Yg[:, ig_if], dtype=np.float64)
    Ns_g = Yg_face.shape[0]
    Ns_l = Yl_face.shape[0]

    M_g, M_l = _get_molar_masses_from_cfg(cfg, Ns_g=Ns_g, Ns_l=Ns_l)
    eq_model = build_equilibrium_model(cfg, Ns_g=Ns_g, Ns_l=Ns_l, M_g=M_g, M_l=M_l)

    Pg = float(getattr(cfg.initial, "P_inf", 101325.0))
    Yg_eq, y_cond, psat = compute_interface_equilibrium(
        eq_model,
        Ts=Ts,
        Pg=Pg,
        Yl_face=Yl_face,
        Yg_face=Yg_face,
    )
    eq_result = {"Yg_eq": Yg_eq, "y_cond": y_cond, "psat": psat}
    return _complete_Yg_eq_with_closure(cfg, eq_result)


def _get_molar_masses_from_cfg(
    cfg: CaseConfig, *, Ns_g: int, Ns_l: int
) -> Tuple[np.ndarray, np.ndarray]:
    mw_map = dict(getattr(cfg.species, "mw_kg_per_mol", {}))

    gas_names = list(getattr(cfg.species, "gas_species_full", []) or [])
    liq_names = list(getattr(cfg.species, "liq_species", []) or [])
    if len(gas_names) != Ns_g:
        raise ValueError(f"gas_species_full length {len(gas_names)} != Ns_g {Ns_g}")
    if len(liq_names) != Ns_l:
        raise ValueError(f"liq_species length {len(liq_names)} != Ns_l {Ns_l}")
    missing_l = [name for name in liq_names if name not in mw_map]
    if missing_l:
        raise ValueError(f"mw_kg_per_mol missing liquid entries: {missing_l}")

    gas_model, _ = get_or_build_models(cfg)
    mech_names = list(getattr(gas_model.gas, "species_names", []))
    if mech_names and mech_names != gas_names:
        raise ValueError("gas_species_full does not match Cantera mechanism species order.")
    M_g = np.asarray(gas_model.gas.molecular_weights, dtype=np.float64) / 1000.0
    if M_g.size != Ns_g:
        raise ValueError(f"Cantera molecular_weights size {M_g.size} != Ns_g {Ns_g}")

    def _build(names: list[str], Ns: int) -> np.ndarray:
        arr = np.ones(Ns, dtype=np.float64)
        for i, name in enumerate(names):
            if i >= Ns:
                break
            arr[i] = float(mw_map[name])
        return arr

    M_l = _build(liq_names, Ns_l)
    return M_g, M_l


def _complete_Yg_eq_with_closure(cfg: CaseConfig, eq_result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    Y = np.asarray(eq_result["Yg_eq"], dtype=np.float64)
    Ns = len(getattr(cfg.species, "gas_species_full", Y))
    if Y.ndim != 1 or Y.size != Ns:
        raise ValueError(f"eq_result['Yg_eq'] must be 1D length {Ns}, got shape {Y.shape}")

    gas_balance = cfg.species.gas_balance_species
    gas_names = list(getattr(cfg.species, "gas_species_full", []) or [])
    k_cl = gas_names.index(gas_balance) if gas_balance in gas_names else None

    Y = np.clip(Y, 0.0, 1.0)
    if k_cl is not None and 0 <= k_cl < Ns:
        sum_other = float(np.sum(Y) - Y[k_cl])
        Y[k_cl] = max(0.0, 1.0 - sum_other)
    s = float(np.sum(Y))
    if s > 1e-14 and not np.isclose(s, 1.0, rtol=1e-12, atol=1e-12):
        Y = Y / s
    eq_result = dict(eq_result)
    eq_result["Yg_eq"] = Y
    return eq_result


def _apply_iface_solution(
    *,
    state: State,
    evaporation: dict | None,
    Ts: float,
    mpp: float,
    Yg_sat: np.ndarray,
) -> dict:
    state.Ts = float(Ts)
    state.mpp = float(mpp)
    evap_out = dict(evaporation or {})
    evap_out["Ts"] = float(Ts)
    evap_out["mpp"] = float(mpp)
    evap_out["Yg_sat"] = np.asarray(Yg_sat, dtype=np.float64)
    return evap_out
