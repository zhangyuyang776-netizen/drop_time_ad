"""
Output helpers:
- ScalarsWriter: configurable scalar output to CSV with optional cumulative diagnostics.
- write_step_scalars: stateless per-step scalars (fields controlled by cfg.io.fields.scalars).
- write_step_spatial: spatial snapshot output.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.types import CaseConfig, Grid1D, State
from properties.equilibrium import compute_interface_equilibrium

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from solvers.timestepper import StepDiagnostics

logger = logging.getLogger(__name__)


def _ensure_parent(path: Path) -> None:
    """Ensure parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _scalar_fields(cfg: CaseConfig) -> list[str]:
    fields = getattr(getattr(cfg, "io", None), "fields", None)
    return list(getattr(fields, "scalars", []) or [])


def _scalar_output_path(cfg: CaseConfig, out_dir: Path | str | None = None) -> Path:
    base = Path(out_dir) if out_dir is not None else (Path(cfg.paths.case_dir) / "scalars")
    return base / "scalars.csv"


def _as_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _is_finite(value: float) -> bool:
    try:
        return np.isfinite(value)
    except Exception:
        return False


def _compute_scalar_row(
    fields: list[str],
    *,
    cfg: CaseConfig,
    grid: Grid1D | None,
    state: State,
    props,
    diag: "StepDiagnostics" | None,
    step_id: int | None,
    t: float | None,
    integ: dict | None,
    eq_model,
    unknown_fields: set[str] | None = None,
) -> list[float]:
    gas_names = list(getattr(cfg.species, "gas_species_full", []))

    t_old = _as_float(getattr(diag, "t_old", math.nan)) if diag is not None else math.nan
    t_new = _as_float(getattr(diag, "t_new", math.nan)) if diag is not None else math.nan
    dt = _as_float(getattr(diag, "dt", math.nan)) if diag is not None else math.nan
    t_val = _as_float(t) if t is not None else (t_new if _is_finite(t_new) else math.nan)

    Ts = _as_float(getattr(state, "Ts", math.nan))
    Rd = _as_float(getattr(state, "Rd", math.nan))
    mpp = _as_float(getattr(state, "mpp", math.nan))

    A_if = 4.0 * math.pi * Rd * Rd if _is_finite(Rd) else math.nan
    m_dot = A_if * mpp if _is_finite(A_if) and _is_finite(mpp) else math.nan

    m_evap_cum = math.nan
    m_drop0 = math.nan
    m_drop = math.nan
    mass_closure_err = math.nan
    if integ is not None:
        m_drop0 = _as_float(integ.get("m_drop0", math.nan))
        if _is_finite(m_dot):
            if _is_finite(dt):
                if integ.get("m_dot_prev") is None:
                    integ["m_evap_cum"] = _as_float(integ.get("m_evap_cum", 0.0)) + m_dot * dt
                else:
                    integ["m_evap_cum"] = _as_float(integ.get("m_evap_cum", 0.0)) + 0.5 * (
                        integ["m_dot_prev"] + m_dot
                    ) * dt
            integ["m_dot_prev"] = m_dot
        m_evap_cum = _as_float(integ.get("m_evap_cum", math.nan))

    rho_l_mean = math.nan
    if props is not None and hasattr(props, "rho_l"):
        try:
            if props.rho_l is not None and props.rho_l.size:
                rho_l_mean = float(np.mean(props.rho_l))
        except Exception:
            pass
    if _is_finite(rho_l_mean) and _is_finite(Rd):
        m_drop = 4.0 / 3.0 * math.pi * rho_l_mean * Rd**3
    if _is_finite(m_drop) and _is_finite(m_drop0):
        mass_closure_err = m_drop + m_evap_cum - m_drop0

    Tg = getattr(state, "Tg", np.array([], dtype=float))
    Tl = getattr(state, "Tl", np.array([], dtype=float))
    Tg_min = _as_float(getattr(diag, "Tg_min", math.nan)) if diag is not None else math.nan
    Tg_max = _as_float(getattr(diag, "Tg_max", math.nan)) if diag is not None else math.nan
    if not _is_finite(Tg_min) and Tg.size:
        Tg_min = float(np.min(Tg))
    if not _is_finite(Tg_max) and Tg.size:
        Tg_max = float(np.max(Tg))

    Tg_mean = float(np.mean(Tg)) if Tg.size else math.nan
    Tl_mean = float(np.mean(Tl)) if Tl.size else math.nan
    Tg0 = float(Tg[0]) if Tg.size else math.nan
    Tg_if = Tg0
    Tg_far = float(Tg[-1]) if Tg.size else math.nan
    Tl_center = float(Tl[0]) if Tl.size else math.nan
    Tl_if = float(Tl[-1]) if Tl.size else math.nan
    Tl_min = float(np.min(Tl)) if Tl.size else math.nan
    Tl_max = float(np.max(Tl)) if Tl.size else math.nan

    Yg0 = state.Yg[:, 0] if getattr(state, "Yg", np.array([], dtype=float)).size else np.array([], dtype=float)
    sumYg0 = float(np.sum(Yg0)) if Yg0.size else math.nan

    cp_g0 = math.nan
    k_g0 = math.nan
    if props is not None:
        try:
            if hasattr(props, "cp_g") and props.cp_g.size:
                cp_g0 = float(props.cp_g[0])
        except Exception:
            pass
        try:
            if hasattr(props, "k_g") and props.k_g.size:
                k_g0 = float(props.k_g[0])
        except Exception:
            pass

    energy_balance_if = _as_float(getattr(diag, "energy_balance_if", math.nan)) if diag is not None else math.nan
    mass_balance_rd = _as_float(getattr(diag, "mass_balance_rd", math.nan)) if diag is not None else math.nan

    Yg_eq = None
    need_eq = any(name.startswith("Yg_eq_") for name in fields)
    if need_eq and eq_model is not None and grid is not None and gas_names:
        try:
            il_if = grid.Nl - 1
            ig_if = 0
            Yl_face = state.Yl[:, il_if] if state.Yl.size else np.zeros((len(cfg.species.liq_species),), float)
            Yg_face = state.Yg[:, ig_if] if state.Yg.size else np.zeros((len(gas_names),), float)
            Pg = float(getattr(cfg.initial, "P_inf", 101325.0))
            Yg_eq, _, _ = compute_interface_equilibrium(
                eq_model, Ts=float(state.Ts), Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face
            )
        except Exception:
            Yg_eq = None

    base = {
        "step": step_id if step_id is not None else math.nan,
        "t": t_val,
        "t_old": t_old,
        "t_new": t_new,
        "dt": dt,
        "Rd": Rd,
        "mpp": mpp,
        "Ts": Ts,
        "A_if": A_if,
        "m_dot": m_dot,
        "m_evap_cum": m_evap_cum,
        "m_drop": m_drop,
        "m_drop0": m_drop0,
        "mass_closure_err": mass_closure_err,
        "Tg0": Tg0,
        "Tg_min": Tg_min,
        "Tg_max": Tg_max,
        "Tg_if": Tg_if,
        "Tg_far": Tg_far,
        "Tg_mean": Tg_mean,
        "Tl_center": Tl_center,
        "Tl_if": Tl_if,
        "Tl_min": Tl_min,
        "Tl_max": Tl_max,
        "Tl_mean": Tl_mean,
        "sumYg0": sumYg0,
        "cp_g0": cp_g0,
        "k_g0": k_g0,
        "energy_balance_if": energy_balance_if,
        "mass_balance_rd": mass_balance_rd,
    }

    row: list[float] = []
    for field in fields:
        if field in base:
            row.append(_as_float(base[field]))
            continue
        if field.startswith("Yg0_"):
            name = field[len("Yg0_") :]
            val = math.nan
            if name in gas_names:
                idx = gas_names.index(name)
                if idx < Yg0.shape[0]:
                    val = float(Yg0[idx])
            row.append(val)
            continue
        if field.startswith("Yg_eq_"):
            name = field[len("Yg_eq_") :]
            val = math.nan
            if Yg_eq is not None and name in gas_names:
                idx = gas_names.index(name)
                if idx < Yg_eq.shape[0]:
                    val = float(Yg_eq[idx])
            row.append(val)
            continue
        if diag is not None and hasattr(diag, field):
            val = getattr(diag, field)
            if np.isscalar(val):
                row.append(_as_float(val))
                continue
        if hasattr(state, field):
            val = getattr(state, field)
            if np.isscalar(val):
                row.append(_as_float(val))
                continue
        if unknown_fields is not None and field not in unknown_fields:
            logger.warning("Unknown scalar field '%s'; writing NaN.", field)
            unknown_fields.add(field)
        row.append(math.nan)
    return row


class ScalarsWriter:
    def __init__(
        self,
        cfg: CaseConfig,
        state0: State | None = None,
        props0=None,
        *,
        out_dir: Path | str | None = None,
    ) -> None:
        self.cfg = cfg
        self.fields = _scalar_fields(cfg)
        self.scalars_every = int(getattr(cfg.io, "scalars_write_every", getattr(cfg.io, "write_every", 1)) or 0)
        self.enabled = self.scalars_every > 0
        self.out_path = _scalar_output_path(cfg, out_dir=out_dir)
        self._fh = None
        self._writer = None
        self._unknown_fields: set[str] = set()
        self._integ = {"m_evap_cum": 0.0, "m_dot_prev": None, "m_drop0": math.nan}
        if state0 is not None and props0 is not None:
            self._init_mass_integrator(state0, props0)
        if self.enabled:
            self._open()

    def _open(self) -> None:
        _ensure_parent(self.out_path)
        self._fh = self.out_path.open("w", newline="")
        self._writer = csv.writer(self._fh)
        if self.fields:
            self._writer.writerow(self.fields)
        try:
            self._fh.flush()
        except Exception:
            pass

    def _init_mass_integrator(self, state0: State, props0) -> None:
        try:
            rho_l_mean = float(np.mean(props0.rho_l)) if hasattr(props0, "rho_l") else math.nan
            Rd0 = float(state0.Rd)
            if np.isfinite(rho_l_mean) and Rd0 > 0.0:
                self._integ["m_drop0"] = 4.0 / 3.0 * math.pi * rho_l_mean * Rd0**3
        except Exception:
            pass

    def maybe_write(
        self,
        *,
        step_id: int,
        grid: Grid1D,
        state: State,
        props,
        diag: "StepDiagnostics",
        t: float | None = None,
        eq_model=None,
    ) -> None:
        if not self.enabled:
            return
        if step_id % self.scalars_every != 0:
            return
        if self._writer is None or self._fh is None:
            self._open()
        if not self.fields:
            return
        row = _compute_scalar_row(
            self.fields,
            cfg=self.cfg,
            grid=grid,
            state=state,
            props=props,
            diag=diag,
            step_id=step_id,
            t=t,
            integ=self._integ,
            eq_model=eq_model,
            unknown_fields=self._unknown_fields,
        )
        self._writer.writerow(row)
        try:
            self._fh.flush()
        except Exception:
            pass

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None
                self._writer = None


def write_step_scalars(cfg: CaseConfig, t: float, state: State, diag: "StepDiagnostics") -> None:
    fields = _scalar_fields(cfg)
    if not fields:
        return
    out_path = _scalar_output_path(cfg, out_dir=None)
    _ensure_parent(out_path)

    row = _compute_scalar_row(
        fields,
        cfg=cfg,
        grid=None,
        state=state,
        props=None,
        diag=diag,
        step_id=None,
        t=t,
        integ=None,
        eq_model=None,
        unknown_fields=None,
    )

    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header and fields:
            writer.writerow(fields)
        if fields:
            writer.writerow(row)


def write_step_spatial(cfg: CaseConfig, grid: Grid1D, state: State, step_id: int | None = None, t: float | None = None) -> None:
    """
    Write spatial snapshot to ``spatial/snapshot_XXXXXX.npz`` under the case directory.

    - Uses cfg.io.fields.{gas, liquid, interface} as allow-lists.
    - Includes r_c/r_f (if present) and a running r_index for convenience.
    - Adds step_id/t when provided by the caller.
    """
    out_dir = (Path(cfg.paths.case_dir) / "spatial") if hasattr(cfg, "paths") else Path("spatial")
    out_dir.mkdir(parents=True, exist_ok=True)

    counter_path = out_dir / "_spatial_index.txt"
    try:
        idx = int(counter_path.read_text().strip())
    except FileNotFoundError:
        idx = 0
    except ValueError:
        idx = 0
    counter_path.write_text(str(idx + 1))

    out_path = out_dir / f"snapshot_{idx:06d}.npz"

    data = {}

    # Grid metadata (best effort)
    try:
        rc = getattr(grid, "r_c", None)
        rf = getattr(grid, "r_f", None)
        if rc is not None:
            rc_arr = np.asarray(rc)
            data["r_c"] = rc_arr
            data["r_index"] = np.arange(rc_arr.size)
        if rf is not None:
            data["r_f"] = np.asarray(rf)
    except Exception:
        pass

    if step_id is not None:
        data["step_id"] = np.asarray(step_id)
    if t is not None:
        data["t"] = np.asarray(t)

    fields = getattr(cfg, "io", None)
    field_cfg = getattr(fields, "fields", None) if fields is not None else None

    gas_fields = getattr(field_cfg, "gas", []) or []
    for name in gas_fields:
        if not hasattr(state, name):
            continue
        data[name] = np.asarray(getattr(state, name))

    liq_fields = getattr(field_cfg, "liquid", []) or []
    for name in liq_fields:
        if not hasattr(state, name):
            continue
        data[name] = np.asarray(getattr(state, name))

    iface_fields = getattr(field_cfg, "interface", []) or []
    for name in iface_fields:
        if not hasattr(state, name):
            continue
        data[name] = np.asarray(getattr(state, name))

    np.savez(out_path, **data)
