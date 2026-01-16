"""
Interface boundary conditions with Ts energy jump and Stefan mass flux.

Responsibilities (current version):
- Resolve interface geometry and unknown indices (Ts / mpp / Rd).
- Build fully physical matrix-row definitions:
  * Ts: gas/liquid conduction + gas-side diffusive enthalpy + latent heat jump,
        aligned with the framework R_energy_if.
  * mpp: single-condensable Stefan condition using zero-sum-corrected
        gas diffusive flux at the interface (R_mass_if).
  * Rd: index recorded only; its evolution equation is handled elsewhere
        (e.g. radius_eq.py).
- Record diagnostic info (geometry, indices, equilibrium preview, flux splits).

Direction and sign conventions:
- Radial coordinate r increases outward (from droplet center to far field).
- Interface normal n = +e_r, pointing from liquid to gas.
- mpp and face-based Fourier flux diagnostics are defined positive along +e_r
  ("out of the droplet"); mpp > 0 means evaporation (liquid -> gas).
- For the Ts energy jump, we internally use "heat into the interface" as
  positive when assembling the matrix row; diagnostics expose both this
  convention and the outward (+r) Fourier convention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from core.types import CaseConfig, Grid1D, State, Props
from core.layout import UnknownLayout
from physics.energy_flux import split_energy_flux_cond_diff_single

# Placeholder type for equilibrium results; used in interface rows and diagnostics.
EqResultLike = Optional[Mapping[str, np.ndarray]]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InterfaceRow:
    """Single matrix row definition for an interface unknown.

    Attributes
    ----------
    row : int
        Global row index in the linear system (matches UnknownLayout).
    cols : List[int]
        Global column indices; skeleton stage: can be left empty.
    vals : List[float]
        Coefficients corresponding to cols; skeleton stage: left empty.
    rhs : float
        Right-hand-side contribution.
    """

    row: int
    cols: List[int] = field(default_factory=list)
    vals: List[float] = field(default_factory=list)
    rhs: float = 0.0


@dataclass(slots=True)
class InterfaceCoeffs:
    """Container for all interface-related matrix rows + diagnostics."""

    rows: List[InterfaceRow] = field(default_factory=list)
    diag: Dict[str, Any] = field(default_factory=dict)


def _require_block(layout: UnknownLayout, name: str) -> None:
    """Raise if a required block is missing to catch config/layout mismatch early."""
    if name not in layout.blocks:
        logger.error("Layout missing required block '%s' while cfg requests it.", name)
        raise ValueError(f"Layout missing required block '{name}' while cfg requests it.")


def build_interface_coeffs(
    grid: Grid1D,
    state: State,
    props: Props,
    layout: UnknownLayout,
    cfg: CaseConfig,
    eq_result: EqResultLike = None,
) -> InterfaceCoeffs:
    """
    Build matrix-row definitions for interface unknowns (Ts, mpp, Rd).

    This version:
    - assembles the Ts energy-jump equation using:
      gas/liquid conduction + gas-side diffusive enthalpy (from j_corr)
      + latent heat term (mpp * L_v),
      consistent with the framework R_energy_if.
    - assembles the mpp Stefan equation in the form
         j_corr,b - mpp * DeltaY_eff = 0
      where j_corr,b is the zero-sum-corrected diffusive flux of the
      balance gas species at the interface (R_mass_if).
    - records Rd index (its time evolution is handled elsewhere).
    - does not mutate State or Props.

    Parameters
    ----------
    grid : Grid1D
        Spherical mesh with liquid and gas segments; iface_f defines the interface face.
    state : State
        Current state (Tg, Yg, Tl, Yl, Ts, mpp, Rd); values are read but not
        modified.
    props : Props
        Cell-centered properties (rho, cp, k, D, h_gk, etc.) used to build
        conductive and diffusive fluxes at the interface; not modified.
    layout : UnknownLayout
        Global unknown layout; used to query indices for Ts/mpp/Rd and nearby cells.
    cfg : CaseConfig
        Full case configuration; `cfg.physics.include_*` toggles decide which
        rows exist, and `cfg.species` / `cfg.physics.interface` supply the
        balance-species mapping and DeltaY regularization.
    eq_result : EqResultLike, optional
        Interface-equilibrium result produced by `properties.equilibrium`
        (must provide Yg_eq in full mechanism order) and required when
        `cfg.physics.include_mpp` (and Ts) are enabled.

    Returns
    -------
    InterfaceCoeffs
        Container with:
        - rows: list of InterfaceRow (one per interface scalar unknown),
        - diag: diagnostic dictionary (geometry, indices, flux splits, etc.).
    """
    rows: List[InterfaceRow] = []
    diag: Dict[str, Any] = {}

    # Basic availability checks for mpp-dependent equations
    if cfg.physics.include_mpp and layout.has_block("mpp") and eq_result is None:
        logger.error("Interface mpp equation requires eq_result (with Yg_eq).")
        raise ValueError("eq_result must be provided when cfg.physics.include_mpp is True.")

    # Geometry and indexing near the interface
    iface_f = grid.iface_f
    il_global = grid.Nl - 1  # last liquid cell (global)
    ig_global = grid.Nl      # first gas cell (global)

    il_local = grid.Nl - 1   # Tl local index
    ig_local = 0             # Tg local index

    diag["iface_f"] = iface_f
    diag["cell_liq_global"] = il_global
    diag["cell_gas_global"] = ig_global
    diag["cell_liq_local"] = il_local
    diag["cell_gas_local"] = ig_local
    diag["r_if"] = float(grid.r_f[iface_f])
    diag["A_if"] = float(grid.A_f[iface_f])

    # Which interface unknowns exist
    phys = cfg.physics
    has_Ts = phys.include_Ts and ("Ts" in layout.blocks)
    has_mpp = phys.include_mpp and ("mpp" in layout.blocks)
    has_Rd = phys.include_Rd and ("Rd" in layout.blocks)

    diag["include_Ts"] = has_Ts
    diag["include_mpp"] = has_mpp
    diag["include_Rd"] = has_Rd

    # Defensive: cfg requests a variable but layout misses it
    if phys.include_Ts and not has_Ts:
        _require_block(layout, "Ts")
    if phys.include_mpp and not has_mpp:
        _require_block(layout, "mpp")
    if phys.include_Rd and not has_Rd:
        _require_block(layout, "Rd")

    # Global unknown indices (no manual offsets)
    if has_Ts:
        idx_Ts = layout.idx_Ts()
        diag["idx_Ts"] = idx_Ts

        if not has_mpp:
            logger.error("Ts equation (interface energy jump) requires mpp unknown.")
            raise ValueError("Ts equation requires mpp unknown (cfg.physics.include_mpp).")

        idx_mpp = layout.idx_mpp()
        diag["idx_mpp"] = idx_mpp

        ts_row, ts_diag = _build_Ts_row(
            grid=grid,
            state=state,
            props=props,
            layout=layout,
            cfg=cfg,
            eq_result=eq_result,
            il_global=il_global,
            ig_global=ig_global,
            il_local=il_local,
            ig_local=ig_local,
            iface_f=iface_f,
            idx_Ts=idx_Ts,
            idx_mpp=idx_mpp,
        )
        rows.append(ts_row)
        diag.update(ts_diag)

    if has_mpp:
        if "idx_mpp" not in diag:
            idx_mpp = layout.idx_mpp()
            diag["idx_mpp"] = idx_mpp
        else:
            idx_mpp = diag["idx_mpp"]
        mpp_row, mpp_diag = _build_mpp_row(
            grid=grid,
            state=state,
            props=props,
            layout=layout,
            cfg=cfg,
            eq_result=eq_result,
            il_global=il_global,
            il_local=il_local,
            ig_global=ig_global,
            ig_local=ig_local,
            iface_f=iface_f,
            idx_mpp=idx_mpp,
        )
        rows.append(mpp_row)
        diag.update(mpp_diag)

    if has_Rd:
        idx_Rd = layout.idx_Rd()
        diag["idx_Rd"] = idx_Rd
        # Rd equation row usually built elsewhere (e.g., radius_eq.py); index recorded for diagnostics only.

    # Optional snapshots of nearby state (read-only)
    if state.Tl.size:
        diag["Tl_if"] = float(state.Tl[il_local])
    if state.Tg.size:
        diag["Tg_if"] = float(state.Tg[ig_local])
    if state.Yg.size:
        diag["Yg_if"] = np.array(state.Yg[:, ig_local], copy=True)

    # Equilibrium preview passthrough
    if eq_result is not None:
        diag["equilibrium"] = eq_result

    # Compatibility/legacy diagnostics expected by older tests
    if "Ts_energy" in diag:
        fourier = diag["Ts_energy"].get("fourier_plus_r", {})
        q_g = fourier.get("q_g_plus_r", 0.0)
        q_l = fourier.get("q_l_plus_r", 0.0)
        q_lat = fourier.get("q_lat", 0.0)
        # diffusive enthalpy (species) uses interface sign into balance; compat uses outward sign
        q_diff = diag["Ts_energy"].get("species_diffusion_enthalpy", {}).get("q_diff_species_pow", 0.0)
        diag["Ts_energy"]["q_g"] = q_g
        diag["Ts_energy"]["q_l"] = q_l
        diag["Ts_energy"]["q_lat"] = q_lat
        diag["Ts_energy"]["q_diff"] = q_diff
        diag["Ts_energy"]["balance"] = q_g + q_l - q_lat + q_diff

    if "evaporation" in diag:
        evap = diag["evaporation"]
        k_b_full = evap.get("k_b_full", 0)
        deltaY = float(evap.get("DeltaY_eff", 0.0))
        j_corr_full = np.asarray(evap.get("j_corr_full", np.zeros(1, dtype=float)), dtype=float)
        Yg_eq_full = np.asarray(evap.get("Yg_eq_full", np.zeros_like(j_corr_full)), dtype=float)
        j_sum_val = float(evap.get("j_sum", 0.0))
        j_raw_full = j_corr_full + Yg_eq_full * j_sum_val
        j_cond = float(j_raw_full[k_b_full]) if j_raw_full.size > k_b_full else 0.0
        mpp_val = float(evap.get("mpp_unconstrained", evap.get("mpp_eval", 0.0)))
        residual = j_cond - deltaY * mpp_val
        diag["mpp_mass"] = {
            "J_cond": j_cond,
            "mpp": mpp_val,
            "DeltaY": deltaY,
            "residual": residual,
        }

    # Direction/sign conventions for diagnostics
    diag["direction_convention"] = {
        "n": "+e_r (liquid -> gas)",
        "mpp_positive": "evaporation (liquid -> gas)",
        "flux_positive": (
            "outward along +r for face-based fluxes in transport/assembly; "
            "Ts_energy.balance_into_interface uses 'heat into interface' as positive."
        ),
    }


    coeffs = InterfaceCoeffs(rows=rows, diag=diag)
    logger.debug(
        "interface_bc built: %d rows (Ts=%s, mpp=%s, Rd=%s)",
        len(rows),
        has_Ts,
        has_mpp,
        has_Rd,
    )
    return coeffs


def _build_Ts_row(
    grid: Grid1D,
    state: State,
    props: Props,
    layout: UnknownLayout,
    cfg: CaseConfig,
    eq_result: EqResultLike,
    il_global: int,
    ig_global: int,
    il_local: int,
    ig_local: int,
    iface_f: int,
    idx_Ts: int,
    idx_mpp: int,
) -> Tuple[InterfaceRow, Dict[str, Any]]:
    """
    Energy balance at the interface with "heat into interface" as positive.

    Continuous form (framework R_energy_if, rearranged):

        q_g,in + q_l,in + q_diff,in - q_lat = 0

    where
        q_g,in    = k_g * (Tg1 - Ts) / dr_g * A_if      # gas -> interface (conduction)
        q_l,in    = k_l * (TlN - Ts) / dr_l * A_if      # liquid -> interface (conduction)
        q_diff,in = - A_if * sum_k( h_gk * j_corr,k )   # gas diffusive enthalpy into interface
        q_lat     = mpp * L_v * A_if                   # latent heat to gas (evaporation)

    Discretely we move q_diff,in to the RHS and build:

        A_if * [ (k_g/dr_g) * Tg1
                + (k_l/dr_l) * TlN
                - (k_g/dr_g + k_l/dr_l) * Ts
                - L_v * mpp ] = q_diff_species_pow

    with q_diff_species_pow = A_if * sum_k( h_gk * j_corr,k ), where j_corr is the
    zero-sum-corrected diffusive flux vector at the interface (sum_k j_corr,k = 0).

    """
    A_if = float(grid.A_f[iface_f])
    r_if = float(grid.r_f[iface_f])
    dr_g = float(grid.r_c[ig_global] - r_if)
    dr_l = float(r_if - grid.r_c[il_global])
    if dr_g <= 0.0 or dr_l <= 0.0:
        logger.error("Non-positive interface spacings: dr_g=%g, dr_l=%g", dr_g, dr_l)
        raise ValueError(f"Interface spacings must be positive: dr_g={dr_g}, dr_l={dr_l}")

    # Conductivities: follow Props naming used across gas/liquid modules
    try:
        k_g = float(props.k_g[ig_local])
    except Exception as exc:
        logger.error("Failed to access gas thermal conductivity k_g at cell %d: %s", ig_local, exc)
        raise
    try:
        k_l = float(props.k_l[il_local])
    except Exception as exc:
        logger.error("Failed to access liquid thermal conductivity k_l at cell %d: %s", il_local, exc)
        raise

    L_v = _get_latent_heat(props, cfg)

    # Unknown indices near the interface
    idx_Tg = layout.idx_Tg(ig_local)

    # Energy balance with heat INTO interface as positive:
    # q_g_in + q_l_in + q_diff_in - q_lat = 0
    # => A_if * [(k_g/dr_g) Tg + (k_l/dr_l) Tl - (k_g/dr_g + k_l/dr_l) Ts - L_v mpp] = q_diff_species_pow
    coeff_Tg = A_if * k_g / dr_g
    coeff_Tl = A_if * k_l / dr_l
    coeff_Ts = -A_if * (k_g / dr_g + k_l / dr_l)
    coeff_mpp = -A_if * L_v

    # Gas-side diffusive enthalpy term Σ h_k j_corr_k
    q_diff_species_area = 0.0
    q_diff_species_pow = 0.0
    if (
        eq_result is not None
        and "Yg_eq" in eq_result
        and getattr(props, "h_gk", None) is not None
        and props.h_gk is not None
    ):
        Yg_eq_full = np.asarray(eq_result["Yg_eq"], dtype=np.float64)
        Ns_g = Yg_eq_full.shape[0]
        if state.Yg.shape[0] >= Ns_g:
            try:
                rho_g = float(props.rho_g[ig_local])
                D_full = _get_gas_diffusivity_vec(props, cfg, ig_local=ig_local, Ns_g=Ns_g)
                Yg_cell_full = np.asarray(state.Yg[:, ig_local], dtype=np.float64).reshape(Ns_g)
                alpha = rho_g * D_full / dr_g
                j_raw = -alpha * (Yg_cell_full - Yg_eq_full)
                j_sum = float(np.sum(j_raw))
                j_corr = j_raw - Yg_eq_full * j_sum
                h_gk_vec = np.asarray(props.h_gk[:Ns_g, ig_local], dtype=np.float64).reshape(Ns_g)
                q_diff_species_area = float(np.dot(h_gk_vec, j_corr))
                q_diff_species_pow = q_diff_species_area * A_if
            except Exception:
                q_diff_species_area = 0.0
                q_diff_species_pow = 0.0

    # Check if Tl is in layout (coupled) or fixed (explicit Gauss-Seidel)
    if layout.has_block("Tl"):
        # Fully coupled: Tl is an unknown in this system
        idx_Tl = layout.idx_Tl(il_local)
        cols = [idx_Tg, idx_Ts, idx_Tl, idx_mpp]
        vals = [coeff_Tg, coeff_Ts, coeff_Tl, coeff_mpp]
        rhs = q_diff_species_pow
    else:
        # Gauss-Seidel split: Tl fixed at old value
        Tl_fixed = float(state.Tl[il_local]) if state.Tl.size > il_local else 0.0
        cols = [idx_Tg, idx_Ts, idx_mpp]
        vals = [coeff_Tg, coeff_Ts, coeff_mpp]
        const_Tl = coeff_Tl * Tl_fixed
        rhs = const_Tl + q_diff_species_pow

    # Diagnostic preview using current state (not mutating state)
    Ts_cur = float(state.Ts)
    Tg_cur = float(state.Tg[ig_local]) if state.Tg.size else np.nan
    Tl_cur = float(state.Tl[il_local]) if state.Tl.size else np.nan
    mpp_cur = float(state.mpp)

    # 1) Heat INTO the interface (positive), consistent with Ts energy equation
    q_g_in = k_g * (Tg_cur - Ts_cur) / dr_g * A_if      # gas -> interface
    q_l_in = k_l * (Tl_cur - Ts_cur) / dr_l * A_if      # liquid -> interface
    q_lat = mpp_cur * L_v * A_if                        # interface -> gas (latent)
    q_diff_in = -q_diff_species_pow  # into interface >0
    balance_eq = q_g_in + q_l_in + q_diff_in - q_lat

    # 2) Fourier-style fluxes positive along +r (for transport/assembly consistency)
    q_g_plus_r = -k_g * (Tg_cur - Ts_cur) / dr_g * A_if
    q_l_plus_r = -k_l * (Ts_cur - Tl_cur) / dr_l * A_if
    balance_plus_r = q_g_plus_r + q_l_plus_r - q_lat


    # Enthalpy-based split diagnostics (no matrix impact)
    enthalpy_diag = None
    if hasattr(props, "h_g") and props.h_g is not None and hasattr(props, "h_l") and props.h_l is not None:
        h_g_if = float(props.h_g[ig_local])
        h_l_if = float(props.h_l[il_local])

        dTdr_g_if = (Tg_cur - Ts_cur) / dr_g
        dTdr_l_if = (Ts_cur - Tl_cur) / dr_l
        J_if = mpp_cur  # outward (+r) positive

        q_tot_g_area, q_cond_g_area, q_diff_g_area = split_energy_flux_cond_diff_single(
            cfg, k_g, dTdr_g_if, h_g_if, J_if
        )
        q_tot_l_area, q_cond_l_area, q_diff_l_area = split_energy_flux_cond_diff_single(
            cfg, k_l, dTdr_l_if, h_l_if, J_if
        )

        q_cond_g_pow = q_cond_g_area * A_if
        q_cond_l_pow = q_cond_l_area * A_if
        q_diff_g_pow = q_diff_g_area * A_if
        q_diff_l_pow = q_diff_l_area * A_if
        q_tot_g_pow = q_tot_g_area * A_if
        q_tot_l_pow = q_tot_l_area * A_if

        Lv_eff = h_g_if - h_l_if
        q_lat_eff_pow = q_diff_g_pow - q_diff_l_pow
        latent_mismatch_pow = q_lat - q_lat_eff_pow

        enthalpy_diag = {
            "h_g_if": h_g_if,
            "h_l_if": h_l_if,
            "Lv_eff": Lv_eff,
            "q_cond_g_area": q_cond_g_area,
            "q_cond_l_area": q_cond_l_area,
            "q_diff_g_area": q_diff_g_area,
            "q_diff_l_area": q_diff_l_area,
            "q_total_g_area": q_tot_g_area,
            "q_total_l_area": q_tot_l_area,
            "q_cond_g_pow": q_cond_g_pow,
            "q_cond_l_pow": q_cond_l_pow,
            "q_diff_g_pow": q_diff_g_pow,
            "q_diff_l_pow": q_diff_l_pow,
            "q_total_g_pow": q_tot_g_pow,
            "q_total_l_pow": q_tot_l_pow,
            "q_lat_old_pow": q_lat,
            "q_lat_eff_pow": q_lat_eff_pow,
            "latent_mismatch_pow": latent_mismatch_pow,
            "balance_old_pow": balance_plus_r,
            "balance_eff_pow": q_g_plus_r + q_l_plus_r - q_lat_eff_pow,
            "units": {"area": "W/m^2", "pow": "W"},
        }

    diag_update: Dict[str, Any] = {
        "Ts_energy": {
            "A_if": A_if,
            "dr_g": dr_g,
            "dr_l": dr_l,
            "k_g": k_g,
            "k_l": k_l,
            "L_v": L_v,
            "coeffs": {
                "Tg": coeff_Tg,
                "Ts": coeff_Ts,
                "Tl": coeff_Tl,
                "mpp": coeff_mpp,
            },
            "state": {
                "Tg_if": Tg_cur,
                "Ts": Ts_cur,
                "Tl_if": Tl_cur,
                "mpp": mpp_cur,
            },
            # 1) 与 Ts 方程直接对应的“流入界面为正”能量平衡
            "balance_into_interface": {
                "q_g_in": q_g_in,
                "q_l_in": q_l_in,
                "q_lat": q_lat,
                "q_diff_in": q_diff_in,
                "balance_eq": balance_eq,
                "sign_convention": (
                    "q_g_in/q_l_in/q_diff_in > 0 => heat into interface; "
                    "q_lat > 0 => latent heat leaving interface (evaporation)"
                ),
            },
            # 2) Fourier 版，沿 +r 为正，方便和其它模块 cross-check
            "fourier_plus_r": {
                "q_g_plus_r": q_g_plus_r,
                "q_l_plus_r": q_l_plus_r,
                "q_lat": q_lat,
                "balance_plus_r": balance_plus_r,
                "sign_convention": "flux > 0 => outward along +r",
            },
            # 3) enthalpy 分解，仍然用 Fourier 方向（跟 energy_flux 模块一致）
            #    如果缺少 h_g/h_l，则为 None，避免强制要求 props.h_g/props.h_l
            "enthalpy_split": enthalpy_diag,
            "species_diffusion_enthalpy": {
                "q_diff_species_area": q_diff_species_area,
                "q_diff_species_pow": q_diff_species_pow,
            },
        }
    }

    return InterfaceRow(row=idx_Ts, cols=cols, vals=vals, rhs=rhs), diag_update




def _build_mpp_row(
    grid: Grid1D,
    state: State,
    props: Props,
    layout: UnknownLayout,
    cfg: CaseConfig,
    eq_result: EqResultLike,
    il_global: int,
    il_local: int,
    ig_global: int,
    ig_local: int,
    iface_f: int,
    idx_mpp: int,
) -> Tuple[InterfaceRow, Dict[str, Any]]:
    """
    Numeric-Jacobian-friendly mpp row.

    Discrete Stefan condition for the balance species b:

        R_mass_if,b = j_corr,b - mpp * DeltaY_eff = 0

    where j_corr,b is the zero-sum-corrected diffusive flux of species b
    at the interface (sum_k j_corr,k = 0), and

        DeltaY_eff approx Y_l,b,s - Y_g,b,eq

    with a small regularization to avoid division by ~0.

    Only the analytic dependence on mpp is kept in the Jacobian; j_corr and
    Yg_eq are frozen per residual build (good for numeric Jacobian).
    """
    if eq_result is None or "Yg_eq" not in eq_result:
        logger.error("mpp equation requires eq_result with 'Yg_eq'.")
        raise ValueError("eq_result with 'Yg_eq' is required for mpp equation.")
    Yg_eq_full = np.asarray(eq_result["Yg_eq"], dtype=np.float64)
    Ns_g = Yg_eq_full.shape[0]
    if Ns_g == 0:
        raise ValueError("eq_result['Yg_eq'] is empty; cannot build mpp row.")

    A_if = float(grid.A_f[iface_f])  # diagnostic only
    r_if = float(grid.r_f[iface_f])
    dr_g = float(grid.r_c[ig_global] - r_if)
    if dr_g <= 0.0:
        logger.error("Non-positive gas-side spacing at interface: dr_g=%g", dr_g)
        raise ValueError(f"Gas-side spacing must be positive: dr_g={dr_g}")

    rho_g = float(props.rho_g[ig_local])
    bal = _get_balance_species_indices(cfg, layout)
    k_b_full = bal["k_g_full"]
    k_b_red = bal["k_g_red"]
    g_name = bal["g_name"]
    l_name = bal["l_name"]

    if Yg_eq_full.shape[0] <= k_b_full:
        raise ValueError(f"eq_result['Yg_eq'] too short for species index {k_b_full}")
    if state.Yg.shape[0] < Ns_g:
        raise ValueError(f"state.Yg has {state.Yg.shape[0]} species, expected at least {Ns_g}")

    D_full = _get_gas_diffusivity_vec(props, cfg, ig_local=ig_local, Ns_g=Ns_g)
    alpha = rho_g * D_full / dr_g

    Yg_cell_full = np.asarray(state.Yg[:, ig_local], dtype=np.float64).reshape(Ns_g)
    Yl_b = float(state.Yl[bal["k_l_full"], il_local])
    Yg_eq_b = float(Yg_eq_full[k_b_full])

    # 1) raw and corrected diffusive fluxes (per-area, +r outward)
    j_raw = -alpha * (Yg_cell_full - Yg_eq_full)
    j_sum = float(np.sum(j_raw))
    j_corr = j_raw - Yg_eq_full * j_sum

    # 2) effective DeltaY with soft regularization to avoid hard zeroing
    delta_Y_raw = Yl_b - Yg_eq_b
    iface_cfg = getattr(cfg, "physics", None)
    iface_cfg = getattr(iface_cfg, "interface", None)
    deltaY_min = getattr(iface_cfg, "min_deltaY", 1e-6)
    if abs(delta_Y_raw) < deltaY_min:
        sign = 1.0 if delta_Y_raw >= 0.0 else -1.0
        delta_Y_eff = sign * deltaY_min
    else:
        delta_Y_eff = delta_Y_raw

    mpp_unconstrained = float(j_corr[k_b_full] / delta_Y_eff) if delta_Y_eff != 0.0 else 0.0
    mpp_state = float(state.mpp)

    # No-condensation handled as diagnostic only (no clamp here)
    interface_type = getattr(cfg.physics.interface, "type", "no_condensation")
    no_condensation_applied = interface_type == "no_condensation" and mpp_unconstrained < 0.0

    # Residual row: delta_Y_eff * mpp = j_corr_b
    cols: List[int] = [idx_mpp]
    vals: List[float] = [delta_Y_eff]
    rhs = float(j_corr[k_b_full])

    # Diagnostics
    J_full = mpp_unconstrained * Yg_eq_full + j_corr
    J_full_state = mpp_state * Yg_eq_full + j_corr
    k_cl = layout.gas_closure_index

    diag_update: Dict[str, Any] = {
        "evaporation": {
            "balance_liq": l_name,
            "balance_gas": g_name,
            "k_b_full": k_b_full,
            "k_b_red": k_b_red,
            "k_closure_full": k_cl,
            "dr_g": dr_g,
            "rho_g": rho_g,
            "DeltaY_raw": delta_Y_raw,
            "DeltaY_eff": delta_Y_eff,
            "j_sum": j_sum,
            "j_raw_sum": float(j_raw.sum()),
            "j_corr_sum": float(j_corr.sum()),
            "mpp_eval": mpp_unconstrained,
            "mpp_unconstrained": mpp_unconstrained,
            "mpp_state": mpp_state,
            "no_condensation_applied": no_condensation_applied,
            "interface_type": interface_type,
            "sumJ_minus_mpp": float(J_full.sum() - mpp_unconstrained),
            "sumJ_minus_mpp_state": float(J_full_state.sum() - mpp_state),
            "A_if": A_if,
            "Yg_eq_full": Yg_eq_full,
            "j_corr_full": j_corr,
            "J_full": J_full,
            "J_full_state": J_full_state,
        }
    }

    return InterfaceRow(row=idx_mpp, cols=cols, vals=vals, rhs=rhs), diag_update

def _get_latent_heat(props: Props, cfg: CaseConfig) -> float:
    """Resolve latent heat L_v; prefer props if available, else cfg fallback."""
    candidates = (
        getattr(props, "h_vap_if", None),
        getattr(props, "lv", None),
        getattr(props, "latent_heat", None),
    )
    for cand in candidates:
        if cand is None:
            continue
        try:
            return float(cand)
        except Exception:
            continue

    L_v_cfg = getattr(cfg.physics, "latent_heat_default", None)
    if L_v_cfg is not None:
        return float(L_v_cfg)

    raise ValueError("Latent heat L_v not provided in props or cfg.physics.latent_heat_default.")


def _get_balance_species_indices(
    cfg: CaseConfig,
    layout: UnknownLayout,
) -> Dict[str, Any]:
    """
    Return indices/names for the liquid balance species and its gas counterpart.

    Returns dict with:
        l_name, g_name, k_l_full, k_g_full, k_g_red (may be None if closure)
    """
    l_name = cfg.species.liq_balance_species
    g_map = cfg.species.liq2gas_map
    if l_name not in g_map:
        raise ValueError(f"liq_balance_species '{l_name}' not found in liq2gas_map {g_map}")
    g_name = g_map[l_name]

    if l_name not in layout.liq_species_full:
        raise ValueError(f"Liquid balance species '{l_name}' not found in liq_species_full {layout.liq_species_full}")
    k_l_full = layout.liq_species_full.index(l_name)

    if g_name not in layout.gas_species_full:
        raise ValueError(f"Gas balance species '{g_name}' not found in gas_species_full {layout.gas_species_full}")
    k_g_full = layout.gas_species_full.index(g_name)

    k_g_red = layout.gas_full_to_reduced.get(g_name)

    return {
        "l_name": l_name,
        "g_name": g_name,
        "k_l_full": k_l_full,
        "k_g_full": k_g_full,
        "k_g_red": k_g_red,
    }


def _get_gas_diffusivity(
    props: Props,
    cfg: CaseConfig,
    k_full: int,
    ig_local: int,
) -> float:
    """
    Return D_g for given full-species index and gas cell.

    Prefer props.D_g if present; optionally fall back to cfg.physics.default_D_g.
    """
    if props.D_g is not None:
        if props.D_g.shape[0] <= k_full or props.D_g.shape[1] <= ig_local:
            raise ValueError(
                f"D_g shape {props.D_g.shape} too small for species {k_full} or cell {ig_local}"
            )
        D_val = float(props.D_g[k_full, ig_local])
        if D_val <= 0.0:
            raise ValueError(f"Non-positive D_g[{k_full},{ig_local}]={D_val}")
        return D_val

    D_default = getattr(cfg.physics, "default_D_g", None)
    if D_default is not None:
        return float(D_default)

    raise ValueError("No D_g available in props and cfg.physics.default_D_g missing.")


def _get_gas_diffusivity_vec(
    props: Props,
    cfg: CaseConfig,
    ig_local: int,
    Ns_g: int,
) -> np.ndarray:
    """
    Return D_g vector (Ns_g,) at gas cell ig_local; prefer props.D_g.
    """
    if props.D_g is not None:
        if props.D_g.shape[0] < Ns_g or props.D_g.shape[1] <= ig_local:
            raise ValueError(
                f"D_g shape {props.D_g.shape} too small for Ns_g={Ns_g} or cell {ig_local}"
            )
        D_vec = np.asarray(props.D_g[:Ns_g, ig_local], dtype=np.float64)
        if np.any(D_vec <= 0.0):
            raise ValueError(f"D_g contains non-positive entries at cell {ig_local}: {D_vec}")
        return D_vec

    D_default = getattr(cfg.physics, "default_D_g", None)
    if D_default is not None:
        val = float(D_default)
        if val <= 0.0:
            raise ValueError(f"default_D_g must be positive, got {val}")
        return np.full((Ns_g,), val, dtype=np.float64)

    raise ValueError("No D_g available in props and cfg.physics.default_D_g missing.")
