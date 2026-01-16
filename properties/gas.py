"""
Gas-phase property evaluation (MVP):
- Uses Cantera mixture-averaged transport; fixed pressure P_inf from cfg.
- Outputs core props: rho_g, cp_g, k_g, D_g (mix-averaged diffusion).
- Extras (returned separately): h_g (mass enthalpy), h_gk (species enthalpy).

Future extensions (not implemented here):
- Variable pressure / low-Mach coupling.
- Stefan–Maxwell multicomponent diffusion, Soret/Dufour.
- Face interpolation with consistency constraints.
- Caching TPY calls and parallel loops for performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import logging
import os
import numpy as np

try:
    import cantera as ct
except Exception as e:  # pragma: no cover - environment dependent
    ct = None
    _ct_import_error = e
else:
    _ct_import_error = None

from core.types import CaseConfig, Grid1D, State

FloatArray = np.ndarray

logger = logging.getLogger(__name__)

_SANITIZE_Y = bool(int(os.environ.get("DROPLET_SANITIZE_Y", "1")))
_SANITIZE_Y_ONCE = bool(int(os.environ.get("DROPLET_SANITIZE_Y_ONCE", "1")))
_SANITIZE_PRINTED: set[tuple[int, int]] = set()
_CURRENT_GAS_PHASE: Optional[str] = None


def set_gas_eval_phase(phase: Optional[str]) -> None:
    """Set the current gas evaluation phase for debug logging."""
    global _CURRENT_GAS_PHASE
    _CURRENT_GAS_PHASE = phase


def get_gas_eval_phase() -> Optional[str]:
    """Get the current gas evaluation phase for debug logging."""
    return _CURRENT_GAS_PHASE


def _get_mpi_rank() -> int:
    try:
        from mpi4py import MPI  # type: ignore

        return int(MPI.COMM_WORLD.Get_rank())
    except Exception:
        return -1


def _sanitize_mass_fractions(Y: np.ndarray, eps: float = 1.0e-30) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float64)
    Y = np.maximum(Y, 0.0)
    s = float(np.sum(Y))
    if s <= eps or not np.isfinite(s):
        j = int(np.argmax(Y)) if np.any(Y > 0.0) else (Y.size - 1)
        Y[:] = 0.0
        Y[j] = 1.0
        return Y
    Y /= s
    return Y


@dataclass(slots=True)
class GasPropertiesModel:
    gas: "ct.Solution"
    P_ref: float
    gas_names: Tuple[str, ...]
    name_to_idx: Dict[str, int]


def build_gas_model(cfg: CaseConfig) -> GasPropertiesModel:
    """Construct gas property model; requires Cantera solution from gas_mech path."""
    if ct is None:
        raise ImportError(f"Cantera is required for gas properties: {_ct_import_error}")

    mech_path = cfg.paths.gas_mech
    if cfg.paths.mechanism_dir is not None:
        mech_path = cfg.paths.mechanism_dir / cfg.paths.gas_mech
    phase = getattr(cfg.species, "gas_mechanism_phase", None)
    gas = ct.Solution(str(mech_path), phase) if phase else ct.Solution(str(mech_path))
    gas_names = tuple(gas.species_names)
    name_to_idx = {name: i for i, name in enumerate(gas_names)}
    return GasPropertiesModel(
        gas=gas,
        P_ref=float(cfg.initial.P_inf),
        gas_names=gas_names,
        name_to_idx=name_to_idx,
    )


def compute_gas_props(
    model: GasPropertiesModel, state: State, grid: Grid1D
) -> Tuple[Dict[str, FloatArray], Dict[str, FloatArray]]:
    """
    Compute gas mixture properties; returns (core_props, extras).

    Contract:
    - state.Yg species count must equal mechanism species count and be in the same order.
    - state.Yg must already be full-length and normalized; no mapping or renormalization done here.
    - core_props["D_g"] has shape (Ns_g, Ng): mechanism-order mixture-averaged mass diffusivities [m^2/s],
      aligned with state.Yg.
    """
    Ng = grid.Ng
    Ns_state = state.Yg.shape[0]
    Ns_mech = model.gas.n_species
    if Ns_state != Ns_mech:
        raise ValueError(
            f"state.Yg has {Ns_state} species, but mechanism has {Ns_mech}. "
            "Provide full mechanism-length Yg."
        )
    rho = np.zeros(Ng, dtype=np.float64)
    cp = np.zeros(Ng, dtype=np.float64)
    k = np.zeros(Ng, dtype=np.float64)
    D = np.zeros((Ns_state, Ng), dtype=np.float64)
    h_mix = np.zeros(Ng, dtype=np.float64)
    h_species = np.zeros((Ns_state, Ng), dtype=np.float64)

    gas = model.gas
    P = model.P_ref

    for ig in range(Ng):
        T = float(state.Tg[ig])
        Y_mech = np.asarray(state.Yg[:, ig], dtype=np.float64)

        sY = float(np.sum(Y_mech))
        minY = float(np.min(Y_mech)) if Y_mech.size else 0.0
        maxY = float(np.max(Y_mech)) if Y_mech.size else 0.0
        bad_sum = (not np.isfinite(sY)) or sY <= 0.0 or abs(sY - 1.0) > 1.0e-8
        bad_min = minY < -1.0e-12
        if _SANITIZE_Y and (bad_sum or bad_min):
            rank = _get_mpi_rank()
            key = (rank, int(ig))
            if (not _SANITIZE_Y_ONCE) or (key not in _SANITIZE_PRINTED):
                _SANITIZE_PRINTED.add(key)
                phase = get_gas_eval_phase()
                y_for_sort = np.where(np.isfinite(Y_mech), Y_mech, -np.inf)
                top_idx = np.argsort(-y_for_sort)[:3]
                top = [(int(i), float(Y_mech[i])) for i in top_idx]
                logger.warning(
                    "[SANITIZE_Y][rank=%s][cell=%d][phase=%s] sum=%.6g min=%.3e max=%.3e top=%s",
                    rank,
                    ig,
                    phase,
                    sY,
                    minY,
                    maxY,
                    top,
                )
            Y_mech = _sanitize_mass_fractions(Y_mech)
            sY = float(np.sum(Y_mech))
            minY = float(np.min(Y_mech)) if Y_mech.size else 0.0
            maxY = float(np.max(Y_mech)) if Y_mech.size else 0.0

        if not np.isfinite(sY) or sY <= 0.0:
            raise ValueError(f"Gas mass fractions at cell {ig} are invalid: sum(Y)={sY}")
        if abs(sY - 1.0) > 1e-6:
            raise ValueError(
                f"Gas mass fractions at cell {ig} are not normalized: sum(Y)={sY}. "
                "state.Yg must be a full, normalized mechanism-length vector."
            )

        gas.TPY = T, P, Y_mech

        rho_i = float(gas.density)
        cp_i = float(gas.cp_mass)
        k_i = float(gas.thermal_conductivity)
        # mixture-averaged mass diffusivities [m^2/s], mechanism order
        if hasattr(gas, "mix_diff_coeffs"):
            D_raw = gas.mix_diff_coeffs  # common Cantera interface
        elif hasattr(gas, "mix_diff_coeffs_mass"):
            D_raw = gas.mix_diff_coeffs_mass  # fallback name in some versions
        else:
            raise AttributeError(
                "Gas object has no mix_diff_coeffs or mix_diff_coeffs_mass; cannot build D_g."
            )
        D_mech = np.asarray(D_raw, dtype=np.float64)
        h_mix_i = float(gas.enthalpy_mass)
        h_pm = np.asarray(gas.partial_molar_enthalpies, dtype=np.float64)  # J/kmol
        MW = np.asarray(gas.molecular_weights, dtype=np.float64)  # kg/kmol
        h_species_mech = h_pm / np.maximum(MW, 1e-30)  # J/kg

        # sanity
        if not np.isfinite(rho_i) or rho_i <= 0.0:
            raise ValueError(f"Non-physical gas density at cell {ig}: {rho_i}")
        if not np.isfinite(cp_i) or cp_i <= 0.0:
            raise ValueError(
                "Non-physical gas cp at cell {ig}: cp={cp} T={T} P={P} "
                "sumY={sumY} minY={minY} maxY={maxY}".format(
                    ig=ig,
                    cp=cp_i,
                    T=T,
                    P=P,
                    sumY=sY,
                    minY=minY,
                    maxY=maxY,
                )
            )
        if not np.isfinite(k_i) or k_i <= 0.0:
            raise ValueError(f"Non-physical gas k at cell {ig}: {k_i}")
        if D_mech.shape != (Ns_mech,):
            raise ValueError(f"Expected D_mech shape ({Ns_mech},), got {D_mech.shape}")
        if np.any(~np.isfinite(D_mech)) or np.any(D_mech < 0.0):
            raise ValueError(f"Non-physical gas diffusion coeffs at cell {ig}")
        if np.any(D_mech < 1e-20):
            bad = np.where(D_mech < 1e-20)[0]
            bad_names = [gas.species_names[int(i)] for i in bad[:8]]
            raise ValueError(
                f"Gas diffusion coeffs too small at cell {ig}: "
                f"min(D)={float(np.min(D_mech)):.3e}, bad_species={bad_names}, total_bad={int(bad.size)}"
            )
        if np.any(~np.isfinite(h_species_mech)):
            raise ValueError(f"Non-physical gas species enthalpy at cell {ig}")

        rho[ig] = rho_i
        cp[ig] = cp_i
        k[ig] = k_i
        # D_g: (Ns, Ng), column ig stores mechanism-order mixture-averaged D at cell ig
        D[:, ig] = D_mech
        h_mix[ig] = h_mix_i
        h_species[:, ig] = h_species_mech

    core = {"rho_g": rho, "cp_g": cp, "k_g": k, "D_g": D}
    extra = {"h_g": h_mix, "h_gk": h_species}
    return core, extra
