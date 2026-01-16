from __future__ import annotations

from pathlib import Path

import cantera as ct

from core.types import CaseConfig


def preprocess_case(cfg: CaseConfig) -> CaseConfig:
    """
    Load mechanism (Cantera) and fill runtime species lists/mappings.
    """
    if getattr(cfg.species, "gas_species_full", None):
        return cfg

    mech_path = Path(cfg.paths.gas_mech)
    if not mech_path.is_absolute():
        mech_path = Path(cfg.paths.mechanism_dir) / mech_path
    mech_path = mech_path.expanduser().resolve()

    if not mech_path.exists():
        raise FileNotFoundError(f"Mechanism file not found: {mech_path}")

    phase_name = getattr(cfg.species, "gas_mechanism_phase", "gas")
    try:
        gas = ct.Solution(str(mech_path), phase_name)
    except Exception:
        gas = ct.Solution(str(mech_path))

    mech_names = list(gas.species_names)
    mech_map = {nm: i for i, nm in enumerate(mech_names)}

    cfg.species.gas_species_full = mech_names
    cfg.species.gas_name_to_index = mech_map

    cl = cfg.species.gas_balance_species
    if cl and cl not in mech_map:
        raise ValueError(f"gas_balance_species '{cl}' not found in mechanism species.")

    conds = list(getattr(cfg.physics.interface.equilibrium, "condensables_gas", []) or [])
    missing = [s for s in conds if s not in mech_map]
    if missing:
        raise ValueError(f"condensables_gas not found in mechanism species: {missing}")

    cfg.species.liq_name_to_index = {nm: i for i, nm in enumerate(cfg.species.liq_species)}
    return cfg
