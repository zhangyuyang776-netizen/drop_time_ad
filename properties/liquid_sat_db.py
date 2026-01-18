"""
P4: Liquid saturation parameter database loader.

This module handles loading liquid_sat_params.yaml and provides:
- Canonical name → LiquidSatParams mapping
- Alias resolution (NC12H26 → n-Dodecane → params)
- Case-insensitive, whitespace/hyphen-insensitive matching

Usage:
    db = load_liquid_sat_db("mechanism/liquid_sat_params.yaml")
    params = db.get_params("NC12H26")  # Returns LiquidSatParams for n-Dodecane
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from properties.saturation_models import LiquidSatParams


def _normalize_name(name: str) -> str:
    """
    Normalize species name for matching.

    Rules:
    - Convert to lowercase
    - Remove hyphens, underscores, spaces
    - Collapse whitespace

    Examples:
        "n-Dodecane" → "ndodecane"
        "NC12H26" → "nc12h26"
        "C12 H26" → "c12h26"

    Args:
        name: Raw species name

    Returns:
        Normalized name for case-insensitive matching
    """
    normalized = name.lower()
    normalized = re.sub(r"[-_\s]+", "", normalized)
    return normalized


class LiquidSatDB:
    """
    Liquid saturation parameter database.

    Stores canonical species data and provides alias-aware lookup.
    """

    def __init__(self):
        """Initialize empty database."""
        self._params: Dict[str, LiquidSatParams] = {}  # canonical → params
        self._alias_map: Dict[str, str] = {}  # normalized_alias → canonical

    def add_species(
        self,
        canonical_name: str,
        aliases: List[str],
        W: float,
        Tb_patm: float,
        patm: float,
        Tc: float,
        Pc: float,
        Lb_Tb: float,
        watson_n: float,
    ):
        """
        Add a species to the database.

        Args:
            canonical_name: Primary species identifier
            aliases: List of alternative names
            W: Molar mass [kg/mol]
            Tb_patm: Normal boiling point [K]
            patm: Reference pressure [Pa]
            Tc: Critical temperature [K]
            Pc: Critical pressure [Pa]
            Lb_Tb: Latent heat at Tb [J/mol]
            watson_n: Watson exponent

        Raises:
            ValueError: If parameters are invalid or canonical name already exists
        """
        if canonical_name in self._params:
            raise ValueError(f"Duplicate canonical name: {canonical_name}")

        # Create LiquidSatParams (validates on construction)
        params = LiquidSatParams(
            canonical_name=canonical_name,
            W=W,
            Tb_patm=Tb_patm,
            patm=patm,
            Tc=Tc,
            Pc=Pc,
            Lb_Tb=Lb_Tb,
            watson_n=watson_n,
        )

        # Store params
        self._params[canonical_name] = params

        # Register canonical name and all aliases for lookup
        all_names = [canonical_name] + aliases
        for name in all_names:
            normalized = _normalize_name(name)
            if normalized in self._alias_map:
                existing_canonical = self._alias_map[normalized]
                if existing_canonical != canonical_name:
                    raise ValueError(
                        f"Alias collision: '{name}' (normalized: '{normalized}') "
                        f"already maps to '{existing_canonical}', "
                        f"cannot add to '{canonical_name}'"
                    )
            self._alias_map[normalized] = canonical_name

    def get_params(self, name: str) -> LiquidSatParams:
        """
        Get saturation parameters for a species by name (canonical or alias).

        Args:
            name: Species name (canonical or any registered alias)

        Returns:
            LiquidSatParams for the species

        Raises:
            KeyError: If species not found in database
        """
        normalized = _normalize_name(name)
        if normalized not in self._alias_map:
            raise KeyError(
                f"Species '{name}' (normalized: '{normalized}') not found in database. "
                f"Available species: {list(self._params.keys())}"
            )

        canonical = self._alias_map[normalized]
        return self._params[canonical]

    def has_species(self, name: str) -> bool:
        """Check if species exists in database (by name or alias)."""
        normalized = _normalize_name(name)
        return normalized in self._alias_map

    def list_species(self) -> List[str]:
        """Return list of canonical species names in database."""
        return list(self._params.keys())


def load_liquid_sat_db(yaml_path: str | Path) -> LiquidSatDB:
    """
    Load liquid saturation database from YAML file.

    Args:
        yaml_path: Path to liquid_sat_params.yaml

    Returns:
        LiquidSatDB instance

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If YAML structure is invalid or parameters are missing
    """
    import yaml

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Liquid saturation params file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "species" not in data:
        raise ValueError(
            f"Invalid YAML structure in {yaml_path}: expected top-level 'species' key"
        )

    db = LiquidSatDB()

    for canonical_name, spec_data in data["species"].items():
        if not isinstance(spec_data, dict):
            raise ValueError(
                f"Invalid species data for '{canonical_name}': expected dict, got {type(spec_data)}"
            )

        # Extract required fields
        required_fields = ["W", "Tb_patm", "patm", "Tc", "Pc", "Lb_Tb", "watson_n"]
        for field in required_fields:
            if field not in spec_data:
                raise ValueError(
                    f"Missing required field '{field}' for species '{canonical_name}' in {yaml_path}"
                )

        # Extract aliases (optional, defaults to empty list)
        aliases = spec_data.get("aliases", [])
        if not isinstance(aliases, list):
            raise ValueError(
                f"Invalid 'aliases' for '{canonical_name}': expected list, got {type(aliases)}"
            )

        # Add species to database
        db.add_species(
            canonical_name=canonical_name,
            aliases=aliases,
            W=float(spec_data["W"]),
            Tb_patm=float(spec_data["Tb_patm"]),
            patm=float(spec_data["patm"]),
            Tc=float(spec_data["Tc"]),
            Pc=float(spec_data["Pc"]),
            Lb_Tb=float(spec_data["Lb_Tb"]),
            watson_n=float(spec_data["watson_n"]),
        )

    return db
