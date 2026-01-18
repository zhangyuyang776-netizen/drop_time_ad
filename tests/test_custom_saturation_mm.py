"""
P4: Unit tests for custom saturation model (Millán-Merino + Watson).

Tests:
1. Parameter database loading and alias resolution
2. psat(T) monotonicity (must increase with T)
3. hvap(T) reasonableness (must decrease with T, Watson behavior)
4. Numerical robustness (fail-fast on invalid inputs)
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from properties.liquid_sat_db import load_liquid_sat_db
from properties.saturation_models import (
    LiquidSatParams,
    SaturationModelMM,
    create_saturation_model,
)


# ============================================================================
# Test 1: Parameter database loading and alias resolution
# ============================================================================


def test_load_liquid_sat_db():
    """P4: Load liquid saturation database from YAML."""
    yaml_path = pathlib.Path(__file__).parent.parent / "mechanism" / "liquid_sat_params.yaml"

    db = load_liquid_sat_db(yaml_path)

    # Should have at least n-Dodecane
    assert db.has_species("n-Dodecane")
    assert "n-Dodecane" in db.list_species()


def test_alias_resolution_nc12h26():
    """P4: Alias 'NC12H26' should resolve to 'n-Dodecane'."""
    yaml_path = pathlib.Path(__file__).parent.parent / "mechanism" / "liquid_sat_params.yaml"
    db = load_liquid_sat_db(yaml_path)

    # All these should resolve to same species
    params1 = db.get_params("NC12H26")
    params2 = db.get_params("n-Dodecane")
    params3 = db.get_params("Dodecane")
    params4 = db.get_params("nC12H26")

    assert params1.canonical_name == "n-Dodecane"
    assert params2.canonical_name == "n-Dodecane"
    assert params3.canonical_name == "n-Dodecane"
    assert params4.canonical_name == "n-Dodecane"


def test_alias_case_insensitive():
    """P4: Alias matching should be case-insensitive."""
    yaml_path = pathlib.Path(__file__).parent.parent / "mechanism" / "liquid_sat_params.yaml"
    db = load_liquid_sat_db(yaml_path)

    # Case variations should all work
    params1 = db.get_params("nc12h26")  # lowercase
    params2 = db.get_params("NC12H26")  # uppercase
    params3 = db.get_params("Nc12H26")  # mixed

    assert params1.canonical_name == "n-Dodecane"
    assert params2.canonical_name == "n-Dodecane"
    assert params3.canonical_name == "n-Dodecane"


def test_get_params_nonexistent_species():
    """P4: Getting params for nonexistent species should raise KeyError."""
    yaml_path = pathlib.Path(__file__).parent.parent / "mechanism" / "liquid_sat_params.yaml"
    db = load_liquid_sat_db(yaml_path)

    with pytest.raises(KeyError, match="not found in database"):
        db.get_params("Nonexistent")


# ============================================================================
# Test 2: psat(T) monotonicity
# ============================================================================


def test_psat_monotonicity():
    """
    P4: psat(T) must be monotonically increasing with T.

    Check over temperature range [400K, 480K] (below Tb=489.41K).
    """
    yaml_path = pathlib.Path(__file__).parent.parent / "mechanism" / "liquid_sat_params.yaml"
    db = load_liquid_sat_db(yaml_path)
    model = SaturationModelMM()

    params = db.get_params("NC12H26")

    # Temperature range below boiling point
    T_range = np.linspace(400, 480, 20)
    psat_values = [model.psat(params, T) for T in T_range]

    # Check monotonicity
    for i in range(len(psat_values) - 1):
        assert psat_values[i + 1] > psat_values[i], (
            f"psat not monotonic: psat(T={T_range[i]:.1f})={psat_values[i]:.1e}, "
            f"psat(T={T_range[i+1]:.1f})={psat_values[i+1]:.1e}"
        )


def test_psat_at_tb_equals_patm():
    """
    P4: psat(Tb) should equal patm (within numerical tolerance).

    By definition, saturation pressure at normal boiling point equals 1 atm.
    """
    yaml_path = pathlib.Path(__file__).parent.parent / "mechanism" / "liquid_sat_params.yaml"
    db = load_liquid_sat_db(yaml_path)
    model = SaturationModelMM()

    params = db.get_params("NC12H26")
    Tb = params.Tb_patm
    patm = params.patm

    psat_at_Tb = model.psat(params, Tb)

    # Should match patm within 0.1%
    relative_error = abs(psat_at_Tb - patm) / patm
    assert relative_error < 1e-3, (
        f"psat(Tb={Tb}) = {psat_at_Tb:.1f} Pa, expected patm = {patm:.1f} Pa "
        f"(rel_error = {relative_error:.2e})"
    )


# ============================================================================
# Test 3: hvap(T) reasonableness (Watson behavior)
# ============================================================================


def test_hvap_decreases_with_temperature():
    """
    P4: hvap(T) must decrease with increasing T (Watson correlation property).

    Check over temperature range [400K, 480K].
    """
    yaml_path = pathlib.Path(__file__).parent.parent / "mechanism" / "liquid_sat_params.yaml"
    db = load_liquid_sat_db(yaml_path)
    model = SaturationModelMM()

    params = db.get_params("NC12H26")

    # Temperature range below boiling point
    T_range = np.linspace(400, 480, 20)
    hvap_values = [model.hvap(params, T) for T in T_range]

    # Check monotonic decrease
    for i in range(len(hvap_values) - 1):
        assert hvap_values[i + 1] < hvap_values[i], (
            f"hvap not monotonically decreasing: hvap(T={T_range[i]:.1f})={hvap_values[i]:.1e}, "
            f"hvap(T={T_range[i+1]:.1f})={hvap_values[i+1]:.1e}"
        )


def test_hvap_positive():
    """P4: hvap(T) must be positive for all valid T < Tc."""
    yaml_path = pathlib.Path(__file__).parent.parent / "mechanism" / "liquid_sat_params.yaml"
    db = load_liquid_sat_db(yaml_path)
    model = SaturationModelMM()

    params = db.get_params("NC12H26")

    # Check over wide temperature range
    T_range = np.linspace(300, 600, 30)  # Up to ~90% of Tc
    for T in T_range:
        if T >= params.Tc:
            continue  # Skip T >= Tc (model raises error, which is correct)
        hvap_val = model.hvap(params, T)
        assert hvap_val > 0, f"hvap(T={T:.1f}) = {hvap_val:.1e} <= 0 (must be positive)"


# ============================================================================
# Test 4: Numerical robustness (fail-fast on invalid inputs)
# ============================================================================


def test_psat_invalid_temperature_negative():
    """P4: psat(T<0) must raise ValueError."""
    yaml_path = pathlib.Path(__file__).parent.parent / "mechanism" / "liquid_sat_params.yaml"
    db = load_liquid_sat_db(yaml_path)
    model = SaturationModelMM()

    params = db.get_params("NC12H26")

    with pytest.raises(ValueError, match="Invalid temperature"):
        model.psat(params, T=-10.0)


def test_psat_invalid_temperature_nan():
    """P4: psat(T=nan) must raise ValueError."""
    yaml_path = pathlib.Path(__file__).parent.parent / "mechanism" / "liquid_sat_params.yaml"
    db = load_liquid_sat_db(yaml_path)
    model = SaturationModelMM()

    params = db.get_params("NC12H26")

    with pytest.raises(ValueError, match="Invalid temperature"):
        model.psat(params, T=np.nan)


def test_psat_temperature_above_tc():
    """P4: psat(T >= Tc) must raise ValueError (not defined beyond critical point)."""
    yaml_path = pathlib.Path(__file__).parent.parent / "mechanism" / "liquid_sat_params.yaml"
    db = load_liquid_sat_db(yaml_path)
    model = SaturationModelMM()

    params = db.get_params("NC12H26")
    Tc = params.Tc

    with pytest.raises(ValueError, match="exceeds critical temperature"):
        model.psat(params, T=Tc + 10.0)


def test_hvap_temperature_above_tc():
    """P4: hvap(T >= Tc) must raise ValueError."""
    yaml_path = pathlib.Path(__file__).parent.parent / "mechanism" / "liquid_sat_params.yaml"
    db = load_liquid_sat_db(yaml_path)
    model = SaturationModelMM()

    params = db.get_params("NC12H26")
    Tc = params.Tc

    with pytest.raises(ValueError, match="exceeds critical temperature"):
        model.hvap(params, T=Tc + 10.0)


def test_create_saturation_model_factory():
    """P4: Factory function should create correct model type."""
    model = create_saturation_model("mm_integral_watson")
    assert isinstance(model, SaturationModelMM)


def test_create_saturation_model_unknown():
    """P4: Factory function should raise on unknown model name."""
    with pytest.raises(ValueError, match="Unknown saturation model"):
        create_saturation_model("unknown_model")


# ============================================================================
# Test 5: Parameter validation on construction
# ============================================================================


def test_liquid_sat_params_validation_invalid_W():
    """P4: LiquidSatParams should raise on invalid molar mass."""
    with pytest.raises(ValueError, match="Invalid molar mass"):
        LiquidSatParams(
            canonical_name="Test",
            W=-0.1,  # Invalid: negative
            Tb_patm=400.0,
            patm=101325.0,
            Tc=600.0,
            Pc=1e6,
            Lb_Tb=40000.0,
            watson_n=0.38,
        )


def test_liquid_sat_params_validation_tb_above_tc():
    """P4: LiquidSatParams should raise if Tb >= Tc."""
    with pytest.raises(ValueError, match="must be 0 < Tb < Tc"):
        LiquidSatParams(
            canonical_name="Test",
            W=0.17,
            Tb_patm=700.0,  # Invalid: above Tc
            patm=101325.0,
            Tc=600.0,
            Pc=1e6,
            Lb_Tb=40000.0,
            watson_n=0.38,
        )


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])
