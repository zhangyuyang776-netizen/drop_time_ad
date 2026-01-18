"""
P4: Integration tests for equilibrium.py using custom saturation model.

Tests:
1. EquilibriumModel can be built with sat_source="custom"
2. compute_interface_equilibrium_full() uses custom psat/hvap
3. Meta dict reports psat_source="custom"
4. Results are physically reasonable (psat finite, y_all valid, etc.)
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from properties.equilibrium import (
    EquilibriumModel,
    build_equilibrium_model,
    compute_interface_equilibrium_full,
)
from properties.liquid_sat_db import load_liquid_sat_db
from properties.saturation_models import create_saturation_model


def _make_minimal_cfg_custom_sat():
    """
    Create minimal CaseConfig with sat_source="custom".

    Gas species: NC12H26, O2, N2
    Liquid species: NC12H26
    """
    yaml_path = pathlib.Path(__file__).parent.parent / "mechanism" / "liquid_sat_params.yaml"

    cfg = SimpleNamespace(
        physics=SimpleNamespace(
            interface=SimpleNamespace(
                equilibrium=SimpleNamespace(
                    method="raoult_psat",
                    psat_model="clausius",  # Not used when sat_source="custom"
                    condensables_gas=[],  # Infer from liq2gas_map
                    coolprop=SimpleNamespace(backend="HEOS", fluids=[]),
                    # P4: Custom saturation config
                    sat_source="custom",
                    custom_sat=SimpleNamespace(
                        model="mm_integral_watson",
                        params_file=str(yaml_path),
                    ),
                )
            )
        ),
        species=SimpleNamespace(
            gas_species_full=["NC12H26", "O2", "N2"],
            liq_species=["NC12H26"],
            liq2gas_map={"NC12H26": "NC12H26"},
            molar_mass={"NC12H26": 170.33, "O2": 32.0, "N2": 28.0},
        ),
        initial=SimpleNamespace(
            P_inf=101325.0,
            Yg={"NC12H26": 0.0, "O2": 0.21, "N2": 0.79},
        ),
    )

    return cfg


# ============================================================================
# Test 1: Build EquilibriumModel with sat_source="custom"
# ============================================================================


def test_build_equilibrium_model_custom_sat():
    """P4: EquilibriumModel should be built successfully with sat_source='custom'."""
    cfg = _make_minimal_cfg_custom_sat()

    Ns_g = 3
    Ns_l = 1
    M_g = np.array([170.33, 32.0, 28.0])
    M_l = np.array([170.33])

    model = build_equilibrium_model(cfg, Ns_g, Ns_l, M_g, M_l)

    # Verify sat_source is set
    assert model.sat_source == "custom"

    # Verify sat_model and sat_db are loaded
    assert model.sat_model is not None
    assert model.sat_db is not None

    # Verify DB contains NC12H26
    assert model.sat_db.has_species("NC12H26")


# ============================================================================
# Test 2: compute_interface_equilibrium_full uses custom psat
# ============================================================================


def test_equilibrium_full_uses_custom_psat():
    """
    P4: compute_interface_equilibrium_full should use custom psat when sat_source='custom'.

    Check meta dict for psat_source="custom".
    """
    cfg = _make_minimal_cfg_custom_sat()

    Ns_g = 3
    Ns_l = 1
    M_g = np.array([170.33, 32.0, 28.0])
    M_l = np.array([170.33])

    model = build_equilibrium_model(cfg, Ns_g, Ns_l, M_g, M_l)

    # Interface conditions (below boiling point)
    Ts = 450.0  # K (below Tb=489.41K)
    Pg = 101325.0  # Pa
    Yl_face = np.array([1.0])  # Pure dodecane liquid
    Yg_face = np.array([0.0, 0.21, 0.79])  # Air (no dodecane vapor yet)

    res = compute_interface_equilibrium_full(model, Ts, Pg, Yl_face, Yg_face)

    # Check meta dict
    assert "psat_source" in res.meta
    assert res.meta["psat_source"] == "custom", (
        f"Expected psat_source='custom', got '{res.meta['psat_source']}'"
    )


# ============================================================================
# Test 3: Results are physically reasonable
# ============================================================================


def test_equilibrium_custom_psat_reasonable_results():
    """
    P4: Results from custom saturation should be physically reasonable.

    Check:
    - psat is finite and positive
    - y_all is valid simplex (sum=1, all non-negative)
    - Yg_eq is finite and normalized
    """
    cfg = _make_minimal_cfg_custom_sat()

    Ns_g = 3
    Ns_l = 1
    M_g = np.array([170.33, 32.0, 28.0])
    M_l = np.array([170.33])

    model = build_equilibrium_model(cfg, Ns_g, Ns_l, M_g, M_l)

    # Interface conditions
    Ts = 450.0  # K
    Pg = 101325.0  # Pa
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.0, 0.21, 0.79])

    res = compute_interface_equilibrium_full(model, Ts, Pg, Yl_face, Yg_face)

    # Check psat
    assert res.psat.shape == (Ns_l,)
    assert np.all(np.isfinite(res.psat)), f"psat has non-finite values: {res.psat}"
    assert np.all(res.psat > 0), f"psat has non-positive values: {res.psat}"

    # Check y_all (mole fractions)
    assert res.y_all.shape == (Ns_g,)
    assert np.all(np.isfinite(res.y_all)), f"y_all has non-finite values: {res.y_all}"
    assert np.all(res.y_all >= -1e-14), f"y_all has negative values: {res.y_all}"
    y_sum = float(np.sum(res.y_all))
    assert abs(y_sum - 1.0) < 1e-10, f"y_all sum = {y_sum}, expected 1.0"

    # Check Yg_eq (mass fractions)
    assert res.Yg_eq.shape == (Ns_g,)
    assert np.all(np.isfinite(res.Yg_eq)), f"Yg_eq has non-finite values: {res.Yg_eq}"
    assert np.all(res.Yg_eq >= -1e-14), f"Yg_eq has negative values: {res.Yg_eq}"
    Y_sum = float(np.sum(res.Yg_eq))
    assert abs(Y_sum - 1.0) < 1e-10, f"Yg_eq sum = {Y_sum}, expected 1.0"


# ============================================================================
# Test 4: Custom psat at different temperatures
# ============================================================================


def test_equilibrium_custom_psat_temperature_sweep():
    """
    P4: Custom psat should work across temperature range.

    Test T = [400, 420, 440, 460, 480] K (all below Tb=489.41K).
    """
    cfg = _make_minimal_cfg_custom_sat()

    Ns_g = 3
    Ns_l = 1
    M_g = np.array([170.33, 32.0, 28.0])
    M_l = np.array([170.33])

    model = build_equilibrium_model(cfg, Ns_g, Ns_l, M_g, M_l)

    Pg = 101325.0
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.0, 0.21, 0.79])

    T_range = [400, 420, 440, 460, 480]
    psat_values = []

    for Ts in T_range:
        res = compute_interface_equilibrium_full(model, Ts, Pg, Yl_face, Yg_face)

        # Check meta
        assert res.meta["psat_source"] == "custom"

        # Check psat is finite
        assert np.all(np.isfinite(res.psat))
        psat_values.append(res.psat[0])  # NC12H26 psat

    # Check psat increases with T (monotonicity)
    for i in range(len(psat_values) - 1):
        assert psat_values[i + 1] > psat_values[i], (
            f"psat not monotonic: psat(T={T_range[i]})={psat_values[i]:.1e}, "
            f"psat(T={T_range[i+1]})={psat_values[i+1]:.1e}"
        )


# ============================================================================
# Test 5: Boiling guard behavior with custom psat
# ============================================================================


def test_equilibrium_custom_psat_near_boiling():
    """
    P4: Custom psat near boiling point should trigger boiling guard.

    At T close to Tb, psat → Pg, y_cond → 1 - EPS_BG, ys_cap_hit should be True.
    """
    cfg = _make_minimal_cfg_custom_sat()

    Ns_g = 3
    Ns_l = 1
    M_g = np.array([170.33, 32.0, 28.0])
    M_l = np.array([170.33])

    model = build_equilibrium_model(cfg, Ns_g, Ns_l, M_g, M_l)

    # Very close to boiling point (but below, to avoid T >= Tb issues)
    Ts = 488.0  # K (Tb = 489.41K)
    Pg = 101325.0
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.0, 0.21, 0.79])

    res = compute_interface_equilibrium_full(model, Ts, Pg, Yl_face, Yg_face)

    # At T~Tb, psat should be close to Pg
    psat_over_P = res.meta["psat_over_P"]
    assert psat_over_P > 0.8, (
        f"Expected psat_over_P > 0.8 near Tb, got {psat_over_P:.3f}"
    )

    # Boiling guard should trigger (ys_cap_hit = True)
    ys_cap_hit = res.meta["ys_cap_hit"]
    assert ys_cap_hit is True, (
        f"Expected ys_cap_hit=True near Tb, got {ys_cap_hit}"
    )

    # y_bg_total should be >= EPS_BG
    y_bg_total = res.meta["y_bg_total"]
    EPS_BG = 1e-5
    assert y_bg_total >= EPS_BG - 1e-10, (
        f"Expected y_bg_total >= EPS_BG={EPS_BG}, got {y_bg_total:.6e}"
    )


# ============================================================================
# Test 6: Fail-fast if liquid species not in DB
# ============================================================================


def test_build_model_fails_if_species_not_in_db():
    """
    P4: build_equilibrium_model should raise if liquid species not in custom DB.

    Create a config with unknown liquid species "UnknownFluid".
    """
    yaml_path = pathlib.Path(__file__).parent.parent / "mechanism" / "liquid_sat_params.yaml"

    cfg = SimpleNamespace(
        physics=SimpleNamespace(
            interface=SimpleNamespace(
                equilibrium=SimpleNamespace(
                    method="raoult_psat",
                    psat_model="clausius",
                    condensables_gas=[],
                    coolprop=SimpleNamespace(backend="HEOS", fluids=[]),
                    sat_source="custom",
                    custom_sat=SimpleNamespace(
                        model="mm_integral_watson",
                        params_file=str(yaml_path),
                    ),
                )
            )
        ),
        species=SimpleNamespace(
            gas_species_full=["UnknownFluid", "O2", "N2"],
            liq_species=["UnknownFluid"],  # Not in DB
            liq2gas_map={"UnknownFluid": "UnknownFluid"},
            molar_mass={"UnknownFluid": 100.0, "O2": 32.0, "N2": 28.0},
        ),
        initial=SimpleNamespace(
            P_inf=101325.0,
            Yg={"UnknownFluid": 0.0, "O2": 0.21, "N2": 0.79},
        ),
    )

    Ns_g = 3
    Ns_l = 1
    M_g = np.array([100.0, 32.0, 28.0])
    M_l = np.array([100.0])

    # Should raise InterfaceEquilibriumError
    from properties.equilibrium import InterfaceEquilibriumError

    with pytest.raises(InterfaceEquilibriumError, match="not found in custom saturation database"):
        build_equilibrium_model(cfg, Ns_g, Ns_l, M_g, M_l)


# ============================================================================
# Test 7: CoolProp decoupling verification (Stage 2)
# ============================================================================


def test_custom_psat_does_not_call_coolprop(monkeypatch):
    """
    P4 Stage 2: Verify custom saturation does NOT call CoolProp.

    Strategy: Monkeypatch CoolProp.PropsSI to raise an exception.
    If sat_source="custom", compute_interface_equilibrium_full() should still work
    because it doesn't call CoolProp for psat calculation.
    """
    # Monkeypatch CoolProp.PropsSI to always raise
    def fake_props_si_that_raises(*args, **kwargs):
        raise RuntimeError(
            "CoolProp.PropsSI was called! This should not happen when sat_source='custom'"
        )

    # Try to patch CoolProp if available
    try:
        import CoolProp.CoolProp as CP
        monkeypatch.setattr(CP, "PropsSI", fake_props_si_that_raises)
    except ImportError:
        # CoolProp not installed, skip this verification
        # (but test should still pass because custom doesn't need it)
        pass

    # Build model with sat_source="custom"
    cfg = _make_minimal_cfg_custom_sat()

    Ns_g = 3
    Ns_l = 1
    M_g = np.array([170.33, 32.0, 28.0])
    M_l = np.array([170.33])

    model = build_equilibrium_model(cfg, Ns_g, Ns_l, M_g, M_l)

    # Compute equilibrium with custom psat
    Ts = 450.0  # K (below Tb)
    Pg = 101325.0  # Pa
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.0, 0.21, 0.79])

    # This should NOT raise "CoolProp.PropsSI was called!" if custom is properly decoupled
    res = compute_interface_equilibrium_full(model, Ts, Pg, Yl_face, Yg_face)

    # Verify it used custom source
    assert res.meta["psat_source"] == "custom"

    # Verify results are reasonable
    assert np.all(np.isfinite(res.psat))
    assert np.all(res.psat > 0)


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])
