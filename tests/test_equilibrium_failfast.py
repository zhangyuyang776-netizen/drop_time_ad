"""
P3: Fail-fast behavior tests for equilibrium.py

Purpose:
- Verify P3 fail-fast behavior (no silent fallback/clip)
- Test that invalid inputs raise InterfaceEquilibriumError

Tests:
1. Negative psat must raise InterfaceEquilibriumError
2. Zero farfield background must raise InterfaceEquilibriumError
3. Large negative mole fractions must raise (in _finalize_simplex)
4. Mole fraction sum far from 1.0 must raise (in _finalize_simplex)
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from properties.equilibrium import (
    EPS_BG,
    InterfaceEquilibriumError,
    _apply_condensable_guard,
    _fill_background,
    _finalize_simplex,
    build_equilibrium_model,
    compute_interface_equilibrium_full,
)


def _make_minimal_eq_model(psat_ref: float = 1000.0, Yg_farfield_custom=None):
    """
    Create minimal equilibrium model for fail-fast testing.

    Args:
        psat_ref: Reference saturation pressure [Pa]
        Yg_farfield_custom: Custom farfield gas composition (optional)

    Returns:
        (cfg, eq_model)
    """
    gas_species = ["NC12H26", "O2", "N2"]
    liq_species = ["NC12H26"]

    # Minimal cfg
    cfg = SimpleNamespace(
        physics=SimpleNamespace(
            interface=SimpleNamespace(
                equilibrium=SimpleNamespace(
                    method="raoult_psat",
                    psat_model="clausius",
                    condensables_gas=[],
                    coolprop=SimpleNamespace(backend="HEOS", fluids=[]),
                )
            )
        ),
        species=SimpleNamespace(
            gas_species_full=gas_species,
            liq_species=liq_species,
            liq2gas_map={"NC12H26": "NC12H26"},
            molar_mass={"NC12H26": 170.33, "O2": 32.0, "N2": 28.0},
        ),
        initial=SimpleNamespace(
            P_inf=101325.0,
            Yg={"NC12H26": 0.0, "O2": 0.21, "N2": 0.79}
            if Yg_farfield_custom is None
            else Yg_farfield_custom,
        ),
    )

    # Build equilibrium model
    Ns_g = len(gas_species)
    Ns_l = len(liq_species)
    M_g = np.array([170.33, 32.0, 28.0])
    M_l = np.array([170.33])
    eq_model = build_equilibrium_model(cfg, Ns_g, Ns_l, M_g, M_l)
    eq_model.psat_ref["NC12H26"] = psat_ref

    return cfg, eq_model


# ============================================================================
# Test 1: Negative psat must raise
# ============================================================================


def test_failfast_negative_psat_raises():
    """
    P3: Negative psat must raise InterfaceEquilibriumError.

    Strategy: Force negative psat by setting psat_ref < 0 in Clausius model.
    """
    cfg, eq_model = _make_minimal_eq_model(psat_ref=1000.0)

    # Force negative psat_ref (invalid)
    eq_model.psat_ref["NC12H26"] = -100.0

    Ts = 350.0
    Pg = 101325.0
    Ns_l = 1
    Ns_g = 3
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.0, 0.21, 0.79])

    # P3: Must raise on negative psat (no silent nan_to_num)
    with pytest.raises(InterfaceEquilibriumError, match="Invalid psat"):
        compute_interface_equilibrium_full(eq_model, Ts, Pg, Yl_face, Yg_face)


# ============================================================================
# Test 2: Zero farfield background must raise
# ============================================================================


def test_failfast_zero_farfield_background_raises():
    """
    P3: Zero farfield background sum must raise InterfaceEquilibriumError.

    Strategy: Set farfield Yg to pure condensable (no background species).
    This causes _fill_background to raise because it cannot normalize.
    """
    # Custom farfield: 100% NC12H26, 0% O2, 0% N2
    Yg_farfield_zero_bg = {"NC12H26": 1.0, "O2": 0.0, "N2": 0.0}
    cfg, eq_model = _make_minimal_eq_model(
        psat_ref=1e5,  # High psat to trigger guard
        Yg_farfield_custom=Yg_farfield_zero_bg,
    )

    Ts = 400.0
    Pg = 101325.0
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.0, 0.21, 0.79])

    # P3: Must raise when farfield background sum is zero
    with pytest.raises(InterfaceEquilibriumError, match="Farfield background sum is zero"):
        compute_interface_equilibrium_full(eq_model, Ts, Pg, Yl_face, Yg_face)


# ============================================================================
# Test 3: _finalize_simplex fails on large negative values
# ============================================================================


def test_failfast_finalize_simplex_large_negative():
    """
    P3: _finalize_simplex must raise on large negative mole fractions.

    Light correction allowed (<1e-14), but large negatives must raise.
    """
    # Large negative value
    y_all = np.array([0.5, 0.5, -0.001])  # -0.001 >> tol_neg=1e-14

    with pytest.raises(InterfaceEquilibriumError, match="Large negative mole fraction"):
        _finalize_simplex(y_all, tol_neg=1e-14, tol_sum=1e-10)


# ============================================================================
# Test 4: _finalize_simplex fails on sum far from 1.0
# ============================================================================


def test_failfast_finalize_simplex_sum_far_from_one():
    """
    P3: _finalize_simplex must raise if sum deviates too much from 1.0.

    Light renormalization allowed (<1e-10), but large deviations must raise.
    """
    # Sum = 1.5 >> tol_sum=1e-10
    y_all = np.array([0.5, 0.5, 0.5])

    with pytest.raises(InterfaceEquilibriumError, match="sum far from 1.0"):
        _finalize_simplex(y_all, tol_neg=1e-14, tol_sum=1e-10)


# ============================================================================
# Test 5: _finalize_simplex allows light correction
# ============================================================================


def test_finalize_simplex_light_correction_ok():
    """
    P3: _finalize_simplex allows light numerical correction.

    - Tiny negative values (<1e-14) set to 0
    - Tiny sum deviation (<1e-10) renormalized
    """
    # Tiny negative (should be corrected)
    y_all = np.array([0.3333, 0.3333, 0.3334 - 1e-15])
    y_corrected = _finalize_simplex(y_all, tol_neg=1e-14, tol_sum=1e-10)

    assert np.all(y_corrected >= 0), "All values should be non-negative after correction"
    assert abs(np.sum(y_corrected) - 1.0) < 1e-12, "Sum should be ~1.0 after renormalization"


# ============================================================================
# Test 6: _fill_background raises on zero farfield background
# ============================================================================


def test_fill_background_zero_farfield_raises():
    """
    P3: _fill_background must raise if farfield background sum is zero.
    """
    Ns_g = 3
    y_bg_total = 0.1
    X_farfield_background = np.array([0.9, 0.0, 0.0])  # Only condensable, no background
    idx_bg = np.array([False, True, True])  # O2, N2 are background

    # Sum of farfield background = 0.0 + 0.0 = 0 → must raise
    with pytest.raises(InterfaceEquilibriumError, match="Farfield background sum is zero"):
        _fill_background(y_bg_total, X_farfield_background, idx_bg, Ns_g)


# ============================================================================
# Test 7: _apply_condensable_guard does NOT raise (just scales)
# ============================================================================


def test_condensable_guard_scales_not_raises():
    """
    P3: _apply_condensable_guard scales condensables, does not raise.

    This is a sanity check - the guard function itself should NOT fail-fast,
    it should just enforce the boundary by scaling.
    """
    y_cond = np.array([0.6, 0.6])  # Sum = 1.2 > boundary=0.99999
    boundary = 1.0 - EPS_BG

    # Should scale, not raise
    y_cond_guarded, meta = _apply_condensable_guard(y_cond, boundary)

    assert meta["ys_cap_hit"] is True, "Guard should have triggered"
    y_sum = float(np.sum(y_cond_guarded))
    assert abs(y_sum - boundary) < 1e-12, f"Sum should be scaled to boundary, got {y_sum}"


# ============================================================================
# Test 8: Non-finite inputs must raise (deterministic injection)
# ============================================================================


def test_failfast_nonfinite_temperature():
    """
    P3: Non-finite temperature must raise InterfaceEquilibriumError.

    Strategy: Directly inject np.nan as Ts (deterministic, not relying on overflow).
    """
    cfg, eq_model = _make_minimal_eq_model(psat_ref=1000.0)

    # Directly inject non-finite temperature
    Ts = np.nan
    Pg = 101325.0
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.0, 0.21, 0.79])

    # P3: Must raise on non-finite Ts
    with pytest.raises(InterfaceEquilibriumError, match="Invalid surface temperature"):
        compute_interface_equilibrium_full(eq_model, Ts, Pg, Yl_face, Yg_face)


def test_failfast_nonfinite_pressure():
    """
    P3: Non-finite pressure must raise InterfaceEquilibriumError.

    Strategy: Directly inject np.inf as Pg.
    """
    cfg, eq_model = _make_minimal_eq_model(psat_ref=1000.0)

    Ts = 350.0
    Pg = np.inf  # Non-finite pressure
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.0, 0.21, 0.79])

    # P3: Must raise on non-finite Pg
    with pytest.raises(InterfaceEquilibriumError, match="Invalid gas pressure"):
        compute_interface_equilibrium_full(eq_model, Ts, Pg, Yl_face, Yg_face)


def test_failfast_nonfinite_composition():
    """
    P3: Non-finite composition arrays must raise InterfaceEquilibriumError.

    Strategy: Inject np.nan in Yg_face.
    """
    cfg, eq_model = _make_minimal_eq_model(psat_ref=1000.0)

    Ts = 350.0
    Pg = 101325.0
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.0, np.nan, 0.79])  # Non-finite in composition

    # P3: Must raise on non-finite Yg_face
    with pytest.raises(InterfaceEquilibriumError, match="Non-finite values in Yg_face"):
        compute_interface_equilibrium_full(eq_model, Ts, Pg, Yl_face, Yg_face)


# ============================================================================
# Test 9-10: Static regression tests (prevent hidden clip/cap from coming back)
# ============================================================================


def test_static_no_hidden_cap_in_equilibrium():
    """
    P3: Static check to prevent "0.995" cap from regressing.

    This test scans the source code to ensure no hidden caps remain.
    """
    import pathlib

    eq_file = pathlib.Path(__file__).parent.parent / "properties" / "equilibrium.py"
    content = eq_file.read_text()

    # Check for old 0.995 cap
    assert "0.995" not in content, (
        "Found '0.995' in equilibrium.py - old Pg cap may have regressed! "
        "P3 requires all caps to go through unified constraint functions."
    )


def test_static_no_np_clip_in_equilibrium():
    """
    P3: Static check to prevent np.clip from regressing in critical paths.

    np.clip is forbidden in composition/constraint logic (P3 requires fail-fast).
    """
    import pathlib

    eq_file = pathlib.Path(__file__).parent.parent / "properties" / "equilibrium.py"
    content = eq_file.read_text()

    # Count np.clip occurrences
    clip_count = content.count("np.clip")

    # P3: Should be zero (all hard clips removed)
    assert clip_count == 0, (
        f"Found {clip_count} instances of 'np.clip' in equilibrium.py. "
        "P3 forbids hard clipping - use unified constraint functions with fail-fast instead."
    )


if __name__ == "__main__":
    # Run fail-fast tests with pytest
    pytest.main([__file__, "-v"])
