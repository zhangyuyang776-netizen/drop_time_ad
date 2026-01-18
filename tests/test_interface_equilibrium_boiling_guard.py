"""
P2: Unit tests for interface equilibrium boiling guard (STRICT VERSION).

Tests validate:
1. Pure function _apply_boiling_guard correctness (forced trigger)
2. Boiling guard triggers correctly in compute_interface_equilibrium_full
3. Guard does NOT trigger when y_cond is low (contrast test)
4. Global constraint: y_bg_total >= EPS_BG for all scenarios
5. Background species scale proportionally (O2/N2 ratio preserved)
6. No "N2 floor=0" segmentation behavior
7. Sum of mole fractions equals 1.0 and all components non-negative
"""

from __future__ import annotations

import numpy as np
import pytest

from properties.equilibrium import (
    EPS_BG,
    EquilibriumModel,
    _apply_boiling_guard,
    compute_interface_equilibrium_full,
)


def _make_mock_model(
    gas_names: list[str],
    liq_names: list[str],
    idx_cond_l: list[int],
    idx_cond_g: list[int],
    M_g: np.ndarray,
    M_l: np.ndarray,
    Yg_farfield: np.ndarray,
) -> EquilibriumModel:
    """Create a mock EquilibriumModel for testing."""
    Ns_g = len(gas_names)

    # Convert mass to mole fractions for farfield
    Y_far = np.asarray(Yg_farfield, dtype=np.float64).reshape(Ns_g)
    s_far = np.sum(Y_far)
    if s_far > 1e-30:
        Y_far = Y_far / s_far

    denom = np.sum(Y_far / np.maximum(M_g, 1e-30))
    if denom > 0:
        X_far = (Y_far / np.maximum(M_g, 1e-30)) / denom
    else:
        X_far = np.zeros_like(Y_far)

    return EquilibriumModel(
        method="raoult_psat",
        psat_model="clausius",  # Use Clausius to avoid CoolProp dependency
        gas_names=gas_names,
        liq_names=liq_names,
        idx_cond_l=np.array(idx_cond_l, dtype=int),
        idx_cond_g=np.array(idx_cond_g, dtype=int),
        M_g=M_g,
        M_l=M_l,
        Yg_farfield=Y_far,
        Xg_farfield=X_far,
        cp_backend="HEOS",
        cp_fluids=liq_names,
        T_ref=298.15,
        psat_ref={"NC12H26": 100.0},  # Mock reference psat
    )


# ============================================================================
# P2-T1: Pure function tests (FORCED TRIGGER, STRONG ASSERTIONS)
# ============================================================================


def test_pure_guard_triggered_exact_boundary():
    """
    Test _apply_boiling_guard: Exactly at boundary (y_cond_sum = 1 - EPS_BG).

    Strong assertion: ys_cap_hit=False, y_bg_total=EPS_BG
    """
    y_cond_in = np.array([1.0 - EPS_BG])
    y_cond_out, ys_cap_hit, y_bg_total = _apply_boiling_guard(y_cond_in, EPS_BG)

    # At boundary, guard should NOT trigger (strict >)
    assert ys_cap_hit is False, "Guard should not trigger at exact boundary"
    assert abs(float(np.sum(y_cond_out)) - (1.0 - EPS_BG)) < 1e-15
    assert abs(y_bg_total - EPS_BG) < 1e-15


def test_pure_guard_triggered_beyond_boundary():
    """
    Test _apply_boiling_guard: Forced trigger (y_cond_sum > 1 - EPS_BG).

    Strong assertions:
    - ys_cap_hit MUST be True
    - sum(y_cond_new) MUST equal 1 - EPS_BG
    - y_bg_total MUST equal EPS_BG
    """
    # Input: y_cond that would leave y_bg < EPS_BG
    y_cond_in = np.array([0.9999999])  # sum > 1 - EPS_BG

    y_cond_out, ys_cap_hit, y_bg_total = _apply_boiling_guard(y_cond_in, EPS_BG)

    # STRONG ASSERTIONS (no "if ys_cap_hit")
    assert ys_cap_hit is True, "Guard MUST trigger when y_cond_sum > 1-EPS_BG"

    y_cond_sum_out = float(np.sum(y_cond_out))
    assert abs(y_cond_sum_out - (1.0 - EPS_BG)) < 1e-12, \
        f"Expected y_cond_sum={1.0-EPS_BG:.15f}, got {y_cond_sum_out:.15f}"

    assert abs(y_bg_total - EPS_BG) < 1e-12, \
        f"Expected y_bg_total={EPS_BG:.15f}, got {y_bg_total:.15f}"


def test_pure_guard_triggered_multiple_condensables():
    """
    Test _apply_boiling_guard: Multiple condensables, proportional scaling.

    Input: [0.7, 0.3] (sum=1.0 > 1-EPS_BG)
    Expected: Both scaled by same factor to sum=1-EPS_BG, ratio preserved
    """
    y_cond_in = np.array([0.7, 0.3])  # sum=1.0
    original_ratio = 0.7 / 0.3

    y_cond_out, ys_cap_hit, y_bg_total = _apply_boiling_guard(y_cond_in, EPS_BG)

    assert ys_cap_hit is True, "Guard must trigger when sum=1.0"

    y_cond_sum_out = float(np.sum(y_cond_out))
    assert abs(y_cond_sum_out - (1.0 - EPS_BG)) < 1e-12

    # Check ratio preservation
    new_ratio = y_cond_out[0] / y_cond_out[1]
    assert abs(new_ratio - original_ratio) < 1e-10, \
        f"Ratio changed: {original_ratio:.6f} -> {new_ratio:.6f}"


def test_pure_guard_not_triggered():
    """
    Test _apply_boiling_guard: No trigger when y_cond_sum is low.

    Strong assertions:
    - ys_cap_hit MUST be False
    - y_cond unchanged
    - y_bg_total = 1 - sum(y_cond)
    """
    y_cond_in = np.array([0.3, 0.2])  # sum=0.5

    y_cond_out, ys_cap_hit, y_bg_total = _apply_boiling_guard(y_cond_in, EPS_BG)

    assert ys_cap_hit is False, "Guard must NOT trigger when y_cond_sum << 1-EPS_BG"

    np.testing.assert_array_almost_equal(y_cond_out, y_cond_in, decimal=15)

    expected_y_bg = 1.0 - 0.5
    assert abs(y_bg_total - expected_y_bg) < 1e-15, \
        f"Expected y_bg_total={expected_y_bg}, got {y_bg_total}"


# ============================================================================
# P2-T2: Integration tests with compute_interface_equilibrium_full
# ============================================================================


def test_integration_guard_triggered_high_psat():
    """
    Integration test: Guard triggers in compute_interface_equilibrium_full.

    Use high psat_ref to force trigger, then strong assertions.
    """
    gas_names = ["NC12H26", "O2", "N2"]
    liq_names = ["NC12H26"]
    M_g = np.array([170.33, 32.0, 28.0])
    M_l = np.array([170.33])
    Yg_far = np.array([0.0, 0.21, 0.79])

    # Use very high psat_ref to force high y_cond
    model = _make_mock_model(
        gas_names=gas_names,
        liq_names=liq_names,
        idx_cond_l=[0],
        idx_cond_g=[0],
        M_g=M_g,
        M_l=M_l,
        Yg_farfield=Yg_far,
    )
    # Override psat_ref to force trigger
    model.psat_ref["NC12H26"] = 1e6  # Very high to force y_cond ~ 1

    Ts = 450.0
    Pg = 101325.0
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.1, 0.189, 0.711])

    result = compute_interface_equilibrium_full(
        model, Ts=Ts, Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face
    )

    # STRONG ASSERTIONS (no "if ys_cap_hit")
    ys_cap_hit = result.meta["ys_cap_hit"]
    assert ys_cap_hit is True, \
        "Guard must trigger with high psat_ref (got ys_cap_hit=False)"

    y_cond_sum = result.meta["y_cond_sum"]
    y_bg_total = result.meta["y_bg_total"]

    assert abs(y_cond_sum - (1.0 - EPS_BG)) < 1e-10, \
        f"Expected y_cond_sum={1.0-EPS_BG}, got {y_cond_sum}"

    assert abs(y_bg_total - EPS_BG) < 1e-10, \
        f"Expected y_bg_total={EPS_BG}, got {y_bg_total}"

    # Background species must be positive
    y_O2 = result.y_all[1]
    y_N2 = result.y_all[2]
    assert y_O2 > 0, f"O2 mole fraction must be positive, got {y_O2}"
    assert y_N2 > 0, f"N2 mole fraction must be positive, got {y_N2}"


def test_integration_guard_not_triggered_low_psat():
    """
    Contrast test: Guard does NOT trigger with low psat.

    Strong assertions:
    - ys_cap_hit MUST be False
    - y_bg_total > EPS_BG (significantly)
    """
    gas_names = ["NC12H26", "O2", "N2"]
    liq_names = ["NC12H26"]
    M_g = np.array([170.33, 32.0, 28.0])
    M_l = np.array([170.33])
    Yg_far = np.array([0.0, 0.21, 0.79])

    model = _make_mock_model(
        gas_names=gas_names,
        liq_names=liq_names,
        idx_cond_l=[0],
        idx_cond_g=[0],
        M_g=M_g,
        M_l=M_l,
        Yg_farfield=Yg_far,
    )
    # Use low psat_ref
    model.psat_ref["NC12H26"] = 10.0

    Ts = 300.0  # Low temperature
    Pg = 101325.0
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.001, 0.21, 0.789])

    result = compute_interface_equilibrium_full(
        model, Ts=Ts, Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face
    )

    ys_cap_hit = result.meta["ys_cap_hit"]
    y_bg_total = result.meta["y_bg_total"]

    assert ys_cap_hit is False, \
        "Guard must NOT trigger with low psat (got ys_cap_hit=True)"

    assert y_bg_total > EPS_BG + 0.1, \
        f"Expected y_bg_total >> EPS_BG, got {y_bg_total}"


# ============================================================================
# P2-T3: Global constraint tests (y_bg_total >= EPS_BG for all scenarios)
# ============================================================================


def test_global_constraint_y_bg_always_above_eps():
    """
    Global constraint test: y_bg_total >= EPS_BG for all temperature scenarios.

    Sweeps temperature and ALWAYS asserts:
    - y_bg_total >= EPS_BG - tolerance
    - y_cond_sum <= 1 - EPS_BG + tolerance
    """
    gas_names = ["NC12H26", "O2", "N2"]
    liq_names = ["NC12H26"]
    M_g = np.array([170.33, 32.0, 28.0])
    M_l = np.array([170.33])
    Yg_far = np.array([0.0, 0.21, 0.79])

    model = _make_mock_model(
        gas_names=gas_names,
        liq_names=liq_names,
        idx_cond_l=[0],
        idx_cond_g=[0],
        M_g=M_g,
        M_l=M_l,
        Yg_farfield=Yg_far,
    )
    # Use moderate psat_ref
    model.psat_ref["NC12H26"] = 1000.0

    Pg = 101325.0
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.1, 0.189, 0.711])

    # Temperature sweep
    temperatures = [300.0, 350.0, 400.0, 450.0, 500.0]

    for Ts in temperatures:
        result = compute_interface_equilibrium_full(
            model, Ts=Ts, Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face
        )

        y_bg_total = result.meta["y_bg_total"]
        y_cond_sum = result.meta["y_cond_sum"]

        # GLOBAL CONSTRAINT: y_bg_total >= EPS_BG (always, no conditions)
        assert y_bg_total >= EPS_BG - 1e-12, \
            f"At Ts={Ts}K: y_bg_total={y_bg_total:.6e} < EPS_BG={EPS_BG:.6e}"

        # GLOBAL CONSTRAINT: y_cond_sum <= 1 - EPS_BG
        assert y_cond_sum <= 1.0 - EPS_BG + 1e-10, \
            f"At Ts={Ts}K: y_cond_sum={y_cond_sum:.15f} > 1-EPS_BG={1.0-EPS_BG:.15f}"


def test_global_constraint_random_scenarios():
    """
    Global constraint test: Random scenarios, always enforce y_bg >= EPS_BG.

    20 random temperature/composition scenarios.
    """
    gas_names = ["NC12H26", "O2", "N2"]
    liq_names = ["NC12H26"]
    M_g = np.array([170.33, 32.0, 28.0])
    M_l = np.array([170.33])
    Yg_far = np.array([0.0, 0.21, 0.79])

    model = _make_mock_model(
        gas_names=gas_names,
        liq_names=liq_names,
        idx_cond_l=[0],
        idx_cond_g=[0],
        M_g=M_g,
        M_l=M_l,
        Yg_farfield=Yg_far,
    )

    np.random.seed(42)
    Pg = 101325.0

    for i in range(20):
        # Random temperature
        Ts = 300.0 + 200.0 * np.random.rand()

        # Random psat_ref
        model.psat_ref["NC12H26"] = 10.0 + 1e5 * np.random.rand()

        # Random compositions
        Yl_face = np.random.rand(1)
        Yl_face /= np.sum(Yl_face)

        Yg_face = np.random.rand(3)
        Yg_face /= np.sum(Yg_face)

        result = compute_interface_equilibrium_full(
            model, Ts=Ts, Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face
        )

        y_bg_total = result.meta["y_bg_total"]
        y_cond_sum = result.meta["y_cond_sum"]

        # GLOBAL CONSTRAINT (no exceptions)
        assert y_bg_total >= EPS_BG - 1e-12, \
            f"Scenario {i}: y_bg_total={y_bg_total:.6e} < EPS_BG={EPS_BG:.6e}"

        assert y_cond_sum <= 1.0 - EPS_BG + 1e-10, \
            f"Scenario {i}: y_cond_sum={y_cond_sum:.15f} > 1-EPS_BG"


# ============================================================================
# Original tests (kept for completeness)
# ============================================================================


def test_background_proportional_scaling():
    """
    Background species scale proportionally as condensables increase.

    Assert: O2/N2 ratio remains constant (equal to farfield ratio).
    """
    gas_names = ["NC12H26", "O2", "N2"]
    liq_names = ["NC12H26"]
    M_g = np.array([170.33, 32.0, 28.0])
    M_l = np.array([170.33])
    Yg_far = np.array([0.0, 0.21, 0.79])

    model = _make_mock_model(
        gas_names=gas_names,
        liq_names=liq_names,
        idx_cond_l=[0],
        idx_cond_g=[0],
        M_g=M_g,
        M_l=M_l,
        Yg_farfield=Yg_far,
    )

    Pg = 101325.0
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.05, 0.20, 0.75])

    # Two scenarios with different psat
    model.psat_ref["NC12H26"] = 100.0
    result1 = compute_interface_equilibrium_full(
        model, Ts=350.0, Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face
    )

    model.psat_ref["NC12H26"] = 5000.0
    result2 = compute_interface_equilibrium_full(
        model, Ts=400.0, Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face
    )

    # Extract O2, N2 mole fractions
    y_O2_1 = result1.y_all[1]
    y_N2_1 = result1.y_all[2]
    y_O2_2 = result2.y_all[1]
    y_N2_2 = result2.y_all[2]

    # Compute O2/N2 ratios
    ratio1 = y_O2_1 / y_N2_1 if y_N2_1 > 1e-14 else 0.0
    ratio2 = y_O2_2 / y_N2_2 if y_N2_2 > 1e-14 else 0.0

    # Farfield ratio
    X_far = model.Xg_farfield
    ratio_far = X_far[1] / X_far[2] if X_far[2] > 1e-14 else 0.0

    # Assert ratios match farfield
    if y_N2_1 > 1e-14:
        assert abs(ratio1 - ratio_far) < 0.01 * ratio_far, \
            f"Scenario 1: O2/N2={ratio1:.6f}, expected {ratio_far:.6f}"

    if y_N2_2 > 1e-14:
        assert abs(ratio2 - ratio_far) < 0.01 * ratio_far, \
            f"Scenario 2: O2/N2={ratio2:.6f}, expected {ratio_far:.6f}"


def test_no_n2_floor_segmentation():
    """
    N2 does not hit floor=0 while O2 remains nonzero.

    Assert: O2>0 => N2>0 (no segmentation).
    """
    gas_names = ["NC12H26", "O2", "N2"]
    liq_names = ["NC12H26"]
    M_g = np.array([170.33, 32.0, 28.0])
    M_l = np.array([170.33])
    Yg_far = np.array([0.0, 0.21, 0.79])

    model = _make_mock_model(
        gas_names=gas_names,
        liq_names=liq_names,
        idx_cond_l=[0],
        idx_cond_g=[0],
        M_g=M_g,
        M_l=M_l,
        Yg_farfield=Yg_far,
    )

    Pg = 101325.0
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.1, 0.189, 0.711])

    temperatures = [320.0, 350.0, 380.0, 410.0, 440.0]

    for Ts in temperatures:
        model.psat_ref["NC12H26"] = 100.0 * (Ts / 320.0) ** 5  # Increasing psat

        result = compute_interface_equilibrium_full(
            model, Ts=Ts, Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face
        )

        y_O2 = result.y_all[1]
        y_N2 = result.y_all[2]

        # No segmentation: if O2>0, then N2>0
        if y_O2 > 1e-14:
            assert y_N2 > 1e-14, \
                f"N2=0 while O2={y_O2:.6g} at T={Ts}"


def test_sum_equals_one_and_nonnegative():
    """
    Sum of y_all equals 1.0 and all components non-negative.
    """
    gas_names = ["NC12H26", "O2", "N2"]
    liq_names = ["NC12H26"]
    M_g = np.array([170.33, 32.0, 28.0])
    M_l = np.array([170.33])
    Yg_far = np.array([0.0, 0.21, 0.79])

    model = _make_mock_model(
        gas_names=gas_names,
        liq_names=liq_names,
        idx_cond_l=[0],
        idx_cond_g=[0],
        M_g=M_g,
        M_l=M_l,
        Yg_farfield=Yg_far,
    )

    np.random.seed(42)
    Pg = 101325.0

    for _ in range(10):
        Ts = 300.0 + 150.0 * np.random.rand()
        model.psat_ref["NC12H26"] = 10.0 + 1e5 * np.random.rand()

        Yl_face = np.random.rand(1)
        Yl_face /= np.sum(Yl_face)

        Yg_face = np.random.rand(3)
        Yg_face /= np.sum(Yg_face)

        result = compute_interface_equilibrium_full(
            model, Ts=Ts, Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face
        )

        y_sum = float(np.sum(result.y_all))
        y_min = float(np.min(result.y_all))

        assert abs(y_sum - 1.0) < 1e-10, \
            f"y_sum={y_sum} at Ts={Ts:.2f}"
        assert y_min >= -1e-14, \
            f"Negative component: min={y_min} at Ts={Ts:.2f}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
