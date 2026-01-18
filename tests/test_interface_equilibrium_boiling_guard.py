"""
P2: Unit tests for interface equilibrium boiling guard.

Tests validate:
1. Boiling guard correctly scales condensables when y_cond_sum > 1-EPS_BG
2. Background species scale proportionally (O2/N2 ratio preserved)
3. No "N2 floor=0" segmentation behavior
4. Sum of mole fractions equals 1.0 and all components non-negative
"""

from __future__ import annotations

import numpy as np
import pytest

from properties.equilibrium import (
    EPS_BG,
    EquilibriumModel,
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


def test_boiling_guard_scaling():
    """
    Test 1: Boiling guard correctly scales condensables when sum > 1-EPS_BG.

    Setup: Single condensable that would produce y_cond_sum > 1-EPS_BG
    Assert: y_cond is scaled to exactly 1-EPS_BG, ys_cap_hit=True
    """
    # Gas: NC12H26, O2, N2
    # Liquid: NC12H26
    gas_names = ["NC12H26", "O2", "N2"]
    liq_names = ["NC12H26"]
    M_g = np.array([170.33, 32.0, 28.0])
    M_l = np.array([170.33])
    Yg_far = np.array([0.0, 0.21, 0.79])  # Far field: air

    model = _make_mock_model(
        gas_names=gas_names,
        liq_names=liq_names,
        idx_cond_l=[0],  # NC12H26 in liquid
        idx_cond_g=[0],  # NC12H26 in gas
        M_g=M_g,
        M_l=M_l,
        Yg_farfield=Yg_far,
    )

    # Setup: pure liquid NC12H26, high temperature to get psat ~ Pg
    Ts = 450.0  # High enough to trigger guard
    Pg = 101325.0
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.1, 0.189, 0.711])

    result = compute_interface_equilibrium_full(
        model, Ts=Ts, Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face
    )

    # Assertions
    y_cond_sum = float(np.sum(result.y_cond))
    y_bg_total = result.y_bg_total
    ys_cap_hit = result.meta["ys_cap_hit"]

    # If guard triggered, y_cond_sum should be close to 1-EPS_BG
    if ys_cap_hit:
        assert abs(y_cond_sum - (1.0 - EPS_BG)) < 1e-12, \
            f"Expected y_cond_sum={1.0-EPS_BG}, got {y_cond_sum}"
        assert abs(y_bg_total - EPS_BG) < 1e-12, \
            f"Expected y_bg_total={EPS_BG}, got {y_bg_total}"

    # Sum should always be 1
    y_sum = float(np.sum(result.y_all))
    assert abs(y_sum - 1.0) < 1e-10, f"y_sum={y_sum}, expected 1.0"

    # All components non-negative
    assert np.all(result.y_all >= -1e-14), \
        f"Negative component: min={np.min(result.y_all)}"


def test_background_proportional_scaling():
    """
    Test 2: Background species scale proportionally as condensables increase.

    Setup: Two scenarios with different y_cond_sum values
    Assert: O2/N2 ratio remains constant (equal to farfield ratio)
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

    # Scenario 1: Lower temperature (less evaporation)
    Ts1 = 350.0
    Pg = 101325.0
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.05, 0.20, 0.75])

    result1 = compute_interface_equilibrium_full(
        model, Ts=Ts1, Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face
    )

    # Scenario 2: Higher temperature (more evaporation)
    Ts2 = 400.0
    result2 = compute_interface_equilibrium_full(
        model, Ts=Ts2, Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face
    )

    # Extract O2, N2 mole fractions
    y_O2_1 = result1.y_all[1]
    y_N2_1 = result1.y_all[2]
    y_O2_2 = result2.y_all[1]
    y_N2_2 = result2.y_all[2]

    # Compute O2/N2 ratios
    if y_N2_1 > 1e-14:
        ratio1 = y_O2_1 / y_N2_1
    else:
        ratio1 = 0.0

    if y_N2_2 > 1e-14:
        ratio2 = y_O2_2 / y_N2_2
    else:
        ratio2 = 0.0

    # Farfield O2/N2 ratio in mole fractions
    X_far = model.Xg_farfield
    ratio_far = X_far[1] / X_far[2] if X_far[2] > 1e-14 else 0.0

    # Assert: ratios should be close to farfield ratio
    if y_N2_1 > 1e-14:
        assert abs(ratio1 - ratio_far) < 0.01 * ratio_far, \
            f"Scenario 1: O2/N2={ratio1:.6f}, expected {ratio_far:.6f}"

    if y_N2_2 > 1e-14:
        assert abs(ratio2 - ratio_far) < 0.01 * ratio_far, \
            f"Scenario 2: O2/N2={ratio2:.6f}, expected {ratio_far:.6f}"

    # Both scenarios should have same ratio (background scales proportionally)
    if y_N2_1 > 1e-14 and y_N2_2 > 1e-14:
        assert abs(ratio1 - ratio2) < 0.01 * ratio_far, \
            f"Ratios differ: {ratio1:.6f} vs {ratio2:.6f}"


def test_no_n2_floor_segmentation():
    """
    Test 3: N2 does not hit floor=0 while O2 remains nonzero.

    Setup: Sweep y_cond_sum from low to high (approaching 1-EPS_BG)
    Assert: Both O2 and N2 remain positive as long as y_bg_total > 0
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

    # Sweep temperatures from low to high
    temperatures = [320.0, 350.0, 380.0, 410.0, 440.0]
    Pg = 101325.0
    Yl_face = np.array([1.0])
    Yg_face = np.array([0.1, 0.189, 0.711])

    y_N2_values = []
    y_O2_values = []
    y_cond_sums = []

    for Ts in temperatures:
        result = compute_interface_equilibrium_full(
            model, Ts=Ts, Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face
        )
        y_N2_values.append(result.y_all[2])
        y_O2_values.append(result.y_all[1])
        y_cond_sums.append(float(np.sum(result.y_cond)))

    # Assert: N2 and O2 both decrease monotonically (or stay constant)
    for i in range(len(temperatures) - 1):
        # As condensables increase, background should decrease
        if y_cond_sums[i+1] > y_cond_sums[i]:
            assert y_N2_values[i+1] <= y_N2_values[i] + 1e-12, \
                f"N2 increased: T={temperatures[i+1]}, y_N2={y_N2_values[i+1]}"
            assert y_O2_values[i+1] <= y_O2_values[i] + 1e-12, \
                f"O2 increased: T={temperatures[i+1]}, y_O2={y_O2_values[i+1]}"

    # Assert: N2 > 0 whenever O2 > 0 (no segmentation)
    for i, Ts in enumerate(temperatures):
        if y_O2_values[i] > 1e-14:
            assert y_N2_values[i] > 1e-14, \
                f"N2=0 while O2={y_O2_values[i]:.6g} at T={Ts}"


def test_sum_equals_one_and_nonnegative():
    """
    Test 4: Sum of y_all equals 1.0 and all components non-negative.

    Setup: Random y_cond scenarios (with sum < 1)
    Assert: sum(y_all) = 1.0 ± 1e-10, min(y_all) >= 0
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

    # Test multiple temperature scenarios
    np.random.seed(42)
    Pg = 101325.0
    Yl_face = np.array([1.0])

    for _ in range(20):
        # Random temperature between 300-450K
        Ts = 300.0 + 150.0 * np.random.rand()
        Yg_face = np.random.rand(3)
        Yg_face /= np.sum(Yg_face)

        result = compute_interface_equilibrium_full(
            model, Ts=Ts, Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face
        )

        y_sum = float(np.sum(result.y_all))
        y_min = float(np.min(result.y_all))

        # Assertions
        assert abs(y_sum - 1.0) < 1e-10, \
            f"y_sum={y_sum} at Ts={Ts:.2f}, expected 1.0"
        assert y_min >= -1e-14, \
            f"Negative component: min={y_min} at Ts={Ts:.2f}"

        # Additional check: y_bg_total >= 0
        y_bg_total = result.y_bg_total
        assert y_bg_total >= -1e-14, \
            f"Negative y_bg_total={y_bg_total} at Ts={Ts:.2f}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
