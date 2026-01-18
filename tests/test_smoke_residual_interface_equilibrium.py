"""
P2: Smoke tests for boiling guard in residual assembly path (方案B).

Purpose:
- Verify P2 boiling guard is actually called in residual assembly
- NOT testing numerical correctness (that's for unit tests)
- ONLY testing: equilibrium function called + guard triggered/not-triggered

Tests:
1. Smoke test: residual path calls equilibrium
2. Smoke test: guard triggers with high psat_ref
3. Smoke test: guard does NOT trigger with low psat_ref
4. Smoke test: global constraint y_bg >= EPS_BG enforced
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

# Import test-only helper from residual assembly
from assembly.residual_global import _smoke_call_interface_equilibrium
from properties.equilibrium import EPS_BG, build_equilibrium_model


def _make_minimal_mock_objects(Ts: float = 350.0, psat_ref: float = 1000.0):
    """
    Create minimal mock objects for smoke test.

    Args:
        Ts: Interface temperature [K]
        psat_ref: Reference saturation pressure [Pa] for Clausius model

    Returns:
        (cfg, layout, grid, state_props, eq_model)
    """
    # Gas species: NC12H26, O2, N2
    # Liquid species: NC12H26
    gas_species = ["NC12H26", "O2", "N2"]
    liq_species = ["NC12H26"]

    # Minimal cfg
    cfg = SimpleNamespace(
        physics=SimpleNamespace(include_mpp=True),  # Required for needs_eq
        species=SimpleNamespace(
            gas_species_full=gas_species,
            liq_species=liq_species,
            liq2gas_map={"NC12H26": "NC12H26"},
            molar_mass={
                "NC12H26": 170.33,
                "O2": 32.0,
                "N2": 28.0,
            },
        ),
        initial=SimpleNamespace(
            P_inf=101325.0,
            Yg={"NC12H26": 0.0, "O2": 0.21, "N2": 0.79},
        ),
    )

    # Add equilibrium config (use Clausius to avoid CoolProp dependency)
    cfg.physics.interface = SimpleNamespace(
        equilibrium=SimpleNamespace(
            method="raoult_psat",
            psat_model="clausius",  # Use Clausius for deterministic tests
            condensables_gas=[],  # Auto-infer from liq2gas_map
            coolprop=SimpleNamespace(
                backend="HEOS",
                fluids=[],  # Auto-infer
            ),
        )
    )

    # Minimal layout with mpp block
    layout = SimpleNamespace()
    layout.has_block = lambda name: name == "mpp"  # Required for needs_eq

    # Minimal grid
    Nl = 10
    Ng = 20
    grid = SimpleNamespace(Nl=Nl, Ng=Ng)

    # Minimal state_props (only need Yl, Yg, Ts at interface)
    Ns_l = len(liq_species)
    Ns_g = len(gas_species)

    Yl = np.zeros((Ns_l, Nl))
    Yl[0, :] = 1.0  # Pure NC12H26 liquid

    Yg = np.zeros((Ns_g, Ng))
    Yg[1, :] = 0.21  # O2
    Yg[2, :] = 0.79  # N2

    state_props = SimpleNamespace(
        Yl=Yl,
        Yg=Yg,
        Ts=Ts,  # Scalar or array, just need float()
    )

    # Build equilibrium model explicitly
    M_g = np.array([170.33, 32.0, 28.0])
    M_l = np.array([170.33])
    eq_model = build_equilibrium_model(cfg, Ns_g, Ns_l, M_g, M_l)

    # Override psat_ref for controlled testing
    eq_model.psat_ref["NC12H26"] = psat_ref

    return cfg, layout, grid, state_props, eq_model


# ============================================================================
# Smoke Test 1: Residual path calls equilibrium
# ============================================================================


def test_smoke_residual_calls_equilibrium():
    """
    Smoke test: _smoke_call_interface_equilibrium successfully calls equilibrium.

    Verifies:
    - No exceptions raised
    - Returns EquilibriumResult
    - Result has expected attributes (Yg_eq, meta, etc.)
    """
    cfg, layout, grid, state_props, eq_model = _make_minimal_mock_objects(
        Ts=350.0, psat_ref=1000.0
    )

    # Call smoke test helper (should not raise)
    result = _smoke_call_interface_equilibrium(
        cfg, layout, grid, state_props, eq_model
    )

    # Verify result structure
    assert hasattr(result, "Yg_eq"), "Result must have Yg_eq"
    assert hasattr(result, "meta"), "Result must have meta"
    assert hasattr(result, "y_cond"), "Result must have y_cond"
    assert hasattr(result, "psat"), "Result must have psat"

    # Verify meta contains P2 fields
    assert "ys_cap_hit" in result.meta, "meta must contain ys_cap_hit"
    assert "y_bg_total" in result.meta, "meta must contain y_bg_total"
    assert "y_cond_sum" in result.meta, "meta must contain y_cond_sum"

    # Basic sanity: Yg_eq sums to 1
    Yg_eq_sum = float(np.sum(result.Yg_eq))
    assert abs(Yg_eq_sum - 1.0) < 1e-10, f"Yg_eq sum={Yg_eq_sum}, expected 1.0"


# ============================================================================
# Smoke Test 2: Guard triggers with high psat_ref
# ============================================================================


def test_smoke_guard_triggered_high_psat():
    """
    Smoke test: Boiling guard triggers in residual path with high psat_ref.

    Verifies:
    - ys_cap_hit=True (guard triggered)
    - y_cond_sum ≈ 1 - EPS_BG
    - y_bg_total ≈ EPS_BG
    - Background species (O2, N2) > 0
    """
    # Use very high psat_ref to force trigger
    cfg, layout, grid, state_props, eq_model = _make_minimal_mock_objects(
        Ts=450.0, psat_ref=1e6  # Very high psat → y_cond >> 1 → guard triggers
    )

    result = _smoke_call_interface_equilibrium(
        cfg, layout, grid, state_props, eq_model
    )

    # SMOKE ASSERTION: Guard must trigger
    ys_cap_hit = result.meta["ys_cap_hit"]
    assert ys_cap_hit is True, \
        "Guard must trigger with psat_ref=1e6 (got ys_cap_hit=False)"

    # SMOKE ASSERTION: y_cond_sum scaled to 1-EPS_BG
    y_cond_sum = result.meta["y_cond_sum"]
    assert abs(y_cond_sum - (1.0 - EPS_BG)) < 1e-9, \
        f"Expected y_cond_sum≈{1.0-EPS_BG}, got {y_cond_sum}"

    # SMOKE ASSERTION: y_bg_total ≈ EPS_BG
    y_bg_total = result.meta["y_bg_total"]
    assert abs(y_bg_total - EPS_BG) < 1e-9, \
        f"Expected y_bg_total≈{EPS_BG}, got {y_bg_total}"

    # SMOKE ASSERTION: Background species positive
    Yg_eq = result.Yg_eq
    # Gas species order: NC12H26, O2, N2
    Y_O2 = Yg_eq[1]
    Y_N2 = Yg_eq[2]

    assert Y_O2 > 0, f"O2 mass fraction must be positive, got {Y_O2}"
    assert Y_N2 > 0, f"N2 mass fraction must be positive, got {Y_N2}"


# ============================================================================
# Smoke Test 3: Guard does NOT trigger with low psat_ref
# ============================================================================


def test_smoke_guard_not_triggered_low_psat():
    """
    Smoke test: Boiling guard does NOT trigger with low psat_ref (contrast).

    Verifies:
    - ys_cap_hit=False (guard not triggered)
    - y_bg_total >> EPS_BG
    """
    # Use low psat_ref
    cfg, layout, grid, state_props, eq_model = _make_minimal_mock_objects(
        Ts=300.0, psat_ref=10.0  # Low psat → y_cond << 1 → guard silent
    )

    result = _smoke_call_interface_equilibrium(
        cfg, layout, grid, state_props, eq_model
    )

    # SMOKE ASSERTION: Guard must NOT trigger
    ys_cap_hit = result.meta["ys_cap_hit"]
    assert ys_cap_hit is False, \
        "Guard must NOT trigger with psat_ref=10.0 (got ys_cap_hit=True)"

    # SMOKE ASSERTION: y_bg_total >> EPS_BG
    y_bg_total = result.meta["y_bg_total"]
    assert y_bg_total > EPS_BG + 0.1, \
        f"Expected y_bg_total >> EPS_BG, got {y_bg_total}"


# ============================================================================
# Smoke Test 4: Global constraint y_bg >= EPS_BG enforced
# ============================================================================


def test_smoke_global_constraint_y_bg_always_above_eps():
    """
    Smoke test: Global constraint y_bg >= EPS_BG enforced in residual path.

    Sweeps psat_ref from low to high, ALWAYS asserts y_bg_total >= EPS_BG.
    """
    cfg_template, layout, grid, state_props_template, _ = _make_minimal_mock_objects()

    # Sweep psat_ref (low → medium → high)
    psat_refs = [10.0, 100.0, 1000.0, 10000.0, 1e5, 1e6]

    for psat_ref in psat_refs:
        # Rebuild eq_model with new psat_ref
        Ns_g = len(cfg_template.species.gas_species_full)
        Ns_l = len(cfg_template.species.liq_species)
        M_g = np.array([170.33, 32.0, 28.0])
        M_l = np.array([170.33])
        eq_model = build_equilibrium_model(cfg_template, Ns_g, Ns_l, M_g, M_l)
        eq_model.psat_ref["NC12H26"] = psat_ref

        result = _smoke_call_interface_equilibrium(
            cfg_template, layout, grid, state_props_template, eq_model
        )

        y_bg_total = result.meta["y_bg_total"]
        y_cond_sum = result.meta["y_cond_sum"]

        # GLOBAL CONSTRAINT (no exceptions)
        assert y_bg_total >= EPS_BG - 1e-12, \
            f"psat_ref={psat_ref}: y_bg_total={y_bg_total:.6e} < EPS_BG={EPS_BG:.6e}"

        assert y_cond_sum <= 1.0 - EPS_BG + 1e-10, \
            f"psat_ref={psat_ref}: y_cond_sum={y_cond_sum:.15f} > 1-EPS_BG"


# ============================================================================
# Smoke Test 5: needs_eq condition validation
# ============================================================================


def test_smoke_needs_eq_condition_required():
    """
    Smoke test: Verify needs_eq condition is enforced.

    Should raise ValueError if:
    - include_mpp=False
    - layout does not have 'mpp' block
    """
    cfg, layout, grid, state_props, eq_model = _make_minimal_mock_objects()

    # Test 1: include_mpp=False
    cfg.physics.include_mpp = False
    with pytest.raises(ValueError, match="include_mpp=True"):
        _smoke_call_interface_equilibrium(cfg, layout, grid, state_props, eq_model)

    # Restore
    cfg.physics.include_mpp = True

    # Test 2: layout without mpp block
    layout.has_block = lambda name: False  # No blocks
    with pytest.raises(ValueError, match="mpp.*block"):
        _smoke_call_interface_equilibrium(cfg, layout, grid, state_props, eq_model)


if __name__ == "__main__":
    # Run smoke tests with pytest
    pytest.main([__file__, "-v"])
