#!/usr/bin/env python3
"""
Test script to verify the background species allocation fix.

This script tests that when condensable species (NC12H26) increases,
background species (N2, O2) are compressed proportionally while maintaining
their relative ratio, instead of incorrectly compressing N2 to zero.
"""

import numpy as np
from properties.equilibrium import build_equilibrium_model, compute_interface_equilibrium_full
from core.types import CaseConfig
import yaml


def test_background_species_ratio():
    """Test that background species maintain correct ratio after equilibrium calculation."""

    # Load the configuration
    config_path = "cases/p3_accept_single_petsc_mpi_schur_with_u_output.yaml"
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # Create a minimal config object
    from types import SimpleNamespace

    cfg = SimpleNamespace()
    cfg.initial = SimpleNamespace()
    cfg.initial.Yg = {"N2": 0.79, "O2": 0.21}
    cfg.initial.P_inf = 101325.0

    cfg.physics = SimpleNamespace()
    cfg.physics.interface = SimpleNamespace()
    cfg.physics.interface.equilibrium = SimpleNamespace()
    cfg.physics.interface.equilibrium.method = "raoult_psat"
    cfg.physics.interface.equilibrium.psat_model = "coolprop"
    cfg.physics.interface.equilibrium.condensables_gas = ["NC12H26"]
    cfg.physics.interface.equilibrium.coolprop = SimpleNamespace()
    cfg.physics.interface.equilibrium.coolprop.backend = "HEOS"
    cfg.physics.interface.equilibrium.coolprop.fluids = ["n-Dodecane"]

    cfg.species = SimpleNamespace()
    cfg.species.liq_species = ["n-Dodecane"]
    cfg.species.liq2gas_map = {"n-Dodecane": "NC12H26"}
    cfg.species.mw_kg_per_mol = {"n-Dodecane": 0.17034}

    # Set up species and molar masses
    gas_names = ["N2", "O2", "NC12H26"]
    liq_names = ["n-Dodecane"]
    M_g = np.array([0.028014, 0.031999, 0.17034])  # kg/mol
    M_l = np.array([0.17034])

    # Build equilibrium model
    eq_model = build_equilibrium_model(cfg, Ns_g=3, Ns_l=1, M_g=M_g, M_l=M_l)

    print("=" * 70)
    print("Testing Background Species Ratio Fix")
    print("=" * 70)

    print("\nInitial farfield composition:")
    print(f"  N2 : {cfg.initial.Yg['N2']:.4f}")
    print(f"  O2 : {cfg.initial.Yg['O2']:.4f}")
    print(f"  Ratio N2/O2 = {cfg.initial.Yg['N2']/cfg.initial.Yg['O2']:.4f} (expected: 3.76)")

    # Test case: High NC12H26 saturation
    Ts = 300.0  # K
    Pg = 101325.0  # Pa
    Yl_face = np.array([1.0])  # Pure n-Dodecane liquid

    # Interface gas composition with high NC12H26
    # This simulates the state after closure reconstruction where N2 might be ~0
    Yg_face = np.array([0.0, 0.21, 0.79])  # N2=0 (closure compressed), O2=0.21, NC12H26=0.79

    print("\nInterface gas composition (input to equilibrium):")
    print(f"  N2 : {Yg_face[0]:.4f} (artificially compressed by closure)")
    print(f"  O2 : {Yg_face[1]:.4f}")
    print(f"  NC12H26 : {Yg_face[2]:.4f}")

    # Compute equilibrium
    result = compute_interface_equilibrium_full(
        eq_model,
        Ts=Ts,
        Pg=Pg,
        Yl_face=Yl_face,
        Yg_face=Yg_face,
    )

    Yg_eq = result.Yg_eq

    print("\nEquilibrium gas composition (output):")
    print(f"  N2 : {Yg_eq[0]:.6f}")
    print(f"  O2 : {Yg_eq[1]:.6f}")
    print(f"  NC12H26 : {Yg_eq[2]:.6f}")
    print(f"  Sum : {np.sum(Yg_eq):.6f}")

    # Calculate ratios
    if Yg_eq[1] > 1e-10:
        ratio_N2_O2 = Yg_eq[0] / Yg_eq[1]
        expected_ratio = cfg.initial.Yg['N2'] / cfg.initial.Yg['O2']

        print(f"\nBackground species ratio:")
        print(f"  N2/O2 = {ratio_N2_O2:.4f}")
        print(f"  Expected (from farfield) = {expected_ratio:.4f}")
        print(f"  Error = {abs(ratio_N2_O2 - expected_ratio):.6f}")

        # Check if ratio is preserved (within 1% tolerance)
        if abs(ratio_N2_O2 - expected_ratio) < 0.05:
            print("\n✓ PASS: Background species ratio is correctly preserved!")
            status = "PASS"
        else:
            print("\n✗ FAIL: Background species ratio is NOT preserved!")
            status = "FAIL"
    else:
        print("\n✗ FAIL: O2 is too small to calculate ratio!")
        status = "FAIL"

    # Check that N2 is not zero
    if Yg_eq[0] > 1e-6:
        print(f"✓ PASS: N2 is not compressed to zero (N2 = {Yg_eq[0]:.6f})")
    else:
        print(f"✗ FAIL: N2 is incorrectly compressed to zero (N2 = {Yg_eq[0]:.6f})")
        status = "FAIL"

    print("=" * 70)

    return status == "PASS"


if __name__ == "__main__":
    try:
        success = test_background_species_ratio()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
