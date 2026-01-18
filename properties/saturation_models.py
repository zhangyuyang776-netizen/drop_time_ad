"""
P4: Custom saturation models (Millán-Merino formulation + Watson correlation).

This module implements saturation pressure psat(T) and latent heat hvap(T)
using physics-based correlations instead of relying on CoolProp black-box calls.

Key features:
- Continuous and differentiable (better for Newton/SNES solvers)
- Controllable and diagnosable (know exactly which parameter affects what)
- Naturally supports multi-component systems (one parameter set per species)

References:
- Millán-Merino et al.: Chemical potential equilibrium → Clausius integral form
- Watson correlation: L(T) extrapolation from L(Tb)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Gas constant [J/(mol·K)]
R_GAS = 8.314462618

# Minimum epsilon for numerical stability
EPS = 1e-30


@dataclass(slots=True)
class LiquidSatParams:
    """
    Saturation parameters for a single liquid species.

    Required for Millán-Merino saturation model:
    - W: Molar mass [kg/mol]
    - Tb_patm: Normal boiling point at 1 atm [K]
    - patm: Reference pressure [Pa]
    - Tc: Critical temperature [K]
    - Lb_Tb: Latent heat at Tb [J/mol]
    - watson_n: Watson correlation exponent (dimensionless)
    """

    canonical_name: str
    W: float  # kg/mol
    Tb_patm: float  # K
    patm: float  # Pa
    Tc: float  # K
    Pc: float  # Pa (optional, for future use)
    Lb_Tb: float  # J/mol
    watson_n: float  # dimensionless

    def __post_init__(self):
        """Validate parameters on construction."""
        if self.W <= 0:
            raise ValueError(f"Invalid molar mass W={self.W} for {self.canonical_name}")
        if self.Tb_patm <= 0 or self.Tb_patm >= self.Tc:
            raise ValueError(
                f"Invalid Tb={self.Tb_patm} (must be 0 < Tb < Tc={self.Tc}) "
                f"for {self.canonical_name}"
            )
        if self.Tc <= 0:
            raise ValueError(f"Invalid Tc={self.Tc} for {self.canonical_name}")
        if self.Lb_Tb <= 0:
            raise ValueError(f"Invalid Lb_Tb={self.Lb_Tb} for {self.canonical_name}")
        if self.watson_n <= 0 or self.watson_n > 1.0:
            raise ValueError(
                f"Invalid watson_n={self.watson_n} (should be ~0.38) "
                f"for {self.canonical_name}"
            )
        if self.patm <= 0:
            raise ValueError(f"Invalid patm={self.patm} for {self.canonical_name}")


class SaturationModelMM:
    """
    Millán-Merino saturation model.

    Computes psat(T) and hvap(T) using:
    1. Watson correlation: L(T) = Lb * [(Tc - T)/(Tc - Tb)]^n
    2. Clausius integral: ln[psat(T)/patm] = ∫_{Tb}^{T} L(τ)/(R·τ²) dτ

    This formulation:
    - Is continuous and smooth (good for Newton solvers)
    - Naturally enforces psat(Tb) = patm
    - Requires only fundamental thermodynamic parameters (no empirical multi-term fits)
    """

    def __init__(self):
        """Initialize Millán-Merino saturation model."""
        pass

    def _watson_hvap_molar(self, sp: LiquidSatParams, T: float) -> float:
        """
        Watson correlation for latent heat of vaporization.

        L(T) = Lb * [(Tc - T) / (Tc - Tb)]^n

        Args:
            sp: Liquid species parameters
            T: Temperature [K]

        Returns:
            L(T) in J/mol

        Raises:
            ValueError: If T is out of valid range (0 < T < Tc)
        """
        # Input validation
        if not np.isfinite(T) or T <= 0.0:
            raise ValueError(
                f"Invalid temperature T={T} for Watson correlation (must be finite and > 0)"
            )
        if T >= sp.Tc:
            raise ValueError(
                f"Temperature T={T} exceeds critical temperature Tc={sp.Tc} "
                f"for {sp.canonical_name} (Watson correlation not valid)"
            )

        # Watson formula
        numerator = sp.Tc - T
        denominator = sp.Tc - sp.Tb_patm
        if denominator <= 0:
            raise ValueError(
                f"Invalid Tc-Tb={denominator} for {sp.canonical_name} (check parameters)"
            )

        reduced_temp_factor = numerator / denominator
        if reduced_temp_factor <= 0:
            # T >= Tc case (should be caught above, but defensive)
            raise ValueError(f"Watson reduced temp factor <= 0 at T={T} for {sp.canonical_name}")

        L_molar = sp.Lb_Tb * (reduced_temp_factor ** sp.watson_n)

        # Sanity check
        if not np.isfinite(L_molar) or L_molar < 0:
            raise ValueError(
                f"Watson correlation produced invalid L={L_molar} at T={T} "
                f"for {sp.canonical_name}"
            )

        return L_molar

    def hvap(self, sp: LiquidSatParams, T: float) -> float:
        """
        Latent heat of vaporization in J/kg (mass-specific).

        Args:
            sp: Liquid species parameters
            T: Temperature [K]

        Returns:
            hvap in J/kg
        """
        L_molar = self._watson_hvap_molar(sp, T)
        hvap_mass = L_molar / sp.W  # J/mol → J/kg
        return hvap_mass

    def psat(self, sp: LiquidSatParams, T: float) -> float:
        """
        Saturation pressure using Millán-Merino Clausius integral.

        ln[psat(T)/patm] = ∫_{Tb}^{T} L(τ)/(R·τ²) dτ

        where L(τ) is given by Watson correlation.

        Args:
            sp: Liquid species parameters
            T: Temperature [K]

        Returns:
            psat(T) in Pa

        Raises:
            ValueError: If T is out of valid range or integration fails
        """
        # Input validation (same as hvap)
        if not np.isfinite(T) or T <= 0.0:
            raise ValueError(
                f"Invalid temperature T={T} for psat calculation (must be finite and > 0)"
            )
        if T >= sp.Tc:
            raise ValueError(
                f"Temperature T={T} exceeds critical temperature Tc={sp.Tc} "
                f"for {sp.canonical_name} (psat not defined beyond Tc)"
            )

        # Special case: T = Tb → psat = patm (by definition)
        if abs(T - sp.Tb_patm) < 1e-6:
            return sp.patm

        # Numerical integration: ∫_{Tb}^{T} L(τ)/(R·τ²) dτ
        # Use scipy.integrate.quad for robust quadrature
        try:
            from scipy.integrate import quad
        except ImportError:
            raise ImportError(
                "scipy is required for Millán-Merino psat integration. "
                "Install with: pip install scipy"
            )

        def integrand(tau: float) -> float:
            """Integrand: L(τ)/(R·τ²)"""
            L_tau = self._watson_hvap_molar(sp, tau)
            return L_tau / (R_GAS * tau**2)

        # Integrate from Tb to T
        # quad returns (integral, error_estimate)
        try:
            integral_value, error_estimate = quad(
                integrand,
                sp.Tb_patm,
                T,
                limit=100,  # Max number of subdivisions
                epsabs=1e-10,  # Absolute tolerance
                epsrel=1e-8,  # Relative tolerance
            )
        except Exception as e:
            raise ValueError(
                f"Numerical integration failed for psat at T={T} "
                f"for {sp.canonical_name}: {e}"
            )

        # psat = patm * exp(integral)
        psat_value = sp.patm * np.exp(integral_value)

        # Sanity check
        if not np.isfinite(psat_value) or psat_value <= 0:
            raise ValueError(
                f"psat calculation produced invalid result psat={psat_value} "
                f"at T={T} for {sp.canonical_name}"
            )

        return psat_value


# Factory function for easy model selection
def create_saturation_model(model_name: str = "mm_integral_watson") -> SaturationModelMM:
    """
    Factory to create saturation model by name.

    Args:
        model_name: Model identifier (currently only "mm_integral_watson" supported)

    Returns:
        SaturationModelMM instance

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name == "mm_integral_watson":
        return SaturationModelMM()
    else:
        raise ValueError(
            f"Unknown saturation model: {model_name}. "
            f"Available: 'mm_integral_watson'"
        )
