"""
Unified liquid property entry point.

Business layers should call get_liquid_props instead of reaching into backend modules.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from core.types import CaseConfig, Grid1D, State
from properties.liquid import LiquidPropertiesModel, build_liquid_model, compute_liquid_props

FloatArray = np.ndarray


def get_liquid_props(
    cfg: CaseConfig,
    grid: Grid1D,
    state: State,
    model: LiquidPropertiesModel | None = None,
) -> Tuple[Dict[str, FloatArray], Dict[str, FloatArray]]:
    """
    Compute liquid properties using the configured backend (p2db).

    If model is not provided, it will be built from cfg.
    """
    if model is None:
        model = build_liquid_model(cfg)
    return compute_liquid_props(model, state, grid)
