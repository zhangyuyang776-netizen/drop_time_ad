"""
Nonlinear solve context for a single timestep.

This packages the data required by a global Newton residual so residual and solver
only depend on a context object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.types import CaseConfig, Grid1D, Props, State
from core.layout import UnknownLayout, VarEntry, pack_state, apply_u_to_state


@dataclass(slots=True)
class NonlinearContext:
    """
    Nonlinear solve context for a single timestep.

    Responsibilities:
      - store configuration, grid, and reference state/props for time n
      - provide u <-> State conversion helpers
      - keep variable scales for residual/Jacobian scaling
      - keep props_old for fallback diagnostics when prop recompute fails
    """

    # Configuration and grid
    cfg: CaseConfig
    grid_ref: Grid1D
    layout: UnknownLayout

    # Time info
    dt: float

    # State/props at time n
    state_old: State
    props_old: Props

    # Variable scaling and metadata (from layout.pack_state)
    scale_u: np.ndarray

    # Reference time level for state_old
    t_old: float = 0.0
    entries: List[VarEntry] = field(default_factory=list)

    # Extension point for diagnostics/counters
    meta: Dict[str, Any] = field(default_factory=dict)

    def make_state(
        self,
        u: np.ndarray,
        *,
        clip_negative_closure: bool = True,
    ) -> State:
        """
        Map a global unknown vector u to a State for t^{n+1} guesses.

        Notes:
          - state_old is used as a template, so untouched variables keep old values.
          - closure species are reconstructed in apply_u_to_state.
          - sum(Y) tolerance follows cfg.checks.sumY_tol.
        """
        u = np.asarray(u, dtype=np.float64)
        tol_sumY = float(getattr(self.cfg.checks, "sumY_tol", 1.0e-10))
        return apply_u_to_state(
            state=self.state_old,
            u=u,
            layout=self.layout,
            tol_closure=tol_sumY,
            clip_negative_closure=clip_negative_closure,
        )

    def make_state_from_u(
        self,
        u: np.ndarray,
        *,
        clip_negative_closure: bool = True,
    ) -> State:
        """Alias for make_state to keep solver call sites explicit."""
        return self.make_state(u, clip_negative_closure=clip_negative_closure)

    def to_scaled_u(self, u: np.ndarray) -> np.ndarray:
        """Convert physical u to scaled u_scaled = u / scale_u."""
        u = np.asarray(u, dtype=np.float64)
        if u.shape != self.scale_u.shape:
            raise ValueError(f"u shape {u.shape} incompatible with scale_u {self.scale_u.shape}")
        return u / self.scale_u

    def from_scaled_u(self, u_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled u_scaled to physical u = u_scaled * scale_u."""
        u_scaled = np.asarray(u_scaled, dtype=np.float64)
        if u_scaled.shape != self.scale_u.shape:
            raise ValueError(f"u_scaled shape {u_scaled.shape} incompatible with scale_u {self.scale_u.shape}")
        return u_scaled * self.scale_u


def build_nonlinear_context_for_step(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props_old: Props,
    *,
    t_old: float,
    dt: Optional[float] = None,
) -> Tuple[NonlinearContext, np.ndarray]:
    """
    Build NonlinearContext for a single timestep and return the initial guess u0.

    Current policy:
      - u0 is packed from state_old
      - scale_u comes from layout.pack_state default refs
    """
    dt_val = float(cfg.time.dt if dt is None else dt)
    if dt_val <= 0.0:
        raise ValueError(f"dt must be positive, got {dt_val}")

    u0, scale_u, entries = pack_state(state_old, layout)

    ctx = NonlinearContext(
        cfg=cfg,
        grid_ref=grid,
        layout=layout,
        dt=dt_val,
        t_old=float(t_old),
        state_old=state_old.copy(),
        props_old=props_old,
        scale_u=scale_u,
        entries=list(entries),
    )
    return ctx, u0
