"""
Unknown layout definition and state pack/unpack utilities.

Principles:
- Residual block order matches layout order: (1) Tg, (2) Yg (reduced), (3) Tl, (4) Yl (reduced),
  (5) interface scalars (Ts, mpp, Rd as enabled).
- Closure species never appear in the unknown vector; they are reconstructed explicitly.
- Unknown indices must come from UnknownLayout helpers (no hand-rolled math).
- Modules read/write unknowns only via State + this layout; no direct mutation of u elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, Tuple

import logging
import os
import numpy as np

from .types import CaseConfig, Grid1D, State
from .simplex import project_Y_cellwise

FloatArray = np.ndarray
UnknownVector = np.ndarray
ScaleVector = np.ndarray

# Global unknown vector for Step 19 fully implicit Newton
LAYOUT_VERSION = "step19-global-newton"

logger = logging.getLogger(__name__)

# Debug tag propagated from solver/FD to apply_u_to_state for attribution
_APPLY_U_TAG: Optional[str] = None

# Debug toggle for apply_u_to_state instrumentation
APPLY_U_DEBUG = os.getenv("DROPLET_APPLY_U_DEBUG", "0") == "1"


def set_apply_u_tag(tag: Optional[str]) -> None:
    """Set a debug tag for subsequent apply_u_to_state calls (debug only)."""
    global _APPLY_U_TAG
    _APPLY_U_TAG = tag


def get_apply_u_tag() -> Optional[str]:
    """Get the current debug tag used by apply_u_to_state."""
    return _APPLY_U_TAG


@dataclass(slots=True)
class VarEntry:
    """Single unknown entry metadata."""

    i: int
    kind: str  # "Yg" | "Tg" | "Yl" | "Tl" | "Ts" | "Rd" | "mpp"
    phase: str  # "gas" | "liq" | "interface"
    cell: Optional[int]
    spec: Optional[int]  # reduced species index
    name: str


@dataclass(frozen=True, slots=True)
class SplitDef:
    name: str
    blocks: Tuple[str, ...]
    policy: str = "by_layout"


@dataclass(slots=True)
class UnknownLayout:
    """Layout of the global unknown vector."""

    size: int
    entries: List[VarEntry]
    blocks: Dict[str, slice]
    block_slices: Dict[str, slice] = field(init=False)
    Ng: int
    Nl: int
    Ns_g_full: int
    Ns_g_eff: int
    Ns_l_full: int
    Ns_l_eff: int
    gas_species_full: List[str]
    gas_species_reduced: List[str]
    gas_closure_species: Optional[str]
    gas_full_to_reduced: Dict[str, Optional[int]]
    gas_reduced_to_full_idx: List[int]
    gas_closure_index: Optional[int]
    liq_species_full: List[str]
    liq_species_reduced: List[str]
    liq_closure_species: Optional[str]
    liq_full_to_reduced: Dict[str, Optional[int]]
    liq_reduced_to_full_idx: List[int]
    liq_closure_index: Optional[int]

    def __post_init__(self) -> None:
        self.block_slices = self._build_block_slices_from_entries()

    def n_dof(self) -> int:
        return self.size

    def has_block(self, name: str) -> bool:
        return name in self.block_slices

    def require_block(self, name: str) -> slice:
        if name not in self.block_slices:
            raise ValueError(f"Unknown block '{name}' not present in layout.")
        return self.block_slices[name]

    def _build_block_slices_from_entries(self) -> Dict[str, slice]:
        """
        Build contiguous slices for each VarEntry.kind.
        Requires: for each kind, indices are a single contiguous span.
        """
        kind_to_indices: Dict[str, List[int]] = {}
        for entry in self.entries:
            k = str(entry.kind)
            kind_to_indices.setdefault(k, []).append(int(entry.i))

        out: Dict[str, slice] = {}
        for k, idxs in kind_to_indices.items():
            if not idxs:
                continue
            idxs_sorted = sorted(idxs)
            i_min, i_max = idxs_sorted[0], idxs_sorted[-1]
            count = len(idxs_sorted)

            span = i_max - i_min + 1
            if span != count:
                missing = sorted(set(range(i_min, i_max + 1)) - set(idxs_sorted))
                raise RuntimeError(
                    f"Layout block '{k}' is not contiguous: "
                    f"min={i_min}, max={i_max}, count={count}, missing={missing[:10]}"
                )

            out[k] = slice(i_min, i_max + 1)

        return out

    def block_slice(self, name: str) -> slice:
        if name not in self.block_slices:
            raise KeyError(f"Unknown block '{name}'. Available: {list(self.block_slices)}")
        return self.block_slices[name]

    def block_size(self, name: str) -> int:
        sl = self.block_slice(name)
        return int(sl.stop - sl.start)

    def block_range(self, name: str) -> Tuple[int, int]:
        """Return (start, end_exclusive) for a base block."""
        sl = self.block_slice(name)
        return int(sl.start), int(sl.stop)

    def iter_blocks(self) -> Iterator[Tuple[str, slice]]:
        """Iterate base blocks in layout order (by slice.start)."""
        items = list(self.block_slices.items())
        items.sort(key=lambda kv: int(kv[1].start))
        for name, sl in items:
            yield name, sl

    def idx_Tg(self, ig: int) -> int:
        sl = self.require_block("Tg")
        if ig < 0 or ig >= self.Ng:
            raise IndexError(f"Tg cell index {ig} out of range [0,{self.Ng})")
        return sl.start + ig

    def idx_Yg(self, k_red: int, ig: int) -> int:
        sl = self.require_block("Yg")
        if k_red < 0 or k_red >= self.Ns_g_eff:
            raise IndexError(f"Yg reduced species index {k_red} out of range [0,{self.Ns_g_eff})")
        if ig < 0 or ig >= self.Ng:
            raise IndexError(f"Yg cell index {ig} out of range [0,{self.Ng})")
        return sl.start + ig * self.Ns_g_eff + k_red

    def idx_Tl(self, il: int) -> int:
        sl = self.require_block("Tl")
        if il < 0 or il >= self.Nl:
            raise IndexError(f"Tl cell index {il} out of range [0,{self.Nl})")
        return sl.start + il

    def idx_Yl(self, k_red: int, il: int) -> int:
        sl = self.require_block("Yl")
        if k_red < 0 or k_red >= self.Ns_l_eff:
            raise IndexError(f"Yl reduced species index {k_red} out of range [0,{self.Ns_l_eff})")
        if il < 0 or il >= self.Nl:
            raise IndexError(f"Yl cell index {il} out of range [0,{self.Nl})")
        return sl.start + il * self.Ns_l_eff + k_red

    def idx_Ts(self) -> int:
        sl = self.require_block("Ts")
        return sl.start

    def idx_mpp(self) -> int:
        sl = self.require_block("mpp")
        return sl.start

    def idx_Rd(self) -> int:
        sl = self.require_block("Rd")
        return sl.start

    def default_fieldsplit_plan(
        self,
        cfg: Optional[CaseConfig] = None,
        *,
        scheme: str = "by_layout",
    ) -> List[SplitDef]:
        """
        Return a list of SplitDef describing the recommended FieldSplit splits.

        scheme:
          - "by_layout": Tg, Yg, Tl, Yl, iface(Ts+mpp+Rd)
          - "bulk_iface": bulk(Tg+Yg+Tl+Yl), iface(Ts+mpp+Rd)
        """
        scheme = str(scheme).lower()
        if cfg is not None:
            solver_cfg = getattr(cfg, "solver", None)
            linear_cfg = getattr(solver_cfg, "linear", None)
            fs_cfg = getattr(linear_cfg, "fieldsplit", None)
            if fs_cfg is not None:
                if isinstance(fs_cfg, Mapping):
                    scheme = fs_cfg.get("scheme", fs_cfg.get("split_mode", scheme))
                else:
                    scheme = getattr(fs_cfg, "scheme", getattr(fs_cfg, "split_mode", scheme))
            scheme = str(scheme).lower()

        def exists(block: str) -> bool:
            return (block in self.block_slices) and (self.block_size(block) > 0)

        iface_blocks = tuple(b for b in ("Ts", "mpp", "Rd") if exists(b))

        if scheme == "bulk_iface":
            bulk_blocks = tuple(b for b in ("Tg", "Yg", "Tl", "Yl") if exists(b))
            splits: List[SplitDef] = []
            if bulk_blocks:
                splits.append(SplitDef(name="bulk", blocks=bulk_blocks, policy=scheme))
            if iface_blocks:
                splits.append(SplitDef(name="iface", blocks=iface_blocks, policy=scheme))
            return splits

        if scheme != "by_layout":
            raise ValueError(f"Unknown fieldsplit scheme '{scheme}'")

        splits = []
        for b in ("Tg", "Yg", "Tl", "Yl"):
            if exists(b):
                splits.append(SplitDef(name=b, blocks=(b,), policy=scheme))
        if iface_blocks:
            splits.append(SplitDef(name="iface", blocks=iface_blocks, policy=scheme))
        return splits

    def describe_fieldsplits(self, plan: Optional[List[SplitDef]] = None) -> Dict[str, object]:
        if plan is None:
            plan = self.default_fieldsplit_plan()

        def blocks_span(blocks: Tuple[str, ...]) -> Dict[str, object]:
            parts = []
            size = 0
            for b in blocks:
                sl = self.block_slice(b)
                part_size = int(sl.stop - sl.start)
                parts.append(
                    {
                        "block": b,
                        "start": int(sl.start),
                        "stop": int(sl.stop),
                        "size": part_size,
                    }
                )
                size += part_size
            return {"size": size, "parts": parts}

        return {
            "n_dof": int(self.n_dof()),
            "blocks": {
                k: {"start": int(v.start), "stop": int(v.stop), "size": int(v.stop - v.start)}
                for k, v in self.block_slices.items()
            },
            "splits": [{"name": s.name, "policy": s.policy, **blocks_span(s.blocks)} for s in plan],
        }

    def build_is_petsc(
        self,
        comm,
        ownership_range: Optional[Tuple[int, int]] = None,
        plan: str = "by_layout",
    ) -> Dict[str, Any]:
        """
        Build PETSc IS for FieldSplit plan.

        Returns: dict[split_name] = PETSc.IS

        If comm size == 1: prefer Stride IS for contiguous ranges.
        If parallel OR ownership_range is provided: create General IS with local owned indices.
        """
        from petsc4py import PETSc

        if comm is None:
            comm = PETSc.COMM_SELF

        comm_size = int(comm.getSize())
        is_parallel = comm_size > 1

        plan_name = str(plan).lower()
        plan_defs = self.default_fieldsplit_plan(cfg=None, scheme=plan_name)

        def _split_global_indices(split_blocks: Tuple[str, ...]) -> np.ndarray:
            parts = []
            for b in split_blocks:
                sl = self.block_slice(b)
                parts.append(np.arange(int(sl.start), int(sl.stop), dtype=np.int64))
            if not parts:
                return np.zeros((0,), dtype=np.int64)
            return np.concatenate(parts)

        def _is_contiguous(idxs: np.ndarray) -> bool:
            if idxs.size == 0:
                return True
            idxs_sorted = np.sort(idxs)
            return bool(idxs_sorted[-1] - idxs_sorted[0] + 1 == idxs_sorted.size)

        def _apply_ownership(idxs: np.ndarray) -> np.ndarray:
            if ownership_range is None:
                return idxs
            r0, r1 = int(ownership_range[0]), int(ownership_range[1])
            if idxs.size == 0:
                return idxs
            mask = (idxs >= r0) & (idxs < r1)
            return idxs[mask]

        out: Dict[str, Any] = {}

        for sd in plan_defs:
            name = sd.name
            idxs_global = _split_global_indices(sd.blocks)
            if idxs_global.size == 0:
                continue

            if is_parallel or ownership_range is not None:
                idxs_local = _apply_ownership(idxs_global).astype(PETSc.IntType, copy=False)
                out[name] = PETSc.IS().createGeneral(idxs_local, comm=comm)
                continue

            if _is_contiguous(idxs_global):
                i0 = int(np.min(idxs_global))
                n = int(idxs_global.size)
                out[name] = PETSc.IS().createStride(size=n, first=i0, step=1, comm=comm)
            else:
                out[name] = PETSc.IS().createGeneral(
                    idxs_global.astype(PETSc.IntType, copy=False),
                    comm=comm,
                )

        return out

    def debug_dump_is(self, is_dict: Mapping[str, Any]) -> Dict[str, Dict[str, Optional[int]]]:
        out: Dict[str, Dict[str, Optional[int]]] = {}
        for name, iset in is_dict.items():
            idx = iset.getIndices()
            try:
                arr = np.asarray(idx, dtype=np.int64)
                if arr.size:
                    out[name] = {"n": int(arr.size), "min": int(arr.min()), "max": int(arr.max())}
                else:
                    out[name] = {"n": 0, "min": None, "max": None}
            finally:
                if hasattr(iset, "restoreIndices"):
                    try:
                        iset.restoreIndices()
                    except TypeError:
                        iset.restoreIndices(idx)
        return out


def _build_species_mapping(full: List[str], closure: Optional[str], active: Optional[Set[str]] = None):
    """
    Map mechanism-ordered full species list to reduced indices.

    active=None: all species except closure are active (backward compatible)
    active=set(...): only names in active become reduced unknowns; others map to None
    """
    if len(full) != len(set(full)):
        raise ValueError(f"Duplicate species names in list: {full}")
    if closure is not None and closure not in full:
        raise ValueError(f"Closure species '{closure}' not found in species list {full}")
    if active is not None:
        missing = [name for name in active if name not in full]
        if missing:
            raise ValueError(f"Active species not present in full list: {missing}")

    reduced: List[str] = []
    reduced_to_full_idx: List[int] = []
    full_to_reduced: Dict[str, Optional[int]] = {}
    closure_idx: Optional[int] = None

    for i_full, name in enumerate(full):
        if closure is not None and name == closure:
            full_to_reduced[name] = None
            closure_idx = i_full
            continue
        if active is not None and name not in active:
            full_to_reduced[name] = None
            continue
        k_red = len(reduced)
        reduced.append(name)
        reduced_to_full_idx.append(i_full)
        full_to_reduced[name] = k_red

    return reduced, full_to_reduced, reduced_to_full_idx, closure_idx

def build_layout(cfg: CaseConfig, grid: Grid1D) -> UnknownLayout:
    """
    Build the unknown layout following residual ordering.

    Block order (aligned with residual assembly):
      1) gas energy (Tg)
      2) gas species (reduced)
      3) liquid energy (Tl)
      4) liquid species (reduced)
      5) interface scalars (Ts, mpp, Rd in that order if enabled)
    """
    gas_full = list(cfg.species.gas_species_full)
    if not gas_full:
        raise ValueError("cfg.species.gas_species_full is empty. Preprocess must load mechanism and fill it.")
    gas_closure = cfg.species.gas_balance_species

    gas_active: Set[str] = set(gas_full)
    gas_active.discard(gas_closure)

    gas_reduced, gas_full_to_reduced, gas_reduced_to_full_idx, gas_closure_idx = _build_species_mapping(
        gas_full, gas_closure, active=gas_active
    )

    liq_full = list(cfg.species.liq_species)
    liq_closure = cfg.species.liq_balance_species
    liq_reduced, liq_full_to_reduced, liq_reduced_to_full_idx, liq_closure_idx = _build_species_mapping(
        liq_full, liq_closure
    )

    Ns_g_eff = len(gas_reduced)
    Ns_l_eff = len(liq_reduced)

    include_gas_energy = cfg.physics.solve_Tg
    include_gas_species = cfg.physics.solve_Yg
    include_liq_energy = cfg.physics.solve_Tl
    include_liq_species = cfg.physics.solve_Yl
    include_Ts = cfg.physics.include_Ts
    include_mpp = cfg.physics.include_mpp
    include_Rd = cfg.physics.include_Rd

    entries: List[VarEntry] = []
    blocks: Dict[str, slice] = {}
    cursor = 0

    # 1) Gas energy (temperature)
    if include_gas_energy and grid.Ng > 0:
        start = cursor
        for ig in range(grid.Ng):
            entries.append(
                VarEntry(
                    i=cursor,
                    kind="Tg",
                    phase="gas",
                    cell=ig,
                    spec=None,
                    name=f"Tg[{ig}]",
                )
            )
            cursor += 1
        blocks["Tg"] = slice(start, cursor)

    # 2) Gas species (cell outer, reduced species inner)
    if include_gas_species and Ns_g_eff > 0 and grid.Ng > 0:
        start = cursor
        for ig in range(grid.Ng):
            for k_red, name in enumerate(gas_reduced):
                entries.append(
                    VarEntry(
                        i=cursor,
                        kind="Yg",
                        phase="gas",
                        cell=ig,
                        spec=k_red,
                        name=f"Yg[{name}@{ig}]",
                    )
                )
                cursor += 1
        blocks["Yg"] = slice(start, cursor)

    # 3) Liquid energy (temperature)
    if include_liq_energy and grid.Nl > 0:
        start = cursor
        for il in range(grid.Nl):
            entries.append(
                VarEntry(
                    i=cursor,
                    kind="Tl",
                    phase="liq",
                    cell=il,
                    spec=None,
                    name=f"Tl[{il}]",
                )
            )
            cursor += 1
        blocks["Tl"] = slice(start, cursor)

    # 4) Liquid species (cell outer, reduced species inner)
    if include_liq_species and Ns_l_eff > 0 and grid.Nl > 0:
        start = cursor
        for il in range(grid.Nl):
            for k_red, name in enumerate(liq_reduced):
                entries.append(
                    VarEntry(
                        i=cursor,
                        kind="Yl",
                        phase="liq",
                        cell=il,
                        spec=k_red,
                        name=f"Yl[{name}@{il}]",
                    )
                )
                cursor += 1
        blocks["Yl"] = slice(start, cursor)

    # 5) Interface scalars
    if include_Ts:
        start = cursor
        entries.append(
            VarEntry(
                i=cursor,
                kind="Ts",
                phase="interface",
                cell=None,
                spec=None,
                name="Ts",
            )
        )
        cursor += 1
        blocks["Ts"] = slice(start, cursor)

    if include_mpp:
        start = cursor
        entries.append(
            VarEntry(
                i=cursor,
                kind="mpp",
                phase="interface",
                cell=None,
                spec=None,
                name="mpp",
            )
        )
        cursor += 1
        blocks["mpp"] = slice(start, cursor)

    if include_Rd:
        start = cursor
        entries.append(
            VarEntry(
                i=cursor,
                kind="Rd",
                phase="interface",
                cell=None,
                spec=None,
                name="Rd",
            )
        )
        cursor += 1
        blocks["Rd"] = slice(start, cursor)

    return UnknownLayout(
        size=cursor,
        entries=entries,
        blocks=blocks,
        Ng=grid.Ng,
        Nl=grid.Nl,
        Ns_g_full=len(gas_full),
        Ns_g_eff=Ns_g_eff,
        Ns_l_full=len(liq_full),
        Ns_l_eff=Ns_l_eff,
        gas_species_full=gas_full,
        gas_species_reduced=gas_reduced,
        gas_closure_species=gas_closure,
        gas_full_to_reduced=gas_full_to_reduced,
        gas_reduced_to_full_idx=gas_reduced_to_full_idx,
        gas_closure_index=gas_closure_idx,
        liq_species_full=liq_full,
        liq_species_reduced=liq_reduced,
        liq_closure_species=liq_closure,
        liq_full_to_reduced=liq_full_to_reduced,
        liq_reduced_to_full_idx=liq_reduced_to_full_idx,
        liq_closure_index=liq_closure_idx,
    )


def pack_state(
    state: State,
    layout: UnknownLayout,
    *,
    refs: Optional[Mapping[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[VarEntry]]:
    """
    Pack State into a 1D unknown vector following the given layout.

    Typical usage:
      - For a fully implicit Newton step, call pack_state(state_old, layout) once to
        get u0 (initial guess) and scale_u (variable scales) for the solver/context.
      - For residual evaluations, map u back to State with apply_u_to_state.

    refs:
      - "T_ref": temperature scaling reference (default 298.15)
      - "Rd_ref": radius scaling reference (default max(|Rd|, 1.0))
      - "mpp_ref": evaporation scaling reference (default 1e-3)
    """
    refs = refs or {}
    T_ref = float(refs.get("T_ref", 298.15))
    Rd_ref = refs.get("Rd_ref", None)
    m_ref = float(refs.get("mpp_ref", 1.0e-3))

    u = np.zeros(layout.size, dtype=np.float64)
    scale_u = np.ones(layout.size, dtype=np.float64)

    for entry in layout.entries:
        i = entry.i
        kind = entry.kind

        if kind == "Yg":
            ig = entry.cell
            k_red = entry.spec
            k_full = layout.gas_reduced_to_full_idx[k_red]
            u[i] = state.Yg[k_full, ig]
            scale_u[i] = 1.0
        elif kind == "Tg":
            ig = entry.cell
            val = float(state.Tg[ig])
            u[i] = val
            scale_u[i] = max(abs(val), T_ref)
        elif kind == "Yl":
            il = entry.cell
            k_red = entry.spec
            k_full = layout.liq_reduced_to_full_idx[k_red]
            u[i] = state.Yl[k_full, il]
            scale_u[i] = 1.0
        elif kind == "Tl":
            il = entry.cell
            val = float(state.Tl[il])
            u[i] = val
            scale_u[i] = max(abs(val), T_ref)
        elif kind == "Ts":
            val = float(state.Ts)
            u[i] = val
            scale_u[i] = max(abs(val), T_ref)
        elif kind == "Rd":
            val = float(state.Rd)
            u[i] = val
            if Rd_ref is None:
                scale_u[i] = max(abs(val), 1.0)
            else:
                scale_u[i] = max(abs(val), float(Rd_ref))
        elif kind == "mpp":
            val = float(state.mpp)
            u[i] = val
            scale_u[i] = max(abs(val), m_ref)
        else:
            raise ValueError(f"Unknown variable kind: {kind}")

    return u, scale_u, list(layout.entries)


def _reconstruct_closure(
    Y_full: np.ndarray,
    reduced_to_full_idx: List[int],
    closure_idx: Optional[int],
    *,
    tol: float,
    clip_negative: bool,
    phase: str,
) -> None:
    """Compute closure species from reduced ones; mutate Y_full in place."""
    if closure_idx is None:
        return
    if Y_full.shape[0] <= closure_idx:
        raise ValueError(f"{phase} closure index {closure_idx} out of bounds for shape {Y_full.shape}")
    sum_other = np.sum(Y_full, axis=0) - Y_full[closure_idx, :]
    closure = 1.0 - sum_other

    if clip_negative:
        # Automatic correction for numerical errors (clamp first, then check)
        # Clamp slightly negative values to 0.0
        closure = np.where((closure < 0.0) & (closure >= -tol), 0.0, closure)
        # Clamp slightly exceeding 1.0 values to 1.0
        closure = np.where((closure > 1.0) & (closure <= 1.0 + tol), 1.0, closure)

        # Force strict physical bounds [0, 1] to handle floating-point precision issues
        closure = np.clip(closure, 0.0, 1.0)

        # After clamping, verify no catastrophic errors remain
        if np.any(~np.isfinite(closure)):
            raise ValueError(f"{phase} closure species contains non-finite values")

        # Assign clipped closure back to Y_full
        Y_full[closure_idx, :] = closure

        # Renormalize to enforce sum(Y) = 1.0 after clipping
        # This is necessary because clipping can break the sum constraint
        # (e.g., if sum_other=1.05, closure gets clipped to 0, sum(Y)=1.05 != 1.0)
        sums = np.sum(Y_full, axis=0)
        mask = sums > 1e-14
        if np.any(mask):
            Y_full[:, mask] /= sums[np.newaxis, mask]
    else:
        # Strict checking without clamping
        if np.any(closure < -tol):
            raise ValueError(
                f"{phase} closure species negative beyond tol={tol}: min={float(np.min(closure)):.3e}"
            )
        if np.any(closure > 1.0 + tol):
            raise ValueError(
                f"{phase} closure species exceeds 1 beyond tol={tol}: max={float(np.max(closure)):.3e}"
            )

        Y_full[closure_idx, :] = closure


def apply_u_to_state(
    state: State,
    u: np.ndarray,
    layout: UnknownLayout,
    *,
    tol_closure: float = 1e-12,
    clip_negative_closure: bool = False,
) -> State:
    """Apply unknown vector to a State and reconstruct closure species."""
    u = np.asarray(u, dtype=np.float64)
    if u.size < layout.size:
        raise ValueError(f"u size {u.size} is smaller than layout size {layout.size}")

    Tg = np.array(state.Tg, copy=True, dtype=np.float64)
    Yg = np.array(state.Yg, copy=True, dtype=np.float64)
    Tl = np.array(state.Tl, copy=True, dtype=np.float64)
    Yl = np.array(state.Yl, copy=True, dtype=np.float64)
    Ts = float(state.Ts)
    Rd = float(state.Rd)
    mpp = float(state.mpp)

    # Debug: track large values written to Yg
    _debug_cell = 7
    _debug_Yg_anomaly_threshold = 1.5  # Y value > 1.5 is clearly anomalous
    tag = get_apply_u_tag()

    for entry in layout.entries:
        i = entry.i
        kind = entry.kind
        val = float(u[i])

        if kind == "Yg":
            ig = entry.cell
            k_red = entry.spec
            k_full = layout.gas_reduced_to_full_idx[k_red]
            if APPLY_U_DEBUG and ig == _debug_cell and abs(val) > _debug_Yg_anomaly_threshold:
                logger.warning(
                    "[APPLY_U_DEBUG] Writing LARGE Yg value! tag=%s i=%d k_red=%d k_full=%d cell=%d val=%.6e u[i]=%.6e",
                    tag, i, k_red, k_full, ig, val, u[i],
                )
                try:
                    yg_slice = np.asarray(state.Yg[:, ig], dtype=np.float64).copy()
                    logger.warning(
                        "[APPLY_U_DEBUG] Yg[cell=%d] BEFORE assign (tag=%s): sum=%g min=%g max=%g vals=%s",
                        ig,
                        tag,
                        float(yg_slice.sum()) if yg_slice.size else 0.0,
                        float(yg_slice.min()) if yg_slice.size else 0.0,
                        float(yg_slice.max()) if yg_slice.size else 0.0,
                        np.array2string(yg_slice, precision=6, floatmode="maxprec"),
                    )
                except Exception:
                    logger.warning("[APPLY_U_DEBUG] Yg slice dump failed (cell=%d, tag=%s)", ig, tag)
            Yg[k_full, ig] = val
        elif kind == "Tg":
            ig = entry.cell
            Tg[ig] = val
        elif kind == "Yl":
            il = entry.cell
            k_red = entry.spec
            k_full = layout.liq_reduced_to_full_idx[k_red]
            Yl[k_full, il] = val
        elif kind == "Tl":
            il = entry.cell
            Tl[il] = val
        elif kind == "Ts":
            Ts = val
        elif kind == "Rd":
            Rd = val
        elif kind == "mpp":
            mpp = val
        else:
            raise ValueError(f"Unknown variable kind: {kind}")

    # Debug: check Yg before closure reconstruction
    if APPLY_U_DEBUG and Yg.ndim == 2 and Yg.shape[1] > _debug_cell:
        Y_before = Yg[:, _debug_cell]
        Y_sum_before = float(np.sum(Y_before))
        if abs(Y_sum_before - 1.0) > 0.01:
            logger.warning(
                "[APPLY_U_DEBUG] Yg@cell%d BEFORE closure (tag=%s): sum=%.6g values=%s closure_idx=%s",
                _debug_cell,
                tag,
                Y_sum_before,
                Y_before.tolist(),
                layout.gas_closure_index,
            )

    if ("Yg" in layout.blocks) and (layout.gas_closure_index is not None):
        _reconstruct_closure(
            Y_full=Yg,
            reduced_to_full_idx=layout.gas_reduced_to_full_idx,
            closure_idx=layout.gas_closure_index,
            tol=tol_closure,
            clip_negative=clip_negative_closure,
            phase="Gas",
        )

    # Debug: check Yg after closure reconstruction
    if APPLY_U_DEBUG and Yg.ndim == 2 and Yg.shape[1] > _debug_cell:
        Y_after = Yg[:, _debug_cell]
        Y_sum_after = float(np.sum(Y_after))
        if abs(Y_sum_after - 1.0) > 0.01:
            logger.warning(
                "[APPLY_U_DEBUG] Yg@cell%d AFTER closure (tag=%s): sum=%.6g values=%s",
                _debug_cell,
                tag,
                Y_sum_after,
                Y_after.tolist(),
            )

    if ("Yl" in layout.blocks) and (layout.liq_closure_index is not None):
        _reconstruct_closure(
            Y_full=Yl,
            reduced_to_full_idx=layout.liq_reduced_to_full_idx,
            closure_idx=layout.liq_closure_index,
            tol=tol_closure,
            clip_negative=clip_negative_closure,
            phase="Liquid",
        )

    # FIXED: Apply simplex projection after Newton solve to enforce sum(Y) = 1.0
    # Newton solver doesn't know about sum constraint, so output may violate it.
    # Simplex projection corrects with minimum perturbation.
    if "Yg" in layout.blocks and Yg.size > 0:
        Yg = project_Y_cellwise(Yg, min_Y=1e-14, axis=0)

    if "Yl" in layout.blocks and Yl.size > 0:
        Yl = project_Y_cellwise(Yl, min_Y=1e-14, axis=0)

    return State(Tg=Tg, Yg=Yg, Tl=Tl, Yl=Yl, Ts=Ts, mpp=mpp, Rd=Rd)


def assert_pack_apply_consistency(state: State, layout: UnknownLayout, rtol=1e-12, atol=1e-14) -> None:
    """Pack then apply and assert the state is unchanged (for tests)."""
    u, _, _ = pack_state(state, layout)
    state2 = apply_u_to_state(state, u, layout)

    if "Tg" in layout.blocks:
        np.testing.assert_allclose(state2.Tg, state.Tg, rtol=rtol, atol=atol)
    if "Yg" in layout.blocks:
        np.testing.assert_allclose(state2.Yg, state.Yg, rtol=rtol, atol=atol)
    if "Tl" in layout.blocks:
        np.testing.assert_allclose(state2.Tl, state.Tl, rtol=rtol, atol=atol)
    if "Yl" in layout.blocks:
        np.testing.assert_allclose(state2.Yl, state.Yl, rtol=rtol, atol=atol)
    if "Ts" in layout.blocks:
        np.testing.assert_allclose(state2.Ts, state.Ts, rtol=rtol, atol=atol)
    if "Rd" in layout.blocks:
        np.testing.assert_allclose(state2.Rd, state.Rd, rtol=rtol, atol=atol)
    if "mpp" in layout.blocks:
        np.testing.assert_allclose(state2.mpp, state.mpp, rtol=rtol, atol=atol)
