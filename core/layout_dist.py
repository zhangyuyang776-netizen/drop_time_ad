from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

bootstrap_mpi_before_petsc()

from petsc4py import PETSc


def _as_scalar(x) -> int:
    if isinstance(x, (tuple, list, np.ndarray)):
        if len(x) == 0:
            raise RuntimeError("Unexpected empty DMDA corners entry.")
        x = x[0]
    return int(x)


def _dmda_get_corners_1d(dm: PETSc.DM) -> Tuple[int, int]:
    c = dm.getCorners()
    if isinstance(c, tuple):
        if len(c) == 2:
            return _as_scalar(c[0]), _as_scalar(c[1])
        if len(c) >= 4:
            return _as_scalar(c[0]), _as_scalar(c[3])
    raise RuntimeError(f"Unexpected DMDA.getCorners() return: {c}")


def _dmda_get_ghost_corners_1d(dm: PETSc.DM) -> Tuple[int, int]:
    if hasattr(dm, "getGhostCorners"):
        c = dm.getGhostCorners()
        if isinstance(c, tuple):
            if len(c) == 2:
                return _as_scalar(c[0]), _as_scalar(c[1])
            if len(c) >= 4:
                return _as_scalar(c[0]), _as_scalar(c[3])
    return _dmda_get_corners_1d(dm)


def _dmda_get_stencil_width(dm: PETSc.DM, default: int = 1) -> int:
    try:
        return int(dm.getStencilWidth())
    except Exception:
        return int(default)


def _layout_has_block(layout, name: str) -> bool:
    try:
        return layout.has_block(name) and layout.block_size(name) > 0
    except Exception:
        return False


@dataclass(slots=True)
class LayoutDistributed:
    comm: PETSc.Comm
    mgr: object
    base_layout: object

    Nl: int
    Ng: int
    dof_liq: int
    dof_gas: int
    n_if: int

    off_liq: int
    off_gas: int
    off_if: int
    N_total: int
    local_size: int
    global_size: int
    ownership_range: Tuple[int, int]
    ownership_ranges: Optional[Tuple[int, ...]]

    liq_xs: int
    liq_xm: int
    gas_xs: int
    gas_xm: int
    liq_gxs: int
    liq_gxm: int
    gas_gxs: int
    gas_gxm: int
    ghost_width: int

    comp_liq: Dict[str, List[int]]
    comp_gas: Dict[str, List[int]]
    comp_if: Dict[str, List[int]]

    @staticmethod
    def build(comm: PETSc.Comm, mgr, layout) -> "LayoutDistributed":
        Nl = int(getattr(layout, "Nl", mgr.Nl))
        Ng = int(getattr(layout, "Ng", mgr.Ng))
        dof_liq = int(mgr.dof_liq)
        dof_gas = int(mgr.dof_gas)
        n_if = int(mgr.n_if)

        Xg = mgr.dm.createGlobalVec()
        try:
            r0, r1 = mgr.dm.getOwnershipRange()
        except Exception:
            r0, r1 = Xg.getOwnershipRange()
        try:
            global_size = int(mgr.dm.getGlobalSize())
        except Exception:
            global_size = int(Xg.getSize())
        N_total = int(Xg.getSize())
        if global_size != N_total:
            raise ValueError(
                f"LayoutDistributed.build: DM global_size={global_size} != Vec size={N_total}."
            )
        ownership_range = (int(r0), int(r1))
        local_size = int(r1 - r0)
        try:
            ownership_ranges = tuple(int(x) for x in Xg.getOwnershipRanges())
        except Exception:
            ownership_ranges = None

        vL = mgr.dm_liq.createGlobalVec()
        vG = mgr.dm_gas.createGlobalVec()
        vI = mgr.dm_if.createGlobalVec()
        nloc_liq = int(vL.getLocalSize())
        nloc_gas = int(vG.getLocalSize())
        nloc_if = int(vI.getLocalSize())

        if local_size != (nloc_liq + nloc_gas + nloc_if):
            raise RuntimeError(
                f"DMComposite local size mismatch: total={local_size}, "
                f"liq={nloc_liq}, gas={nloc_gas}, if={nloc_if}"
            )
        layout_size = int(getattr(layout, "size", layout.n_dof()))
        if global_size != layout_size:
            raise ValueError(
                f"LayoutDistributed.build: DM global_size={global_size} != layout.size={layout_size}."
            )

        off_liq = int(r0)
        off_gas = int(r0 + nloc_liq)
        off_if = int(r0 + nloc_liq + nloc_gas)

        liq_xs, liq_xm = _dmda_get_corners_1d(mgr.dm_liq)
        gas_xs, gas_xm = _dmda_get_corners_1d(mgr.dm_gas)
        liq_gxs, liq_gxm = _dmda_get_ghost_corners_1d(mgr.dm_liq)
        gas_gxs, gas_gxm = _dmda_get_ghost_corners_1d(mgr.dm_gas)

        ghost_width = max(
            _dmda_get_stencil_width(mgr.dm_liq, default=1),
            _dmda_get_stencil_width(mgr.dm_gas, default=1),
        )

        comp_liq: Dict[str, List[int]] = {}
        comp_gas: Dict[str, List[int]] = {}
        comp_if: Dict[str, List[int]] = {}

        c = 0
        if _layout_has_block(layout, "Tl"):
            comp_liq["Tl"] = [c]
            c += 1
        if _layout_has_block(layout, "Yl"):
            ns = int(getattr(layout, "Ns_l_eff", 0))
            if ns <= 0 and Nl > 0:
                ns = int(layout.block_size("Yl") // Nl)
            if ns > 0:
                comp_liq["Yl"] = list(range(c, c + ns))
                c += ns

        c = 0
        if _layout_has_block(layout, "Tg"):
            comp_gas["Tg"] = [c]
            c += 1
        if _layout_has_block(layout, "Yg"):
            ns = int(getattr(layout, "Ns_g_eff", 0))
            if ns <= 0 and Ng > 0:
                ns = int(layout.block_size("Yg") // Ng)
            if ns > 0:
                comp_gas["Yg"] = list(range(c, c + ns))
                c += ns

        c = 0
        for b in ("Ts", "mpp", "Rd"):
            if _layout_has_block(layout, b):
                comp_if[b] = [c]
                c += 1

        return LayoutDistributed(
            comm=comm,
            mgr=mgr,
            base_layout=layout,
            Nl=Nl,
            Ng=Ng,
            dof_liq=dof_liq,
            dof_gas=dof_gas,
            n_if=n_if,
            off_liq=off_liq,
            off_gas=off_gas,
            off_if=off_if,
            N_total=N_total,
            local_size=local_size,
            global_size=global_size,
            ownership_range=ownership_range,
            ownership_ranges=ownership_ranges,
            liq_xs=liq_xs,
            liq_xm=liq_xm,
            gas_xs=gas_xs,
            gas_xm=gas_xm,
            liq_gxs=liq_gxs,
            liq_gxm=liq_gxm,
            gas_gxs=gas_gxs,
            gas_gxm=gas_gxm,
            ghost_width=ghost_width,
            comp_liq=comp_liq,
            comp_gas=comp_gas,
            comp_if=comp_if,
        )

    def owned_global_indices(self, block: str) -> np.ndarray:
        layout = self.base_layout
        if not _layout_has_block(layout, block):
            return np.empty(0, dtype=PETSc.IntType)

        if block in self.comp_liq:
            comps = self.comp_liq[block]
            ii = np.arange(0, self.liq_xm, dtype=np.int64)
            parts = [self.off_liq + self.dof_liq * ii + int(comp) for comp in comps]
            gidx = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
            return gidx.astype(PETSc.IntType, copy=False)

        if block in self.comp_gas:
            comps = self.comp_gas[block]
            ii = np.arange(0, self.gas_xm, dtype=np.int64)
            parts = [self.off_gas + self.dof_gas * ii + int(comp) for comp in comps]
            gidx = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
            return gidx.astype(PETSc.IntType, copy=False)

        if block in self.comp_if:
            if self.comm.getRank() != 0:
                return np.empty(0, dtype=PETSc.IntType)
            j = int(self.comp_if[block][0])
            return np.array([self.off_if + j], dtype=PETSc.IntType)

        return np.empty(0, dtype=PETSc.IntType)

    def build_is_for_block(self, block: str) -> PETSc.IS:
        idx = self.owned_global_indices(block)
        return PETSc.IS().createGeneral(idx, comm=self.comm)

    def build_is_petsc(self, groups: Dict[str, List[str]], name: str) -> PETSc.IS:
        if name not in groups:
            raise KeyError(f"Unknown group '{name}', available: {list(groups.keys())}")
        parts: List[np.ndarray] = []
        for b in groups[name]:
            a = self.owned_global_indices(b)
            if a is None:
                continue
            a = np.asarray(a, dtype=PETSc.IntType).ravel()
            if a.size > 0:
                parts.append(a)
        if not parts:
            return PETSc.IS().createGeneral([], comm=self.comm)
        idx = np.concatenate(parts)
        idx = np.unique(idx).astype(PETSc.IntType, copy=False)
        return PETSc.IS().createGeneral(idx.tolist(), comm=self.comm)
