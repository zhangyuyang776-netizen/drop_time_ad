from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

bootstrap_mpi_before_petsc()

from petsc4py import PETSc


@dataclass(slots=True)
class DMManager:
    comm: PETSc.Comm
    dm: PETSc.DM
    dm_liq: PETSc.DM
    dm_gas: PETSc.DM
    dm_if: PETSc.DM
    n_if: int
    dof_liq: int
    dof_gas: int
    Nl: int
    Ng: int


@contextmanager
def _dmcomposite_access(dm, X):
    acc = dm.getAccess(X)
    if hasattr(acc, "__enter__"):
        with acc as parts:
            yield parts
        return
    parts = acc
    try:
        yield parts
    finally:
        dm.restoreAccess(X, parts)


def _get_mpi_comm(comm: PETSc.Comm):
    try:
        from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

        bootstrap_mpi_before_petsc()
        from mpi4py import MPI
        return MPI, comm.tompi4py()
    except Exception as exc:
        if comm.getSize() > 1:
            raise RuntimeError("mpi4py is required for interface collectives in MPI mode.") from exc
        return None, None


def _iface_bcast_from_root(comm: PETSc.Comm, buf_root: np.ndarray, n_if: int) -> np.ndarray:
    if n_if == 0:
        return np.empty(0, dtype=np.float64)
    MPI, mpicomm = _get_mpi_comm(comm)
    if mpicomm is None:
        return np.asarray(buf_root, dtype=np.float64)
    buf = np.empty(n_if, dtype=np.float64)
    if mpicomm.rank == 0:
        buf[:] = np.asarray(buf_root, dtype=np.float64)
    mpicomm.Bcast([buf, MPI.DOUBLE], root=0)
    return buf


def _iface_allreduce_sum(comm: PETSc.Comm, local: np.ndarray) -> np.ndarray:
    if local.size == 0:
        return np.empty(0, dtype=np.float64)
    MPI, mpicomm = _get_mpi_comm(comm)
    if mpicomm is None:
        return np.asarray(local, dtype=np.float64)
    loc = np.asarray(local, dtype=np.float64)
    out = np.empty_like(loc)
    mpicomm.Allreduce([loc, MPI.DOUBLE], [out, MPI.DOUBLE], op=MPI.SUM)
    return out


def _layout_has_block(layout, name: str) -> bool:
    try:
        return layout.has_block(name) and layout.block_size(name) > 0
    except Exception:
        return False


def _get_counts(cfg, layout) -> Tuple[int, int, int, int, int]:
    if layout is not None:
        Nl = int(getattr(layout, "Nl", getattr(cfg.geometry, "N_liq")))
        Ng = int(getattr(layout, "Ng", getattr(cfg.geometry, "N_gas")))

        dof_liq = 0
        if _layout_has_block(layout, "Tl"):
            dof_liq += 1
        if _layout_has_block(layout, "Yl"):
            dof_liq += int(getattr(layout, "Ns_l_eff", 0))

        dof_gas = 0
        if _layout_has_block(layout, "Tg"):
            dof_gas += 1
        if _layout_has_block(layout, "Yg"):
            dof_gas += int(getattr(layout, "Ns_g_eff", 0))

        n_if = 0
        for b in ("Ts", "mpp", "Rd"):
            if _layout_has_block(layout, b):
                n_if += 1
        return Nl, Ng, dof_liq, dof_gas, n_if

    Nl = int(cfg.geometry.N_liq)
    Ng = int(cfg.geometry.N_gas)
    dof_liq = int(bool(cfg.physics.solve_Tl))
    dof_gas = int(bool(cfg.physics.solve_Tg))

    if cfg.physics.solve_Yl:
        dof_liq += max(0, len(cfg.species.liq_species) - 1)
    if cfg.physics.solve_Yg:
        dof_gas += max(0, len(cfg.species.gas_species_full) - 1)

    n_if = int(bool(cfg.physics.include_Ts)) + int(bool(cfg.physics.include_mpp)) + int(bool(cfg.physics.include_Rd))
    return Nl, Ng, dof_liq, dof_gas, n_if


def _create_dmda_1d(comm, n: int, dof: int, sw: int = 1) -> PETSc.DM:
    if n <= 0:
        raise ValueError(f"DMDA requires positive grid size, got n={n}")
    if dof <= 0:
        raise ValueError(f"DMDA requires positive dof per cell, got dof={dof}")
    if comm.getSize() > n:
        raise ValueError(
            f"DMDA setup fails: n={n} < nproc={comm.getSize()} (More processes than data points). "
            "Increase Nl/Ng in MPI tests."
        )

    dm = PETSc.DMDA().create(
        sizes=[n],
        dof=dof,
        stencil_width=sw,
        boundary_type=(PETSc.DMDA.BoundaryType.NONE,),
        stencil_type=PETSc.DMDA.StencilType.BOX,
        comm=comm,
    )
    dm.setUp()
    return dm


def _create_interface_shell_dm(comm, n_if: int) -> PETSc.DM:
    if n_if < 0:
        raise ValueError(f"Interface dof must be >= 0, got n_if={n_if}")

    dm_if = PETSc.DMShell().create(comm=comm)

    def _create_global_vec(dm):
        loc = n_if if comm.getRank() == 0 else 0
        v = PETSc.Vec().createMPI((loc, n_if), comm=comm)
        v.setBlockSize(1)
        v.setUp()
        return v

    def _create_local_vec(dm):
        v = PETSc.Vec().createSeq(n_if, comm=PETSc.COMM_SELF)
        v.setBlockSize(1)
        v.setUp()
        return v

    dm_if.setCreateGlobalVector(_create_global_vec)
    dm_if.setCreateLocalVector(_create_local_vec)

    def _is_add_mode(mode) -> bool:
        try:
            return mode == PETSc.InsertMode.ADD_VALUES or int(mode) == int(PETSc.InsertMode.ADD_VALUES)
        except Exception:
            return False

    def _g2l_begin(dm, Xg, mode, Xl):
        if n_if == 0:
            return
        xl = Xl.getArray()
        if comm.getRank() == 0:
            try:
                xl[:] = np.asarray(Xg.getArray(readonly=True), dtype=np.float64)
            except TypeError:
                xl[:] = np.asarray(Xg.getArray(), dtype=np.float64)
        else:
            xl[:] = 0.0

    def _g2l_end(dm, Xg, mode, Xl):
        return

    def _l2g_begin(dm, Xl, mode, Xg):
        if n_if == 0 or comm.getRank() != 0:
            return

        try:
            xl = np.asarray(Xl.getArray(readonly=True), dtype=np.float64)
        except TypeError:
            xl = np.asarray(Xl.getArray(), dtype=np.float64)

        xg = Xg.getArray()
        if xg.size != xl.size:
            raise ValueError(f"Interface global vec size mismatch: {xg.size} vs {xl.size}")
        if _is_add_mode(mode):
            xg[:] += xl
        else:
            xg[:] = xl

    def _l2g_end(dm, Xl, mode, Xg):
        return

    dm_if.setGlobalToLocal(_g2l_begin, _g2l_end)
    dm_if.setLocalToGlobal(_l2g_begin, _l2g_end)

    dm_if.setUp()
    return dm_if


def build_dm(cfg, layout, comm: Optional[PETSc.Comm] = None) -> DMManager:
    comm = PETSc.COMM_WORLD if comm is None else comm
    Nl, Ng, dof_liq, dof_gas, n_if = _get_counts(cfg, layout)

    dm_liq = _create_dmda_1d(comm, Nl, dof_liq, sw=1)
    dm_gas = _create_dmda_1d(comm, Ng, dof_gas, sw=1)
    dm_if = _create_interface_shell_dm(comm, n_if)

    dm = PETSc.DMComposite().create(comm=comm)
    dm.addDM(dm_liq)
    dm.addDM(dm_gas)
    dm.addDM(dm_if)
    dm.setUp()

    return DMManager(
        comm=comm,
        dm=dm,
        dm_liq=dm_liq,
        dm_gas=dm_gas,
        dm_if=dm_if,
        n_if=n_if,
        dof_liq=dof_liq,
        dof_gas=dof_gas,
        Nl=Nl,
        Ng=Ng,
    )


def create_global_vec(mgr: DMManager) -> PETSc.Vec:
    return mgr.dm.createGlobalVec()


def create_local_vecs(mgr: DMManager):
    liq = mgr.dm_liq.createLocalVec()
    gas = mgr.dm_gas.createLocalVec()
    iface = mgr.dm_if.createLocalVec()
    return liq, gas, iface


def global_to_local(mgr: DMManager, Xg: PETSc.Vec):
    comm = mgr.comm
    n_if = int(mgr.n_if)
    with _dmcomposite_access(mgr.dm, Xg) as (X_liq, X_gas, X_if):
        Xl_liq = mgr.dm_liq.createLocalVec()
        Xl_gas = mgr.dm_gas.createLocalVec()
        mgr.dm_liq.globalToLocal(X_liq, Xl_liq, addv=PETSc.InsertMode.INSERT_VALUES)
        mgr.dm_gas.globalToLocal(X_gas, Xl_gas, addv=PETSc.InsertMode.INSERT_VALUES)
        Xl_if = mgr.dm_if.createLocalVec()
        if n_if > 0:
            if comm.getRank() == 0:
                try:
                    buf_root = np.asarray(X_if.getArray(readonly=True), dtype=np.float64).copy()
                except TypeError:
                    buf_root = np.asarray(X_if.getArray(), dtype=np.float64).copy()
            else:
                buf_root = np.empty(n_if, dtype=np.float64)
            buf = _iface_bcast_from_root(comm, buf_root, n_if)
            xl = Xl_if.getArray()
            xl[:n_if] = buf
    return Xl_liq, Xl_gas, Xl_if


def local_to_global_add(
    mgr: DMManager,
    Fl_liq: PETSc.Vec,
    Fl_gas: PETSc.Vec,
    F_if: PETSc.Vec,
) -> PETSc.Vec:
    comm = mgr.comm
    n_if = int(mgr.n_if)
    Fg = mgr.dm.createGlobalVec()
    Fg.set(0.0)

    with _dmcomposite_access(mgr.dm, Fg) as (F_liq, F_gas, F_if_g):
        mgr.dm_liq.localToGlobal(Fl_liq, F_liq, addv=PETSc.InsertMode.ADD_VALUES)
        mgr.dm_gas.localToGlobal(Fl_gas, F_gas, addv=PETSc.InsertMode.ADD_VALUES)
        if n_if > 0:
            try:
                f_loc = np.asarray(F_if.getArray(readonly=True), dtype=np.float64)
            except TypeError:
                f_loc = np.asarray(F_if.getArray(), dtype=np.float64)
            f_sum = _iface_allreduce_sum(comm, f_loc)
            if comm.getRank() == 0:
                xg = F_if_g.getArray()
                if xg.size != f_sum.size:
                    raise ValueError(f"Interface global vec size mismatch: {xg.size} vs {f_sum.size}")
                xg[:n_if] += f_sum
    return Fg


def local_state_to_global(
    mgr: DMManager,
    Xl_liq: PETSc.Vec,
    Xl_gas: PETSc.Vec,
    Xl_if: PETSc.Vec,
) -> PETSc.Vec:
    def _as_scalar(val) -> int:
        if isinstance(val, (tuple, list, np.ndarray)):
            if len(val) == 0:
                raise RuntimeError("Unexpected empty DMDA corners entry.")
            val = val[0]
        return int(val)

    def _corners_1d(dm) -> Tuple[int, int]:
        c = dm.getCorners()
        if isinstance(c, tuple):
            if len(c) == 2:
                return _as_scalar(c[0]), _as_scalar(c[1])
            if len(c) >= 4:
                return _as_scalar(c[0]), _as_scalar(c[3])
        raise RuntimeError(f"Unexpected DMDA.getCorners() return: {c}")

    comm = mgr.comm
    n_if = int(mgr.n_if)
    Xg = mgr.dm.createGlobalVec()
    Xg.set(0.0)

    with _dmcomposite_access(mgr.dm, Xg) as (X_liq_g, X_gas_g, X_if_g):
        xs, xm = _corners_1d(mgr.dm_liq)
        aL = mgr.dm_liq.getVecArray(Xl_liq)
        aG = mgr.dm_liq.getVecArray(X_liq_g)
        for i in range(xs, xs + xm):
            try:
                aG[i, :] = aL[i, :]
            except Exception:
                aG[i] = aL[i]

        xs, xm = _corners_1d(mgr.dm_gas)
        aL = mgr.dm_gas.getVecArray(Xl_gas)
        aG = mgr.dm_gas.getVecArray(X_gas_g)
        for i in range(xs, xs + xm):
            try:
                aG[i, :] = aL[i, :]
            except Exception:
                aG[i] = aL[i]

        if n_if > 0 and comm.getRank() == 0:
            X_if_g.getArray()[:n_if] = Xl_if.getArray()[:n_if]

    return Xg
