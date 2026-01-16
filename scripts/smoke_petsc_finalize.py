from __future__ import annotations

from parallel.mpi_bootstrap import bootstrap_mpi

bootstrap_mpi()

from petsc4py import PETSc

comm = PETSc.COMM_WORLD
comm.Barrier()
print(f"rank {comm.getRank()} ok", flush=True)
comm.Barrier()
