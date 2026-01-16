# -*- coding: utf-8 -*-
"""
Smoke test for the PETSc parallel SNES backend (solvers.petsc_snes_parallel).

特点 / 目的：
- 走和 tests/test_petsc_snes_parallel_defaults_mpi.py 同一条构建路线：
  * 使用该测试文件中的 `_build_case(...)` 构造 cfg/layout/ctx/u0；
  * 使用 parallel.dm_manager.build_dm 和 core.layout_dist.LayoutDistributed；
  * 调用 solvers.petsc_snes_parallel.solve_nonlinear_petsc_parallel 进行一次全局求解。
- 仅用于基础连通性自检：MPI + DMComposite + MFFD + ASM+ILU 这一整条并行 SNES 线路是否工作正常。
- 不加载 step4_2 这样的真实蒸发算例，避免 Rd < 0 等物理异常干扰基础检查。

用法示例：
    mpiexec -n 1 python scripts/smoke_petsc_parallel_snes.py
    mpiexec -n 2 python scripts/smoke_petsc_parallel_snes.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc


def main() -> None:
    # 1. 先做 MPI / PETSc 初始化（和其他 smoke_* 脚本保持一致）
    bootstrap_mpi_before_petsc()

    # 现在才能安全 import petsc4py 和项目内部依赖
    from petsc4py import PETSc  # type: ignore

    # 日志配置：简单在 root logger 上开 INFO
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [rank %(rank)s] %(message)s",
        style="%",
    )
    logger = logging.getLogger(__name__)

    comm = PETSc.COMM_WORLD
    size = comm.getSize()
    rank = comm.getRank()

    # 小工具，把 rank 信息注入 log record，方便区分不同进程输出
    class RankFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.rank = rank
            return True

    logger.addFilter(RankFilter())

    logger.info("=== smoke_petsc_parallel_snes: start (size=%d) ===", size)

    # 2. 复用测试里的 _build_case，避免在脚本里重复造轮子
    #    这里显式依赖 tests/ 目录：仅用于 smoke 自检，没有物理意义
    from tests.test_petsc_snes_parallel_defaults_mpi import (  # type: ignore
        _build_case,
    )

    # 每个 rank 使用独立的临时输出目录，结构与测试保持一致
    out_root = Path("out") / "smoke_petsc_parallel_snes"
    out_root.mkdir(parents=True, exist_ok=True)
    tmp_rank = out_root / f"rank_{rank:03d}"
    tmp_rank.mkdir(parents=True, exist_ok=True)

    # _build_case(tmp_path, size, rank) -> (cfg, layout, ctx, u0)
    cfg, layout, ctx, u0 = _build_case(tmp_rank, size, rank)

    # 保险起见，强制并行 SNES 后端要求的一些配置（即使测试中已设置）
    if not hasattr(cfg, "petsc"):
        raise RuntimeError("cfg.petsc is missing; _build_case did not produce PETSc config.")

    # 并行后端目前只支持矩阵自由 jacobian_mode="mf"
    cfg.petsc.jacobian_mode = "mf"

    # 确保非线性求解开关是开的
    if not hasattr(cfg, "nonlinear"):
        raise RuntimeError("cfg.nonlinear is missing; _build_case did not produce nonlinear config.")
    cfg.nonlinear.enabled = True

    # 3. 构造 DM / LayoutDistributed，写回 ctx.meta
    from parallel.dm_manager import build_dm  # type: ignore
    from core.layout_dist import LayoutDistributed  # type: ignore

    dm_mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, dm_mgr, layout)

    # 填充 meta，和测试里保持一致
    ctx.meta.setdefault("dm", dm_mgr.dm)
    ctx.meta.setdefault("dm_manager", dm_mgr)
    ctx.meta.setdefault("layout_dist", ld)

    # 4. 调用并行 SNES 求解
    from solvers.petsc_snes_parallel import (  # type: ignore
        solve_nonlinear_petsc_parallel,
    )

    logger.info("Calling solve_nonlinear_petsc_parallel ...")
    result = solve_nonlinear_petsc_parallel(ctx, u0)

    # 5. 输出收敛信息（只在 rank 0 上打印即可）
    if rank == 0:
        diag = result.diag
        logger.info(
            "SNES parallel smoke done: converged=%s, reason=%s, n_iter=%d, "
            "||F||_2=%.3e, ||F||_inf=%.3e, ksp_its_total=%d",
            diag.converged,
            diag.extra.get("snes_reason", None),
            diag.n_iter,
            diag.res_norm_2,
            diag.res_norm_inf,
            diag.extra.get("ksp_its_total", -1),
        )

        # 简单给个非 0 退出判据（方便 CI 脚本）
        if not diag.converged:
            raise SystemExit(1)

    # 所有进程同步一下再结束
    comm.Barrier()
    if rank == 0:
        logger.info("=== smoke_petsc_parallel_snes: all ranks finished ===")


if __name__ == "__main__":
    main()
