from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _parse_case_id_and_out_root(cfg_path: Path, override_out: str | None) -> tuple[str, Path]:
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    case_id = str(raw.get("case", {}).get("id", "case"))
    base = cfg_path.parent
    out_raw = override_out if override_out is not None else raw.get("paths", {}).get("output_root", str(ROOT / "out"))
    out_root = _resolve_path(base, out_raw)
    return case_id, out_root


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate P3 acceptance report (PETSc MPI n=1 vs n=4).")
    p.add_argument(
        "cases",
        nargs="+",
        help="Case YAML(s) used for the runs (e.g. cases/p3_accept_single_petsc_mpi_schur.yaml).",
    )
    p.add_argument("--out", default=None, help="Override output root (default: paths.output_root from YAML).")
    p.add_argument("--n-serial", type=int, default=1, help="MPI size for the serial baseline (default: 1).")
    p.add_argument("--n-parallel", type=int, default=4, help="MPI size for the parallel run (default: 4).")

    # Tolerances for 'small scalar error' checks (default: conservative / engineering-friendly).
    p.add_argument("--tol-Ts-abs", type=float, default=1.0e-3, help="Max abs Ts diff (K).")
    p.add_argument("--tol-Rd-rel", type=float, default=1.0e-6, help="Max rel Rd diff (-).")
    p.add_argument(
        "--tol-mass-rel",
        type=float,
        default=1.0e-6,
        help="Max rel mass_closure_err diff (normalized by m_drop0).",
    )
    p.add_argument("--speedup-min", type=float, default=1.2, help="Minimum speedup to claim '明显加速'.")
    return p.parse_args(argv)


def _find_latest_run_dir(base_dir: Path, nproc: int) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Missing base dir: {base_dir}")
    prefix = f"n{int(nproc)}"
    candidates: list[Path] = []
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith(prefix):
            continue
        summary = p / "summary.json"
        scalars = p / "scalars" / "scalars.csv"
        if summary.exists() and scalars.exists():
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No completed runs found under {base_dir} for {prefix}*")
    candidates.sort(key=lambda d: (d / "summary.json").stat().st_mtime, reverse=True)
    return candidates[0]


def _read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_scalars(path: Path) -> dict[str, np.ndarray]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}

    cols: dict[str, list[float]] = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k, v in row.items():
            try:
                cols[k].append(float(v))
            except Exception:
                cols[k].append(float("nan"))

    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def _safe_rel(diff: float, denom: float, *, tiny: float = 1.0e-30) -> float:
    d = abs(float(denom))
    return abs(float(diff)) / max(d, tiny)


@dataclass(frozen=True)
class ScalarDiff:
    max_abs: float
    max_rel: float
    final_abs: float
    final_rel: float


def _diff_series(a: np.ndarray, b: np.ndarray, *, rel_denom: np.ndarray | None = None) -> ScalarDiff:
    n = int(min(a.size, b.size))
    if n <= 0:
        return ScalarDiff(math.nan, math.nan, math.nan, math.nan)
    a0 = a[:n]
    b0 = b[:n]
    diff = b0 - a0
    max_abs = float(np.nanmax(np.abs(diff)))
    final_abs = float(abs(diff[-1]))

    denom = a0 if rel_denom is None else rel_denom[:n]
    denom_safe = np.maximum(np.abs(denom), 1.0e-30)
    rel = np.abs(diff) / denom_safe
    max_rel = float(np.nanmax(rel))
    final_rel = float(rel[-1])
    return ScalarDiff(max_abs=max_abs, max_rel=max_rel, final_abs=final_abs, final_rel=final_rel)


def _fmt(x: float, *, fmt: str = ".3e") -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return format(float(x), fmt)


def _fmt_s(x: float, *, fmt: str = ".3f") -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return format(float(x), fmt)


def main() -> int:
    args = _parse_args()
    case_paths = [Path(p).expanduser().resolve() for p in args.cases]
    for p in case_paths:
        if not p.exists():
            print(f"[ERROR] case YAML not found: {p}", file=sys.stderr)
            return 2

    # Assume a single output root for the report.
    case0_id, out_root = _parse_case_id_and_out_root(case_paths[0], args.out)
    for p in case_paths[1:]:
        _, out_root_i = _parse_case_id_and_out_root(p, args.out)
        if out_root_i != out_root:
            print(f"[ERROR] Mixed output_root across cases: {out_root} vs {out_root_i}", file=sys.stderr)
            return 2

    report_dir = out_root / "p3_stage3_serial_vs_parallel"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "p3_stage3_report.md"

    lines: list[str] = []
    lines.append("# P3 阶段验收：PETSc MPI 串行 vs 并行")
    lines.append("")
    lines.append(f"- 生成时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- 串行: `mpiexec -n {int(args.n_serial)}`  并行: `mpiexec -n {int(args.n_parallel)}`")
    lines.append("")

    any_fail = False

    for cfg_path in case_paths:
        case_id, _ = _parse_case_id_and_out_root(cfg_path, args.out)
        base_dir = out_root / "p3_stage3_serial_vs_parallel" / case_id
        try:
            dir_s = _find_latest_run_dir(base_dir, args.n_serial)
            dir_p = _find_latest_run_dir(base_dir, args.n_parallel)
        except Exception as exc:
            lines.append(f"## {case_id}")
            lines.append("")
            lines.append(f"- [ERROR] 找不到完整的 n={args.n_serial}/n={args.n_parallel} 结果目录: {exc}")
            lines.append("")
            any_fail = True
            continue

        sum_s = _read_summary(dir_s / "summary.json")
        sum_p = _read_summary(dir_p / "summary.json")
        t_s = float(sum_s.get("wall_time_sec_max", float("nan")))
        t_p = float(sum_p.get("wall_time_sec_max", float("nan")))
        speedup = t_s / t_p if np.isfinite(t_s) and np.isfinite(t_p) and t_p > 0.0 else float("nan")

        scal_s = _read_scalars(dir_s / "scalars" / "scalars.csv")
        scal_p = _read_scalars(dir_p / "scalars" / "scalars.csv")

        Ts_diff = _diff_series(scal_s.get("Ts", np.array([])), scal_p.get("Ts", np.array([])))
        Rd_diff = _diff_series(scal_s.get("Rd", np.array([])), scal_p.get("Rd", np.array([])))

        m0 = scal_s.get("m_drop0", np.array([]))
        mc_s = scal_s.get("mass_closure_err", np.array([]))
        mc_p = scal_p.get("mass_closure_err", np.array([]))
        # Compare mass_closure_err normalized by m_drop0 (serial baseline).
        if mc_s.size and mc_p.size and m0.size:
            denom = np.maximum(np.abs(m0[: min(m0.size, mc_s.size)]), 1.0e-30)
            mc_rel_s = mc_s[: denom.size] / denom
            mc_rel_p = mc_p[: denom.size] / denom
            mc_rel_diff = _diff_series(mc_rel_s, mc_rel_p, rel_denom=np.ones_like(mc_rel_s))
        else:
            mc_rel_diff = ScalarDiff(math.nan, math.nan, math.nan, math.nan)

        # Acceptance decisions (heuristic thresholds; report always includes raw numbers).
        ok_Ts = np.isfinite(Ts_diff.max_abs) and Ts_diff.max_abs <= float(args.tol_Ts_abs)
        ok_Rd = np.isfinite(Rd_diff.max_rel) and Rd_diff.max_rel <= float(args.tol_Rd_rel)
        ok_mass = np.isfinite(mc_rel_diff.max_abs) and mc_rel_diff.max_abs <= float(args.tol_mass_rel)
        ok_speed = np.isfinite(speedup) and speedup >= float(args.speedup_min)

        if not (ok_Ts and ok_Rd and ok_mass):
            any_fail = True

        lines.append(f"## {case_id}")
        lines.append("")
        lines.append(f"- 配置: `{cfg_path}`")
        lines.append(f"- 串行结果目录: `{dir_s}`")
        lines.append(f"- 并行结果目录: `{dir_p}`")
        lines.append("")
        lines.append("### 性能")
        lines.append(f"- wall_time (n={args.n_serial}): `{_fmt_s(t_s, fmt='.3f')}` s")
        lines.append(f"- wall_time (n={args.n_parallel}): `{_fmt_s(t_p, fmt='.3f')}` s")
        lines.append(f"- speedup = T{args.n_serial}/T{args.n_parallel}: `{_fmt_s(speedup, fmt='.3f')}`  ({'PASS' if ok_speed else 'INFO'})")
        lines.append("")
        lines.append("### 标量误差（并行 vs 串行）")
        lines.append(
            f"- Ts: max_abs=`{_fmt(Ts_diff.max_abs)}` final_abs=`{_fmt(Ts_diff.final_abs)}`  ({'PASS' if ok_Ts else 'FAIL'})"
        )
        lines.append(
            f"- Rd: max_rel=`{_fmt(Rd_diff.max_rel)}` final_rel=`{_fmt(Rd_diff.final_rel)}`  ({'PASS' if ok_Rd else 'FAIL'})"
        )
        lines.append(
            f"- mass_closure_err/m_drop0: max_abs=`{_fmt(mc_rel_diff.max_abs)}` final_abs=`{_fmt(mc_rel_diff.final_abs)}`  ({'PASS' if ok_mass else 'FAIL'})"
        )
        lines.append("")
        lines.append("### 结论（该 case）")
        conclusions: list[str] = []
        if ok_Ts and ok_Rd and ok_mass:
            conclusions.append("关键标量误差小（Ts/Rd/质量闭合差异在阈值内）。")
        else:
            conclusions.append("关键标量误差偏大（需要复查数值稳定性/并行一致性）。")
        if ok_speed:
            conclusions.append(f"在该规模下并行有明显加速（speedup≥{args.speedup_min:g}）。")
        else:
            conclusions.append("在该规模下加速不明显（可能受规模/通信/IO/冷启动影响）。")
        lines.append(f"- {' '.join(conclusions)}")
        lines.append("")

    # Overall conclusion.
    lines.append("## 总结")
    lines.append("")
    if any_fail:
        lines.append("- 结论：存在未满足阈值的项；建议先看各 case 的误差/耗时明细，再决定是否需要加大规模或调整 solver/IO。")
    else:
        lines.append("- 结论：关键标量误差满足阈值；并行在至少一个 case 上有加速空间，满足阶段 P3 验收口径。")
    lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote P3 acceptance report: {report_path}")
    return 0 if not any_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())

