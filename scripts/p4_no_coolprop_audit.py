#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


PATTERN = re.compile(r"(CoolProp|PropsSI|AbstractState|HEOS::|coolprop)")

EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    "out",
    "outputs",
    "results",
    "data",
    "datasets",
}

ALLOW_PREFIXES = (
    "docs/",
    "tools/fit_coolprop/",
    "tools/audit_",
    "code/tests/",
)


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def _rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_allowed(path: Path, root: Path) -> bool:
    rel = _rel_posix(path, root)
    return any(rel.startswith(prefix) for prefix in ALLOW_PREFIXES)


def _run(cmd: List[str]) -> Tuple[int, str]:
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    msg = out if out else err
    return proc.returncode, msg


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _is_excluded(path):
            continue
        yield path


def _scan_files(root: Path) -> Tuple[int, int, List[dict]]:
    allowed = 0
    blocked = 0
    hits: List[dict] = []
    for path in _iter_files(root):
        rel = _rel_posix(path, root)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not PATTERN.search(line):
                continue
            item = {
                "path": rel,
                "line": line_no,
                "text": line.strip(),
                "allowed": _is_allowed(path, root),
            }
            hits.append(item)
            if item["allowed"]:
                allowed += 1
            else:
                blocked += 1
    return allowed, blocked, hits


def main() -> int:
    ap = argparse.ArgumentParser(description="P4-E no-CoolProp audit")
    ap.add_argument("--root", default=".", help="Repo root")
    ap.add_argument("--out", default="out/p4_no_coolprop_audit", help="Output directory")
    ap.add_argument("--skip-uninstall", action="store_true", help="Skip pip uninstall CoolProp")
    ap.add_argument("--skip-compile", action="store_true", help="Skip python -m compileall")
    ap.add_argument("--skip-pytest", action="store_true", help="Skip pytest -q")
    ap.add_argument("--skip-scan", action="store_true", help="Skip string scan")
    ap.add_argument("--pytest-args", default="", help="Extra pytest args (quoted)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = {}

    if args.skip_uninstall:
        steps["pip_uninstall"] = {"ran": False, "return_code": 0, "message": "skipped"}
    else:
        rc, msg = _run([sys.executable, "-m", "pip", "uninstall", "-y", "CoolProp"])
        steps["pip_uninstall"] = {"ran": True, "return_code": rc, "message": msg}

    if args.skip_compile:
        steps["compileall"] = {"ran": False, "return_code": 0, "message": "skipped"}
    else:
        rc, msg = _run([sys.executable, "-m", "compileall", str(root)])
        steps["compileall"] = {"ran": True, "return_code": rc, "message": msg}

    if args.skip_pytest:
        steps["pytest"] = {"ran": False, "return_code": 0, "message": "skipped"}
    else:
        pytest_cmd = [sys.executable, "-m", "pytest", "-q"]
        if args.pytest_args:
            pytest_cmd.extend(args.pytest_args.split())
        rc, msg = _run(pytest_cmd)
        steps["pytest"] = {"ran": True, "return_code": rc, "message": msg}

    allowed_hits = 0
    blocked_hits = 0
    hits: List[dict] = []
    if args.skip_scan:
        steps["scan"] = {"ran": False, "return_code": 0, "message": "skipped"}
    else:
        allowed_hits, blocked_hits, hits = _scan_files(root)
        steps["scan"] = {"ran": True, "return_code": 0, "message": "ok"}

    ok = True
    for step in ("pip_uninstall", "compileall", "pytest"):
        info = steps.get(step, {})
        if info.get("ran") and info.get("return_code", 0) != 0:
            ok = False
    if not args.skip_scan and blocked_hits > 0:
        ok = False

    report = {
        "ok": ok,
        "root": str(root),
        "pattern": PATTERN.pattern,
        "allow_prefixes": list(ALLOW_PREFIXES),
        "excluded_dirs": sorted(EXCLUDE_DIRS),
        "steps": steps,
        "scan": {
            "allowed_hits": allowed_hits,
            "blocked_hits": blocked_hits,
            "hits": hits,
        },
    }

    out_path = out_dir / "p4_no_coolprop_audit.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
