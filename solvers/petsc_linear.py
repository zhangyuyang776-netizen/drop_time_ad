"""
PETSc-based linear solver backend (mirrors SciPy interface).

Design goals:
- Pure PETSc/NumPy (no SciPy dependency).
- API mirrors SciPy backend: returns LinearSolveResult.
- Strict shape checks; no state/layout mutations.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, Optional

import numpy as np

from core.types import CaseConfig
from solvers.linear_types import LinearSolveResult

logger = logging.getLogger(__name__)

# Track options injected by this module to allow YAML overrides without
# clobbering CLI-provided options in subsequent calls.
_INJECTED_PETSC_OPTIONS: Dict[str, str] = {}


def _cfg_get(obj, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalize_pc_type(pc_type: Optional[str]) -> Optional[str]:
    if pc_type is None:
        return None
    val = str(pc_type).strip().lower()
    if val in ("blockjacobi", "block_jacobi", "block-jacobi"):
        return "bjacobi"
    return val


def _configure_pc_asm_or_bjacobi_subksps(
    pc,
    *,
    sub_ksp_type: Optional[str],
    sub_pc_type: Optional[str],
    sub_ksp_rtol: Optional[float] = None,
    sub_ksp_atol: Optional[float] = None,
    sub_ksp_max_it: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Optional[int]:
    pctype = _normalize_pc_type(pc.getType())
    if pctype not in ("asm", "bjacobi"):
        return None

    if pctype == "asm" and overlap is not None:
        try:
            pc.setASMOverlap(int(overlap))
        except Exception:
            pass
    try:
        pc.setUp()
    except Exception:
        pass

    sub = None
    if pctype == "asm":
        try:
            sub = pc.getASMSubKSP()
        except Exception:
            sub = None
    else:
        for name in ("getBJACOBISubKSP", "getBJacobiSubKSP"):
            if hasattr(pc, name):
                try:
                    sub = getattr(pc, name)()
                except Exception:
                    sub = None
                break

    if sub is None:
        return None

    if isinstance(sub, tuple) and len(sub) == 2:
        _, subksps = sub
    else:
        subksps = sub

    for subksp in subksps:
        if sub_ksp_type:
            try:
                subksp.setType(str(sub_ksp_type))
            except Exception:
                pass
        if sub_ksp_rtol is not None or sub_ksp_atol is not None or sub_ksp_max_it is not None:
            try:
                subksp.setTolerances(rtol=sub_ksp_rtol, atol=sub_ksp_atol, max_it=sub_ksp_max_it)
            except Exception:
                pass
        try:
            spc = subksp.getPC()
            if sub_pc_type:
                spc.setType(str(sub_pc_type))
        except Exception:
            pass
        try:
            subksp.setFromOptions()
        except Exception:
            pass
        try:
            subksp.setUp()
        except Exception:
            pass

    return len(subksps)


def _get_pc_use_amat(pc) -> Optional[bool]:
    if hasattr(pc, "getUseAmat"):
        try:
            return bool(pc.getUseAmat())
        except Exception:
            return None
    return None


def _disable_pc_use_amat(pc, ksp_prefix: str, PETSc) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "setUseAmat": None,
        "opt_pc_use_amat": None,
        "setFromOptions": None,
        "uses_amat_after": None,
    }
    if hasattr(pc, "setUseAmat"):
        try:
            pc.setUseAmat(False)
            info["setUseAmat"] = "ok"
        except Exception as exc:
            info["setUseAmat"] = f"error:{type(exc).__name__}"
    try:
        opts = PETSc.Options(ksp_prefix or "")
        try:
            opts.setValue("pc_use_amat", "0")
        except Exception:
            opts["pc_use_amat"] = "0"
        info["opt_pc_use_amat"] = "0"
        try:
            pc.setFromOptions()
            info["setFromOptions"] = "ok"
        except Exception as exc:
            info["setFromOptions"] = f"error:{type(exc).__name__}"
    except Exception as exc:
        info["opt_pc_use_amat"] = f"error:{type(exc).__name__}"
    info["uses_amat_after"] = _get_pc_use_amat(pc)
    return info


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _opts_has(opts, key: str) -> bool:
    has_name = getattr(opts, "hasName", None)
    if callable(has_name):
        try:
            return bool(has_name(key))
        except Exception:
            pass
    getter = getattr(opts, "getString", None)
    if callable(getter):
        try:
            return getter(key, default="") not in (None, "")
        except Exception:
            pass
    try:
        _ = opts[key]
        return True
    except Exception:
        return False


def _opts_set_if_absent(opts, key: str, value: Any) -> bool:
    if _opts_has(opts, key):
        return False
    setter = getattr(opts, "setValue", None)
    if callable(setter):
        setter(key, str(value))
    else:
        opts[key] = str(value)
    return True


def _opts_get_str(opts, key: str) -> Optional[str]:
    for getter in ("getString", "getInt", "getBool"):
        g = getattr(opts, getter, None)
        if callable(g):
            try:
                v = g(key, default="")
                if v is None:
                    return None
                s = str(v)
                return s if s != "" else None
            except Exception:
                pass
    try:
        v = opts[key]
        return str(v) if v is not None else None
    except Exception:
        return None


def _opts_set_guarded(
    opts,
    *,
    ksp_prefix: str,
    key: str,
    value: Any,
    diag_injected: Dict[str, str],
    diag_skipped: Dict[str, Any],
    diag_overwritten: Optional[Dict[str, Any]] = None,
) -> None:
    full = f"{ksp_prefix}{key}" if ksp_prefix else key

    if not _opts_has(opts, key):
        setter = getattr(opts, "setValue", None)
        if callable(setter):
            setter(key, str(value))
        else:
            opts[key] = str(value)
        _INJECTED_PETSC_OPTIONS[full] = str(value)
        diag_injected[key] = str(value)
        return

    cur = _opts_get_str(opts, key)
    prev = _INJECTED_PETSC_OPTIONS.get(full, None)

    if prev is not None and cur == prev:
        setter = getattr(opts, "setValue", None)
        if callable(setter):
            setter(key, str(value))
        else:
            opts[key] = str(value)
        _INJECTED_PETSC_OPTIONS[full] = str(value)
        diag_injected[key] = str(value)
        if diag_overwritten is not None:
            diag_overwritten[key] = True
        return

    diag_skipped[key] = True
    if prev is not None and cur != prev:
        _INJECTED_PETSC_OPTIONS.pop(full, None)


def _get_fieldsplit_subksps(pc) -> list:
    sub = None
    try:
        sub = pc.getFieldSplitSubKSP()
    except Exception:
        return []
    if sub is None:
        return []
    if isinstance(sub, tuple) and len(sub) == 2:
        _, subksps = sub
    else:
        subksps = sub
    return list(subksps)


def _is_aij(mat_type: str) -> bool:
    return "aij" in (mat_type or "").lower()


def _is_shell_like(mat_type: str) -> bool:
    t = (mat_type or "").lower()
    return ("shell" in t) or ("mffd" in t) or ("python" in t)


def apply_structured_pc(
    ksp,
    cfg: CaseConfig,
    layout,
    A,
    P,
    *,
    pc_type_override: Optional[str] = None,
) -> Dict[str, Any]:
    from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

    bootstrap_mpi_before_petsc()
    from petsc4py import PETSc

    diag: Dict[str, Any] = {"enabled": False}

    linear_cfg = getattr(getattr(cfg, "solver", None), "linear", None)
    petsc_cfg = getattr(cfg, "petsc", None)
    default_ksp_rtol = float(getattr(petsc_cfg, "rtol", 1.0e-8)) if petsc_cfg is not None else 1.0e-8
    default_ksp_atol = float(getattr(petsc_cfg, "atol", 1.0e-12)) if petsc_cfg is not None else 1.0e-12
    default_ksp_max_it = int(getattr(petsc_cfg, "max_it", 200)) if petsc_cfg is not None else 200

    def _as_float(value, default):
        if value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default

    def _as_int(value, default):
        if value is None:
            return default
        try:
            return int(value)
        except Exception:
            return default

    def _resolve_sub_tols(ksp_type, *, base_rtol, base_atol, base_max_it, cfg_rtol, cfg_atol, cfg_max_it):
        rtol = _as_float(cfg_rtol, base_rtol)
        atol = _as_float(cfg_atol, base_atol)
        max_it = _as_int(cfg_max_it, base_max_it)
        if str(ksp_type).lower() == "preonly":
            if cfg_rtol is None:
                rtol = 0.0
            if cfg_atol is None:
                atol = 0.0
            if cfg_max_it is None:
                max_it = 1
        return rtol, atol, max_it
    pc_type = pc_type_override if pc_type_override is not None else _cfg_get(linear_cfg, "pc_type", None)
    if pc_type is None:
        return diag
    pc_type = _normalize_pc_type(pc_type)
    if pc_type is None:
        return diag
    diag["global"] = {"pc_type": pc_type}

    if pc_type != "fieldsplit":
        pc = ksp.getPC()
        try:
            pc.setType(pc_type)
        except Exception:
            logger.warning("Unknown pc_type='%s', falling back to jacobi", pc_type)
            pc.setType("jacobi")

        if pc_type == "asm":
            asm_overlap = _cfg_get(linear_cfg, "asm_overlap", None)
            if asm_overlap is not None:
                try:
                    asm_overlap = int(asm_overlap)
                    pc.setASMOverlap(asm_overlap)
                    diag["global"]["asm_overlap"] = asm_overlap
                except Exception:
                    pass
        return diag
    if layout is None:
        raise ValueError("fieldsplit pc requires a layout to build split IS.")

    pc = ksp.getPC()
    pc.setType("fieldsplit")
    diag["enabled"] = True
    diag["global"] = {"pc_type": "fieldsplit"}
    diag["ksp_prefix"] = str(ksp.getOptionsPrefix() or "")
    diag["options_prefix"] = diag["ksp_prefix"]

    fs_cfg = _cfg_get(linear_cfg, "fieldsplit", None)
    fs_type = str(_cfg_get(fs_cfg, "type", "additive")).lower()
    split_mode = _cfg_get(fs_cfg, "split_mode", None)
    if split_mode is None:
        split_mode = _cfg_get(fs_cfg, "scheme", "by_layout")
    split_mode = str(split_mode).lower()
    split_plan = "bulk_iface" if fs_type == "schur" else split_mode
    if split_plan not in ("by_layout", "bulk_iface"):
        raise ValueError(f"Unsupported fieldsplit split_mode '{split_plan}'")

    comm = None
    try:
        comm = ksp.getComm()
    except Exception:
        comm = None
    if comm is None and P is not None:
        try:
            comm = P.getComm()
        except Exception:
            comm = None
    if comm is None:
        try:
            comm = A.getComm()
        except Exception:
            comm = None
    comm_size = 1
    if comm is not None:
        try:
            comm_size = int(comm.getSize())
        except Exception:
            pass
    is_parallel = comm_size > 1

    Aop_ksp = None
    Pop_ksp = None
    try:
        Aop_ksp, Pop_ksp = ksp.getOperators()
    except Exception:
        pass

    A_eff = Aop_ksp if Aop_ksp is not None else A
    P_eff = Pop_ksp if Pop_ksp is not None else P

    def _fs_has(name: str) -> bool:
        if fs_cfg is None:
            return False
        if isinstance(fs_cfg, Mapping):
            return name in fs_cfg and fs_cfg.get(name) is not None
        if not hasattr(fs_cfg, name):
            return False
        return getattr(fs_cfg, name) is not None

    def _try_get_ownership_range(mat):
        if mat is None:
            return None
        try:
            r0, r1 = mat.getOwnershipRange()
            return int(r0), int(r1)
        except Exception:
            return None

    ownership_range = None
    ownership_src = None
    for src, mat in (
        ("ksp_pmat", Pop_ksp),
        ("arg_P", P),
        ("ksp_amat", Aop_ksp),
        ("arg_A", A),
    ):
        rng = _try_get_ownership_range(mat)
        if rng is not None:
            ownership_range = rng
            ownership_src = src
            break

    diag["ownership_range_source"] = ownership_src
    diag["ownership_range"] = ownership_range

    comm_for_is = None
    if Pop_ksp is not None:
        try:
            comm_for_is = Pop_ksp.getComm()
        except Exception:
            comm_for_is = None
    if comm_for_is is None and P is not None:
        try:
            comm_for_is = P.getComm()
        except Exception:
            comm_for_is = None
    if comm_for_is is None:
        comm_for_is = comm
    if comm_for_is is None and Aop_ksp is not None:
        try:
            comm_for_is = Aop_ksp.getComm()
        except Exception:
            comm_for_is = None
    if comm_for_is is None and A is not None:
        try:
            comm_for_is = A.getComm()
        except Exception:
            comm_for_is = None

    splits = layout.build_is_petsc(comm=comm_for_is, ownership_range=ownership_range, plan=split_plan)
    diag["splits"] = {k: int(v.getSize()) for k, v in splits.items()}
    diag["split_names"] = list(splits.keys())
    split_sizes_local = {}
    for name, iset in splits.items():
        try:
            split_sizes_local[name] = int(iset.getLocalSize())
        except Exception:
            try:
                split_sizes_local[name] = int(iset.getSize())
            except Exception:
                split_sizes_local[name] = 0
    diag["split_sizes_local"] = split_sizes_local
    diag["fieldsplit"] = {"type": fs_type, "plan": split_plan, "splits": {}}

    pairs = [(str(name), iset) for name, iset in splits.items()]
    try:
        pc.setFieldSplitIS(*pairs)
    except TypeError:
        for name, iset in pairs:
            try:
                pc.setFieldSplitIS(name, iset)
            except TypeError:
                try:
                    pc.setFieldSplitIS(iset, name)
                except TypeError:
                    pc.setFieldSplitIS((name, iset))

    try:
        pc.setFieldSplitType(fs_type)
    except Exception:
        type_map = {
            "additive": PETSc.PC.CompositeType.ADDITIVE,
            "multiplicative": PETSc.PC.CompositeType.MULTIPLICATIVE,
            "schur": PETSc.PC.CompositeType.SCHUR,
        }
        if fs_type in type_map:
            pc.setFieldSplitType(type_map[fs_type])
        else:
            raise

    diag["fieldsplit_type"] = fs_type

    Aop_ksp = None
    Pop_ksp = None
    try:
        Aop_ksp, Pop_ksp = ksp.getOperators()
    except Exception:
        pass

    A_eff = Aop_ksp if Aop_ksp is not None else A
    P_eff = Pop_ksp if Pop_ksp is not None else P

    uses_amat = None
    should_disable_amat = False
    if P_eff is not None:
        if A_eff is None:
            should_disable_amat = True
        else:
            try:
                should_disable_amat = bool(A_eff.handle != P_eff.handle)
            except Exception:
                should_disable_amat = bool(A_eff is not P_eff)

    try:
        A_type = ""
        P_type = ""
        if A_eff is not None:
            try:
                A_type = str(A_eff.getType())
            except Exception:
                A_type = ""
        if P_eff is not None:
            try:
                P_type = str(P_eff.getType())
            except Exception:
                P_type = ""
        diag["outer_ops"] = {
            "A_type": A_type,
            "P_type": P_type,
            "A_from_ksp": Aop_ksp is not None,
            "P_from_ksp": Pop_ksp is not None,
            "should_disable_amat": bool(should_disable_amat),
        }
    except Exception:
        pass
    failfast = (
        _env_flag("DROPLET_PETSC_FAILFAST_FIELDSPLIT")
        or _env_flag("DROPLET_STRICT_FIELDSPLIT_USE_AMAT")
        or _env_flag("DROPLET_PETSC_DEBUG")
    )
    if failfast:
        A_type = str(diag.get("outer_ops", {}).get("A_type", "") or "").lower()
        P_type = str(diag.get("outer_ops", {}).get("P_type", "") or "").lower()
        is_shell = ("shell" in A_type) or ("mffd" in A_type) or ("python" in A_type)
        is_aij = "aij" in P_type

        def _opt_requests_use_amat(opts) -> bool:
            for getter in ("getBool", "getInt", "getString"):
                if hasattr(opts, getter):
                    try:
                        val = getattr(opts, getter)("pc_use_amat", default="")
                        return str(val).strip().lower() in {"1", "true", "yes", "on"}
                    except Exception:
                        pass
            try:
                val = opts["pc_use_amat"]
                return str(val).strip().lower() in {"1", "true", "yes", "on"}
            except Exception:
                return False

        opt_use_amat = False
        try:
            opt_use_amat = _opt_requests_use_amat(PETSc.Options(diag.get("ksp_prefix", "")))
        except Exception:
            opt_use_amat = False
        if not opt_use_amat:
            try:
                opt_use_amat = _opt_requests_use_amat(PETSc.Options())
            except Exception:
                opt_use_amat = False
        if opt_use_amat and is_shell and is_aij:
            raise RuntimeError(
                "[P3.4.1-5 failfast] PCFieldSplit option pc_use_amat=1 while outer A is shell-like and P is AIJ; "
                "this can hang/crash in MatGetSubMatrix. "
                f"ksp_prefix='{diag.get('ksp_prefix','')}' fieldsplit_type='{fs_type}' "
                f"scheme='{split_plan}' A_type='{A_type}' P_type='{P_type}'"
            )

    disable_attempt = None
    if should_disable_amat:
        disable_attempt = _disable_pc_use_amat(pc, diag.get("ksp_prefix", ""), PETSc)
        uses_amat = disable_attempt.get("uses_amat_after")
        diag["disable_amat_attempt"] = disable_attempt
    if uses_amat is None:
        uses_amat = _get_pc_use_amat(pc)
    if uses_amat is None and should_disable_amat:
        uses_amat = False
    if uses_amat is not None:
        diag["uses_amat"] = bool(uses_amat)

    if failfast:
        A_type = str(diag.get("outer_ops", {}).get("A_type", "") or "").lower()
        P_type = str(diag.get("outer_ops", {}).get("P_type", "") or "").lower()
        uses_amat_eff = _get_pc_use_amat(pc)
        if uses_amat_eff is None:
            uses_amat_eff = uses_amat
        is_shell = ("shell" in A_type) or ("mffd" in A_type) or ("python" in A_type)
        is_aij = "aij" in P_type
        if bool(uses_amat_eff) and is_shell and is_aij:
            raise RuntimeError(
                "[P3.4.1-5 failfast] PCFieldSplit uses Amat while outer A is shell-like and P is AIJ; "
                "this can hang/crash in MatGetSubMatrix. "
                f"ksp_prefix='{diag.get('ksp_prefix','')}' fieldsplit_type='{fs_type}' "
                f"scheme='{split_plan}' A_type='{A_type}' P_type='{P_type}' "
                f"disable_attempt={disable_attempt}"
            )

    if fs_type == "schur":
        schur_fact = str(_cfg_get(fs_cfg, "schur_fact_type", "lower")).lower()
        try:
            pc.setFieldSplitSchurFactType(schur_fact)
        except Exception:
            schur_map = {
                "full": PETSc.PC.SchurFactType.FULL,
                "diag": PETSc.PC.SchurFactType.DIAG,
                "lower": PETSc.PC.SchurFactType.LOWER,
                "upper": PETSc.PC.SchurFactType.UPPER,
            }
            if schur_fact in schur_map:
                pc.setFieldSplitSchurFactType(schur_map[schur_fact])
            else:
                raise
        diag["schur_fact_type"] = schur_fact
        diag["uses_amat"] = False

    def _get_subsolvers_cfg(raw_cfg):
        if raw_cfg is None:
            return None
        if isinstance(raw_cfg, Mapping):
            return raw_cfg.get("subsolvers") or raw_cfg.get("subksp")
        sub = getattr(raw_cfg, "subsolvers", None)
        if sub is None:
            sub = getattr(raw_cfg, "subksp", None)
        return sub

    def _normalize_block(raw_block: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        defaults = {
            "ksp_type": "preonly",
            "pc_type": "asm",
            "asm_overlap": 1,
            "asm_sub_pc_type": "ilu",
            "ksp_rtol": None,
            "ksp_max_it": None,
        }
        if raw_block and not isinstance(raw_block, Mapping):
            raw_block = {
                "ksp_type": getattr(raw_block, "ksp_type", None),
                "pc_type": getattr(raw_block, "pc_type", None),
                "asm_overlap": getattr(raw_block, "asm_overlap", None),
                "asm_sub_pc_type": getattr(raw_block, "asm_sub_pc_type", None),
                "ksp_rtol": getattr(raw_block, "ksp_rtol", None),
                "ksp_max_it": getattr(raw_block, "ksp_max_it", None),
            }
        if raw_block:
            for key in defaults:
                if key in raw_block and raw_block[key] is not None:
                    defaults[key] = raw_block[key]
        defaults["ksp_type"] = str(defaults["ksp_type"]).lower()
        defaults["pc_type"] = _normalize_pc_type(defaults["pc_type"]) or "asm"
        defaults["asm_sub_pc_type"] = _normalize_pc_type(defaults["asm_sub_pc_type"]) or "ilu"
        try:
            defaults["asm_overlap"] = int(defaults["asm_overlap"])
        except Exception:
            defaults["asm_overlap"] = 1
        if defaults.get("ksp_rtol") is not None:
            defaults["ksp_rtol"] = float(defaults["ksp_rtol"])
        if defaults.get("ksp_max_it") is not None:
            defaults["ksp_max_it"] = int(defaults["ksp_max_it"])
        return defaults

    sub_cfg = _get_subsolvers_cfg(fs_cfg)
    raw_bulk = None
    raw_iface = None
    if isinstance(sub_cfg, Mapping):
        raw_bulk = sub_cfg.get("bulk")
        raw_iface = sub_cfg.get("iface")
    elif sub_cfg is not None:
        raw_bulk = getattr(sub_cfg, "bulk", None)
        raw_iface = getattr(sub_cfg, "iface", None)

    if fs_type == "schur":
        if raw_bulk is None and _cfg_get(fs_cfg, "bulk_ksp_type", None) is not None:
            raw_bulk = {
                "ksp_type": _cfg_get(fs_cfg, "bulk_ksp_type", None),
                "pc_type": _cfg_get(fs_cfg, "bulk_pc_type", None),
                "asm_sub_pc_type": _cfg_get(fs_cfg, "bulk_pc_asm_sub_pc_type", None),
                "asm_overlap": _cfg_get(fs_cfg, "bulk_pc_asm_overlap", None),
            }
        if raw_iface is None and _cfg_get(fs_cfg, "iface_ksp_type", None) is not None:
            raw_iface = {
                "ksp_type": _cfg_get(fs_cfg, "iface_ksp_type", None),
                "pc_type": _cfg_get(fs_cfg, "iface_pc_type", None),
                "asm_sub_pc_type": _cfg_get(fs_cfg, "iface_pc_asm_sub_pc_type", None),
                "asm_overlap": _cfg_get(fs_cfg, "iface_pc_asm_overlap", None),
            }

        if raw_iface is not None and isinstance(raw_iface, Mapping):
            if raw_iface.get("asm_sub_pc_type") is None:
                if isinstance(raw_iface, dict):
                    raw_iface["asm_sub_pc_type"] = "lu"

        bulk_cfg = _normalize_block(raw_bulk)
        iface_cfg = _normalize_block(raw_iface)
        if iface_cfg["asm_sub_pc_type"] == "ilu":
            iface_cfg["asm_sub_pc_type"] = "lu"

        diag.setdefault("options_injected", {})
        diag.setdefault("options_skipped_cli", {})
        diag.setdefault("options_overwritten", {})

        options_prefix = diag.get("ksp_prefix", "")
        try:
            opts = PETSc.Options(options_prefix or "")
        except Exception:
            opts = PETSc.Options()

        def _inject_block(name: str, block_cfg: Dict[str, Any]) -> None:
            base = f"fieldsplit_{name}_"
            kv = {
                f"{base}ksp_type": block_cfg["ksp_type"],
                f"{base}pc_type": block_cfg["pc_type"],
            }
            if block_cfg.get("ksp_rtol") is not None:
                kv[f"{base}ksp_rtol"] = block_cfg["ksp_rtol"]
            if block_cfg.get("ksp_max_it") is not None:
                kv[f"{base}ksp_max_it"] = block_cfg["ksp_max_it"]
            if block_cfg["pc_type"] in ("asm", "bjacobi"):
                kv[f"{base}pc_asm_overlap"] = block_cfg["asm_overlap"]
                kv[f"{base}sub_ksp_type"] = "preonly"
                kv[f"{base}sub_pc_type"] = block_cfg["asm_sub_pc_type"]

            for key, val in kv.items():
                _opts_set_guarded(
                    opts,
                    ksp_prefix=options_prefix,
                    key=key,
                    value=val,
                    diag_injected=diag["options_injected"],
                    diag_skipped=diag["options_skipped_cli"],
                    diag_overwritten=diag["options_overwritten"],
                )

        _inject_block("bulk", bulk_cfg)
        _inject_block("iface", iface_cfg)

        def _split_info(block_cfg: Dict[str, Any]) -> Dict[str, Any]:
            info = {
                "ksp_type": block_cfg["ksp_type"],
                "pc_type": block_cfg["pc_type"],
                "asm_overlap": block_cfg["asm_overlap"],
                "subdomain_ksp_type": "preonly",
                "subdomain_pc_type": block_cfg["asm_sub_pc_type"],
            }
            if block_cfg.get("ksp_rtol") is not None:
                info["ksp_rtol"] = float(block_cfg["ksp_rtol"])
            if block_cfg.get("ksp_max_it") is not None:
                info["ksp_max_it"] = int(block_cfg["ksp_max_it"])
            return info

        fs_diag = diag.setdefault("fieldsplit", {})
        fs_splits = fs_diag.setdefault("splits", {})
        fs_splits.setdefault("bulk", _split_info(bulk_cfg))
        fs_splits.setdefault("iface", _split_info(iface_cfg))

        diag["sub_defaults"] = {
            "default": {
                "ksp_type": bulk_cfg["ksp_type"],
                "pc_type": bulk_cfg["pc_type"],
                "pc_asm_overlap": bulk_cfg["asm_overlap"],
                "subdomain_ksp_type": "preonly",
                "subdomain_pc_type": bulk_cfg["asm_sub_pc_type"],
            },
            "by_name": {
                "bulk": {
                    "ksp_type": bulk_cfg["ksp_type"],
                    "pc_type": bulk_cfg["pc_type"],
                    "pc_asm_overlap": bulk_cfg["asm_overlap"],
                    "subdomain_ksp_type": "preonly",
                    "subdomain_pc_type": bulk_cfg["asm_sub_pc_type"],
                },
                "iface": {
                    "ksp_type": iface_cfg["ksp_type"],
                    "pc_type": iface_cfg["pc_type"],
                    "pc_asm_overlap": iface_cfg["asm_overlap"],
                    "subdomain_ksp_type": "preonly",
                    "subdomain_pc_type": iface_cfg["asm_sub_pc_type"],
                },
            },
        }
    else:
        if _env_flag("DROPLET_MPI_LINEAR_STRICT"):
            from solvers.mpi_linear_support import assert_mpi_additive_subsolvers_filled

            assert_mpi_additive_subsolvers_filled(cfg, strict=True)

        def _get_subsolvers_cfg(raw_cfg):
            if raw_cfg is None:
                return None
            if isinstance(raw_cfg, Mapping):
                return raw_cfg.get("subsolvers") or raw_cfg.get("subksp")
            sub = getattr(raw_cfg, "subsolvers", None)
            if sub is None:
                sub = getattr(raw_cfg, "subksp", None)
            return sub

        def _normalize_block(raw_block: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
            defaults = {
                "ksp_type": "preonly",
                "pc_type": "asm",
                "asm_overlap": 1,
                "asm_sub_pc_type": "ilu",
                "ksp_rtol": None,
                "ksp_max_it": None,
            }
            if raw_block and not isinstance(raw_block, Mapping):
                raw_block = {
                    "ksp_type": getattr(raw_block, "ksp_type", None),
                    "pc_type": getattr(raw_block, "pc_type", None),
                    "asm_overlap": getattr(raw_block, "asm_overlap", None),
                    "asm_sub_pc_type": getattr(raw_block, "asm_sub_pc_type", None),
                    "ksp_rtol": getattr(raw_block, "ksp_rtol", None),
                    "ksp_max_it": getattr(raw_block, "ksp_max_it", None),
                }
            if raw_block:
                for key in defaults:
                    if key in raw_block and raw_block[key] is not None:
                        defaults[key] = raw_block[key]
            defaults["ksp_type"] = str(defaults["ksp_type"]).lower()
            defaults["pc_type"] = _normalize_pc_type(defaults["pc_type"]) or "asm"
            defaults["asm_sub_pc_type"] = _normalize_pc_type(defaults["asm_sub_pc_type"]) or "ilu"
            try:
                defaults["asm_overlap"] = int(defaults["asm_overlap"])
            except Exception:
                defaults["asm_overlap"] = 1
            if defaults.get("ksp_rtol") is not None:
                defaults["ksp_rtol"] = float(defaults["ksp_rtol"])
            if defaults.get("ksp_max_it") is not None:
                defaults["ksp_max_it"] = int(defaults["ksp_max_it"])
            return defaults

        sub_cfg = _get_subsolvers_cfg(fs_cfg)
        raw_bulk = None
        raw_iface = None
        if isinstance(sub_cfg, Mapping):
            raw_bulk = sub_cfg.get("bulk")
            raw_iface = sub_cfg.get("iface")
        elif sub_cfg is not None:
            raw_bulk = getattr(sub_cfg, "bulk", None)
            raw_iface = getattr(sub_cfg, "iface", None)

        legacy = {
            "ksp_type": _cfg_get(fs_cfg, "sub_ksp_type", None),
            "pc_type": _cfg_get(fs_cfg, "sub_pc_type", None),
            "asm_overlap": _cfg_get(fs_cfg, "sub_pc_asm_overlap", None),
            "asm_sub_pc_type": _cfg_get(fs_cfg, "sub_pc_asm_sub_pc_type", None),
            "ksp_rtol": _cfg_get(fs_cfg, "sub_ksp_rtol", None),
            "ksp_max_it": _cfg_get(fs_cfg, "sub_ksp_max_it", None),
        }
        legacy = {k: v for k, v in legacy.items() if v is not None}
        bulk_cfg = _normalize_block(raw_bulk)
        iface_cfg = _normalize_block(raw_iface)
        if legacy:
            default_cfg = _normalize_block(None)
            if bulk_cfg == default_cfg and iface_cfg == default_cfg:
                bulk_cfg = _normalize_block(legacy)
                iface_cfg = _normalize_block(legacy)

        diag.setdefault("options_injected", {})
        diag.setdefault("options_skipped_cli", {})
        diag.setdefault("options_overwritten", {})

        options_prefix = diag.get("ksp_prefix", "")
        try:
            opts = PETSc.Options(options_prefix or "")
        except Exception:
            opts = PETSc.Options()

        def _inject_block(name: str, block_cfg: Dict[str, Any]) -> None:
            base = f"fieldsplit_{name}_"
            kv = {
                f"{base}ksp_type": block_cfg["ksp_type"],
                f"{base}pc_type": block_cfg["pc_type"],
            }
            if block_cfg.get("ksp_rtol") is not None:
                kv[f"{base}ksp_rtol"] = block_cfg["ksp_rtol"]
            if block_cfg.get("ksp_max_it") is not None:
                kv[f"{base}ksp_max_it"] = block_cfg["ksp_max_it"]
            if block_cfg["pc_type"] in ("asm", "bjacobi"):
                kv[f"{base}pc_asm_overlap"] = block_cfg["asm_overlap"]
                kv[f"{base}sub_ksp_type"] = "preonly"
                kv[f"{base}sub_pc_type"] = block_cfg["asm_sub_pc_type"]

            for key, val in kv.items():
                _opts_set_guarded(
                    opts,
                    ksp_prefix=options_prefix,
                    key=key,
                    value=val,
                    diag_injected=diag["options_injected"],
                    diag_skipped=diag["options_skipped_cli"],
                    diag_overwritten=diag["options_overwritten"],
                )

        _inject_block("bulk", bulk_cfg)
        _inject_block("iface", iface_cfg)

        def _split_info(block_cfg: Dict[str, Any]) -> Dict[str, Any]:
            info = {
                "ksp_type": block_cfg["ksp_type"],
                "pc_type": block_cfg["pc_type"],
                "asm_overlap": block_cfg["asm_overlap"],
                "subdomain_ksp_type": "preonly",
                "subdomain_pc_type": block_cfg["asm_sub_pc_type"],
            }
            if block_cfg.get("ksp_rtol") is not None:
                info["ksp_rtol"] = float(block_cfg["ksp_rtol"])
            if block_cfg.get("ksp_max_it") is not None:
                info["ksp_max_it"] = int(block_cfg["ksp_max_it"])
            return info

        fs_diag = diag.setdefault("fieldsplit", {})
        fs_splits = fs_diag.setdefault("splits", {})
        fs_splits.setdefault("bulk", _split_info(bulk_cfg))
        fs_splits.setdefault("iface", _split_info(iface_cfg))

        diag["sub_defaults"] = {
            "default": {
                "ksp_type": bulk_cfg["ksp_type"],
                "pc_type": bulk_cfg["pc_type"],
                "pc_asm_overlap": bulk_cfg["asm_overlap"],
                "subdomain_ksp_type": "preonly",
                "subdomain_pc_type": bulk_cfg["asm_sub_pc_type"],
            },
            "by_name": {
                "bulk": {
                    "ksp_type": bulk_cfg["ksp_type"],
                    "pc_type": bulk_cfg["pc_type"],
                    "pc_asm_overlap": bulk_cfg["asm_overlap"],
                    "subdomain_ksp_type": "preonly",
                    "subdomain_pc_type": bulk_cfg["asm_sub_pc_type"],
                },
                "iface": {
                    "ksp_type": iface_cfg["ksp_type"],
                    "pc_type": iface_cfg["pc_type"],
                    "pc_asm_overlap": iface_cfg["asm_overlap"],
                    "subdomain_ksp_type": "preonly",
                    "subdomain_pc_type": iface_cfg["asm_sub_pc_type"],
                },
            },
        }

    check_subksp = _env_flag("DROPLET_PETSC_DEBUG") or failfast
    if check_subksp:
        P_type = str(diag.get("outer_ops", {}).get("P_type", "") or "").lower()
        A_type = str(diag.get("outer_ops", {}).get("A_type", "") or "").lower()
        if not P_type and P_eff is not None:
            try:
                P_type = str(P_eff.getType()).lower()
            except Exception:
                P_type = ""
        if not A_type and A_eff is not None:
            try:
                A_type = str(A_eff.getType()).lower()
            except Exception:
                A_type = ""
        if failfast and not _is_aij(P_type):
            raise RuntimeError(
                "[P3.4.3 failfast] Pop must be AIJ/MPIAIJ before PCSetUp in strict mode. "
                f"outer Aop={A_type} Pop={P_type} prefix={diag.get('ksp_prefix','')!r}"
            )
        try:
            pc.setUp()
        except Exception:
            pass
        subksps = _get_fieldsplit_subksps(pc)
        sub_ops = []
        for idx, sksp in enumerate(subksps):
            try:
                As, Ps = sksp.getOperators()
            except Exception:
                continue
            As_t = str(As.getType()).lower() if As is not None else ""
            Ps_t = str(Ps.getType()).lower() if Ps is not None else ""
            sub_ops.append({"index": idx, "Asub_type": As_t, "Psub_type": Ps_t})
            ok_p = _is_aij(Ps_t)
            ok_a = _is_aij(As_t) or (fs_type == "schur" and As_t == "schurcomplement")
            ok_shell = not _is_shell_like(As_t)
            if failfast and ((not ok_p) or (not ok_a) or (not ok_shell)):
                raise RuntimeError(
                    "[P3.4.3 failfast] subKSP operators are unsafe: "
                    f"i={idx} Asub={As_t} Psub={Ps_t}; "
                    f"outer Aop={A_type} Pop={P_type} prefix={diag.get('ksp_prefix','')!r}"
                )
        fs_diag = diag.setdefault("fieldsplit", {})
        fs_diag["subksp_ops"] = sub_ops

    return diag


def apply_fieldsplit_subksp_defaults(ksp, diag: Mapping[str, Any]) -> None:
    if not diag.get("enabled", False):
        return
    fs_type = str(diag.get("fieldsplit_type") or diag.get("fieldsplit", {}).get("type") or "").lower()

    def _fill_splits_from_defaults() -> None:
        fs_diag = diag.setdefault("fieldsplit", {})
        fs_splits = fs_diag.setdefault("splits", {})
        defaults = diag.get("sub_defaults", {})
        if not isinstance(defaults, Mapping):
            return
        by_name = defaults.get("by_name", {}) or {}
        default_cfg = defaults.get("default", None)

        def _fill_split_defaults(name: str, cfg: Mapping[str, Any]) -> None:
            if name in fs_splits:
                return
            entry = dict(cfg)
            pc_type = _normalize_pc_type(entry.get("pc_type", None))
            if pc_type == "asm":
                if "asm_overlap" not in entry:
                    if entry.get("pc_asm_overlap", None) is not None:
                        entry["asm_overlap"] = entry.get("pc_asm_overlap")
                    else:
                        entry["asm_overlap"] = 1
            fs_splits[name] = entry

        if by_name:
            for name, cfg in by_name.items():
                _fill_split_defaults(str(name), cfg)
        elif default_cfg:
            names = diag.get("split_names") or list(diag.get("splits", {}).keys())
            for name in names:
                _fill_split_defaults(str(name), default_cfg)

    if fs_type in ("additive", "schur"):
        _fill_splits_from_defaults()

    if fs_type == "additive":
        if _env_flag("DROPLET_PETSC_DEBUG"):
            fs_diag = diag.setdefault("fieldsplit", {})
            sub_ops = []
            try:
                pc = ksp.getPC()
                if str(pc.getType()).lower() == "fieldsplit":
                    for subksp in _get_fieldsplit_subksps(pc):
                        try:
                            Asub, Psub = subksp.getOperators()
                            sub_ops.append(
                                {
                                    "ksp_type": str(subksp.getType()),
                                    "pc_type": str(subksp.getPC().getType()),
                                    "Asub_type": str(Asub.getType()) if Asub is not None else "",
                                    "Psub_type": str(Psub.getType()) if Psub is not None else "",
                                }
                            )
                        except Exception as exc:
                            sub_ops.append({"error": f"{type(exc).__name__}:{exc}"})
                fs_diag["subksp_ops"] = sub_ops
            except Exception as exc:
                fs_diag["subksp_ops_error"] = f"{type(exc).__name__}:{exc}"
        return

    pc = ksp.getPC()
    try:
        pc.setUp()
    except Exception:
        pass
    sub = None
    err_primary = None
    try:
        sub = pc.getFieldSplitSubKSP()
    except Exception as exc:
        err_primary = exc

    if sub is None:
        A0 = None
        P0 = None
        swapped = False
        err_retry = None
        try:
            A0, P0 = ksp.getOperators()
        except Exception:
            A0 = None
            P0 = None
        try:
            if P0 is not None:
                ksp.setOperators(P0, P0)
                swapped = True
                try:
                    ksp.setUp()
                except Exception:
                    pass
                pc = ksp.getPC()
                sub = pc.getFieldSplitSubKSP()
        except Exception as exc:
            err_retry = exc
        finally:
            if swapped and (A0 is not None) and (P0 is not None):
                try:
                    ksp.setOperators(A0, P0)
                except Exception:
                    pass

        if sub is None:
            fs_diag = diag.setdefault("fieldsplit", {})
            if err_primary is not None or err_retry is not None:
                parts = []
                if err_primary is not None:
                    parts.append(
                        f"getFieldSplitSubKSP failed: {type(err_primary).__name__}: {err_primary}"
                    )
                if err_retry is not None:
                    parts.append(
                        f"retry failed: {type(err_retry).__name__}: {err_retry}"
                    )
                fs_diag["subksp_error"] = "; ".join(parts)
            _fill_splits_from_defaults()
            return

    if isinstance(sub, tuple) and len(sub) == 2:
        names, subksps = sub
    else:
        subksps = sub
        names = diag.get("split_names", None)

    if not names:
        names = list(diag.get("splits", {}).keys())

    defaults = diag.get("sub_defaults", {})
    by_name = defaults.get("by_name", {})
    default_cfg = defaults.get("default", None)
    fs_diag = diag.setdefault("fieldsplit", {})
    fs_splits = fs_diag.setdefault("splits", {})

    for idx, subksp in enumerate(subksps):
        name = names[idx] if idx < len(names) else f"split_{idx}"
        if isinstance(name, bytes):
            name = name.decode()
        name = str(name)
        cfg = by_name.get(name, default_cfg)
        if not cfg:
            continue
        ksp_type = cfg.get("ksp_type", None)
        pc_type = _normalize_pc_type(cfg.get("pc_type", None))
        ksp_rtol = cfg.get("ksp_rtol", None)
        ksp_atol = cfg.get("ksp_atol", None)
        ksp_max_it = cfg.get("ksp_max_it", None)
        asm_overlap = cfg.get("pc_asm_overlap", None)
        if pc_type == "asm" and asm_overlap is None:
            asm_overlap = 1
        subdomain_ksp_type = cfg.get("subdomain_ksp_type", None)
        subdomain_pc_type = _normalize_pc_type(cfg.get("subdomain_pc_type", None))
        subdomain_ksp_rtol = cfg.get("subdomain_ksp_rtol", None)
        subdomain_ksp_atol = cfg.get("subdomain_ksp_atol", None)
        subdomain_ksp_max_it = cfg.get("subdomain_ksp_max_it", None)
        n_sub = None
        spc = None
        if ksp_type:
            try:
                subksp.setType(str(ksp_type))
            except Exception:
                pass
        if ksp_rtol is not None or ksp_atol is not None or ksp_max_it is not None:
            try:
                subksp.setTolerances(rtol=ksp_rtol, atol=ksp_atol, max_it=ksp_max_it)
            except Exception:
                pass
        try:
            spc = subksp.getPC()
            if pc_type:
                spc.setType(str(pc_type))
            if pc_type in ("asm", "bjacobi"):
                n_sub = _configure_pc_asm_or_bjacobi_subksps(
                    spc,
                    sub_ksp_type=subdomain_ksp_type,
                    sub_pc_type=subdomain_pc_type,
                    sub_ksp_rtol=subdomain_ksp_rtol,
                    sub_ksp_atol=subdomain_ksp_atol,
                    sub_ksp_max_it=subdomain_ksp_max_it,
                    overlap=asm_overlap,
                )
        except Exception:
            pass
        try:
            subksp.setFromOptions()
        except Exception:
            pass
        try:
            subksp.setUp()
        except Exception:
            pass

        split_info = fs_splits.setdefault(name, {})
        if ksp_type:
            split_info["ksp_type"] = str(ksp_type)
        if pc_type:
            split_info["pc_type"] = str(pc_type)
        if ksp_rtol is not None:
            split_info["ksp_rtol"] = float(ksp_rtol)
        if ksp_atol is not None:
            split_info["ksp_atol"] = float(ksp_atol)
        if ksp_max_it is not None:
            split_info["ksp_max_it"] = int(ksp_max_it)
        if pc_type == "asm":
            if asm_overlap is not None:
                split_info["asm_overlap"] = asm_overlap
            if spc is not None and hasattr(spc, "getASMOverlap"):
                try:
                    split_info["asm_overlap"] = int(spc.getASMOverlap())
                except Exception:
                    pass
        if pc_type in ("asm", "bjacobi") and subdomain_ksp_type:
            split_info["subdomain_ksp_type"] = str(subdomain_ksp_type)
        if pc_type in ("asm", "bjacobi") and subdomain_pc_type:
            split_info["subdomain_pc_type"] = str(subdomain_pc_type)
        if pc_type in ("asm", "bjacobi") and subdomain_ksp_rtol is not None:
            split_info["subdomain_ksp_rtol"] = float(subdomain_ksp_rtol)
        if pc_type in ("asm", "bjacobi") and subdomain_ksp_atol is not None:
            split_info["subdomain_ksp_atol"] = float(subdomain_ksp_atol)
        if pc_type in ("asm", "bjacobi") and subdomain_ksp_max_it is not None:
            split_info["subdomain_ksp_max_it"] = int(subdomain_ksp_max_it)
        if pc_type == "asm":
            split_info["asm_subdomains"] = int(n_sub) if n_sub is not None else 0
        if pc_type == "bjacobi":
            split_info["bjacobi_subdomains"] = int(n_sub) if n_sub is not None else 0

def solve_linear_system_petsc(
    A,
    b,
    cfg: CaseConfig,
    x0: Optional[np.ndarray] = None,
    method: str = "ksp",
    *,
    layout=None,
    P=None,
) -> LinearSolveResult:
    """Solve Ax=b using PETSc KSP; returns LinearSolveResult (mirrors SciPy backend)."""
    try:
        from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

        bootstrap_mpi_before_petsc()
        from petsc4py import PETSc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("petsc4py is required for PETSc backend.") from exc

    if not isinstance(A, PETSc.Mat):
        raise TypeError(f"Expected PETSc.Mat for A, got {type(A)}")
    if not isinstance(b, PETSc.Vec):
        raise TypeError(f"Expected PETSc.Vec for b, got {type(b)}")

    m, n = A.getSize()
    if m != n:
        raise ValueError(f"A must be square, got size {(m, n)}")
    N = n
    if b.getSize() != N:
        raise ValueError(f"b size {b.getSize()} does not match A dimension {N}")

    if x0 is not None:
        x0 = np.asarray(x0, dtype=np.float64)
        if x0.shape != (N,):
            raise ValueError(f"x0 shape {x0.shape} does not match A dimension {N}")

    petsc_cfg = cfg.petsc
    linear_cfg = getattr(getattr(cfg, "solver", None), "linear", None)
    prefix = getattr(petsc_cfg, "options_prefix", "")
    if prefix is None:
        prefix = ""
    prefix = str(prefix)
    if prefix and not prefix.endswith("_"):
        prefix += "_"
    ksp_type = str(getattr(petsc_cfg, "ksp_type", "gmres"))
    pc_type = _cfg_get(linear_cfg, "pc_type", None)
    if pc_type is None:
        pc_type = str(getattr(petsc_cfg, "pc_type", "ilu"))
    else:
        pc_type = str(pc_type)
    pc_type = _normalize_pc_type(pc_type) or "ilu"
    if method in ("direct", "lu", "preonly"):
        ksp_type, pc_type = "preonly", "lu"
    elif method in ("ksp", "gmres", "fgmres", "lgmres"):
        pass
    rtol = float(getattr(petsc_cfg, "rtol", 1e-8))
    atol = float(getattr(petsc_cfg, "atol", 1e-12))
    max_it = int(getattr(petsc_cfg, "max_it", 200))
    restart = int(getattr(petsc_cfg, "restart", 30))
    monitor = bool(getattr(petsc_cfg, "monitor", False))

    logger.debug(
        "solve_linear_system_petsc: case=%s size=%s method=%s ksp=%s pc=%s rtol=%.3e atol=%.3e max_it=%d",
        getattr(cfg.case, "id", "unknown"),
        (N, N),
        method,
        ksp_type,
        pc_type,
        rtol,
        atol,
        max_it,
    )

    comm = A.getComm()
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix(prefix)
    if P is None:
        P = A
    ksp.setOperators(A, P)

    try:
        ksp.setType(ksp_type)
    except Exception:
        logger.warning("Unknown ksp_type='%s', falling back to gmres", ksp_type)
        ksp.setType("gmres")

    pc = ksp.getPC()
    if str(pc_type).lower() != "fieldsplit":
        try:
            pc.setType(pc_type)
        except Exception:
            logger.warning("Unknown pc_type='%s', falling back to jacobi", pc_type)
            pc.setType("jacobi")

    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)

    try:
        ksp_type_eff = str(ksp.getType()).lower()
        if ksp_type_eff in ("gmres", "fgmres"):
            ksp.setGMRESRestart(restart)
        elif ksp_type_eff == "lgmres":
            if hasattr(ksp, "setLGMRESRestart"):
                ksp.setLGMRESRestart(restart)
    except Exception:
        logger.debug("Unable to set restart for ksp_type='%s'", ksp.getType())

    if monitor:
        def _monitor(ksp_obj, its, rnorm):
            logger.debug("[KSP] its=%d rnorm=%.6e", its, rnorm)
        ksp.setMonitor(_monitor)

    diag_pc = apply_structured_pc(
        ksp=ksp,
        cfg=cfg,
        layout=layout,
        A=A,
        P=P,
        pc_type_override=_normalize_pc_type(pc_type),
    )

    ksp.setFromOptions()
    ksp.setUp()

    apply_fieldsplit_subksp_defaults(ksp, diag_pc)

    x = A.createVecRight()
    x.set(0.0)
    if x0 is not None:
        ksp_type_eff = str(ksp.getType()).lower()
        if ksp_type_eff != "preonly":
            x0_arr = np.ascontiguousarray(x0, dtype=np.float64).copy()
            x.setArray(x0_arr)
            ksp.setInitialGuessNonzero(True)

    ksp.solve(b, x)

    reason = int(ksp.getConvergedReason())
    converged = reason > 0
    n_iter = int(ksp.getIterationNumber())

    res_norm = None
    try:
        res_norm = float(ksp.getResidualNorm())
    except Exception:
        res_norm = None
    if res_norm is None or not np.isfinite(res_norm) or res_norm <= 0.0:
        r = b.duplicate()
        A.mult(x, r)
        r.aypx(-1.0, b)
        res_norm = float(r.norm())

    b_norm = float(b.norm())
    rel = res_norm / (b_norm + 1e-30)

    if not converged:
        logger.warning(
            "PETSc KSP not converged: reason=%d residual=%.3e rel=%.3e ksp=%s pc=%s",
            reason,
            res_norm,
            rel,
            ksp.getType(),
            ksp.getPC().getType(),
        )

    return LinearSolveResult(
        x=np.asarray(x.getArray(), dtype=np.float64).copy(),
        converged=converged,
        n_iter=n_iter,
        residual_norm=res_norm,
        rel_residual=rel,
        method=f"{ksp.getType()}+{ksp.getPC().getType()}",
        message=None if converged else f"PETSc KSP diverged (reason={reason})",
        diag={"pc": diag_pc, "ksp_type": str(ksp.getType()), "pc_type": str(ksp.getPC().getType())},
    )
