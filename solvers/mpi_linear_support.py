from __future__ import annotations

from typing import Any, Mapping, Optional

import os

from solvers.linear_types import (
    FieldSplitScheme,
    FieldSplitType,
    LinearSolverConfig as LinearSolverConfigTyped,
    PCType,
    SchurFactType,
)

MPI_ALLOWED_PC_TYPES = {PCType.ASM.value, PCType.FIELDSPLIT.value}
MPI_ALLOWED_FIELDSPLIT_TYPES = {FieldSplitType.ADDITIVE.value, FieldSplitType.SCHUR.value}
MPI_ALLOWED_FIELDSPLIT_SCHEMES = {FieldSplitScheme.BULK_IFACE.value}
MPI_ALLOWED_SCHUR_FACT_TYPES = {SchurFactType.LOWER.value, SchurFactType.UPPER.value, SchurFactType.FULL.value}

_ALLOWED_SUBSOLVERS_BLOCKS = {"bulk", "iface"}
_ALLOWED_BLOCK_KEYS = {"ksp_type", "pc_type", "asm_overlap", "asm_sub_pc_type", "ksp_rtol", "ksp_max_it"}

_ALLOWED_BLOCK_KSP_TYPES = {"preonly", "gmres", "fgmres"}
_ALLOWED_BLOCK_PC_TYPES = {"asm"}
_ALLOWED_BLOCK_ASM_SUB_PC_TYPES = {"ilu", "lu"}


def _get_backend(cfg: Any) -> str:
    backend = getattr(getattr(cfg, "nonlinear", None), "backend", None)
    if backend is None:
        backend = getattr(getattr(cfg, "solver", None), "backend", None)
    return str(backend or "unknown")


def _fmt_allowed(name: str, allowed: set[str]) -> str:
    return f"{name} allowed={sorted(allowed)}"


def _get_linear(cfg: Any) -> Optional[Any]:
    solver = getattr(cfg, "solver", None)
    if solver is None:
        return None
    return getattr(solver, "linear", None)


def _get_raw_fieldsplit(linear: Any) -> Optional[Any]:
    if linear is None:
        return None
    if isinstance(linear, Mapping):
        return linear.get("fieldsplit", None)
    return getattr(linear, "fieldsplit", None)


def _get_raw_pc_type(linear: Any) -> str:
    if linear is None:
        return PCType.ASM.value
    if isinstance(linear, Mapping):
        return str(linear.get("pc_type", PCType.ASM.value))
    return str(getattr(linear, "pc_type", PCType.ASM.value))

def _get_raw_subsolvers(raw_fieldsplit: Any) -> Optional[Any]:
    if raw_fieldsplit is None:
        return None
    if isinstance(raw_fieldsplit, Mapping):
        if "subsolvers" in raw_fieldsplit:
            return raw_fieldsplit.get("subsolvers")
        if "subksp" in raw_fieldsplit:
            return raw_fieldsplit.get("subksp")
        return None
    if hasattr(raw_fieldsplit, "subsolvers"):
        return getattr(raw_fieldsplit, "subsolvers", None)
    if hasattr(raw_fieldsplit, "subksp"):
        return getattr(raw_fieldsplit, "subksp", None)
    return None


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _mark_fieldsplit_validated(raw_fieldsplit: Any, *, subsolvers_filled: bool) -> None:
    if raw_fieldsplit is None:
        return
    if isinstance(raw_fieldsplit, Mapping):
        try:
            raw_fieldsplit["_mpi_validated"] = True
            raw_fieldsplit["_subsolvers_filled"] = bool(subsolvers_filled)
        except Exception:
            pass
        return
    try:
        setattr(raw_fieldsplit, "_mpi_validated", True)
        setattr(raw_fieldsplit, "_subsolvers_filled", bool(subsolvers_filled))
    except Exception:
        pass


def assert_mpi_additive_subsolvers_filled(cfg: Any, *, strict: Optional[bool] = None) -> None:
    if strict is None:
        strict = _env_flag("DROPLET_MPI_LINEAR_STRICT")
    if not strict:
        return

    linear = _get_linear(cfg)
    raw_fieldsplit = _get_raw_fieldsplit(linear)
    if raw_fieldsplit is None:
        return

    fs_type = None
    if isinstance(raw_fieldsplit, Mapping):
        fs_type = raw_fieldsplit.get("type", None)
    else:
        fs_type = getattr(raw_fieldsplit, "type", None)
    if str(fs_type or "").lower() != FieldSplitType.ADDITIVE.value:
        return

    marker = None
    if isinstance(raw_fieldsplit, Mapping):
        marker = raw_fieldsplit.get("_subsolvers_filled", None)
    else:
        marker = getattr(raw_fieldsplit, "_subsolvers_filled", None)

    if marker is not True:
        raise RuntimeError(
            "[MPI linear strict] additive fieldsplit subsolvers are not marked as filled; "
            "did you bypass validate_mpi_linear_support(cfg)?"
        )

    subsolvers = _get_raw_subsolvers(raw_fieldsplit)
    if subsolvers is None:
        raise RuntimeError(
            "[MPI linear strict] additive fieldsplit subsolvers are missing; "
            "did you bypass validate_mpi_linear_support(cfg)?"
        )

    if isinstance(subsolvers, Mapping):
        bulk = subsolvers.get("bulk", None)
        iface = subsolvers.get("iface", None)
    else:
        bulk = getattr(subsolvers, "bulk", None)
        iface = getattr(subsolvers, "iface", None)

    if bulk is None or iface is None:
        raise RuntimeError(
            "[MPI linear strict] additive fieldsplit subsolvers missing bulk/iface blocks; "
            "did you bypass validate_mpi_linear_support(cfg)?"
        )


def _validate_subsolvers_mapping(raw: Any, *, ctx: str) -> None:
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise TypeError(f"{ctx}: expected mapping, got {type(raw).__name__}")

    for k in raw.keys():
        if k not in _ALLOWED_SUBSOLVERS_BLOCKS:
            raise ValueError(f"{ctx}: unknown block '{k}', allowed={sorted(_ALLOWED_SUBSOLVERS_BLOCKS)}")

    for blk_name, blk in raw.items():
        if not isinstance(blk, Mapping):
            raise TypeError(f"{ctx}.{blk_name}: expected mapping, got {type(blk).__name__}")
        for kk in blk.keys():
            if kk not in _ALLOWED_BLOCK_KEYS:
                raise ValueError(
                    f"{ctx}.{blk_name}: unknown key '{kk}', allowed={sorted(_ALLOWED_BLOCK_KEYS)}"
                )
        if "asm_overlap" in blk:
            try:
                ov = int(blk["asm_overlap"])
            except Exception as exc:
                raise TypeError(
                    f"{ctx}.{blk_name}.asm_overlap: expected int-like, got {blk['asm_overlap']!r}"
                ) from exc
            if ov < 0:
                raise ValueError(f"{ctx}.{blk_name}.asm_overlap: must be >= 0, got {ov}")


def _validate_subsolvers_values(raw: Any, *, ctx: str, backend: str, pc_type: str, fs_type: str, fs_scheme: str, strict_iface_preonly: bool = False) -> None:
    """Validate subsolver field values against whitelist."""
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        return

    for blk_name, blk in raw.items():
        if not isinstance(blk, Mapping):
            continue

        if "ksp_type" in blk:
            ksp_t = str(blk["ksp_type"])
            allowed_ksp = _ALLOWED_BLOCK_KSP_TYPES
            if strict_iface_preonly and blk_name == "iface":
                allowed_ksp = {"preonly"}
            if ksp_t not in allowed_ksp:
                raise ValueError(
                    f"MPI linear support: invalid {ctx}.{blk_name}.ksp_type='{ksp_t}', allowed={sorted(allowed_ksp)} "
                    f"(backend={backend}, pc_type={pc_type}, fieldsplit.type={fs_type}, scheme={fs_scheme})"
                )

        if "pc_type" in blk:
            pc_t = str(blk["pc_type"])
            if pc_t not in _ALLOWED_BLOCK_PC_TYPES:
                raise ValueError(
                    f"MPI linear support: invalid {ctx}.{blk_name}.pc_type='{pc_t}', allowed={sorted(_ALLOWED_BLOCK_PC_TYPES)} "
                    f"(backend={backend}, pc_type={pc_type}, fieldsplit.type={fs_type}, scheme={fs_scheme})"
                )

        if "asm_sub_pc_type" in blk:
            sub_pc_t = str(blk["asm_sub_pc_type"])
            if sub_pc_t not in _ALLOWED_BLOCK_ASM_SUB_PC_TYPES:
                raise ValueError(
                    f"MPI linear support: invalid {ctx}.{blk_name}.asm_sub_pc_type='{sub_pc_t}', allowed={sorted(_ALLOWED_BLOCK_ASM_SUB_PC_TYPES)} "
                    f"(backend={backend}, pc_type={pc_type}, fieldsplit.type={fs_type}, scheme={fs_scheme})"
                )


def _subsolvers_to_dict(subsolvers) -> dict:
    def _block_to_dict(block) -> dict:
        out = {
            "ksp_type": block.ksp_type.value,
            "pc_type": block.pc_type.value,
            "asm_overlap": int(block.asm_overlap),
            "asm_sub_pc_type": block.asm_sub_pc_type.value,
        }
        if block.ksp_rtol is not None:
            out["ksp_rtol"] = float(block.ksp_rtol)
        if block.ksp_max_it is not None:
            out["ksp_max_it"] = int(block.ksp_max_it)
        return out

    return {
        "bulk": _block_to_dict(subsolvers.bulk),
        "iface": _block_to_dict(subsolvers.iface),
    }


def validate_mpi_linear_support(cfg: Any) -> None:
    backend = _get_backend(cfg)
    linear = _get_linear(cfg)
    if linear is None:
        raise ValueError(f"[MPI linear validate] backend={backend}: missing cfg.solver.linear")

    raw_fieldsplit = _get_raw_fieldsplit(linear)
    raw_pc_type = _get_raw_pc_type(linear)
    raw_subsolvers_pre = _get_raw_subsolvers(raw_fieldsplit)

    if raw_pc_type == PCType.ASM.value and raw_fieldsplit is not None:
        raise ValueError(
            "[MPI linear validate] pc_type=asm must not provide fieldsplit config\n"
            f"  backend={backend}\n"
        )

    typed = LinearSolverConfigTyped.from_cfg(cfg)
    pc_type = typed.pc_type.value

    if pc_type not in MPI_ALLOWED_PC_TYPES:
        raise ValueError(
            "[MPI linear validate] unsupported pc_type\n"
            f"  backend={backend}\n"
            f"  pc_type={pc_type}\n"
            f"  {_fmt_allowed('pc_type', MPI_ALLOWED_PC_TYPES)}\n"
        )

    if pc_type == PCType.ASM.value:
        return

    if raw_fieldsplit is None:
        raise ValueError(
            "[MPI linear validate] pc_type=fieldsplit requires cfg.solver.linear.fieldsplit\n"
            f"  backend={backend}\n"
        )

    fs_cfg = typed.fieldsplit
    if fs_cfg is None:
        raise ValueError(
            "[MPI linear validate] pc_type=fieldsplit requires cfg.solver.linear.fieldsplit\n"
            f"  backend={backend}\n"
        )

    fs_type = fs_cfg.type.value
    fs_scheme = fs_cfg.scheme.value

    if fs_type not in MPI_ALLOWED_FIELDSPLIT_TYPES:
        raise ValueError(
            "[MPI linear validate] unsupported fieldsplit.type\n"
            f"  backend={backend}\n"
            f"  pc_type={pc_type}\n"
            f"  fieldsplit.type={fs_type}\n"
            f"  fieldsplit.scheme={fs_scheme}\n"
            f"  {_fmt_allowed('fieldsplit.type', MPI_ALLOWED_FIELDSPLIT_TYPES)}\n"
        )

    if fs_scheme not in MPI_ALLOWED_FIELDSPLIT_SCHEMES:
        raise ValueError(
            "[MPI linear validate] unsupported fieldsplit.scheme\n"
            f"  backend={backend}\n"
            f"  pc_type={pc_type}\n"
            f"  fieldsplit.type={fs_type}\n"
            f"  fieldsplit.scheme={fs_scheme}\n"
            f"  {_fmt_allowed('fieldsplit.scheme', MPI_ALLOWED_FIELDSPLIT_SCHEMES)}\n"
        )

    if fs_type == FieldSplitType.SCHUR.value:
        typed_sft = fs_cfg.schur_fact_type.value if fs_cfg.schur_fact_type is not None else None
        raw_sft = None
        if isinstance(raw_fieldsplit, Mapping):
            raw_sft = raw_fieldsplit.get("schur_fact_type", None)
        else:
            try:
                raw_sft = getattr(raw_fieldsplit, "schur_fact_type", None)
            except Exception:
                raw_sft = None

        effective_sft = raw_sft or typed_sft or SchurFactType.LOWER.value

        if raw_sft is None:
            if isinstance(raw_fieldsplit, Mapping):
                try:
                    raw_fieldsplit["schur_fact_type"] = effective_sft
                except Exception:
                    pass
            else:
                try:
                    setattr(raw_fieldsplit, "schur_fact_type", effective_sft)
                except Exception:
                    pass

        if effective_sft not in MPI_ALLOWED_SCHUR_FACT_TYPES:
            raise ValueError(
                "[MPI linear validate] unsupported schur_fact_type\n"
                f"  backend={backend}\n"
                f"  pc_type={pc_type}\n"
                f"  fieldsplit.type={fs_type}\n"
                f"  fieldsplit.scheme={fs_scheme}\n"
                f"  schur_fact_type={effective_sft}\n"
                f"  {_fmt_allowed('schur_fact_type', MPI_ALLOWED_SCHUR_FACT_TYPES)}\n"
            )

        _validate_subsolvers_mapping(raw_subsolvers_pre, ctx="linear.fieldsplit.subsolvers")
        _validate_subsolvers_values(
            raw_subsolvers_pre,
            ctx="subsolvers",
            backend=backend,
            pc_type=pc_type,
            fs_type=fs_type,
            fs_scheme=fs_scheme,
            strict_iface_preonly=True,
        )

        if isinstance(raw_fieldsplit, Mapping):
            sub_defaults = _subsolvers_to_dict(fs_cfg.subsolvers)
            raw_subsolvers = raw_fieldsplit.get("subsolvers")
            if raw_subsolvers is None:
                raw_subsolvers = raw_fieldsplit.get("subksp")
            if raw_subsolvers is None:
                raw_fieldsplit["subsolvers"] = sub_defaults
            else:
                if not isinstance(raw_subsolvers, Mapping):
                    raise TypeError(
                        f"linear.fieldsplit.subsolvers: expected mapping, got {type(raw_subsolvers).__name__}"
                    )
                for blk_name, blk_defaults in sub_defaults.items():
                    if blk_name not in raw_subsolvers or raw_subsolvers[blk_name] is None:
                        raw_subsolvers[blk_name] = dict(blk_defaults)
                        continue
                    blk = raw_subsolvers[blk_name]
                    if not isinstance(blk, Mapping):
                        raise TypeError(
                            f"linear.fieldsplit.subsolvers.{blk_name}: expected mapping, got {type(blk).__name__}"
                        )
                    for kk, vv in blk_defaults.items():
                        if kk not in blk:
                            blk[kk] = vv
                raw_fieldsplit["subsolvers"] = raw_subsolvers
            _mark_fieldsplit_validated(raw_fieldsplit, subsolvers_filled=True)
        else:
            _mark_fieldsplit_validated(raw_fieldsplit, subsolvers_filled=True)
        return

    if fs_type == FieldSplitType.ADDITIVE.value:
        _validate_subsolvers_mapping(raw_subsolvers_pre, ctx="linear.fieldsplit.subsolvers")
        _validate_subsolvers_values(
            raw_subsolvers_pre,
            ctx="subsolvers",
            backend=backend,
            pc_type=pc_type,
            fs_type=fs_type,
            fs_scheme=fs_scheme,
            strict_iface_preonly=False,
        )

        if isinstance(raw_fieldsplit, Mapping):
            sub_defaults = _subsolvers_to_dict(fs_cfg.subsolvers)
            raw_subsolvers = raw_fieldsplit.get("subsolvers")
            if raw_subsolvers is None:
                raw_subsolvers = raw_fieldsplit.get("subksp")
            if raw_subsolvers is None:
                raw_fieldsplit["subsolvers"] = sub_defaults
            else:
                if not isinstance(raw_subsolvers, Mapping):
                    raise TypeError(
                        f"linear.fieldsplit.subsolvers: expected mapping, got {type(raw_subsolvers).__name__}"
                    )
                for blk_name, blk_defaults in sub_defaults.items():
                    if blk_name not in raw_subsolvers or raw_subsolvers[blk_name] is None:
                        raw_subsolvers[blk_name] = dict(blk_defaults)
                        continue
                    blk = raw_subsolvers[blk_name]
                    if not isinstance(blk, Mapping):
                        raise TypeError(
                            f"linear.fieldsplit.subsolvers.{blk_name}: expected mapping, got {type(blk).__name__}"
                        )
                    for kk, vv in blk_defaults.items():
                        if kk not in blk:
                            blk[kk] = vv
                raw_fieldsplit["subsolvers"] = raw_subsolvers
            _mark_fieldsplit_validated(raw_fieldsplit, subsolvers_filled=True)
        else:
            _mark_fieldsplit_validated(raw_fieldsplit, subsolvers_filled=True)
