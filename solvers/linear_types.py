"""
Shared linear solver result types.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional

import numpy as np


@dataclass
class LinearSolveResult:
    x: np.ndarray
    converged: bool
    n_iter: int
    residual_norm: float
    rel_residual: float
    method: str
    message: Optional[str] = None
    diag: Optional[Dict[str, Any]] = None


class JacobianMode:
    """
    Canonical names and helpers for PETSc Jacobian / preconditioner modes.

    These are strings to stay compatible with cfg.petsc.jacobian_mode.
    """

    FD: str = "fd"
    MF: str = "mf"
    MFPC_SPARSE_FD: str = "mfpc_sparse_fd"
    MFPC_AIJA: str = "mfpc_aija"

    _ALIASES = {
        "mfpc_sparse": MFPC_SPARSE_FD,
        "mfpc_sparse_fd": MFPC_SPARSE_FD,
        "mfpc_aij": MFPC_AIJA,
        "mfpc_aija": MFPC_AIJA,
    }

    @classmethod
    def normalize(cls, value: Optional[str]) -> str:
        """
        Normalize user/cfg input into one of the canonical strings.

        Returns the lowercased string if it's not a known alias.
        """
        if value is None:
            return cls.FD
        v = str(value).strip().lower()
        if not v:
            return cls.FD
        if v in cls._ALIASES:
            return cls._ALIASES[v]
        return v

    @classmethod
    def canonical_set(cls) -> set[str]:
        return {cls.FD, cls.MF, cls.MFPC_SPARSE_FD, cls.MFPC_AIJA}


class PCType(str, Enum):
    ASM = "asm"
    FIELDSPLIT = "fieldsplit"


class FieldSplitType(str, Enum):
    ADDITIVE = "additive"
    SCHUR = "schur"


class FieldSplitScheme(str, Enum):
    BY_LAYOUT = "by_layout"
    BULK_IFACE = "bulk_iface"


class SchurFactType(str, Enum):
    LOWER = "lower"
    UPPER = "upper"
    FULL = "full"


class FieldSplitBlockKSPType(str, Enum):
    PREONLY = "preonly"
    GMRES = "gmres"
    FGMRES = "fgmres"


class FieldSplitBlockPCType(str, Enum):
    ASM = "asm"


class FieldSplitBlockASMSubPCType(str, Enum):
    ILU = "ilu"
    LU = "lu"


def _coerce_enum(enum_cls: type[Enum], value: Any, where: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except Exception:
            allowed = [e.value for e in enum_cls]
            raise ValueError(f"{where}: invalid value {value!r}, allowed={allowed}")
    raise TypeError(f"{where}: expected str or {enum_cls.__name__}, got {type(value).__name__}")


@dataclass(slots=True)
class FieldSplitBlockConfig:
    ksp_type: FieldSplitBlockKSPType = FieldSplitBlockKSPType.PREONLY
    pc_type: FieldSplitBlockPCType = FieldSplitBlockPCType.ASM
    asm_overlap: int = 1
    asm_sub_pc_type: FieldSplitBlockASMSubPCType = FieldSplitBlockASMSubPCType.ILU
    ksp_rtol: Optional[float] = None
    ksp_max_it: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, where: str) -> "FieldSplitBlockConfig":
        raw_ksp_type = d.get("ksp_type", None)
        if raw_ksp_type is None:
            raw_ksp_type = FieldSplitBlockKSPType.PREONLY.value
        ksp_type = _coerce_enum(FieldSplitBlockKSPType, raw_ksp_type, f"{where}.ksp_type")

        raw_pc_type = d.get("pc_type", None)
        if raw_pc_type is None:
            raw_pc_type = FieldSplitBlockPCType.ASM.value
        pc_type = _coerce_enum(FieldSplitBlockPCType, raw_pc_type, f"{where}.pc_type")

        raw_overlap = d.get("asm_overlap", None)
        if raw_overlap is None:
            asm_overlap = 1
        else:
            try:
                asm_overlap = int(raw_overlap)
            except Exception as exc:
                raise ValueError(f"{where}.asm_overlap: invalid value {raw_overlap!r}") from exc

        raw_sub_pc = d.get("asm_sub_pc_type", None)
        if raw_sub_pc is None:
            raw_sub_pc = FieldSplitBlockASMSubPCType.ILU.value
        asm_sub_pc_type = _coerce_enum(
            FieldSplitBlockASMSubPCType,
            raw_sub_pc,
            f"{where}.asm_sub_pc_type",
        )

        ksp_rtol = d.get("ksp_rtol", None)
        if ksp_rtol is not None:
            try:
                ksp_rtol = float(ksp_rtol)
            except Exception as exc:
                raise ValueError(f"{where}.ksp_rtol: invalid value {ksp_rtol!r}") from exc

        ksp_max_it = d.get("ksp_max_it", None)
        if ksp_max_it is not None:
            try:
                ksp_max_it = int(ksp_max_it)
            except Exception as exc:
                raise ValueError(f"{where}.ksp_max_it: invalid value {ksp_max_it!r}") from exc

        return cls(
            ksp_type=ksp_type,
            pc_type=pc_type,
            asm_overlap=asm_overlap,
            asm_sub_pc_type=asm_sub_pc_type,
            ksp_rtol=ksp_rtol,
            ksp_max_it=ksp_max_it,
        )


@dataclass(slots=True)
class FieldSplitSubsolversConfig:
    bulk: FieldSplitBlockConfig
    iface: FieldSplitBlockConfig


@dataclass(slots=True)
class FieldSplitConfig:
    type: FieldSplitType
    scheme: FieldSplitScheme
    schur_fact_type: Optional[SchurFactType] = None
    subsolvers: Optional[FieldSplitSubsolversConfig] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FieldSplitConfig":
        raw_type = d.get("type", None)
        if raw_type is None:
            raw_type = FieldSplitType.ADDITIVE.value
        fs_type = _coerce_enum(FieldSplitType, raw_type, "fieldsplit.type")

        raw_scheme = d.get("scheme", None)
        raw_split_mode = d.get("split_mode", None)
        if raw_scheme is None and raw_split_mode is not None:
            raw_scheme = raw_split_mode
        elif raw_scheme is not None and raw_split_mode is not None:
            if str(raw_scheme).strip().lower() != str(raw_split_mode).strip().lower():
                raise ValueError(
                    f"fieldsplit: scheme {raw_scheme!r} does not match split_mode {raw_split_mode!r}"
                )
        if raw_scheme is None:
            raw_scheme = FieldSplitScheme.BULK_IFACE.value
        scheme = _coerce_enum(FieldSplitScheme, raw_scheme, "fieldsplit.scheme")

        schur_fact = d.get("schur_fact_type", None)
        if schur_fact is not None:
            schur_fact = _coerce_enum(SchurFactType, schur_fact, "fieldsplit.schur_fact_type")
        elif fs_type == FieldSplitType.SCHUR:
            schur_fact = SchurFactType.LOWER
        if fs_type == FieldSplitType.SCHUR and isinstance(d, dict):
            if d.get("schur_fact_type", None) is None:
                if schur_fact is None:
                    d["schur_fact_type"] = SchurFactType.LOWER.value
                else:
                    d["schur_fact_type"] = getattr(schur_fact, "value", str(schur_fact))

        raw_subsolvers = d.get("subsolvers", None)
        raw_subksp = d.get("subksp", None)
        if raw_subsolvers is not None and raw_subksp is not None:
            raise ValueError("fieldsplit: provide only one of 'subsolvers' or 'subksp'")
        if raw_subsolvers is None and raw_subksp is not None:
            raw_subsolvers = raw_subksp

        def _block_to_dict(block: FieldSplitBlockConfig) -> dict:
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

        def _fill_block_defaults(raw_block: Any, block: FieldSplitBlockConfig) -> None:
            if not isinstance(raw_block, dict):
                return
            raw_block.setdefault("ksp_type", block.ksp_type.value)
            raw_block.setdefault("pc_type", block.pc_type.value)
            raw_block.setdefault("asm_overlap", int(block.asm_overlap))
            raw_block.setdefault("asm_sub_pc_type", block.asm_sub_pc_type.value)
            if block.ksp_rtol is not None:
                raw_block.setdefault("ksp_rtol", float(block.ksp_rtol))
            if block.ksp_max_it is not None:
                raw_block.setdefault("ksp_max_it", int(block.ksp_max_it))

        iface_seed = {"asm_sub_pc_type": "lu"} if fs_type == FieldSplitType.SCHUR else {}
        if raw_subsolvers is None:
            bulk_block = FieldSplitBlockConfig.from_dict({}, where="fieldsplit.subsolvers.bulk")
            iface_block = FieldSplitBlockConfig.from_dict(iface_seed, where="fieldsplit.subsolvers.iface")
            subsolvers = FieldSplitSubsolversConfig(bulk=bulk_block, iface=iface_block)
            if isinstance(d, dict):
                d["subsolvers"] = {
                    "bulk": _block_to_dict(bulk_block),
                    "iface": _block_to_dict(iface_block),
                }
        else:
            if not isinstance(raw_subsolvers, Mapping):
                raise TypeError(f"fieldsplit.subsolvers: expected mapping, got {type(raw_subsolvers).__name__}")
            raw_bulk = raw_subsolvers.get("bulk", None)
            raw_iface = raw_subsolvers.get("iface", None)
            if raw_bulk is None:
                raw_bulk = {}
            if raw_iface is None:
                raw_iface = {}
                if isinstance(raw_subsolvers, dict):
                    raw_subsolvers["iface"] = raw_iface
            if fs_type == FieldSplitType.SCHUR and isinstance(raw_iface, dict):
                raw_iface.setdefault("asm_sub_pc_type", "lu")
            if not isinstance(raw_bulk, Mapping):
                raise TypeError(
                    f"fieldsplit.subsolvers.bulk: expected mapping, got {type(raw_bulk).__name__}"
                )
            if not isinstance(raw_iface, Mapping):
                raise TypeError(
                    f"fieldsplit.subsolvers.iface: expected mapping, got {type(raw_iface).__name__}"
                )
            bulk_block = FieldSplitBlockConfig.from_dict(raw_bulk, where="fieldsplit.subsolvers.bulk")
            iface_block = FieldSplitBlockConfig.from_dict(raw_iface, where="fieldsplit.subsolvers.iface")
            subsolvers = FieldSplitSubsolversConfig(bulk=bulk_block, iface=iface_block)
            if isinstance(raw_subsolvers, dict):
                raw_subsolvers.setdefault("bulk", _block_to_dict(bulk_block))
                raw_subsolvers.setdefault("iface", _block_to_dict(iface_block))
                _fill_block_defaults(raw_subsolvers.get("bulk"), bulk_block)
                _fill_block_defaults(raw_subsolvers.get("iface"), iface_block)

        return cls(
            type=fs_type,
            scheme=scheme,
            schur_fact_type=schur_fact,
            subsolvers=subsolvers,
        )


@dataclass(slots=True)
class LinearSolverConfig:
    pc_type: PCType
    fieldsplit: Optional[FieldSplitConfig] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LinearSolverConfig":
        pc_raw = d.get("pc_type", None)
        if pc_raw is None:
            pc_raw = PCType.ASM.value
        pc_type = _coerce_enum(PCType, pc_raw, "linear.pc_type")

        fs = d.get("fieldsplit", None)
        if pc_type == PCType.FIELDSPLIT:
            if fs is None:
                fs_cfg = FieldSplitConfig.from_dict({})
            elif isinstance(fs, Mapping):
                fs_cfg = FieldSplitConfig.from_dict(fs)
            else:
                raise TypeError(f"linear.fieldsplit: expected mapping, got {type(fs).__name__}")
            return cls(pc_type=pc_type, fieldsplit=fs_cfg)

        if fs is not None:
            raise ValueError("linear: pc_type='asm' forbids providing fieldsplit config")
        return cls(pc_type=pc_type, fieldsplit=None)

    @classmethod
    def from_cfg(cls, cfg: Any) -> "LinearSolverConfig":
        linear = getattr(getattr(cfg, "solver", None), "linear", None)
        if linear is None:
            return cls.from_dict({})
        if isinstance(linear, Mapping):
            d = {"pc_type": linear.get("pc_type", None), "fieldsplit": linear.get("fieldsplit", None)}
            return cls.from_dict(d)

        fs = getattr(linear, "fieldsplit", None)
        if fs is not None and not isinstance(fs, Mapping):
            fs = {
                "type": getattr(fs, "type", None),
                "scheme": getattr(fs, "scheme", None),
                "schur_fact_type": getattr(fs, "schur_fact_type", None),
            }
        d = {
            "pc_type": getattr(linear, "pc_type", None),
            "fieldsplit": fs,
        }
        return cls.from_dict(d)
