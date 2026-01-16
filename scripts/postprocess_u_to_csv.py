#!/usr/bin/env python3
"""
Post-processing script: Convert u vector + grid coordinates to human-readable CSV.

Usage:
    python postprocess_u_to_csv.py --run-dir <path/to/3D_out/case_xxx/run_yyy> \
                                    [--t-start <sec>] [--t-end <sec>] \
                                    [--stride <N>] [--out-dir <path>]

Features:
- Reads mapping.json to understand u vector layout
- Scans steps/*.npz files
- Filters by time window [t_start, t_end]
- Selects every Nth file (stride)
- Outputs one CSV per step file with gas/liquid data combined
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_mapping(run_dir: Path) -> dict:
    """Load mapping.json from run directory."""
    mapping_path = run_dir / "mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"mapping.json not found in {run_dir}")

    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    logger.info(f"Loaded mapping.json from {mapping_path}")
    return mapping


def scan_step_files(run_dir: Path) -> List[Path]:
    """Scan steps/ directory for .npz files."""
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        raise FileNotFoundError(f"steps/ directory not found in {run_dir}")

    step_files = sorted(steps_dir.glob("step_*.npz"))
    logger.info(f"Found {len(step_files)} step files in {steps_dir}")
    return step_files


def filter_by_time(
    step_files: List[Path], t_start: float, t_end: float
) -> List[Tuple[Path, float]]:
    """
    Filter step files by time window.

    Returns list of (file_path, t) tuples sorted by time.
    """
    filtered = []

    for step_file in step_files:
        try:
            data = np.load(step_file)
            t = float(data["t"])

            if t_start <= t <= t_end:
                filtered.append((step_file, t))
        except Exception as e:
            logger.warning(f"Failed to read time from {step_file}: {e}")
            continue

    # Sort by time
    filtered.sort(key=lambda x: x[1])

    logger.info(
        f"Filtered to {len(filtered)} files in time range [{t_start:.6e}, {t_end:.6e}]"
    )
    return filtered


def apply_stride(
    files_with_time: List[Tuple[Path, float]], stride: int
) -> List[Tuple[Path, float]]:
    """Apply stride to select every Nth file."""
    if stride <= 0:
        stride = 1

    selected = files_with_time[::stride]
    logger.info(f"Applied stride={stride}, selected {len(selected)} files")
    return selected


def unpack_u(u: np.ndarray, mapping: dict) -> dict:
    """
    Unpack u vector into fields based on mapping.

    Returns dict with field names as keys and arrays as values.

    Note: Yg and Yl are stored as (N_spatial, N_species) in u vector,
    but we transpose them to (N_species, N_spatial) for easier processing.
    """
    blocks = mapping["blocks"]
    meta = mapping["meta"]

    fields = {}

    for block in blocks:
        name = block["name"]
        offset = block["offset"]
        size = block["size"]
        shape = block["shape"]

        # Extract data
        u_block = u[offset : offset + size]

        # Reshape
        if len(shape) == 1:
            # 1D array
            fields[name] = u_block.copy()
        elif len(shape) == 2:
            # 2D array
            # u is stored in C-order (row-major)
            reshaped = u_block.reshape(shape, order="C")

            # For Yg and Yl, the shape is (N_spatial, N_species) from storage
            # but we want (N_species, N_spatial) for processing
            if name in ("Yg", "Yl"):
                fields[name] = reshaped.T  # Transpose to (N_species, N_spatial)
            else:
                fields[name] = reshaped
        else:
            # Fallback: keep 1D
            fields[name] = u_block.copy()

    return fields


def reconstruct_full_Y(
    Y_reduced: np.ndarray,
    species_full: List[str],
    species_reduced: List[str],
    closure_species: str,
) -> np.ndarray:
    """
    Reconstruct full Y array including closure species.

    Y_reduced: (Ns_eff, N) array
    Returns: (Ns_full, N) array
    """
    Ns_full = len(species_full)
    if Y_reduced.ndim == 1:
        # Scalar case (shouldn't happen for Y, but handle it)
        N = 1
        Y_reduced = Y_reduced.reshape(-1, 1)
    else:
        N = Y_reduced.shape[1]

    Y_full = np.zeros((Ns_full, N), dtype=np.float64)

    # Map reduced species to full array
    for i_red, sp_name in enumerate(species_reduced):
        i_full = species_full.index(sp_name)
        Y_full[i_full, :] = Y_reduced[i_red, :]

    # Reconstruct closure species
    if closure_species in species_full:
        i_closure = species_full.index(closure_species)
        Y_full[i_closure, :] = 1.0 - np.sum(Y_full, axis=0)
        # Clamp to [0, 1]
        Y_full[i_closure, :] = np.clip(Y_full[i_closure, :], 0.0, 1.0)

    return Y_full


def convert_step_to_csv(
    step_file: Path, mapping: dict, out_dir: Path
) -> None:
    """
    Convert a single step file to CSV.

    CSV format:
    - Header: phase, r, T, Y_<species1>, Y_<species2>, ...
    - Rows: gas phase first, then liquid phase
    - Scalars (Ts, mpp, Rd) written as comments at the top
    """
    # Load step data
    data = np.load(step_file)
    step_id = int(data["step_id"])
    t = float(data["t"])
    u = data["u"]
    r_g = data["r_g"]
    r_l = data["r_l"]

    # Unpack u into fields
    fields = unpack_u(u, mapping)
    meta = mapping["meta"]

    # Extract scalars
    Ts = float(fields.get("Ts", [np.nan])[0]) if "Ts" in fields else np.nan
    mpp = float(fields.get("mpp", [np.nan])[0]) if "mpp" in fields else np.nan
    Rd = float(fields.get("Rd", [np.nan])[0]) if "Rd" in fields else np.nan

    # Extract and reconstruct gas phase
    Tg = fields.get("Tg", np.array([], dtype=np.float64))
    Yg_reduced = fields.get("Yg", np.array([], dtype=np.float64))

    if Yg_reduced.ndim == 1:
        # Reshape to (Ns_eff, Ng)
        Ns_g_eff = meta["Ns_g_eff"]
        Ng = meta["Ng"]
        if Yg_reduced.size == Ns_g_eff * Ng:
            Yg_reduced = Yg_reduced.reshape((Ns_g_eff, Ng), order="C")

    if Yg_reduced.size > 0 and meta["Ns_g_full"] > 0:
        Yg_full = reconstruct_full_Y(
            Yg_reduced,
            meta["species_g_full"],
            meta["species_g_reduced"],
            meta["species_g_closure"],
        )
    else:
        Yg_full = np.zeros((meta["Ns_g_full"], len(r_g)), dtype=np.float64)

    # Extract and reconstruct liquid phase
    Tl = fields.get("Tl", np.array([], dtype=np.float64))
    Yl_reduced = fields.get("Yl", np.array([], dtype=np.float64))

    if Yl_reduced.ndim == 1 and Yl_reduced.size > 0:
        # Reshape to (Ns_eff, Nl)
        Ns_l_eff = meta["Ns_l_eff"]
        Nl = meta["Nl"]
        if Yl_reduced.size == Ns_l_eff * Nl:
            Yl_reduced = Yl_reduced.reshape((Ns_l_eff, Nl), order="C")

    if Yl_reduced.size > 0 and meta["Ns_l_full"] > 0:
        Yl_full = reconstruct_full_Y(
            Yl_reduced,
            meta["species_l_full"],
            meta["species_l_reduced"],
            meta["species_l_closure"],
        )
    else:
        # If Yl is not solved, initialize with closure species = 1.0
        Yl_full = np.zeros((meta["Ns_l_full"], len(r_l)), dtype=np.float64)
        if meta["Ns_l_full"] > 0 and len(r_l) > 0:
            closure_species = meta["species_l_closure"]
            species_l_full = meta["species_l_full"]
            if closure_species in species_l_full:
                i_closure = species_l_full.index(closure_species)
                Yl_full[i_closure, :] = 1.0

    # Generate output filename
    out_filename = f"step_{step_id:06d}_time_{t:.6e}s.csv"
    out_path = out_dir / out_filename

    # Build CSV header
    species_g = meta["species_g_full"]
    species_l = meta["species_l_full"]

    # Use union of gas and liquid species for header
    all_species = list(species_g)
    for sp in species_l:
        if sp not in all_species:
            all_species.append(sp)

    header = ["phase", "r", "T"] + [f"Y_{sp}" for sp in all_species]

    # Write CSV
    with open(out_path, "w", newline="") as f:
        # Write scalar metadata as comments
        f.write(f"# step_id={step_id}\n")
        f.write(f"# t={t:.12e}\n")
        f.write(f"# Ts={Ts:.12e}\n")
        f.write(f"# mpp={mpp:.12e}\n")
        f.write(f"# Rd={Rd:.12e}\n")
        f.write("#\n")

        writer = csv.writer(f)
        writer.writerow(header)

        # Write gas phase rows
        Ng = len(r_g)
        for ig in range(Ng):
            row = ["gas", f"{r_g[ig]:.12e}", f"{Tg[ig]:.12e}"]

            for sp in all_species:
                if sp in species_g:
                    i_sp = species_g.index(sp)
                    row.append(f"{Yg_full[i_sp, ig]:.12e}")
                else:
                    row.append("")  # Empty if species not present in gas

            writer.writerow(row)

        # Write liquid phase rows
        Nl = len(r_l)
        for il in range(Nl):
            row = ["liq", f"{r_l[il]:.12e}", f"{Tl[il]:.12e}"]

            for sp in all_species:
                if sp in species_l:
                    i_sp = species_l.index(sp)
                    row.append(f"{Yl_full[i_sp, il]:.12e}")
                else:
                    row.append("")  # Empty if species not present in liquid

            writer.writerow(row)

    logger.debug(f"Wrote CSV: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert u vector step files to CSV format"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory (e.g., 3D_out/case_xxx/run_yyy)",
    )
    parser.add_argument(
        "--t-start",
        type=float,
        default=float("-inf"),
        help="Start time (seconds, inclusive)",
    )
    parser.add_argument(
        "--t-end", type=float, default=float("inf"), help="End time (seconds, inclusive)"
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Select every Nth file (default: 1)"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <run-dir>/post_csv)",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory does not exist: {run_dir}")
        sys.exit(1)

    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = run_dir / "post_csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # Load mapping
    try:
        mapping = load_mapping(run_dir)
    except Exception as e:
        logger.error(f"Failed to load mapping: {e}")
        sys.exit(1)

    # Scan step files
    try:
        step_files = scan_step_files(run_dir)
    except Exception as e:
        logger.error(f"Failed to scan step files: {e}")
        sys.exit(1)

    if not step_files:
        logger.error("No step files found")
        sys.exit(1)

    # Filter by time
    files_with_time = filter_by_time(step_files, args.t_start, args.t_end)

    if not files_with_time:
        logger.error("No files in specified time range")
        sys.exit(1)

    # Apply stride
    selected_files = apply_stride(files_with_time, args.stride)

    if not selected_files:
        logger.error("No files after applying stride")
        sys.exit(1)

    # Convert each file to CSV
    logger.info(f"Converting {len(selected_files)} files to CSV...")
    for i, (step_file, t) in enumerate(selected_files):
        try:
            convert_step_to_csv(step_file, mapping, out_dir)
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(selected_files)} files")
        except Exception as e:
            logger.error(f"Failed to convert {step_file}: {e}")
            continue

    logger.info(f"Conversion complete! Output in {out_dir}")


if __name__ == "__main__":
    main()
