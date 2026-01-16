"""
Simplex Projection for Mass Fraction Normalization

Implementation of Duchi et al. 2008 algorithm for projecting vectors onto
the probability simplex and its variants.

References:
    Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008).
    Efficient projections onto the ℓ1-ball for learning in high dimensions.
    ICML 2008.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def project_simplex(y: np.ndarray, z: float = 1.0, eps: float = 1e-30) -> np.ndarray:
    """
    Project vector y onto the simplex {x: x >= 0, sum(x) = z}.

    Uses the efficient O(N log N) algorithm from Duchi et al. 2008.
    The projection minimizes the Euclidean distance ||x - y||_2.

    Args:
        y: Input vector, shape (N,) or (N, 1)
        z: Target sum (default 1.0 for probability simplex)
        eps: Small constant to avoid division by zero

    Returns:
        x: Projected vector on simplex, shape (N,) or (N, 1)

    Mathematical formulation:
        x* = argmin_{x in Delta} ||x - y||_2^2
        where Delta = {x: x_i >= 0, sum(x_i) = z}

    Algorithm:
        1. Sort y in descending order: u = sort(y, reverse=True)
        2. Find rho = max{j: u[j] - (sum(u[:j+1]) - z)/(j+1) > 0}
        3. Compute theta = (sum(u[:rho]) - z) / rho
        4. Project: x = max(y - theta, 0)
    """
    y = np.asarray(y, dtype=np.float64)
    original_shape = y.shape

    # Hard guards (do not allow silent propagation)
    if z < 0.0:
        raise ValueError(f"Simplex sum z must be non-negative, got z={z}")
    if not np.all(np.isfinite(y)):
        raise ValueError("Input vector contains NaN/Inf; cannot project.")

    # Fast path: z == 0 -> all zeros (unique feasible solution)
    if z == 0.0:
        x0 = np.zeros_like(y, dtype=np.float64)
        return x0.reshape(original_shape)

    # Flatten to 1D if needed

    if y.ndim == 2 and y.shape[1] == 1:
        y = y.flatten()
    elif y.ndim != 1:
        raise ValueError(f"Input must be 1D or (N, 1), got shape {original_shape}")

    N = len(y)

    # Handle edge case: empty or single element
    if N == 0:
        # Empty vector can only satisfy sum constraint when z == 0 (handled above).
        raise ValueError("Cannot project an empty vector when z != 0.")
    if N == 1:
        result = np.array([z], dtype=np.float64)
        return result.reshape(original_shape)

    # Step 1: Sort in descending order
    u = np.sort(y)[::-1]

    # Step 2: Find critical index rho
    # We need: u[j] - (cumsum[j] - z) / (j+1) > 0
    cumsum = np.cumsum(u)
    rho_indices = np.arange(1, N + 1)
    condition = u - (cumsum - z) / rho_indices > 0

    # Find last index where condition is True
    valid_indices = np.where(condition)[0]
    if len(valid_indices) == 0:
        # All elements should be zero (y is extremely negative)
        rho = 1
    else:
        rho = valid_indices[-1] + 1

    # Step 3: Compute threshold theta
    theta = (cumsum[rho - 1] - z) / rho

    # Step 4: Project
    x = np.maximum(y - theta, 0.0)

    # Step 5: Numerical stability - active set residual compensation
    # Instead of rescaling (which breaks optimality), distribute residual
    # evenly among active components (x > 0)
    x_sum = float(np.sum(x))
    residual = z - x_sum

    if abs(residual) > eps:
        # Find active set (components with x > 0)
        active_mask = x > eps
        n_active = int(np.sum(active_mask))

        if n_active > 0:
            # Distribute residual evenly among active components
            x[active_mask] += residual / n_active

            # Re-clip to ensure non-negativity (in case residual was large negative)
            x = np.maximum(x, 0.0)

            # Final sum check
            x_sum_final = float(np.sum(x))
            if abs(x_sum_final - z) > 1e-10:
                # Fallback: if still violated, use rescale
                if x_sum_final > eps:
                    x = x * (z / x_sum_final)

    # Restore original shape
    return x.reshape(original_shape)


def project_shifted_simplex(
    y: np.ndarray,
    min_val: float = 0.0,
    z: float = 1.0,
) -> np.ndarray:
    """
    Project vector onto shifted simplex {x: x >= min_val, sum(x) = z}.

    This is useful when we need to enforce a minimum value for each component
    (e.g., avoid extremely small mass fractions that cause numerical issues).

    Args:
        y: Input vector, shape (N,)
        min_val: Minimum value for each component (must satisfy N*min_val <= z)
        z: Target sum (default 1.0)

    Returns:
        x: Projected vector, shape (N,)

    Algorithm:
        1. Check feasibility: N * min_val <= z
        2. Transform: y_shifted = y - min_val
        3. Project y_shifted onto simplex with sum = z - N*min_val
        4. Transform back: x = x_shifted + min_val

    Raises:
        ValueError: If N * min_val > z (infeasible simplex)
    """
    y = np.asarray(y, dtype=np.float64)
    original_shape = y.shape

    if y.ndim == 2 and y.shape[1] == 1:
        y = y.flatten()
    elif y.ndim != 1:
        raise ValueError(f"Input must be 1D or (N, 1), got shape {original_shape}")

    N = len(y)

    # Check feasibility
    # FIXED: Use > instead of >= (equality is feasible), add tolerance for floating-point
    tol = 1e-14
    if min_val < 0:
        raise ValueError(f"min_val must be non-negative, got {min_val}")
    if N * min_val > z + tol:
        raise ValueError(
            f"Infeasible shifted simplex: N*min_val = {N}*{min_val} = {N*min_val} > z = {z}"
        )

    # Special case: min_val = 0 reduces to standard simplex
    if min_val == 0.0:
        return project_simplex(y, z=z).reshape(original_shape)

    # Coordinate transformation
    y_shifted = y - min_val
    target_sum = z - N * min_val

    # Project onto standard simplex
    x_shifted = project_simplex(y_shifted, z=target_sum)

    # Transform back
    x = x_shifted + min_val

    # Verify result
    # NOTE: Do NOT rescale here (rescaling breaks the Euclidean projection optimality).
    # project_simplex() already performs an active-set residual compensation.
    x_sum = float(np.sum(x))


    # FIXED: Replace assert with explicit raise (asserts can be disabled with -O)
    if not np.all(x >= min_val - 1e-12):
        raise ValueError(f"Projected values below min_val: min(x)={np.min(x):.3e}, min_val={min_val}")
    if abs(np.sum(x) - z) >= 1e-10:
        raise ValueError(f"Sum constraint violated: sum(x)={np.sum(x):.3e}, z={z}")

    return x.reshape(original_shape)


def project_fixed_component_simplex(
    y: np.ndarray,
    fixed_idx: int,
    fixed_val: float,
    min_val: float = 0.0,
) -> np.ndarray:
    """
    Project vector onto simplex with one component fixed.

    This is useful for interface conditions where one species (e.g., saturated
    vapor) is fixed at equilibrium concentration, and others must sum to
    complement.

    Args:
        y: Input vector, shape (N,)
        fixed_idx: Index of component to fix
        fixed_val: Value to fix component at
        min_val: Minimum value for free components (default 0.0)

    Returns:
        x: Projected vector with x[fixed_idx] = fixed_val, others projected
           onto simplex with sum = 1 - fixed_val

    Mathematical formulation:
        x* = argmin ||x - y||_2^2
        s.t. x[fixed_idx] = fixed_val
             x[i] >= min_val for i != fixed_idx
             sum(x[i] for i != fixed_idx) = 1 - fixed_val

    Raises:
        ValueError: If fixed_val out of [0, 1] or if free simplex is infeasible
    """
    y = np.asarray(y, dtype=np.float64)
    original_shape = y.shape

    if y.ndim == 2 and y.shape[1] == 1:
        y = y.flatten()
    elif y.ndim != 1:
        raise ValueError(f"Input must be 1D or (N, 1), got shape {original_shape}")

    N = len(y)

    # Validate inputs
    # FIXED: Add tolerance for floating-point comparison
    tol = 1e-14
    if not (0 <= fixed_idx < N):
        raise ValueError(f"fixed_idx={fixed_idx} out of range [0, {N})")
    if not (0.0 - tol <= fixed_val <= 1.0 + tol):
        raise ValueError(f"fixed_val={fixed_val} must be in [0, 1]")

    # FIXED: Check if fixed_val is below min_val (if min_val > 0)
    if min_val > 0 and fixed_val < min_val - tol:
        raise ValueError(
            f"Infeasible: fixed_val={fixed_val} < min_val={min_val}"
        )

    # Remaining sum for free components
    remaining_sum = 1.0 - fixed_val

    if remaining_sum < -tol:
        raise ValueError(f"Infeasible: fixed_val={fixed_val} > 1.0")

    # Extract free components
    mask = np.ones(N, dtype=bool)
    mask[fixed_idx] = False
    y_free = y[mask]

    # Check feasibility for free components
    # FIXED: Add tolerance to avoid false positives from floating-point errors
    N_free = N - 1
    if N_free * min_val > remaining_sum + tol:
        raise ValueError(
            f"Infeasible free simplex: {N_free}*{min_val} = {N_free*min_val:.3e} > remaining_sum = {remaining_sum:.3e}"
        )

    # Project free components
    if min_val > 0:
        x_free = project_shifted_simplex(y_free, min_val=min_val, z=remaining_sum)
    else:
        x_free = project_simplex(y_free, z=remaining_sum)

    # Assemble full vector
    x = np.empty(N, dtype=np.float64)
    x[fixed_idx] = fixed_val
    x[mask] = x_free.flatten()

    # FIXED: Replace assert with explicit raise
    if abs(x[fixed_idx] - fixed_val) >= 1e-14:
        raise ValueError(f"Fixed component changed: x[{fixed_idx}]={x[fixed_idx]:.3e}, expected={fixed_val:.3e}")
    if abs(np.sum(x) - 1.0) >= 1e-10:
        raise ValueError(f"Sum constraint violated: sum={np.sum(x):.3e}, expected=1.0")

    return x.reshape(original_shape)


def project_Y_cellwise(
    Y: np.ndarray,
    *,
    min_Y: float = 0.0,
    axis: int = 0,
    check_result: bool = True,
) -> np.ndarray:
    """
    Project mass fraction array onto simplex, cell-by-cell.

    This is the main interface for use in evaporation code. It handles
    multi-cell arrays and applies simplex projection independently to each cell.

    Args:
        Y: Mass fraction array
           - Shape (Ns, Ncells): species × cells (axis=0 for species)
           - Shape (Ncells, Ns): cells × species (axis=1 for species)
        min_Y: Minimum mass fraction (default 0.0, recommend 1e-14)
        axis: Axis along which species are arranged (default 0)
        check_result: If True, verify projection succeeded (default True)

    Returns:
        Y_proj: Projected mass fractions with same shape as input
                Guarantees: Y_proj >= min_Y, sum(Y_proj, axis=axis) = 1.0

    Example:
        >>> Y = np.array([[0.6, 0.4], [0.5, 0.7], [-0.1, -0.1]])  # 3 species, 2 cells
        >>> Y_proj = project_Y_cellwise(Y, min_Y=1e-14)
        >>> print(Y_proj.sum(axis=0))  # [1.0, 1.0]

    Raises:
        ValueError: If projection fails or result violates constraints
    """
    Y = np.asarray(Y, dtype=np.float64)

    # Handle 1D case (single cell)
    if Y.ndim == 1:
        if min_Y > 0:
            return project_shifted_simplex(Y, min_val=min_Y, z=1.0)
        else:
            return project_simplex(Y, z=1.0)

    # Handle 2D case (multiple cells)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 1D or 2D array, got shape {Y.shape}")

    # Transpose if needed so species are along axis 0
    if axis == 1:
        Y = Y.T

    Ns, Ncells = Y.shape

    # Check feasibility (with tolerance for floating-point)
    tol = 1e-14
    if Ns * min_Y > 1.0 + tol:
        raise ValueError(
            f"Infeasible: Ns*min_Y = {Ns}*{min_Y} = {Ns*min_Y:.3e} > 1.0"
        )

    # Project each cell
    Y_proj = np.empty_like(Y)
    for j in range(Ncells):
        if min_Y > 0:
            Y_proj[:, j] = project_shifted_simplex(
                Y[:, j], min_val=min_Y, z=1.0
            ).flatten()
        else:
            Y_proj[:, j] = project_simplex(Y[:, j], z=1.0).flatten()

    # Transpose back if needed
    if axis == 1:
        Y_proj = Y_proj.T

    # Verify result
    if check_result:
        sum_axis = 1 if axis == 1 else 0
        Y_sums = np.sum(Y_proj, axis=sum_axis)
        max_sum_error = float(np.max(np.abs(Y_sums - 1.0)))

        if max_sum_error > 1e-10:
            raise ValueError(
                f"Projection failed: max|sum(Y)-1| = {max_sum_error:.3e} > 1e-10"
            )

        if min_Y > 0:
            min_value = float(np.min(Y_proj))
            if min_value < min_Y - 1e-12:
                raise ValueError(
                    f"Projection failed: min(Y) = {min_value:.3e} < min_Y = {min_Y:.3e}"
                )

    return Y_proj


def compute_projection_stats(
    Y_raw: np.ndarray,
    Y_proj: np.ndarray,
    axis: int = 0,
) -> dict:
    """
    Compute statistics about simplex projection operation.

    This is useful for diagnostics to understand how much correction was needed.

    Args:
        Y_raw: Original mass fractions before projection
        Y_proj: Projected mass fractions after projection
        axis: Species axis

    Returns:
        Dictionary with statistics:
            - delta_max: max|Y_proj - Y_raw| (L-infinity norm)
            - delta_rms: sqrt(mean((Y_proj - Y_raw)^2)) (L2 norm)
            - sum_error_max_before: max|sum(Y_raw) - 1|
            - sum_error_max_after: max|sum(Y_proj) - 1|
            - min_val_before: min(Y_raw)
            - min_val_after: min(Y_proj)
            - n_cells_corrected: number of cells where correction > 1e-14
    """
    Y_raw = np.asarray(Y_raw, dtype=np.float64)
    Y_proj = np.asarray(Y_proj, dtype=np.float64)

    # Compute deltas
    delta = Y_proj - Y_raw
    delta_max = float(np.max(np.abs(delta)))
    delta_rms = float(np.sqrt(np.mean(delta**2)))

    # Sum errors
    sum_axis = 1 if axis == 1 else 0
    if Y_raw.ndim == 2:
        sum_raw = np.sum(Y_raw, axis=sum_axis)
        sum_proj = np.sum(Y_proj, axis=sum_axis)
        sum_error_before = float(np.max(np.abs(sum_raw - 1.0)))
        sum_error_after = float(np.max(np.abs(sum_proj - 1.0)))

        # FIXED: Count cells with significant correction
        # Use axis (species axis) to compute per-cell max, not (1-sum_axis)
        # axis=0 (Ns×Ncells) → max over species → Ncells values
        # axis=1 (Ncells×Ns) → max over species → Ncells values
        cell_deltas = np.max(np.abs(delta), axis=axis)
        n_cells_corrected = int(np.sum(cell_deltas > 1e-14))
    else:
        sum_error_before = float(abs(np.sum(Y_raw) - 1.0))
        sum_error_after = float(abs(np.sum(Y_proj) - 1.0))
        n_cells_corrected = 1 if delta_max > 1e-14 else 0

    # Min values
    min_val_before = float(np.min(Y_raw))
    min_val_after = float(np.min(Y_proj))

    return {
        "delta_max": delta_max,
        "delta_rms": delta_rms,
        "sum_error_max_before": sum_error_before,
        "sum_error_max_after": sum_error_after,
        "min_val_before": min_val_before,
        "min_val_after": min_val_after,
        "n_cells_corrected": n_cells_corrected,
    }
