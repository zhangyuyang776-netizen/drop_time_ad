"""
Grid construction from CaseConfig geometry settings.

Builds a static 1D spherical mesh (liquid + gas) using tanh stretching on each
segment, with interface-concentrated refinement.
"""

from __future__ import annotations

import logging
import os
import numpy as np

from .types import CaseConfig, Grid1D, FloatArray, RemapPlan

logger = logging.getLogger(__name__)


def _build_segment_tanh(L: float, N: int, *, beta: float = 3.0, center_bias: float = 0.0) -> FloatArray:
    """
    Generate positive cell widths on [0, L] using a tanh mapping.

    center_bias > 0 -> cluster toward the right end; center_bias < 0 -> left end.
    Returns an array of length N whose sum is L.

    Robustness note:
    - For large |beta| the tanh mapping can saturate and create repeated nodes in float64.
      We enforce strict monotonicity of the node mapping to avoid zero widths.
    """
    N = int(N)
    L = float(L)
    beta = float(beta)
    center_bias = float(center_bias)

    if N <= 0:
        return np.array([], dtype=np.float64)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("Segment length L must be positive and finite.")
    if not np.isfinite(beta) or not np.isfinite(center_bias):
        raise ValueError("beta and center_bias must be finite.")

    if abs(beta) < 1.0e-14:
        return np.full(N, L / N, dtype=np.float64)

    s = np.linspace(-1.0, 1.0, N + 1, dtype=np.float64)
    y = np.tanh(beta * (s + center_bias)).astype(np.float64)

    y_m = y.copy()
    for i in range(1, y_m.size):
        if not (y_m[i] > y_m[i - 1]):
            y_m[i] = np.nextafter(y_m[i - 1], np.inf)

    y0 = float(y_m[0])
    y1 = float(y_m[-1])
    den = y1 - y0
    if (not np.isfinite(den)) or den <= 0.0:
        raise ValueError("tanh grid mapping is degenerate; try smaller |beta| or different mapping.")

    xi = (y_m - y0) / den
    xi[0] = 0.0
    xi[-1] = 1.0

    nodes = L * xi
    widths = np.diff(nodes)

    if np.any(~np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError("tanh grid produced non-positive or non-finite widths.")

    widths *= (L / float(np.sum(widths)))
    return widths.astype(np.float64)


def _beta_for_tanh_target_edge_width(
    L: float,
    N: int,
    target_dr: float,
    *,
    edge: str,
    center_bias: float,
    beta_lo: float = 0.05,
    beta_hi_init: float = 2.0,
    beta_hi_max: float = 128.0,
    rtol: float = 1e-12,
    max_iter: int = 80,
) -> float:
    """
    Solve for beta such that the tanh segment produces a specific edge cell width.

    Assumption (holds for our usage): increasing beta strengthens clustering, reducing the
    target edge width on the clustered side.

    This function uses:
    - feasibility checks
    - progressive bracketing of beta_hi (avoid jumping to extreme values)
    - bisection for robustness
    """
    L = float(L)
    N = int(N)
    target_dr = float(target_dr)
    center_bias = float(center_bias)

    if N <= 0:
        raise ValueError("N must be positive.")
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("L must be positive and finite.")
    if not np.isfinite(target_dr) or target_dr <= 0.0:
        raise ValueError("target_dr must be positive and finite.")
    if edge not in ("left", "right"):
        raise ValueError("edge must be 'left' or 'right'.")

    uniform = L / N
    if target_dr > uniform * (1.0 + 1.0e-14):
        raise ValueError(
            f"target_dr too large for clustered tanh grid: target_dr={target_dr:g} > L/N={uniform:g}. "
            "Try decreasing iface_dr or using a different mapping."
        )

    def edge_width(beta: float) -> float:
        widths = _build_segment_tanh(L=L, N=N, beta=float(beta), center_bias=center_bias)
        return float(widths[0] if edge == "left" else widths[-1])

    b_lo = float(beta_lo)
    w_lo = edge_width(b_lo)
    if w_lo <= target_dr * (1.0 + rtol):
        return b_lo

    b_hi = float(max(beta_hi_init, b_lo * 1.2))
    w_hi = edge_width(b_hi)
    while (w_hi > target_dr) and (b_hi < beta_hi_max):
        b_hi = min(beta_hi_max, b_hi * 1.7)
        w_hi = edge_width(b_hi)

    if w_hi > target_dr * (1.0 + rtol):
        raise ValueError(
            f"target_dr too small for tanh grid within beta<= {beta_hi_max:g}: "
            f"target_dr={target_dr:g}, achieved_edge_width={w_hi:g} at beta={b_hi:g}. "
            "Try increasing N or using a geometric mapping."
        )

    for _ in range(max_iter):
        b_mid = 0.5 * (b_lo + b_hi)
        w_mid = edge_width(b_mid)
        if abs(w_mid - target_dr) <= rtol * max(target_dr, 1.0):
            return b_mid
        if w_mid > target_dr:
            b_lo = b_mid
        else:
            b_hi = b_mid

    return 0.5 * (b_lo + b_hi)


def _geometric_w0_from_L_N_q(L: float, N: int, q: float) -> float:
    L = float(L)
    N = int(N)
    q = float(q)
    if N <= 0 or L <= 0.0:
        raise ValueError(f"Invalid geometric segment sizes: L={L}, N={N}.")
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError(f"q must be positive and finite, got {q}.")
    if abs(q - 1.0) < 1.0e-12:
        return L / N
    denom = q**N - 1.0
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError(f"Invalid geometric denominator for q={q}, N={N}.")
    return L * (q - 1.0) / denom


def _geometric_L_from_w0_N_q(w0: float, N: int, q: float) -> float:
    w0 = float(w0)
    N = int(N)
    q = float(q)
    if N <= 0 or w0 <= 0.0:
        raise ValueError(f"Invalid geometric inputs: w0={w0}, N={N}.")
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError(f"q must be positive and finite, got {q}.")
    if abs(q - 1.0) < 1.0e-12:
        return w0 * N
    return w0 * (q**N - 1.0) / (q - 1.0)


def _solve_geometric_q_for_L_N_w0(
    L: float,
    N: int,
    w0: float,
    *,
    q_hi_init: float = 1.1,
    q_hi_max: float = 1.0e6,
    rtol: float = 1.0e-12,
    max_iter: int = 80,
) -> float:
    L = float(L)
    N = int(N)
    w0 = float(w0)
    if N <= 0:
        raise ValueError("N must be positive.")
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("L must be positive and finite.")
    if not np.isfinite(w0) or w0 <= 0.0:
        raise ValueError("w0 must be positive and finite.")
    if N == 1:
        if abs(L - w0) > rtol * max(L, 1.0):
            raise ValueError(f"No geometric q satisfies L={L} with N=1 and w0={w0}.")
        return 1.0

    min_L = w0 * N
    if L < min_L * (1.0 - 1.0e-12):
        raise ValueError(f"Requested L={L} is too small for w0={w0} and N={N}.")
    if abs(L - min_L) <= rtol * max(L, 1.0):
        return 1.0

    q_lo = 1.0
    q_hi = max(q_hi_init, 1.0 + 1.0e-6)
    L_hi = _geometric_L_from_w0_N_q(w0, N, q_hi)
    while L_hi < L and q_hi < q_hi_max:
        q_hi = min(q_hi_max, q_hi * 1.6)
        L_hi = _geometric_L_from_w0_N_q(w0, N, q_hi)

    if L_hi < L:
        raise ValueError(
            f"Failed to bracket geometric q for L={L}, w0={w0}, N={N} (q_hi_max={q_hi_max})."
        )

    for _ in range(max_iter):
        q_mid = 0.5 * (q_lo + q_hi)
        L_mid = _geometric_L_from_w0_N_q(w0, N, q_mid)
        if abs(L_mid - L) <= rtol * max(L, 1.0):
            return q_mid
        if L_mid < L:
            q_lo = q_mid
        else:
            q_hi = q_mid

    return 0.5 * (q_lo + q_hi)


def _build_segment_geometric(
    L: float,
    N: int,
    *,
    q: float | None = None,
    w0_iface: float | None = None,
    cluster_to: str,
) -> FloatArray:
    """
    Generate widths using a geometric progression, clustered to the chosen edge.
    """
    L = float(L)
    N = int(N)
    if N <= 0:
        return np.array([], dtype=np.float64)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("Segment length L must be positive and finite.")
    if (q is None) == (w0_iface is None):
        raise ValueError("Provide exactly one of q or w0_iface for geometric segment.")
    if q is None:
        q = _solve_geometric_q_for_L_N_w0(L=L, N=N, w0=float(w0_iface))
    q = float(q)
    if not np.isfinite(q) or q < 1.0:
        raise ValueError(f"q must be >= 1.0 and finite, got {q}.")
    if cluster_to not in ("left", "right"):
        raise ValueError(f"cluster_to must be 'left' or 'right', got {cluster_to}.")

    if w0_iface is None:
        w0 = _geometric_w0_from_L_N_q(L, N, q)
        widths = w0 * (q ** np.arange(N, dtype=np.float64))
        if cluster_to == "right":
            widths = widths[::-1]
        widths *= (L / float(np.sum(widths)))
        return widths.astype(np.float64)

    w0 = float(w0_iface)
    if not np.isfinite(w0) or w0 <= 0.0:
        raise ValueError(f"w0_iface must be positive and finite, got {w0}.")
    widths = w0 * (q ** np.arange(N, dtype=np.float64))
    if cluster_to == "right":
        widths = widths[::-1]

    sum_w = float(np.sum(widths))
    delta = L - sum_w
    idx = -1 if cluster_to == "left" else 0
    widths[idx] += delta
    if widths[idx] <= 0.0:
        widths *= (L / sum_w)
    return widths.astype(np.float64)


def _exp_L_from_w0_N_alpha_power(w0: float, N: int, alpha: float, power: float) -> float:
    w0 = float(w0)
    N = int(N)
    alpha = float(alpha)
    power = float(power)
    if N <= 0 or w0 <= 0.0:
        raise ValueError("Invalid exp inputs.")
    if N == 1:
        return w0
    if not np.isfinite(alpha) or not np.isfinite(power) or power <= 0.0:
        raise ValueError("alpha must be finite and power>0.")
    i = np.arange(N, dtype=np.float64)
    t = (i / float(N - 1)) ** power
    w = w0 * np.exp(alpha * t)
    return float(np.sum(w))


def _solve_exp_alpha_for_L_N_w0(
    L: float,
    N: int,
    w0: float,
    *,
    power: float,
    alpha_hi_init: float = 1.0,
    alpha_hi_max: float = 1.0e6,
    rtol: float = 1e-12,
    max_iter: int = 80,
) -> float:
    L = float(L)
    N = int(N)
    w0 = float(w0)
    power = float(power)

    if N <= 0:
        raise ValueError("N must be positive.")
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("L must be positive and finite.")
    if not np.isfinite(w0) or w0 <= 0.0:
        raise ValueError("w0 must be positive and finite.")
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("power must be positive and finite.")

    if N == 1:
        if abs(L - w0) > rtol * max(L, 1.0):
            raise ValueError(f"No exp alpha satisfies L={L} with N=1 and w0={w0}.")
        return 0.0

    min_L = w0 * N
    if L < min_L * (1.0 - 1.0e-12):
        raise ValueError(f"Requested L={L} is too small for w0={w0} and N={N}.")
    if abs(L - min_L) <= rtol * max(L, 1.0):
        return 0.0

    a_lo = 0.0
    a_hi = max(float(alpha_hi_init), 1.0e-6)
    L_hi = _exp_L_from_w0_N_alpha_power(w0, N, a_hi, power)
    while L_hi < L and a_hi < alpha_hi_max:
        a_hi = min(alpha_hi_max, a_hi * 1.6)
        L_hi = _exp_L_from_w0_N_alpha_power(w0, N, a_hi, power)
    if L_hi < L:
        raise ValueError(f"Failed to bracket exp alpha for L={L}, w0={w0}, N={N}.")

    for _ in range(max_iter):
        a_mid = 0.5 * (a_lo + a_hi)
        L_mid = _exp_L_from_w0_N_alpha_power(w0, N, a_mid, power)
        if abs(L_mid - L) <= rtol * max(L, 1.0):
            return a_mid
        if L_mid < L:
            a_lo = a_mid
        else:
            a_hi = a_mid
    return 0.5 * (a_lo + a_hi)


def _build_segment_exp_power(
    L: float,
    N: int,
    *,
    w0_iface: float,
    cluster_to: str,
    power: float = 2.0,
) -> FloatArray:
    L = float(L)
    N = int(N)
    w0 = float(w0_iface)
    if N <= 0:
        return np.array([], dtype=np.float64)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("Segment length L must be positive and finite.")
    if not np.isfinite(w0) or w0 <= 0.0:
        raise ValueError("w0_iface must be positive and finite.")
    if cluster_to not in ("left", "right"):
        raise ValueError("cluster_to must be 'left' or 'right'.")
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("power must be positive and finite.")

    alpha = _solve_exp_alpha_for_L_N_w0(L=L, N=N, w0=w0, power=power)

    if N == 1:
        widths = np.array([w0], dtype=np.float64)
    else:
        i = np.arange(N, dtype=np.float64)
        t = (i / float(N - 1)) ** float(power)
        widths = w0 * np.exp(alpha * t)

    if cluster_to == "right":
        widths = widths[::-1]

    sum_w = float(np.sum(widths))
    delta = L - sum_w
    idx = -1 if cluster_to == "left" else 0
    widths[idx] += delta

    if np.any(~np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError("exp_power grid produced non-positive or non-finite widths.")
    return widths.astype(np.float64)


def _segment_widths_to_faces(r0: float, widths: FloatArray) -> FloatArray:
    """Construct face coordinates from a starting radius and cell widths."""
    r0 = float(r0)
    widths = np.asarray(widths, dtype=np.float64)
    faces = np.empty(widths.size + 1, dtype=np.float64)
    faces[0] = r0
    np.cumsum(widths, out=faces[1:])
    faces[1:] += r0
    return faces


def _check_iface_dr_limits(
    *,
    iface_dr: float,
    a0: float,
    Lg: float,
    Nl: int,
    Ng: int,
    strict: bool = True,
    safety: float = 1.0 - 1.0e-12,
    logger: logging.Logger | None = None,
) -> float:
    """
    Ensure iface_dr is feasible; optionally clip instead of raising.

    Returns iface_dr used (possibly clipped).
    """
    if Nl <= 0 or Ng <= 0:
        return float(iface_dr)
    iface_dr = float(iface_dr)
    a0 = float(a0)
    Lg = float(Lg)
    max_liq = a0 / float(Nl)
    max_gas = Lg / float(Ng)
    lim = min(max_liq, max_gas) * float(safety)
    if iface_dr <= lim * (1.0 + 1.0e-15):
        return iface_dr

    if strict:
        raise ValueError(
            "iface_dr is too large for interface clustering: "
            f"iface_dr={iface_dr:.6e}, a0/N_liq={max_liq:.6e}, Lg/N_gas={max_gas:.6e}. "
            "Try increasing N_liq/N_gas, reducing iface_dr, or adjusting Rd/R_inf."
        )

    if logger is not None:
        try:
            logger.warning(
                "iface_dr is too large for interface clustering; "
                "iface_dr=%.6e, a0/N_liq=%.6e, Lg/N_gas=%.6e. Clipping to %.6e.",
                iface_dr,
                max_liq,
                max_gas,
                lim,
            )
        except Exception:
            pass
    return lim


def _build_grid_for_radius(
    cfg: CaseConfig,
    Rd: float,
    Nl: int | None = None,
    Ng: int | None = None,
    *,
    strict_iface_dr: bool = True,
) -> Grid1D:
    """
    Build a 1D spherical grid for a given droplet radius Rd.

    Nl/Ng can be specified to enforce consistency with an existing layout/grid.
    """
    gcfg = cfg.geometry
    mesh = gcfg.mesh

    Nl_base = int(gcfg.N_liq)
    Ng_base = int(gcfg.N_gas)
    if Nl is None:
        Nl = Nl_base
    if Ng is None:
        Ng = Ng_base
    if Nl != Nl_base or Ng != Ng_base:
        raise ValueError(f"N_liq/N_gas mismatch with geometry: ({Nl},{Ng}) vs ({Nl_base},{Ng_base})")

    a0 = float(Rd)
    Rinf = float(gcfg.R_inf)
    if a0 <= 0.0:
        raise ValueError(f"Rd_new must be positive, got {a0}")
    if a0 >= Rinf:
        raise ValueError(f"Rd_new={a0} must be less than R_inf={Rinf}")

    mesh_method = getattr(mesh, "method", None)
    liq_method = getattr(mesh, "liq_method", None) or mesh_method
    gas_method = getattr(mesh, "gas_method", None) or mesh_method
    if not (0.0 < a0 < Rinf):
        raise ValueError(f"Require 0 < a0 < R_inf, got a0={a0}, R_inf={Rinf}.")
    if Nl < 0 or Ng < 0 or (Nl + Ng) <= 0:
        raise ValueError(f"Invalid cell counts: N_liq={Nl}, N_gas={Ng}.")

    control = getattr(mesh, "control", None)
    iface_dr = getattr(mesh, "iface_dr", None)
    if control is None:
        control = "iface_dr" if iface_dr is not None else "gas_beta"
    control = str(control).strip().lower()
    if control not in ("iface_dr", "gas_beta", "auto_health"):
        raise ValueError(f"mesh.control must be 'iface_dr', 'gas_beta', or 'auto_health', got {control}.")

    allowed_methods = {"tanh", "geometric", "exp_power"}
    if liq_method not in allowed_methods or gas_method not in allowed_methods:
        raise ValueError(
            f"Unsupported mesh method (liq={liq_method}, gas={gas_method}); "
            f"allowed: {sorted(allowed_methods)}."
        )

    if control == "auto_health":
        if liq_method != "geometric" or gas_method != "geometric":
            raise ValueError(
                f"auto_health requires geometric mesh method (liq={liq_method}, gas={gas_method})."
            )
    else:
        if control == "gas_beta":
            if liq_method != "tanh" or gas_method != "tanh":
                raise ValueError(
                    f"gas_beta control requires tanh mesh method (liq={liq_method}, gas={gas_method})."
                )
        elif liq_method != gas_method:
            logger.warning(
                "liq_method (%s) differs from gas_method (%s) with iface_dr control; interface continuity will use w0 match.",
                liq_method,
                gas_method,
            )

    # store widths for optional interface continuity check
    wL: FloatArray | None = None
    wG: FloatArray | None = None

    if control == "auto_health":
        if Nl <= 0 or Ng <= 0:
            raise ValueError("auto_health requires N_liq > 0 and N_gas > 0.")

        adj_max_target = float(getattr(mesh, "adj_max_target", 1.30))
        if not np.isfinite(adj_max_target) or adj_max_target < 1.0:
            raise ValueError(f"adj_max_target must be >= 1.0, got {adj_max_target}.")

        global_max_target = getattr(mesh, "global_max_target", None)
        if global_max_target is not None:
            global_max_target = float(global_max_target)
            if not np.isfinite(global_max_target) or global_max_target < 1.0:
                raise ValueError(f"global_max_target must be >= 1.0, got {global_max_target}.")

        liq_adj_max_target = getattr(mesh, "liq_adj_max_target", None)
        if liq_adj_max_target is None:
            liq_adj_max_target = adj_max_target
        else:
            liq_adj_max_target = float(liq_adj_max_target)
            if not np.isfinite(liq_adj_max_target) or liq_adj_max_target < 1.0:
                raise ValueError(f"liq_adj_max_target must be >= 1.0, got {liq_adj_max_target}.")

        auto_adjust_n_liq = bool(getattr(mesh, "auto_adjust_n_liq", False))
        if auto_adjust_n_liq:
            logger.warning("auto_adjust_n_liq requested but not supported; keeping N_liq=%d.", Nl)

        Lg = Rinf - a0
        Ll = a0
        if Ng <= 1:
            q_gas = 1.0
        else:
            q_gas = adj_max_target
            if global_max_target is not None:
                q_global = global_max_target ** (1.0 / max(Ng - 1, 1))
                q_gas = min(q_gas, q_global)
            q_gas = max(q_gas, 1.0)

        dr_if = _geometric_w0_from_L_N_q(Lg, Ng, q_gas)
        needs_liq_adjust = False
        if Nl == 1:
            if abs(Ll - dr_if) > 1.0e-12 * max(Ll, dr_if, 1.0):
                needs_liq_adjust = True
        elif Ll < dr_if * Nl * (1.0 - 1.0e-12):
            needs_liq_adjust = True

        if needs_liq_adjust:
            dr_if = Ll / Nl
            logger.warning(
                "auto_health fallback: L_liq incompatible with gas-derived dr_if; adjusted dr_if=%.6e (L_liq=%.6e, N_liq=%d).",
                dr_if,
                Ll,
                Nl,
            )
            q_gas = _solve_geometric_q_for_L_N_w0(Lg, Ng, dr_if)
            if q_gas > adj_max_target * (1.0 + 1.0e-12):
                logger.warning(
                    "auto_health: q_gas=%.3f exceeds adj_max_target=%.3f after fallback (N_gas=%d).",
                    q_gas,
                    adj_max_target,
                    Ng,
                )
            if global_max_target is not None and Ng > 1:
                q_global = global_max_target ** (1.0 / max(Ng - 1, 1))
                if q_gas > q_global * (1.0 + 1.0e-12):
                    logger.warning(
                        "auto_health: q_gas=%.3f exceeds global_max_target=%.3f after fallback (N_gas=%d).",
                        q_gas,
                        global_max_target,
                        Ng,
                    )

        q_liq = _solve_geometric_q_for_L_N_w0(Ll, Nl, dr_if)
        if q_liq > liq_adj_max_target * (1.0 + 1.0e-12):
            if liq_adj_max_target <= 1.0 + 1.0e-12:
                n_liq_min = int(np.ceil(Ll / dr_if))
            else:
                n_liq_min = int(
                    np.ceil(
                        np.log(1.0 + Ll * (liq_adj_max_target - 1.0) / dr_if) / np.log(liq_adj_max_target)
                    )
                )
            logger.warning(
                "auto_health: q_liq=%.3f exceeds liq_adj_max_target=%.3f; N_liq_min=%d.",
                q_liq,
                liq_adj_max_target,
                n_liq_min,
            )

        wG = _build_segment_geometric(Lg, Ng, q=q_gas, cluster_to="left")
        wL = _build_segment_geometric(Ll, Nl, q=q_liq, cluster_to="right")
        fG = _segment_widths_to_faces(a0, wG)
        fL = _segment_widths_to_faces(0.0, wL)

        logger.info(
            "auto_health geometric: q_gas=%.3f q_liq=%.3f dr_if=%.6e N_liq=%d N_gas=%d L_liq=%.6e L_gas=%.6e",
            q_gas,
            q_liq,
            dr_if,
            Nl,
            Ng,
            Ll,
            Lg,
        )
    else:
        if getattr(mesh, "liq_beta", None) is not None:
            logger.warning("mesh.liq_beta is deprecated and ignored; liquid beta is now matched to interface dr.")
        if getattr(mesh, "liq_center_bias", None) is not None:
            logger.warning("mesh.liq_center_bias is deprecated and ignored; using center_bias=0.5.")
        if getattr(mesh, "gas_center_bias", None) is not None:
            logger.warning("mesh.gas_center_bias is deprecated and ignored; using center_bias=-0.5.")

        beta_gas = None
        if control == "iface_dr":
            if iface_dr is None:
                raise ValueError("mesh.iface_dr must be provided when control='iface_dr'.")
            iface_dr = float(iface_dr)
            if iface_dr <= 0.0:
                raise ValueError(f"mesh.iface_dr must be positive, got {iface_dr}.")
        else:
            beta_gas = getattr(mesh, "gas_beta", None)
            if beta_gas is None:
                raise ValueError("mesh.gas_beta must be provided when control='gas_beta'.")
            beta_gas = float(beta_gas)
            if beta_gas <= 0.0:
                raise ValueError(f"mesh.gas_beta must be >0, got {beta_gas}.")

        bias_liq = 0.5
        bias_gas = -0.5

        # Gas segment [a0, R_inf] (determine interface dr)
        Lg = Rinf - a0
        dr_if: float | None = None
        if Ng > 0:
            if control == "iface_dr":
                iface_dr = _check_iface_dr_limits(
                    iface_dr=iface_dr,
                    a0=a0,
                    Lg=Lg,
                    Nl=Nl,
                    Ng=Ng,
                    strict=strict_iface_dr,
                    logger=logger,
                )
                if gas_method == "tanh":
                    beta_gas = _beta_for_tanh_target_edge_width(
                        L=Lg,
                        N=Ng,
                        target_dr=iface_dr,
                        edge="left",
                        center_bias=bias_gas,
                    )
                    wG = _build_segment_tanh(L=Lg, N=Ng, beta=beta_gas, center_bias=bias_gas)
                elif gas_method == "geometric":
                    wG = _build_segment_geometric(Lg, Ng, w0_iface=iface_dr, cluster_to="left")
                else:
                    exp_power = float(getattr(mesh, "exp_power", 2.0))
                    wG = _build_segment_exp_power(
                        Lg, Ng, w0_iface=iface_dr, cluster_to="left", power=exp_power
                    )
                dr_if = float(wG[0])
            else:
                wG = _build_segment_tanh(L=Lg, N=Ng, beta=beta_gas, center_bias=bias_gas)
                dr_if = float(wG[0])
            fG = _segment_widths_to_faces(a0, wG)
        else:
            fG = np.array([a0, Rinf], dtype=np.float64)

        # Liquid segment [0, a0] (match interface dr)
        if Nl > 0:
            if dr_if is not None:
                dr_if = _check_iface_dr_limits(
                    iface_dr=dr_if,
                    a0=a0,
                    Lg=Lg,
                    Nl=Nl,
                    Ng=Ng,
                    strict=strict_iface_dr,
                    logger=logger,
                )
                if liq_method == "tanh":
                    beta_liq = _beta_for_tanh_target_edge_width(
                        L=a0,
                        N=Nl,
                        target_dr=dr_if,
                        edge="right",
                        center_bias=bias_liq,
                    )
                    wL = _build_segment_tanh(L=a0, N=Nl, beta=beta_liq, center_bias=bias_liq)
                elif liq_method == "geometric":
                    wL = _build_segment_geometric(a0, Nl, w0_iface=dr_if, cluster_to="right")
                else:
                    exp_power = float(getattr(mesh, "exp_power", 2.0))
                    wL = _build_segment_exp_power(
                        a0, Nl, w0_iface=dr_if, cluster_to="right", power=exp_power
                    )
            else:
                beta_liq = getattr(mesh, "liq_beta", None)
                if beta_liq is None:
                    raise ValueError("mesh.liq_beta must be provided when no interface target is available.")
                beta_liq = float(beta_liq)
                if beta_liq <= 0.0:
                    raise ValueError(f"mesh.liq_beta must be >0, got {beta_liq}.")
                wL = _build_segment_tanh(L=a0, N=Nl, beta=beta_liq, center_bias=bias_liq)
            fL = _segment_widths_to_faces(0.0, wL)
        else:
            fL = np.array([0.0, a0], dtype=np.float64)

    # Optional interface continuity check on adjacent cell widths
    enforce_iface = bool(getattr(cfg.checks, "enforce_grid_state_props_split", False)) and bool(
        getattr(mesh, "enforce_interface_continuity", False)
    )
    if enforce_iface and (wL is not None) and (wG is not None):
        dr_liq = float(wL[-1])
        dr_gas = float(wG[0])
        tol = float(getattr(mesh, "continuity_tol", 0.0))
        denom = max(dr_liq, dr_gas, 1.0)
        rel_err = abs(dr_liq - dr_gas) / denom
        if rel_err > tol:
            logger.warning(
                "Interface grid continuity diagnostic: "
                "dr_liq=%.6e, dr_gas=%.6e, rel_err=%.3e > tol=%.3e",
                dr_liq,
                dr_gas,
                rel_err,
                tol,
            )

    faces_r = np.concatenate([fL, fG[1:]], axis=0)

    iface = Nl
    faces_A = 4.0 * np.pi * faces_r * faces_r
    cells_rc = 0.5 * (faces_r[:-1] + faces_r[1:])
    cells_V = (4.0 * np.pi / 3.0) * (
        np.maximum(faces_r[1:], 0.0) ** 3 - np.maximum(faces_r[:-1], 0.0) ** 3
    )

    grid = Grid1D(
        Nl=Nl,
        Ng=Ng,
        Nc=Nl + Ng,
        r_c=cells_rc,
        r_f=faces_r,
        V_c=cells_V,
        A_f=faces_A,
        iface_f=iface,
    )

    # Grid1D.__post_init__ performs consistency checks.
    return grid


def build_grid(cfg: CaseConfig) -> Grid1D:
    """Build grid for initial radius cfg.geometry.a0."""
    Rd0 = float(cfg.geometry.a0)
    return _build_grid_for_radius(cfg, Rd0, strict_iface_dr=True)


def rebuild_grid_with_Rd(cfg: CaseConfig, Rd_new: float, grid_ref: Grid1D) -> Grid1D:
    """Rebuild grid using updated Rd, keeping Nl/Ng consistent with an existing grid."""
    return _build_grid_for_radius(
        cfg,
        Rd_new,
        Nl=grid_ref.Nl,
        Ng=grid_ref.Ng,
        strict_iface_dr=False,
    )


def rebuild_grid_phase2(cfg: CaseConfig, Rd_new: float, grid_ref: Grid1D) -> tuple[Grid1D, RemapPlan | None]:
    """
    Phase 2: rebuild liquid + gas band with fixed interface refinement, keeping far gas fixed.
    """
    gcfg = cfg.geometry
    mesh = gcfg.mesh

    a0 = float(Rd_new)
    Rinf = float(gcfg.R_inf)
    if a0 <= 0.0:
        raise ValueError(f"Rd_new must be positive, got {a0}")
    if a0 >= Rinf:
        raise ValueError(f"Rd_new={a0} must be less than R_inf={Rinf}")

    if not bool(getattr(mesh, "phase2_enabled", False)):
        return rebuild_grid_with_Rd(cfg, a0, grid_ref), None

    phase2_method = str(getattr(mesh, "phase2_method", "band_geometric")).strip().lower()
    if phase2_method != "band_geometric":
        raise ValueError(f"phase2_method must be 'band_geometric', got {phase2_method}.")

    Nl, Ng = int(grid_ref.Nl), int(grid_ref.Ng)
    iface = int(grid_ref.iface_f)
    if Nl <= 0 or Ng <= 0:
        raise ValueError(f"Phase2 requires Nl>0 and Ng>0, got Nl={Nl}, Ng={Ng}.")

    r_f_ref = np.asarray(grid_ref.r_f, dtype=np.float64)
    if iface <= 0 or iface >= r_f_ref.size - 1:
        raise ValueError(f"Invalid interface index for phase2: iface_f={iface}.")

    wL_last_ref = float(r_f_ref[iface] - r_f_ref[iface - 1])
    wG_first_ref = float(r_f_ref[iface + 1] - r_f_ref[iface])
    dr_if_target = 0.5 * (wL_last_ref + wG_first_ref)
    if dr_if_target <= 0.0 or not np.isfinite(dr_if_target):
        raise ValueError("Invalid reference interface width for phase2 rebuild.")

    frac_target = float(getattr(mesh, "phase2_gas_band_frac_target", 0.15))
    frac_min = float(getattr(mesh, "phase2_gas_band_frac_min", 0.10))
    frac_max = float(getattr(mesh, "phase2_gas_band_frac_max", 0.20))
    if not (0.0 < frac_min <= frac_target <= frac_max <= 1.0):
        raise ValueError(
            "phase2 gas band fractions must satisfy 0 < min <= target <= max <= 1.0 "
            f"(min={frac_min}, target={frac_target}, max={frac_max})."
        )

    k_base = int(round(Ng * frac_target))
    k_min = int(np.ceil(Ng * frac_min))
    k_max = int(np.floor(Ng * frac_max))
    k_max = max(1, min(k_max, Ng))
    if Ng >= 2:
        k_max = max(2, k_max)
    k_min = max(1, min(k_min, k_max))
    if Ng >= 2:
        k_min = max(2, k_min)
    k_base = max(1, min(k_base, Ng))
    k = max(k_min, min(k_base, k_max))

    q_max = float(getattr(mesh, "phase2_adj_max_target", 1.30))
    if not np.isfinite(q_max) or q_max < 1.0:
        raise ValueError(f"phase2_adj_max_target must be >= 1.0, got {q_max}.")

    for _ in range(3):
        j_anchor = iface + k
        if j_anchor >= r_f_ref.size:
            break
        r_anchor = float(r_f_ref[j_anchor])
        L_band = r_anchor - a0
        if L_band <= 0.0:
            break
        if q_max <= 1.0 + 1.0e-12:
            k_need = int(np.ceil(L_band / dr_if_target))
        else:
            arg = 1.0 + L_band * (q_max - 1.0) / dr_if_target
            if arg <= 1.0:
                k_need = 1
            else:
                k_need = int(np.ceil(np.log(arg) / np.log(q_max)))
        k_new = max(k_min, min(max(k_base, k_need), k_max))
        if k_new == k:
            break
        k = k_new

    j_anchor = iface + k
    if j_anchor >= r_f_ref.size:
        logger.warning("phase2: band anchor reaches domain end; falling back to full remesh.")
        return rebuild_grid_with_Rd(cfg, a0, grid_ref), None

    r_anchor = float(r_f_ref[j_anchor])
    L_band = r_anchor - a0
    if L_band <= 0.0:
        logger.warning("phase2: non-positive band length; falling back to full remesh.")
        return rebuild_grid_with_Rd(cfg, a0, grid_ref), None

    if q_max <= 1.0 + 1.0e-12:
        k_need = int(np.ceil(L_band / dr_if_target))
    else:
        arg = 1.0 + L_band * (q_max - 1.0) / dr_if_target
        k_need = int(np.ceil(np.log(arg) / np.log(q_max))) if arg > 1.0 else 1
    if k_need > k_max:
        logger.warning(
            "phase2: k_need > k_max (k_need=%d, k_max=%d) for q_max=%.3f; band fraction may be insufficient.",
            k_need,
            k_max,
            q_max,
        )

    eps = float(getattr(mesh, "phase2_dr_if_eps", 1.0e-15))
    if dr_if_target * k > L_band:
        dr_if_target = (L_band / k) * (1.0 - eps)
        logger.warning(
            "phase2: gas band too short for dr_if_target; adjusted dr_if_target=%.6e (L_band=%.6e, k=%d).",
            dr_if_target,
            L_band,
            k,
        )

    L_liq = a0
    if dr_if_target * Nl > L_liq:
        dr_if_target = (L_liq / Nl) * (1.0 - eps)
        logger.warning(
            "phase2: liquid too short for fixed dr_if; capped dr_if_target=%.6e (L_liq=%.6e, N_liq=%d).",
            dr_if_target,
            L_liq,
            Nl,
        )

    if dr_if_target * k > L_band:
        dr_if_target = (L_band / k) * (1.0 - eps)
        logger.warning(
            "phase2: gas band too short after liquid cap; adjusted dr_if_target=%.6e (L_band=%.6e, k=%d).",
            dr_if_target,
            L_band,
            k,
        )

    try:
        q_band = _solve_geometric_q_for_L_N_w0(L_band, k, dr_if_target)
        q_liq = _solve_geometric_q_for_L_N_w0(L_liq, Nl, dr_if_target)
    except Exception as exc:
        logger.warning("phase2: failed to solve geometric ratios (%s); falling back to full remesh.", exc)
        return rebuild_grid_with_Rd(cfg, a0, grid_ref), None

    if q_band > q_max * (1.0 + 1.0e-12):
        logger.warning(
            "phase2: q_band=%.3f exceeds adj_max_target=%.3f (k=%d).",
            q_band,
            q_max,
            k,
        )
    if q_liq > q_max * (1.0 + 1.0e-12):
        if q_max <= 1.0 + 1.0e-12:
            n_liq_min = int(np.ceil(L_liq / dr_if_target))
        else:
            n_liq_min = int(
                np.ceil(
                    np.log(1.0 + L_liq * (q_max - 1.0) / dr_if_target) / np.log(q_max)
                )
            )
        logger.warning(
            "phase2: q_liq=%.3f exceeds adj_max_target=%.3f; N_liq_min=%d.",
            q_liq,
            q_max,
            n_liq_min,
        )

    w_band = dr_if_target * (q_band ** np.arange(k, dtype=np.float64))
    delta_band = L_band - float(np.sum(w_band))
    w_band[-1] += delta_band
    if w_band[-1] <= 0.0:
        logger.warning("phase2: invalid gas band widths; falling back to full remesh.")
        return rebuild_grid_with_Rd(cfg, a0, grid_ref), None

    w_liq_iface = dr_if_target * (q_liq ** np.arange(Nl, dtype=np.float64))
    delta_liq = L_liq - float(np.sum(w_liq_iface))
    w_liq_iface[-1] += delta_liq
    if w_liq_iface[-1] <= 0.0:
        logger.warning("phase2: invalid liquid widths; falling back to full remesh.")
        return rebuild_grid_with_Rd(cfg, a0, grid_ref), None
    wL = w_liq_iface[::-1]

    fL = _segment_widths_to_faces(0.0, wL)
    fG_band = _segment_widths_to_faces(a0, w_band)

    faces_r = np.empty_like(r_f_ref)
    faces_r[: iface + 1] = fL
    faces_r[iface : iface + k + 1] = fG_band
    faces_r[j_anchor:] = r_f_ref[j_anchor:]
    faces_r[iface] = a0
    faces_r[0] = 0.0
    faces_r[-1] = Rinf
    faces_r[j_anchor] = r_anchor

    if not np.all(np.diff(faces_r) > 0.0):
        logger.warning("phase2: non-monotone faces detected; falling back to full remesh.")
        return rebuild_grid_with_Rd(cfg, a0, grid_ref), None

    iface_idx = Nl
    faces_A = 4.0 * np.pi * faces_r * faces_r
    cells_rc = 0.5 * (faces_r[:-1] + faces_r[1:])
    cells_V = (4.0 * np.pi / 3.0) * (
        np.maximum(faces_r[1:], 0.0) ** 3 - np.maximum(faces_r[:-1], 0.0) ** 3
    )

    grid = Grid1D(
        Nl=Nl,
        Ng=Ng,
        Nc=Nl + Ng,
        r_c=cells_rc,
        r_f=faces_r,
        V_c=cells_V,
        A_f=faces_A,
        iface_f=iface_idx,
    )

    remap_plan = RemapPlan(
        liq_remap=True,
        gas_remap_cells=(iface_idx, iface_idx + k - 1),
        gas_copy_cells=(iface_idx + k, iface_idx + Ng - 1) if k < Ng else None,
        k=k,
        j_anchor=j_anchor,
        dr_if_target=dr_if_target,
        q_band=q_band,
        q_liq=q_liq,
    )

    if os.getenv("DROPLET_PHASE2_DEBUG", "0") == "1":
        if not np.array_equal(faces_r[j_anchor:], r_f_ref[j_anchor:]):
            raise AssertionError("phase2 debug: far faces are not an exact copy of reference.")
        drL = faces_r[iface] - faces_r[iface - 1]
        drG = faces_r[iface + 1] - faces_r[iface]
        if abs(drL - drG) > 1.0e-12 * max(drL, drG, 1.0):
            raise AssertionError("phase2 debug: interface continuity violated.")
        if not remap_plan.liq_remap:
            raise AssertionError("phase2 debug: liquid must be remapped in phase2.")
        if remap_plan.gas_copy_cells is None:
            expected = (iface, iface + Ng - 1)
            if remap_plan.gas_remap_cells != expected:
                raise AssertionError("phase2 debug: gas remap cells do not cover full gas domain.")
        else:
            if remap_plan.gas_remap_cells[0] != iface:
                raise AssertionError("phase2 debug: gas_remap_cells must start at interface.")
            if remap_plan.gas_remap_cells[1] + 1 != remap_plan.gas_copy_cells[0]:
                raise AssertionError("phase2 debug: gas_remap_cells must abut gas_copy_cells.")
            if remap_plan.gas_copy_cells[1] != iface + Ng - 1:
                raise AssertionError("phase2 debug: gas_copy_cells must end at last gas cell.")

    logger.info(
        "phase2 band: Rd_new=%.6e dr_if=%.6e k=%d q_band=%.3f q_liq=%.3f j_anchor=%d r_anchor=%.6e",
        a0,
        dr_if_target,
        k,
        q_band,
        q_liq,
        j_anchor,
        r_anchor,
    )

    return grid, remap_plan
