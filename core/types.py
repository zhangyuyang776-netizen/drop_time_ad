"""
Strongly typed containers for case configuration, grid, state, properties, and diagnostics.

Global shape and sign conventions (law of the land):
- Nl: number of liquid cells; Ng: number of gas cells; Nc = Nl + Ng
- Nspec_g: total gas species (includes closure species); Nspec_g_eff = Nspec_g - 1
- Nspec_l: total liquid species; Nspec_l_eff: effective liquid species count (layout decides)
- Tg.shape == (Ng,); Yg.shape == (Nspec_g, Ng)  # columns are space
- Tl.shape == (Nl,); Yl.shape == (Nspec_l, Nl)  # Yl always exists even if not solved
- Interface scalars: Ts [K], mpp [kg/m^2/s], Rd [m]
- Radial coordinate r increases outward; mpp > 0 means liquid -> gas (evaporation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class CaseMeta:
    """Metadata for the case block."""

    id: str
    title: str
    version: int
    notes: Optional[str] = None


@dataclass(slots=True)
class CasePaths:
    """Case-level paths (resolved elsewhere).

    Attributes
    ----------
    case_dir : Path
        Root directory of the case output 
    mechanism_dir : Path
        Directory containing mechanism/property files.
    gas_mech : str
        Relative or absolute path to the gas mechanism file.

    """
    output_root: Path
    case_dir: Path
    mechanism_dir: Path
    gas_mech: str

    def __post_init__(self) -> None:
        if not self.gas_mech:
            raise ValueError("gas_mech must be provided.")

        for name in ("output_root", "case_dir", "mechanism_dir"):
            v = getattr(self, name)
            if not isinstance(v, Path):
                raise TypeError(f"{name} must be pathlib.Path (loader must convert str -> Path).")
            
@dataclass(slots=True)
class CaseSpecies:
    """Species definitions and closure choices (YAML-aligned)."""

    gas_balance_species: str
    gas_mechanism_phase: str = "gas"
    liq_species: List[str] = field(default_factory=list)
    liq_balance_species: str = ""
    liq2gas_map: Mapping[str, str] = field(default_factory=dict)
    mw_kg_per_mol: Mapping[str, float] = field(default_factory=dict)
    molar_volume_cm3_per_mol: Mapping[str, float] = field(default_factory=dict)
    # Runtime-filled mappings (mechanism order)
    gas_species_full: List[str] = field(default_factory=list)
    gas_name_to_index: Mapping[str, int] = field(default_factory=dict)
    liq_name_to_index: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.gas_balance_species:
            raise ValueError("gas_balance_species must be provided.")
        if not self.liq_balance_species:
            raise ValueError("liq_balance_species must be provided.")
        if self.liq_balance_species not in self.liq_species:
            raise ValueError(
                f"liq_balance_species '{self.liq_balance_species}' must be included in liq_species {self.liq_species}"
            )
        if self.gas_species_full:
            if self.gas_balance_species not in self.gas_species_full:
                raise ValueError(
                    f"gas_balance_species '{self.gas_balance_species}' must be included in gas_species_full {self.gas_species_full}"
                )

@dataclass(slots=True)
class CaseConventions:
    """Diagnostics conventions (YAML conventions block)."""

    radial_normal: str
    flux_sign: str
    heat_flux_def: str
    evap_sign: str
    gas_closure_species: str
    index_source: str
    assembly_pure: bool
    grid_state_props_split: bool


@dataclass(slots=True)
class CaseMesh:
    """Mesh distribution options (geometry.mesh)."""

    liq_method: str
    liq_beta: Optional[float]
    liq_center_bias: Optional[float]
    gas_method: str
    gas_beta: Optional[float]
    gas_center_bias: Optional[float]
    enforce_interface_continuity: bool
    continuity_tol: float
    method: Optional[str] = None
    control: Optional[str] = None
    iface_dr: Optional[float] = None
    adj_max_target: float = 1.30
    global_max_target: Optional[float] = None
    liq_adj_max_target: Optional[float] = None
    auto_adjust_n_liq: bool = False
    phase2_enabled: bool = False
    phase2_method: str = "band_geometric"
    phase2_adj_max_target: float = 1.30
    phase2_gas_band_frac_target: float = 0.15
    phase2_gas_band_frac_min: float = 0.10
    phase2_gas_band_frac_max: float = 0.20
    phase2_dr_if_eps: float = 1.0e-15
    exp_power: float = 2.0


@dataclass(slots=True)
class RemapPlan:
    """Partial remap plan for moving-grid updates."""

    liq_remap: bool = True
    gas_remap_cells: Optional[Tuple[int, int]] = None
    gas_copy_cells: Optional[Tuple[int, int]] = None
    k: Optional[int] = None
    j_anchor: Optional[int] = None
    dr_if_target: Optional[float] = None
    q_band: Optional[float] = None
    q_liq: Optional[float] = None


@dataclass(slots=True)
class CaseGeometry:
    """Geometry parameters (YAML geometry block)."""

    a0: float
    R_inf: float
    N_liq: int
    N_gas: int
    mesh: CaseMesh


@dataclass(slots=True)
class CaseTime:
    """Time control settings."""

    t0: float
    dt: float
    t_end: float
    max_steps: Optional[int] = None


@dataclass(slots=True)
class CaseDiscretization:
    """Time discretization settings."""

    time_scheme: str
    theta: float
    mass_matrix: str


@dataclass(slots=True)
class CaseCoolProp:
    backend: str = "HEOS"
    fluids: List[str] = field(default_factory=list)


@dataclass(slots=True)
class CaseCustomSat:
    """P4: Custom saturation model configuration."""
    model: str = "mm_integral_watson"
    params_file: str = ""


@dataclass(slots=True)
class CaseEquilibrium:
    method: str = "raoult_psat"
    psat_model: str = "coolprop"
    condensables_gas: List[str] = field(default_factory=list)
    coolprop: CaseCoolProp = field(default_factory=CaseCoolProp)
    # P4: Custom saturation support
    sat_source: str = "coolprop"  # "coolprop" | "custom"
    custom_sat: CaseCustomSat = field(default_factory=CaseCustomSat)



@dataclass(slots=True)
class CaseInterface:
    type: str = "no_condensation"
    bc_mode: str = "Ts_fixed"
    Ts_fixed: float = 300.0
    equilibrium: CaseEquilibrium = field(default_factory=CaseEquilibrium)

@dataclass(slots=True)
class CasePhysics:
    """Physics toggles."""

    model: str = "droplet_1d_spherical_noChem"

    enable_liquid: bool = True
    include_chemistry: bool = False

    # unknown toggles
    solve_Tg: bool = True
    solve_Yg: bool = True
    solve_Tl: bool = True
    solve_Yl: bool = False  # Yl must exist even if not solved

    # interface/radius unknowns
    include_Ts: bool = False
    include_mpp: bool = True
    include_Rd: bool = True
    latent_heat_default: Optional[float] = None  # J/kg fallback if props do not provide Lv

    # derived fields (not in UnknownLayout)
    stefan_velocity: bool = True
    species_convection: bool = False

    interface: CaseInterface = field(default_factory=CaseInterface)

    def __post_init__(self) -> None:
        if not self.enable_liquid:
            raise ValueError("Always-present droplet: enable_liquid must be True.")
        if self.include_chemistry:
            raise ValueError("NoChem stage: include_chemistry must be False.")
            
@dataclass(slots=True)
class CaseTransport:
    """Transport defaults/overrides."""

    D_l_const: float = 1.0e-9
    D_l_species: Mapping[str, float] = field(default_factory=dict)

@dataclass(slots=True)
class CaseInitial:
    """Initial/farfield conditions."""

    T_inf: float
    P_inf: float
    T_d0: float
    Yg: Mapping[str, float]
    Yl: Mapping[str, float]
    Y_seed: float
    t_init_T: float = 1.0e-6
    t_init_Y: float = 1.0e-6  # legacy, unused
    D_init_Y: float = 1.0e-5  # legacy, unused
    Y_vap_if0: float = 1.0e-6


@dataclass(slots=True)
class CasePETSc:
    """PETSc solver options (minimal)."""

    options_prefix: str = ""
    ksp_type: str = "gmres"
    pc_type: str = "ilu"
    rtol: float = 1.0e-8
    atol: float = 1.0e-12
    max_it: int = 200
    restart: int = 30
    monitor: bool = False

    # SNES-specific options (used by PETSc nonlinear backend)
    snes_type: str = "newtonls"
    linesearch_type: str = "bt"
    jacobian_mode: str = "fd"
    fd_eps: float = 1.0e-8
    snes_monitor: bool = False
    precond_drop_tol: float = 0.0
    precond_max_nnz_row: Optional[int] = None


@dataclass(slots=True)
class LinearSolverConfig:
    """Linear solver options (backend + assembly mode)."""

    backend: str = "scipy"
    method: Optional[str] = None
    assembly_mode: str = "bridge_dense"
    pc_type: Optional[str] = None
    asm_overlap: Optional[int] = None
    fieldsplit: Optional[Any] = None


@dataclass(slots=True)
class CaseSolver:
    """Solver configuration container."""

    linear: LinearSolverConfig = field(default_factory=LinearSolverConfig)


@dataclass(slots=True)
class CaseNonlinear:
    """Nonlinear solver options (global Newton)."""

    enabled: bool = False
    backend: str = "scipy"

    solver: str = "newton_krylov"
    krylov_method: str = "lgmres"
    inner_maxiter: int = 20

    max_outer_iter: int = 20
    f_rtol: float = 1.0e-6
    f_atol: float = 1.0e-10

    use_scaled_unknowns: bool = True
    use_scaled_residual: bool = True
    residual_scale_floor: float = 1.0e-12

    verbose: bool = False
    log_every: int = 5

    # P3.5-5-1: smoke mode for quick SNES smoke tests (limits iterations)
    smoke: bool = False


@dataclass(slots=True)
class CaseIOFields:
    """Fields to output."""

    scalars: List[str] = field(default_factory=list)
    gas: List[str] = field(default_factory=list)
    liquid: List[str] = field(default_factory=list)
    interface: List[str] = field(default_factory=list)


@dataclass(slots=True)
class CaseIO:
    """Output controls."""

    write_every: int
    formats: List[str]
    save_grid: bool
    fields: CaseIOFields
    scalars_write_every: int = 1


@dataclass(slots=True)
class CaseRemap:
    """Remap and post-correction controls."""

    post_correct_interface: bool = False
    post_correct_tol: float = 1.0e-10
    post_correct_rtol: float = 1.0e-3
    post_correct_tol_skip_abs: float = 1.0e-8
    post_correct_eps_rd_rel: float = 1.0e-6
    post_correct_skip_if_uncovered_zero: bool = True
    post_correct_improve_min: float = 1.0e-3
    post_correct_max_iter: int = 12
    post_correct_damping: float = 0.7
    post_correct_fd_eps_T: float = 1.0e-3
    post_correct_fd_eps_m: float = 1.0e-6


@dataclass(slots=True)
class CaseChecks:
    """Diagnostics checks configuration."""

    enforce_sumY: bool
    sumY_tol: float
    clamp_negative_Y: bool
    min_Y: float
    enforce_T_bounds: bool
    T_min: float
    T_max: float
    enforce_unique_index: bool = True
    enforce_grid_state_props_split: bool = True
    enforce_assembly_purity: bool = True


@dataclass(slots=True)
class CaseOutput:
    """Spatial output controls for u vector and grid coordinates."""

    u_enabled: bool = False
    u_every: int = 1


@dataclass(slots=True)
class CaseConfig:
    """Top-level case configuration container."""

    case: CaseMeta
    paths: CasePaths
    conventions: CaseConventions
    physics: CasePhysics
    species: CaseSpecies
    geometry: CaseGeometry
    time: CaseTime
    discretization: CaseDiscretization
    initial: CaseInitial
    petsc: CasePETSc
    io: CaseIO
    checks: CaseChecks
    transport: CaseTransport = field(default_factory=CaseTransport)
    nonlinear: CaseNonlinear = field(default_factory=CaseNonlinear)
    solver: CaseSolver = field(default_factory=CaseSolver)
    remap: CaseRemap = field(default_factory=CaseRemap)
    output: CaseOutput = field(default_factory=CaseOutput)

    def __post_init__(self) -> None:
        if not isinstance(self.conventions, CaseConventions):
            raise TypeError("conventions must be CaseConventions (loader must build dataclass).")
        if not isinstance(self.species, CaseSpecies):
            raise TypeError("species must be CaseSpecies (loader must build dataclass).")
        if not isinstance(self.physics, CasePhysics):
            raise TypeError("physics must be CasePhysics (loader must build dataclass).")
        if not isinstance(self.transport, CaseTransport):
            raise TypeError("transport must be CaseTransport (loader must build dataclass).")
        if not isinstance(self.nonlinear, CaseNonlinear):
            raise TypeError("nonlinear must be CaseNonlinear (loader must build dataclass).")
        if not isinstance(self.solver, CaseSolver):
            raise TypeError("solver must be CaseSolver (loader must build dataclass).")
        if not isinstance(self.remap, CaseRemap):
            raise TypeError("remap must be CaseRemap (loader must build dataclass).")
        if not isinstance(self.output, CaseOutput):
            raise TypeError("output must be CaseOutput (loader must build dataclass).")
        if self.conventions.gas_closure_species != self.species.gas_balance_species:
            raise ValueError(
                f"gas closure species mismatch: conventions='{self.conventions.gas_closure_species}' "
                f"vs species='{self.species.gas_balance_species}'"
            )

@dataclass(slots=True)
class Grid1D:
    """1D spherical grid container (no generation logic).

    Fields
    ------
    Nl, Ng, Nc : int
        Cell counts; Nc must equal Nl + Ng.
    r_c : (Nc,) float64
        Cell-centered radii [m].
    r_f : (Nc+1,) float64
        Face radii [m].
    V_c : (Nc,) float64
        Cell volumes [m^3].
    A_f : (Nc+1,) float64
        Face areas [m^2].
    iface_f : int
        Global face index for the liquid/gas interface (must equal Nl by convention).
    liq_slice, gas_slice : slice
        Standard slices for liquid/gas cell ranges.
    dr_c, dr_f : optional spacings matching r_c/r_f.
    """

    Nl: int
    Ng: int
    Nc: int
    r_c: FloatArray
    r_f: FloatArray
    V_c: FloatArray
    A_f: FloatArray
    iface_f: int
    liq_slice: Optional[slice] = None
    gas_slice: Optional[slice] = None
    dr_c: Optional[FloatArray] = None
    dr_f: Optional[FloatArray] = None

    def __post_init__(self) -> None:
        expected_nc = self.Nl + self.Ng
        if self.Nc != expected_nc:
            raise ValueError(f"Nc mismatch: expected {expected_nc}, got {self.Nc}")
        if self.r_c.shape != (self.Nc,):
            raise ValueError(f"r_c shape {self.r_c.shape} != ({self.Nc},)")
        if self.V_c.shape != (self.Nc,):
            raise ValueError(f"V_c shape {self.V_c.shape} != ({self.Nc},)")
        if self.r_f.shape != (self.Nc + 1,):
            raise ValueError(f"r_f shape {self.r_f.shape} != ({self.Nc + 1},)")
        if self.A_f.shape != (self.Nc + 1,):
            raise ValueError(f"A_f shape {self.A_f.shape} != ({self.Nc + 1},)")
        if self.dr_c is not None and self.dr_c.shape != (self.Nc,):
            raise ValueError(f"dr_c shape {self.dr_c.shape} != ({self.Nc},)")
        if self.dr_f is not None and self.dr_f.shape != (self.Nc + 1,):
            raise ValueError(f"dr_f shape {self.dr_f.shape} != ({self.Nc + 1},)")
        # Project assumption: always has a droplet (both phases exist)
        if self.Nl <= 0 or self.Ng <= 0:
            raise ValueError(
                f"This project assumes an always-present droplet: require Nl>0 and Ng>0, got Nl={self.Nl}, Ng={self.Ng}"
            )

        # Convention: liquid cells are [0..Nl-1], gas cells are [Nl..Nc-1]
        # Interface face is exactly at index Nl (between cell Nl-1 and Nl)
        if self.iface_f != self.Nl:
            raise ValueError(
                f"iface_f convention violated: require iface_f == Nl (= {self.Nl}), got iface_f={self.iface_f}"
            )

        # --- geometry sanity checks ---
        # 1) faces strictly increasing
        if not np.all(np.diff(self.r_f) > 0.0):
            raise ValueError("r_f must be strictly increasing (monotone outward).")

        # 2) centers lie between adjacent faces
        if np.any(self.r_c <= self.r_f[:-1]) or np.any(self.r_c >= self.r_f[1:]):
            raise ValueError("r_c must lie strictly between surrounding faces r_f[i] < r_c[i] < r_f[i+1].")

        # 3) positive volumes and non-negative areas
        if np.any(self.V_c <= 0.0):
            raise ValueError("V_c must be positive for all cells.")
        if np.any(self.A_f < 0.0):
            raise ValueError("A_f must be non-negative for all faces.")

        if self.liq_slice is None:
            self.liq_slice = slice(0, self.Nl)
        if self.gas_slice is None:
            self.gas_slice = slice(self.Nl, self.Nc)

    def split_cells(self) -> Tuple[slice, slice]:
        """Return (liq_slice, gas_slice) to avoid manual index math."""
        return self.liq_slice, self.gas_slice


@dataclass(slots=True)
class State:
    """State variables (all float64 arrays).

    Tg : (Ng,) K
    Yg : (Nspec_g, Ng) mass fractions, columns are spatial positions
    Tl : (Nl,) K
    Yl : (Nspec_l, Nl) mass fractions; must exist even if not solved
    Ts : float, interface temperature [K]
    mpp : float, evaporation mass flux [kg/m^2/s], >0 means liquid -> gas
    Rd : float, droplet radius [m]
    """

    Tg: FloatArray
    Yg: FloatArray
    Tl: FloatArray
    Yl: FloatArray
    Ts: float
    mpp: float
    Rd: float

    def copy(self) -> "State":
        """Deep copy arrays to decouple from the original state."""
        return State(
            Tg=np.array(self.Tg, copy=True),
            Yg=np.array(self.Yg, copy=True),
            Tl=np.array(self.Tl, copy=True),
            Yl=np.array(self.Yl, copy=True),
            Ts=float(self.Ts),
            mpp=float(self.mpp),
            Rd=float(self.Rd),
        )


@dataclass(slots=True)
class Props:
    """Cell-centered thermophysical properties.

    D_g / D_l are diffusion coefficients with shape (Nspec, Ncells) if provided.
    """

    rho_g: FloatArray
    cp_g: FloatArray
    k_g: FloatArray
    D_g: Optional[FloatArray] = None  # expected shape: (Nspec_g, Ng)
    h_g: Optional[FloatArray] = None  # (Ng,) J/kg
    h_gk: Optional[FloatArray] = None  # (Ns_g, Ng) J/kg
    h_l: Optional[FloatArray] = None  # (Nl,) J/kg

    rho_l: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    cp_l: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    k_l: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    D_l: Optional[FloatArray] = None  # expected shape: (Nspec_l, Nl)
    psat_l: Optional[FloatArray] = None  # (Ns_l,) Pa
    hvap_l: Optional[FloatArray] = None  # (Ns_l,) J/kg
    h_vap_if: Optional[float] = None  # J/kg, interface latent heat (balance species)

    def validate_shapes(self, grid: Grid1D, Ns_g: int, Ns_l: int) -> None:
        """Ensure property arrays match grid and species counts."""
        if self.rho_g.shape != (grid.Ng,):
            raise ValueError(f"rho_g shape {self.rho_g.shape} != ({grid.Ng},)")
        if self.cp_g.shape != (grid.Ng,):
            raise ValueError(f"cp_g shape {self.cp_g.shape} != ({grid.Ng},)")
        if self.k_g.shape != (grid.Ng,):
            raise ValueError(f"k_g shape {self.k_g.shape} != ({grid.Ng},)")
        if self.D_g is not None and self.D_g.shape != (Ns_g, grid.Ng):
            raise ValueError(f"D_g shape {self.D_g.shape} != ({Ns_g}, {grid.Ng})")
        if self.h_g is not None and self.h_g.shape != (grid.Ng,):
            raise ValueError(f"h_g shape {self.h_g.shape} != ({grid.Ng},)")
        if self.h_gk is not None and self.h_gk.shape != (Ns_g, grid.Ng):
            raise ValueError(f"h_gk shape {self.h_gk.shape} != ({Ns_g}, {grid.Ng})")

        if self.rho_l.shape != (grid.Nl,):
            raise ValueError(f"rho_l shape {self.rho_l.shape} != ({grid.Nl},)")
        if self.cp_l.shape != (grid.Nl,):
            raise ValueError(f"cp_l shape {self.cp_l.shape} != ({grid.Nl},)")
        if self.k_l.shape != (grid.Nl,):
            raise ValueError(f"k_l shape {self.k_l.shape} != ({grid.Nl},)")
        if self.D_l is not None and self.D_l.shape != (Ns_l, grid.Nl):
            raise ValueError(f"D_l shape {self.D_l.shape} != ({Ns_l}, {grid.Nl})")
        if self.h_l is not None and self.h_l.shape != (grid.Nl,):
            raise ValueError(f"h_l shape {self.h_l.shape} != ({grid.Nl},)")
        if self.psat_l is not None and self.psat_l.shape != (Ns_l,):
            raise ValueError(f"psat_l shape {self.psat_l.shape} != ({Ns_l},)")
        if self.hvap_l is not None and self.hvap_l.shape != (Ns_l,):
            raise ValueError(f"hvap_l shape {self.hvap_l.shape} != ({Ns_l},)")


@dataclass(slots=True)
class Diagnostics:
    """Minimal diagnostics container."""

    t: float
    step: int
    newton_iter: int
    res_norm: float
    warnings: List[str] = field(default_factory=list)


def check_sumY(state: State, *, tol: float = 1e-10) -> None:
    """Verify mass fractions sum to unity column-wise."""
    if state.Yg.size:
        yg_sum = np.sum(state.Yg, axis=0)
        max_err_g = float(np.max(np.abs(yg_sum - 1.0)))
        if max_err_g > tol:
            raise ValueError(f"Gas Y sum off by {max_err_g:.3e} (tol={tol})")
    if state.Yl.size:
        yl_sum = np.sum(state.Yl, axis=0)
        max_err_l = float(np.max(np.abs(yl_sum - 1.0)))
        if max_err_l > tol:
            raise ValueError(f"Liquid Y sum off by {max_err_l:.3e} (tol={tol})")


def check_nonneg(state: State, *, tol: float = -1e-14) -> None:
    """Ensure mass fractions are not negative beyond tolerance.

    Notes
    -----
    Guard against empty arrays (np.min on empty raises).
    """
    if state.Yg.size:
        ymin_g = float(np.min(state.Yg))
        if ymin_g < tol:
            raise ValueError(f"Gas Y has values below tol={tol}: min(Yg)={ymin_g:.3e}")
    if state.Yl.size:
        ymin_l = float(np.min(state.Yl))
        if ymin_l < tol:
            raise ValueError(f"Liquid Y has values below tol={tol}: min(Yl)={ymin_l:.3e}")


def check_positive_props(props: Props, *, tol: float = 0.0) -> None:
    """Ensure densities and heat capacities/conductivities are positive."""
    for name, arr in (
        ("rho_g", props.rho_g),
        ("cp_g", props.cp_g),
        ("k_g", props.k_g),
        ("rho_l", props.rho_l),
        ("cp_l", props.cp_l),
        ("k_l", props.k_l),
    ):
        if arr.size and np.any(arr <= tol):
            raise ValueError(f"{name} contains non-positive entries (tol={tol})")


def check_state_shapes(state: State, grid: Grid1D, *, Ns_g: int, Ns_l: int) -> None:
    """Validate State array shapes against grid and species counts."""
    if state.Tg.shape != (grid.Ng,):
        raise ValueError(f"Tg shape {state.Tg.shape} != ({grid.Ng},)")
    if state.Yg.shape != (Ns_g, grid.Ng):
        raise ValueError(f"Yg shape {state.Yg.shape} != ({Ns_g},{grid.Ng})")

    if state.Tl.shape != (grid.Nl,):
        raise ValueError(f"Tl shape {state.Tl.shape} != ({grid.Nl},)")
    if state.Yl.shape != (Ns_l, grid.Nl):
        raise ValueError(f"Yl shape {state.Yl.shape} != ({Ns_l},{grid.Nl})")

    for name, arr in [("Tg", state.Tg), ("Yg", state.Yg), ("Tl", state.Tl), ("Yl", state.Yl)]:
        if arr.dtype != np.float64:
            raise ValueError(f"{name} dtype must be float64, got {arr.dtype}")
