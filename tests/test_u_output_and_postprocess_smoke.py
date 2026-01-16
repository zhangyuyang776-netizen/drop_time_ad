"""
Smoke test for u vector output and postprocessing.

Tests:
1. get_run_dir() creates correct directory structure
2. build_u_mapping() generates valid mapping.json
3. write_step_u() creates npz files with correct content
4. Postprocessing script can convert npz to CSV
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.layout import UnknownLayout, build_layout
from core.types import CaseConfig, Grid1D, State
from io.writers import (
    build_u_mapping,
    get_run_dir,
    should_write_u,
    write_mapping_json,
    write_step_u,
)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="test_u_output_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config(temp_output_dir):
    """Create a minimal mock CaseConfig."""

    class MockPaths:
        output_root = str(temp_output_dir / "3D_out")
        case_dir = str(temp_output_dir / "out" / "test_case")

    class MockCase:
        id = "test_case_001"

    class MockOutput:
        u_enabled = True
        u_every = 1

    class MockSpecies:
        gas_species_full = ["N2", "NC12H26"]
        gas_balance_species = "N2"
        liq_species = ["n-Dodecane"]
        liq_balance_species = "n-Dodecane"

    class MockPhysics:
        solve_Tg = True
        solve_Yg = True
        solve_Tl = True
        solve_Yl = False
        include_Ts = True
        include_mpp = True
        include_Rd = True

    cfg = CaseConfig.__new__(CaseConfig)
    cfg.paths = MockPaths()
    cfg.case = MockCase()
    cfg.output = MockOutput()
    cfg.species = MockSpecies()
    cfg.physics = MockPhysics()

    return cfg


@pytest.fixture
def mock_grid():
    """Create a minimal Grid1D."""
    Nl = 5
    Ng = 10
    Nc = Nl + Ng

    r_c = np.linspace(0.5e-4, 1.0e-2, Nc)
    r_f = np.linspace(0.0, 1.05e-2, Nc + 1)
    V_c = np.ones(Nc) * 1e-12
    A_f = np.ones(Nc + 1) * 1e-8

    grid = Grid1D(
        Nl=Nl,
        Ng=Ng,
        Nc=Nc,
        r_c=r_c,
        r_f=r_f,
        V_c=V_c,
        A_f=A_f,
        iface_f=Nl,
        liq_slice=slice(0, Nl),
        gas_slice=slice(Nl, Nc),
    )

    return grid


@pytest.fixture
def mock_layout(mock_config, mock_grid):
    """Create a minimal UnknownLayout."""
    try:
        layout = build_layout(mock_config, mock_grid)
        return layout
    except Exception:
        # Fallback: create a minimal manual layout
        from dataclasses import dataclass

        @dataclass
        class MockLayout:
            size = 50
            Ng = 10
            Nl = 5
            Ns_g_full = 2
            Ns_g_eff = 1
            Ns_l_full = 1
            Ns_l_eff = 0
            gas_species_full = ["N2", "NC12H26"]
            gas_species_reduced = ["NC12H26"]
            gas_closure_species = "N2"
            liq_species_full = ["n-Dodecane"]
            liq_species_reduced = []
            liq_closure_species = "n-Dodecane"

            def iter_blocks(self):
                return [
                    ("Tg", slice(0, 10)),
                    ("Yg", slice(10, 20)),
                    ("Tl", slice(20, 25)),
                    ("Ts", slice(25, 26)),
                    ("mpp", slice(26, 27)),
                    ("Rd", slice(27, 28)),
                ]

        return MockLayout()


def test_get_run_dir(mock_config, temp_output_dir):
    """Test that get_run_dir creates correct directory structure."""
    run_dir = get_run_dir(mock_config)

    # Check directory exists
    assert run_dir.exists()
    assert run_dir.is_dir()

    # Check path structure: 3D_out/case_xxx/run_yyy
    assert "3D_out" in str(run_dir)
    assert "case_test_case_001" in str(run_dir)
    assert "run_" in str(run_dir)

    # Check that calling again returns same directory (cached)
    run_dir2 = get_run_dir(mock_config)
    assert run_dir == run_dir2


def test_build_u_mapping(mock_config, mock_grid, mock_layout):
    """Test that build_u_mapping generates valid mapping."""
    mapping = build_u_mapping(mock_config, mock_grid, mock_layout)

    # Check required keys
    assert "version" in mapping
    assert "dtype" in mapping
    assert "endianness" in mapping
    assert "ordering" in mapping
    assert "blocks" in mapping
    assert "meta" in mapping

    # Check blocks
    assert len(mapping["blocks"]) > 0
    for block in mapping["blocks"]:
        assert "name" in block
        assert "offset" in block
        assert "size" in block
        assert "shape" in block

    # Check total size matches layout
    assert mapping["total_size"] == mock_layout.size

    # Check meta contains grid and species info
    meta = mapping["meta"]
    assert "Ng" in meta
    assert "Nl" in meta
    assert "species_g_full" in meta
    assert "species_l_full" in meta


def test_write_mapping_json(mock_config, mock_grid, mock_layout, temp_output_dir):
    """Test that write_mapping_json creates valid JSON file."""
    run_dir = get_run_dir(mock_config)
    write_mapping_json(mock_config, mock_grid, mock_layout, run_dir)

    mapping_path = run_dir / "mapping.json"
    assert mapping_path.exists()

    # Load and validate JSON
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    assert mapping["version"] == 1
    assert len(mapping["blocks"]) > 0


def test_write_step_u(mock_config, mock_grid, mock_layout, temp_output_dir):
    """Test that write_step_u creates valid npz files."""
    run_dir = get_run_dir(mock_config)

    # Create mock u vector
    u = np.random.randn(mock_layout.size)

    # Write step file
    write_step_u(mock_config, step_id=1, t=1.0e-6, u=u, grid=mock_grid, run_dir=run_dir)

    # Check file exists
    steps_dir = run_dir / "steps"
    assert steps_dir.exists()

    step_files = list(steps_dir.glob("step_*.npz"))
    assert len(step_files) == 1

    # Load and validate
    data = np.load(step_files[0])
    assert "step_id" in data
    assert "t" in data
    assert "u" in data
    assert "r_g" in data
    assert "r_l" in data

    assert int(data["step_id"]) == 1
    assert float(data["t"]) == 1.0e-6
    assert len(data["u"]) == len(u)
    assert len(data["r_g"]) == mock_grid.Ng
    assert len(data["r_l"]) == mock_grid.Nl


def test_should_write_u(mock_config):
    """Test output control logic."""
    # Test with u_enabled=True, u_every=1
    assert should_write_u(mock_config, step_id=0)
    assert should_write_u(mock_config, step_id=1)
    assert should_write_u(mock_config, step_id=2)

    # Test with u_every=3
    mock_config.output.u_every = 3
    assert should_write_u(mock_config, step_id=0)
    assert not should_write_u(mock_config, step_id=1)
    assert not should_write_u(mock_config, step_id=2)
    assert should_write_u(mock_config, step_id=3)

    # Test with u_enabled=False
    mock_config.output.u_enabled = False
    assert not should_write_u(mock_config, step_id=0)


def test_postprocess_script_exists():
    """Test that postprocess script exists and is executable."""
    script_path = Path(__file__).parent.parent / "scripts" / "postprocess_u_to_csv.py"
    assert script_path.exists()

    # Try to run with --help
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0


def test_full_workflow(mock_config, mock_grid, mock_layout, temp_output_dir):
    """
    Integration test: write mapping, write steps, run postprocessing.
    """
    run_dir = get_run_dir(mock_config)

    # Write mapping
    write_mapping_json(mock_config, mock_grid, mock_layout, run_dir)

    # Write a few step files
    for step_id in range(5):
        u = np.random.randn(mock_layout.size)
        t = step_id * 1.0e-6
        write_step_u(mock_config, step_id, t, u, mock_grid, run_dir)

    # Verify files exist
    assert (run_dir / "mapping.json").exists()
    steps_dir = run_dir / "steps"
    step_files = sorted(steps_dir.glob("step_*.npz"))
    assert len(step_files) == 5

    # Run postprocessing script
    script_path = Path(__file__).parent.parent / "scripts" / "postprocess_u_to_csv.py"
    out_dir = run_dir / "post_csv"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-dir",
            str(run_dir),
            "--stride",
            "2",
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        timeout=30,
    )

    # Check postprocessing succeeded
    if result.returncode != 0:
        print("STDOUT:", result.stdout.decode())
        print("STDERR:", result.stderr.decode())

    assert result.returncode == 0

    # Check CSV files were created
    csv_files = list(out_dir.glob("step_*.csv"))
    assert len(csv_files) > 0

    # Validate a CSV file
    csv_file = csv_files[0]
    with open(csv_file, "r") as f:
        lines = f.readlines()

    # Check for header and data
    assert any("phase" in line for line in lines)
    assert any("gas" in line for line in lines)
    assert any("liq" in line for line in lines)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
