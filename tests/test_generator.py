from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from metsmiles import (
    generate_metsmiles_record,
    read_xyz_file,
    recognize_directory,
)

GOLDEN = {
    "0": {
        "Metal Center": "Fe",
        "Real Coordination Number": 6,
        "Geometry Type": "Octahedral",
        "Suffix 2": "[Fe&Cl&Cl&N&N&O&O]",
        "Suffix 3": "[{Octahedral}Fe&Cl&Cl&N&N&O&O]",
    },
    "405": {
        "Metal Center": "Fe",
        "Real Coordination Number": 4,
        "Geometry Type": "Tetrahedral",
        "Suffix 2": "[Fe&Cl&Cl&Cl&Cl]",
        "Suffix 3": "[{Tetrahedral}Fe&Cl&Cl&Cl&Cl]",
    },
    "493": {
        "Metal Center": "Fe",
        "Real Coordination Number": 4,
        "Geometry Type": "Tetrahedral",
        "Suffix 2": "[Fe&C&C%CC%CC]",
        "Suffix 3": "[{Tetrahedral}Fe&C&C%CC%CC]",
    },
    "498": {
        "Metal Center": "Fe",
        "Real Coordination Number": 5,
        "Geometry Type": "TrigonalBipyramid",
        "Suffix 2": "[Fe&C&N%CCCCC]",
        "Suffix 3": "[{TrigonalBipyramid}Fe&C&N%CCCCC]",
    },
}


@pytest.mark.parametrize("stem", sorted(GOLDEN))
def test_suffixes_match_reference_values(example_data, stem):
    record = generate_metsmiles_record(
        example_data / f"{stem}.xyz", example_data / f"{stem}.sdf"
    , prefix_mode="suffix-only")
    for key, expected in GOLDEN[stem].items():
        assert record[key] == expected, f"{stem}: {key}"


def test_base_smiles_concatenation(example_data):
    record = generate_metsmiles_record(
        example_data / "0.xyz", example_data / "0.sdf", base_smiles="CCO"
    , prefix_mode="suffix-only")
    assert record["SMILES_2"] == "CCO." + record["Suffix 2"]
    assert record["SMILES_3"] == "CCO." + record["Suffix 3"]


@pytest.mark.parametrize("stem", sorted(GOLDEN))
def test_atom_order_invariance(example_data, stem):
    xyz_path = example_data / f"{stem}.xyz"
    sdf_path = example_data / f"{stem}.sdf"
    atom_count = read_xyz_file(xyz_path).num_atoms()
    order = np.random.default_rng(20260928).permutation(atom_count).tolist()

    baseline = generate_metsmiles_record(xyz_path, sdf_path, prefix_mode="suffix-only")
    permuted = generate_metsmiles_record(xyz_path, sdf_path, atom_order=order, prefix_mode="suffix-only")

    assert baseline["SMILES_3"] == permuted["SMILES_3"]
    assert baseline["Geometry Type"] == permuted["Geometry Type"]


def test_small_coordinate_noise_keeps_geometry(example_data):
    record = generate_metsmiles_record(
        example_data / "0.xyz",
        example_data / "0.sdf",
        coordinate_noise_std=0.001,
        seed=7,
     prefix_mode="suffix-only")
    assert record["Geometry Type"] == "Octahedral"
    assert record["Suffix 2"] == GOLDEN["0"]["Suffix 2"]


def test_generation_parameters_are_recorded(example_data):
    record = generate_metsmiles_record(
        example_data / "405.xyz",
        example_data / "405.sdf",
        bond_tolerance=1.13,
        geometry_sigma=20.0,
     prefix_mode="suffix-only")
    assert record["bond_tolerance"] == 1.13
    assert record["geometry_sigma"] == 20.0
    assert record["coordinate_noise_std"] == 0.0
    assert record["connectivity_source"] == "sdf"


def test_distance_inference_without_sdf(example_data):
    record = generate_metsmiles_record(example_data / "405.xyz", None, prefix_mode="suffix-only")
    assert record["connectivity_source"] == "distance_inference"
    assert record["Suffix 3"] == GOLDEN["405"]["Suffix 3"]


def test_other_metal_center(example_data, tmp_path):
    """Retargeting the metal to a non-present element yields no record."""
    with pytest.raises(ValueError, match="Expected one target metal center"):
        generate_metsmiles_record(
            example_data / "0.xyz", example_data / "0.sdf",
            target_metal_elements=["Co"],
         prefix_mode="suffix-only")


def test_structure_without_metal_raises(no_metal_xyz):
    with pytest.raises(ValueError, match="Expected one target metal center"):
        generate_metsmiles_record(no_metal_xyz, None, prefix_mode="suffix-only")


def test_import_has_no_filesystem_side_effect(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib

    import metsmiles.generator

    importlib.reload(metsmiles.generator)
    assert list(tmp_path.iterdir()) == []


def test_recognize_directory_writes_csv(example_data, tmp_path):
    output_csv = tmp_path / "coordination_analysis.csv"
    rows = recognize_directory(example_data, output_csv=output_csv, prefix_mode="suffix-only")

    assert len(rows) == len(GOLDEN)
    by_index = {row["gdb_idx"]: row for row in rows}
    for stem, expected in GOLDEN.items():
        assert by_index[stem]["Suffix 3"] == expected["Suffix 3"]

    with open(output_csv, newline="", encoding="utf-8") as handle:
        persisted = list(csv.DictReader(handle))
    assert len(persisted) == len(GOLDEN)
    assert {row["gdb_idx"] for row in persisted} == set(GOLDEN)


def test_cli_single_structure_to_json(example_data):
    result = subprocess.run(
        [
            sys.executable, "-m", "metsmiles",
            str(example_data / "498.xyz"),
            "--sdf", str(example_data / "498.sdf"), "--prefix-mode", "suffix-only",
        ],
        capture_output=True, text=True, check=True,
    )
    record = json.loads(result.stdout)
    assert record["Suffix 3"] == GOLDEN["498"]["Suffix 3"]


def test_cli_directory_mode(example_data, tmp_path):
    output_csv = tmp_path / "out.csv"
    result = subprocess.run(
        [
            sys.executable, "-m", "metsmiles",
            str(example_data), "--output-csv", str(output_csv), "--prefix-mode", "suffix-only",
        ],
        capture_output=True, text=True, check=True,
    )
    assert f"Wrote {len(GOLDEN)} rows" in result.stderr
    assert output_csv.exists()


def test_cli_no_metal_error_is_reported(no_metal_xyz):
    result = subprocess.run(
        [sys.executable, "-m", "metsmiles", str(no_metal_xyz)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "error:" in result.stderr
