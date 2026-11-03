from __future__ import annotations

from pathlib import Path

import pytest

from metsmiles import (
    generate_global_metsmiles,
    generate_metsmiles_record,
    generate_metsmiles_records,
    metal_clusters,
    read_xyz_file,
    sharing_map,
)
from metsmiles.multimetal import global_metsmiles_for_molecule


@pytest.fixture()
def fe2o2_xyz(tmp_path: Path) -> Path:
    """A synthetic Fe2(mu-O)2 core with two terminal N donors on each Fe."""
    path = tmp_path / "fe2o2.xyz"
    path.write_text(
        "8\n"
        "synthetic Fe2O2 core\n"
        "Fe 0.0 0.0 0.0\n"
        "Fe 3.4 0.0 0.0\n"
        "O 1.7 1.0 0.0\n"
        "O 1.7 -1.0 0.0\n"
        "N -1.5 0.0 0.0\n"
        "N 0.0 -1.5 0.0\n"
        "N 4.9 0.0 0.0\n"
        "N 3.4 -1.5 0.0\n",
        encoding="utf-8",
    )
    return path


def test_sharing_map_detects_bridging_donors(fe2o2_xyz):
    molecule = read_xyz_file(fe2o2_xyz)
    molecule.infer_bonds_from_distance()
    sharing = sharing_map(molecule)
    # The two bridging O atoms coordinate both Fe centers.
    shared = {idx: ms for idx, ms in sharing.items() if len(ms) == 2}
    assert len(shared) == 2
    assert all(molecule.atoms[i].element == "O" for i in shared)


def test_clusters_group_linked_metals(fe2o2_xyz):
    molecule = read_xyz_file(fe2o2_xyz)
    molecule.infer_bonds_from_distance()
    clusters = metal_clusters(molecule)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_center_suffix_marks_bridging_donors(fe2o2_xyz):
    record = generate_metsmiles_records(fe2o2_xyz, prefix_mode="suffix-only")
    assert len(record) == 2
    for row in record:
        assert row["Suffix 2"] == "[Fe&N&N&O:u2&O:u2]"
        assert row["Shared Donors"] == 2


def test_global_suffix_has_cluster_header(fe2o2_xyz):
    record = generate_global_metsmiles(fe2o2_xyz, prefix_mode="suffix-only")
    assert record["n_metal_centers"] == 2
    block, centers = record["Global Suffix 2"].split(".")
    assert block == "{cluster:Fe2;u2-O2}"
    assert centers == "[Fe&N&N&O:u2&O:u2]x2"


def test_global_matches_single_center_api(example_data):
    """A mononuclear structure produces the classic suffix unchanged."""
    single = generate_metsmiles_record(
        example_data / "0.xyz", example_data / "0.sdf"
    , prefix_mode="suffix-only")
    global_record = generate_global_metsmiles(
        example_data / "0.xyz", example_data / "0.sdf"
    , prefix_mode="suffix-only")
    assert global_record["Global Suffix 2"] == single["Suffix 2"]
    assert global_record["Global Suffix 3"] == single["Suffix 3"]
    assert global_record["n_metal_centers"] == 1


def test_single_center_api_still_rejects_multimetal(fe2o2_xyz):
    with pytest.raises(ValueError, match="generate_global_metsmiles"):
        generate_metsmiles_record(fe2o2_xyz, None, prefix_mode="suffix-only")


def test_no_metal_gives_empty_clusters(no_metal_xyz):
    molecule = read_xyz_file(no_metal_xyz)
    molecule.infer_bonds_from_distance()
    record = global_metsmiles_for_molecule(molecule)
    assert record["n_metal_centers"] == 0
    assert record["Global Suffix 3"] == ""
