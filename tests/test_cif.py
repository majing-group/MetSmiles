from __future__ import annotations

import subprocess
import sys

import pytest

from metsmiles import expand_unit_cell, generate_global_metsmiles_from_cif, read_cif


def test_read_cif_parses_cell_atoms_symops(example_data):
    parsed = read_cif(example_data / "mil-88b.cif")
    assert parsed["cell"]["a"] == pytest.approx(13.9105)
    assert parsed["cell"]["gamma"] == pytest.approx(120.0)
    assert len(parsed["atoms"]) > 0
    assert any(a[1] == "Fe" for a in parsed["atoms"])


def test_global_metsmiles_on_mil88b(example_data):
    pytest.importorskip("spglib")
    record = generate_global_metsmiles_from_cif(example_data / "mil-88b.cif")
    # Two crystallographically independent Fe3(mu3-O) clusters per unit cell.
    assert record["n_metal_centers"] == 6
    assert len(record["clusters"]) == 2
    assert record["periodicity"] == 3
    # The file declares P1 (Materials Studio expansion dump); spglib
    # recovers the true hexagonal group P63/mmc of MIL-88B.
    assert record["space_group"] == "P6_3/mmc"
    assert record["space_group_declared"] == "P1"
    assert record["space_group_source"] == "spglib"
    assert record["point_group"] == "6/mmm"
    assert (
        record["Global Suffix 3"]
        == "{cluster:Fe3;u3-O}x2.[{Octahedral}Fe&O&O&O&O&O&O:u3]x3"
           ".{periodic:3D;sg=P6_3/mmc;pg=6/mmm}"
    )
    for cluster in record["clusters"]:
        assert cluster["metal_composition"] == "Fe3"
        for center in cluster["centers"]:
            assert center["Geometry Type"] == "Octahedral"


def test_topology_token_is_optional(example_data):
    pytest.importorskip("spglib")
    record = generate_global_metsmiles_from_cif(
        example_data / "mil-88b.cif", topology="the"
    )
    assert record["Global Suffix 3"].endswith("{periodic:3D;sg=P6_3/mmc;pg=6/mmm;topo=the}")


def test_ligand_extraction_finds_bdc(example_data):
    """The MIL-88B linker is extracted and reported as BDC(2-)."""
    rdkit = pytest.importorskip("rdkit")
    record = generate_global_metsmiles_from_cif(example_data / "mil-88b.cif")
    linker = record["ligands"][0]
    assert linker["n_atoms"] == 16          # C8H4O4
    assert linker["n_copies"] == 6
    assert linker["smiles"] is not None
    # canonical SMILES of terephthalate(2-)
    assert "C(=O)" in linker["smiles"] and "[O-]" in linker["smiles"]
    assert record["SMILES_3"].startswith(linker["smiles"] + ".{cluster:Fe3")


def test_ligand_smiles_requires_no_rdkit(example_data, monkeypatch):
    """Without RDKit the record still builds; ligand smiles stay None."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("rdkit"):
            raise ImportError("no rdkit")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    record = generate_global_metsmiles_from_cif(example_data / "mil-88b.cif")
    assert record["ligands"][0]["smiles"] is None
    assert "{cluster:Fe3" in record["Global Suffix 3"]


def test_space_group_falls_back_to_file_without_spglib(example_data, monkeypatch):
    """Without spglib the declared group is used as-is."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "spglib":
            raise ImportError("no spglib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    record = generate_global_metsmiles_from_cif(example_data / "mil-88b.cif")
    assert record["space_group_source"] == "file"
    assert record["space_group"] == "P1"


def test_global_metsmiles_with_base_smiles(example_data):
    record = generate_global_metsmiles_from_cif(
        example_data / "mil-88b.cif", base_smiles="OC(=O)c1ccccc1"
    )
    assert record["SMILES_3"].startswith("OC(=O)c1ccccc1.{cluster:Fe3")


def test_cli_cif_mode(example_data):
    import json

    result = subprocess.run(
        [sys.executable, "-m", "metsmiles", str(example_data / "mil-88b.cif")],
        capture_output=True, text=True, check=True,
    )
    record = json.loads(result.stdout)
    assert record["n_metal_centers"] == 6
    assert "{cluster:Fe3" in record["Global Suffix 3"]


def test_point_group_from_hm():
    from metsmiles.cif import point_group_from_hm

    cases = {
        "P6_3/mmc": "6/mmm",   # D6h
        "P-62c": "-62m",       # D3h
        "P2_1/c": "2/m",
        "F432": "432",
        "Fm-3m": "m-3m",
        "Pnma": "mmm",
        "I4/mmm": "4/mmm",
        "Fd-3c": "m-3m",
    }
    for sg, expected in cases.items():
        assert point_group_from_hm(sg) == expected, sg


def test_expand_unit_cell_deduplicates(example_data):
    parsed = read_cif(example_data / "mil-88b.cif")
    sites = expand_unit_cell(parsed["atoms"], parsed["symops"])
    assert len(sites) == len({(round(x, 4), round(y, 4), round(z, 4))
                             for _, _, x, y, z in sites})
    assert len(sites) >= len(parsed["atoms"])
