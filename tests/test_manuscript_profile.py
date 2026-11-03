"""Specification tests for the manuscript-v1 method profile."""
from __future__ import annotations
import builtins
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from metsmiles import (
    Atom, Molecule, generate_metsmiles_record, generate_metsmiles_records,
    generate_global_metsmiles, generate_global_metsmiles_from_cif,
    mask_donor_entries, remove_donor_list, remove_pi_marker, remove_geometry,
    recognize_directory,
)
from metsmiles.prefix import (
    resolve_molecular_prefix, PrefixError, PrefixDependencyError, PrefixChemistryError,
)


@pytest.mark.parametrize("elements", [
    ["C"] * 2, ["C"] * 3, ["C"] * 4, ["C"] * 5, ["C"] * 6,
    ["O", "O"], ["N", "C", "O"], ["C", "N", "C", "O", "C"],
])
def test_connected_groups_use_one_centroid_except_cp(elements):
    mol = Molecule()
    mol.add_atom(Atom("Fe", 0, 0, 0))
    coords = [(.2 * i, .1 * i, 1.8) for i in range(len(elements))]
    for el, xyz in zip(elements, coords):
        mol.add_atom(Atom(el, *xyz))
    mol.bond_info = {(i + 2, i + 3): 1 for i in range(len(elements) - 1)}
    result = mol.analyze_ligands(mol.atoms[0], mol.atoms[1:])
    assert result["pi_ligands"] == [mol.atoms[1:]]
    assert result["normal_ligands"] == []
    assert result["real_coordination_number"] == 1
    assert len(result["pi_coordination_sites"]) == 1
    np.testing.assert_allclose(result["pi_coordination_sites"][0], np.mean(coords, axis=0))
    # Every contact participates: no skipped donors or full-ligand graph walk.
    report = mol.generate_coordination_report()
    assert report["coordination_numbers"] == [len(elements)]
    assert report["real_coordination_numbers"] == [1]
    assert report["geometry_types"] == ["unknown"]


def test_cp_three_sites_and_heterocycle_completion():
    mol = Molecule()
    mol.add_atom(Atom("Fe", 0, 0, 0))
    for angle in np.arange(5) * 2 * math.pi / 5:
        mol.add_atom(Atom("C", 1.1 * math.cos(angle), 1.1 * math.sin(angle), 1.7))
    mol.bond_info = {(i + 2, (i + 1) % 5 + 2): 1 for i in range(5)}
    r = mol.analyze_ligands(mol.atoms[0], mol.atoms[1:])
    assert r["real_coordination_number"] == 3
    assert len(r["pi_coordination_sites"]) == 3
    anchor = mol.find_nearest_point(mol.atoms[0], mol.atoms[1:])
    assert any(np.array_equal(anchor.coordinates, x) for x in r["pi_coordination_sites"])
    mol.atoms[2].element = "N"
    r = mol.analyze_ligands(mol.atoms[0], mol.atoms[1:])
    assert r["real_coordination_number"] == 1
    assert len(r["pi_ligands"]) == 1


def test_gaussian_sigma15_and_outside_template_range():
    mol = Molecule()
    assert mol._calculate_angle_similarity([(90., 90.), (105., 90.)]) == pytest.approx((1 + math.exp(-.5)) / 2)
    mol.geometry_sigma = 30.
    assert mol._calculate_angle_similarity([(105., 90.)]) == pytest.approx(math.exp(-.125))
    for count in [0, 1, 7, 8]:
        center = Atom("Fe", 0, 0, 0)
        sites = [Atom("N", math.cos(i), math.sin(i), 1) for i in range(count)]
        assert mol.analyze_coordination_geometry(center, sites)["geometry_type"] == "unknown"


def test_manuscript_ablations_one_x_per_entry_and_global_grammar():
    prefix = "C%10CCCCC%10.[NH4+]"
    suffix = "{cluster:CoNi;u2-O}x2.[{Linear}Co&N%CO:u2]x3|[{SquarePlanar}Ni&Cl&N&O:u2%CCCCC].{periodic:2D;sg=P1;topo=abc}"
    text = prefix + "." + suffix
    masked = mask_donor_entries(text)
    assert masked == prefix + ".{cluster:CoNi;u2-O}x2.[{Linear}Co&X%X:u2]x3|[{SquarePlanar}Ni&X&X&X:u2%X].{periodic:2D;sg=P1;topo=abc}"
    assert "%XX" not in masked
    assert remove_geometry(text) == prefix + "." + suffix.replace("{Linear}", "").replace("{SquarePlanar}", "")
    assert remove_pi_marker(text) == prefix + "." + suffix.replace("%", "&")
    assert remove_donor_list(text) != masked


def block_extras(monkeypatch):
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name.startswith("rdkit"):
            raise ImportError("RDKit deliberately unavailable")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)


def test_prefix_missing_dependency_raises_and_override_is_explicit(example_data, monkeypatch):
    block_extras(monkeypatch)
    with pytest.raises(PrefixDependencyError, match=r"metsmiles\[molecular\]"):
        generate_metsmiles_record(example_data / "405.xyz", example_data / "405.sdf")
    explicit = generate_metsmiles_record(example_data / "405.xyz", example_data / "405.sdf", base_smiles="[Cl-].[Cl-]")
    assert explicit["prefix_source"] == "explicit_base_smiles"
    assert explicit["SMILES_1"] == "[Cl-].[Cl-]"
    empty = generate_metsmiles_record(example_data / "405.xyz", None, base_smiles="")
    assert empty["prefix_status"] == "supplied" and empty["SMILES_1"] == ""
    suffix = generate_metsmiles_record(example_data / "405.xyz", None, prefix_mode="suffix-only")
    assert suffix["prefix_source"] == "suffix_only"


def test_no_sdf_requires_explicit_boundary(example_data):
    with pytest.raises(PrefixError, match="matching SDF"):
        generate_metsmiles_record(example_data / "405.xyz", None)
    with pytest.raises(ValueError, match="prefix_mode"):
        generate_metsmiles_record(example_data / "405.xyz", None, prefix_mode="silent")
    with pytest.raises(ValueError, match="method_profile"):
        generate_metsmiles_record(example_data / "405.xyz", None, method_profile="legacy")


def sdf_from_smiles(tmp_path, smiles):
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import rdDepictor
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdDepictor.Compute2DCoords(mol)
    path = tmp_path / "prefix.sdf"
    path.write_text(Chem.MolToMolBlock(mol) + "\n$$$$\n")
    return path


@pytest.mark.parametrize("smiles", [
    "[Fe].CCO.CCO.[Na+].[Cl-]", "[Fe].C[C@H](O)F", "[Fe].[CH3]", "[Fe].[O]",
    "[Fe].F/C=C/F", "[Fe].[13CH3]O", "[Fe].[NH4+].[Cl-]",
])
def test_auto_prefix_retains_fragments_charge_stereo_radicals(tmp_path, smiles):
    Chem = pytest.importorskip("rdkit.Chem")
    path = sdf_from_smiles(tmp_path, smiles)
    prefix, metadata = resolve_molecular_prefix(path, {"Fe"})
    expected = sorted(Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=True)
                      for s in smiles.split(".") if s != "[Fe]")
    assert prefix.split(".") == expected
    assert metadata["prefix_source"] == "sdf_rdkit"
    assert metadata["prefix_status"] == "generated"
    assert metadata["rdkit_version"]
    assert "duplicates_retained" in metadata["fragment_policy"]


def test_only_target_metals_removed_and_metal_only_empty_is_legitimate(tmp_path):
    path = sdf_from_smiles(tmp_path, "[Fe].[Co+2].[Cl-]")
    prefix, _ = resolve_molecular_prefix(path, {"Fe"})
    assert "[Co+2]" in prefix and "[Cl-]" in prefix
    path = sdf_from_smiles(tmp_path, "[Fe]")
    prefix, metadata = resolve_molecular_prefix(path, {"Fe"})
    assert prefix == "" and metadata["prefix_status"] == "no_non_target_atoms"


def test_no_guessed_charge_or_valence_repair(tmp_path):
    Chem = pytest.importorskip("rdkit.Chem")
    # Deliberately invalid neutral O with three single bonds must not become O+.
    mol = Chem.RWMol(Chem.MolFromSmiles("[Fe].COC"))
    carbon = mol.AddAtom(Chem.Atom("C"))
    mol.AddBond(2, carbon, Chem.BondType.SINGLE)
    path = tmp_path / "invalid.sdf"
    path.write_text(Chem.MolToMolBlock(mol.GetMol()) + "\n$$$$\n")
    with pytest.raises(PrefixChemistryError, match="without chemical repair"):
        resolve_molecular_prefix(path, {"Fe"})
    assert resolve_molecular_prefix(path, {"Fe"}, base_smiles="COC")[0] == "COC"


def test_new_radical_not_hydrogen_completion_after_metal_removal(tmp_path):
    path = sdf_from_smiles(tmp_path, "[Fe]C")
    prefix, _ = resolve_molecular_prefix(path, {"Fe"})
    assert prefix == "[CH3]"  # deleting the metal bond must not invent a fourth H


def test_single_multi_directory_profile_coherence(example_data, tmp_path):
    args = (example_data / "405.xyz", example_data / "405.sdf")
    single = generate_metsmiles_record(*args, prefix_mode="suffix-only")
    global_record = generate_global_metsmiles(*args, prefix_mode="suffix-only")
    centers = generate_metsmiles_records(*args, prefix_mode="suffix-only")
    rows = recognize_directory(example_data, tmp_path / "out.csv", prefix_mode="suffix-only")
    for record in [single, global_record, *centers, *rows]:
        assert record["method_profile"] == "manuscript-v1"
        assert record["serialization_profile"] == "canonical-sorted-v1"
        assert record["prefix_source"] == "suffix_only"
    assert single["Suffix 3"] == global_record["Global Suffix 3"] == centers[0]["Suffix 3"]


def test_non_fe_two_center_api_and_cli(tmp_path):
    path = tmp_path / "co_ni.xyz"
    path.write_text("3\nsynthetic shared oxygen\nCo -1.7 0 0\nNi 1.7 0 0\nO 0 0 0\n")
    record = generate_global_metsmiles(path, target_metal_elements=["Co", "Ni"], prefix_mode="suffix-only")
    assert record["n_metal_centers"] == 2
    assert record["Global Suffix 2"] == "{cluster:CoNi;u2-O}.[Co&O:u2]|[Ni&O:u2]"
    result = subprocess.run([sys.executable, "-m", "metsmiles", str(path), "--metal", "Co", "--metal", "Ni", "--prefix-mode", "suffix-only"], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["Global Suffix 3"] == record["Global Suffix 3"]


def test_directory_prefix_failure_is_typed_not_silent(example_data, tmp_path):
    import shutil
    shutil.copy2(example_data / "405.xyz", tmp_path / "one.xyz")
    rows = recognize_directory(tmp_path, tmp_path / "out.csv")
    assert rows[0]["status"] == "error" and rows[0]["error_type"] == "PrefixError"
    assert "matching SDF" in rows[0]["error"]
    assert "SMILES_3" not in rows[0]
    result = subprocess.run([sys.executable, "-m", "metsmiles", str(tmp_path), "--output-csv", str(tmp_path / "cli.csv")], text=True, capture_output=True)
    assert result.returncode == 1 and "1 errors" in result.stderr


def test_figure2_actual_structure_and_supplied_prefix_are_separate():
    folder = Path(__file__).resolve().parents[1] / "examples/figure2"
    ref = json.loads((folder / "reference.json").read_text())
    args = (folder / "48.xyz", folder / "48.sdf")
    supplied = generate_metsmiles_record(*args, base_smiles=ref["supplied_base_smiles"])
    assert supplied["Suffix 3"] == ref["expected_canonical_suffix"]
    assert supplied["Real Coordination Number"] == 6
    assert supplied["prefix_source"] == "explicit_base_smiles"
    assert supplied["SMILES_3"] == ref["supplied_base_smiles"] + "." + supplied["Suffix 3"]
    assert supplied["Suffix 3"] != ref["manuscript_diagram_suffix"]


def test_figure2_actual_automatic_sdf_prefix():
    pytest.importorskip("rdkit")
    folder = Path(__file__).resolve().parents[1] / "examples/figure2"
    ref = json.loads((folder / "reference.json").read_text())
    automatic = generate_metsmiles_record(folder / "48.xyz", folder / "48.sdf")
    assert automatic["prefix_source"] == "sdf_rdkit"
    assert automatic["Suffix 3"] == ref["expected_canonical_suffix"]
    assert automatic["SMILES_1"] == "C[C]1C(C)=C(C)C(C)=C1C.[CH2][CH]CC.[C][O].[C][O]"
    assert automatic["SMILES_1"] != ref["supplied_base_smiles"]
    assert automatic["SMILES_1"].count(".") == 3


def test_cif_profile_and_explicit_acs_not_automatic(example_data):
    args = (example_data / "mil-88b.cif",)
    default = generate_global_metsmiles_from_cif(*args, with_ligands=False, detect_symmetry=False)
    explicit = generate_global_metsmiles_from_cif(*args, topology="acs", with_ligands=False, detect_symmetry=False)
    assert "topo=" not in default["SMILES_3"]
    assert explicit["SMILES_3"].endswith("topo=acs}")
    assert explicit["prefix_scope"] == "largest_finite_linker_not_full_composition"
    assert explicit["method_profile"] == "manuscript-v1"
    assert all(c["method_profile"] == explicit["method_profile"] for c in explicit["centers"])


def test_zero_contact_single_and_global_geometry_agree(tmp_path):
    xyz = tmp_path / "metal.xyz"
    xyz.write_text("1\nmetal only\nFe 0 0 0\n")
    single = generate_metsmiles_record(xyz, None, prefix_mode="suffix-only")
    global_record = generate_global_metsmiles(xyz, prefix_mode="suffix-only")
    assert single["Geometry Type"] == "unknown"
    assert single["Suffix 3"] == global_record["Global Suffix 3"]


def test_automatic_prefix_pipeline_consistency(example_data, tmp_path):
    pytest.importorskip("rdkit")
    import shutil
    for ext in ["xyz", "sdf"]:
        shutil.copy2(example_data / ("405." + ext), tmp_path / ("405." + ext))
    args = (tmp_path / "405.xyz", tmp_path / "405.sdf")
    one = generate_metsmiles_record(*args)
    whole = generate_global_metsmiles(*args)
    local = generate_metsmiles_records(*args)[0]
    batch = recognize_directory(tmp_path, tmp_path / "out.csv")[0]
    for item in [one, whole, local, batch]:
        assert item["SMILES_1"] == "[Cl].[Cl].[Cl].[Cl]"
        assert item["SMILES_3"] == one["SMILES_3"]
        assert item["prefix_source"] == "sdf_rdkit"
        assert item["rdkit_version"] == one["rdkit_version"]
        assert item["method_profile"] == "manuscript-v1"


def test_cli_does_not_retry_prefix_error_as_global(example_data, monkeypatch, capsys):
    import metsmiles.cli as cli
    def prefix_failure(*args, **kwargs):
        raise PrefixChemistryError("deliberate chemical failure")
    def forbidden_global(*args, **kwargs):
        raise AssertionError("a single-center chemistry error must not retry globally")
    monkeypatch.setattr(cli, "generate_metsmiles_record", prefix_failure)
    monkeypatch.setattr(cli, "generate_global_metsmiles", forbidden_global)
    code = cli.main([str(example_data / "405.xyz"), "--sdf", str(example_data / "405.sdf")])
    assert code == 1
    assert "PrefixChemistryError" in capsys.readouterr().err
