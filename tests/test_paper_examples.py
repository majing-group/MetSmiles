"""Paper-mapped inputs: independent geometric/graph checks plus fresh goldens."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from metsmiles import (generate_metsmiles_record, generate_global_metsmiles_from_cif,
                       mask_donor_entries, remove_pi_marker, remove_geometry)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CASES = json.loads((EXAMPLES / "paper_cases.json").read_text())
GOLDEN = json.loads((EXAMPLES / "paper_golden.json").read_text())["examples"]
CP = "[{Octahedral}Fe&C&C%CC%CCCCC]"
CL = "[{Tetrahedral}Fe&Cl&Cl&Cl&Cl]"
MIL = "{cluster:Fe3;u3-O}x2.[{Octahedral}Fe&O&O&O&O&O&O:u3]x3.{periodic:3D;sg=P6_3/mmc;pg=6/mmm;topo=acs}"


def molecular(stem, **kwargs):
    return generate_metsmiles_record(EXAMPLES / (stem + ".xyz"), EXAMPLES / (stem + ".sdf"), **kwargs)


def raw_xyz(stem):
    lines = (EXAMPLES / (stem + ".xyz")).read_text().splitlines()
    atoms = [line.split() for line in lines[2:] if line.strip()]
    assert len(atoms) == int(lines[0])
    return [a[0] for a in atoms], np.array([[float(x) for x in a[1:4]] for a in atoms])


@pytest.mark.parametrize("filename,digest", CASES["input_sha256"].items())
def test_exact_paper_input_hash(filename, digest):
    assert hashlib.sha256((EXAMPLES / filename).read_bytes()).hexdigest() == digest


def test_fecl4_geometry_independent_of_generator_and_golden():
    elements, coordinates = raw_xyz("data/405")
    assert elements == ["Fe", "Cl", "Cl", "Cl", "Cl"]
    vectors = coordinates[1:] - coordinates[0]
    lengths = np.linalg.norm(vectors, axis=1)
    assert np.all((lengths > 2.21) & (lengths < 2.24))
    unit = vectors / lengths[:, None]
    angles = [math.degrees(math.acos(np.clip(np.dot(unit[i], unit[j]), -1, 1)))
              for i in range(4) for j in range(i)]
    assert len(angles) == 6
    assert max(abs(a - math.degrees(math.acos(-1 / 3))) for a in angles) < .10
    record = molecular("data/405", prefix_mode="suffix-only")
    assert record["Suffix 3"] == CL
    assert record["Real Coordination Number"] == 4
    assert record["Geometry Type"] == "Tetrahedral"
    assert record["Confidence"] > .999


def test_cp_count_from_raw_coordinates_and_sdf_induced_graph():
    elements, xyz = raw_xyz("figure2/48")
    fe = elements.index("Fe")
    # Manuscript-v1 radii: Fe 1.52 A, C 0.69 A; gamma 1.16.
    # Calculate contacts independently of the API donor-selection implementation.
    cutoff = 1.16 * (1.52 + 0.69)
    assert cutoff == pytest.approx(2.5636)
    donors = {i for i, el in enumerate(elements)
              if el == "C" and np.linalg.norm(xyz[i] - xyz[fe]) <= cutoff}
    assert len(donors) == 9
    lines = (EXAMPLES / "figure2/48.sdf").read_text().splitlines()
    na, nb = int(lines[3][:3]), int(lines[3][3:6])
    graph = {i: set() for i in donors}
    bonds = []
    for line in lines[4 + na:4 + na + nb]:
        a, b, order = int(line[:3]) - 1, int(line[3:6]) - 1, int(line[6:9])
        bonds.append((a, b, order))
        if a in donors and b in donors:
            graph[a].add(b)
            graph[b].add(a)
    unseen = set(donors)
    groups = []
    while unseen:
        stack, group = [unseen.pop()], set()
        while stack:
            i = stack.pop()
            group.add(i)
            neighbors = graph[i] & unseen
            unseen -= neighbors
            stack.extend(neighbors)
        groups.append(group)
    assert sorted(map(len, groups)) == [1, 1, 2, 5]
    ring = next(g for g in groups if len(g) == 5)
    assert all(len(graph[i]) == 2 for i in ring)  # actual five-cycle
    # Two CO contacts, one alkene pair, three Cp sites; not raw CN9.
    assert sum(3 if len(g) == 5 else 1 for g in groups) == 6
    for group in groups:
        if len(group) == 1:
            carbon = next(iter(group))
            assert any(carbon in (a, b) and elements[b if a == carbon else a] == "O"
                       for a, b, _ in bonds)
    record = molecular("figure2/48", prefix_mode="suffix-only")
    assert record["Suffix 3"] == CP
    assert record["Real Coordination Number"] == 6
    assert record["Geometry Type"] == "Octahedral"
    assert record["Confidence"] == pytest.approx(.449501717853311)


@pytest.mark.parametrize("name,stem", [("cp_star", "figure2/48"), ("fecl4", "data/405")])
def test_auto_prefix_golden_without_supplied_base(name, stem):
    Chem = pytest.importorskip("rdkit.Chem")
    record = molecular(stem)  # no golden or reference prefix passed as input
    for key, value in GOLDEN[name].items():
        if key != "variants":
            assert record[key] == value
    assert record["prefix_source"] == "sdf_rdkit"
    assert record["prefix_mode"] == "auto"
    assert record["charge_policy"] == "retain_sdf_no_charge_search_no_implicit_H_completion"
    ligand = Chem.MolFromSmiles(record["SMILES_1"])
    assert ligand is not None and all(a.GetSymbol() != "Fe" for a in ligand.GetAtoms())
    assert Chem.GetFormalCharge(ligand) == 0  # no chloride anion invented
    assert len(Chem.GetMolFrags(ligand)) == 4  # duplicates retained
    if name == "fecl4":
        assert all(a.GetSymbol() == "Cl" and a.GetNumRadicalElectrons() == 1 for a in ligand.GetAtoms())
    else:
        parts = record["SMILES_1"].split(".")
        assert parts.count("[C][O]") == 2  # single C-O bonds are SDF-authoritative
        assert "[C]=O" not in parts
        assert sum(a.GetNumRadicalElectrons() for a in ligand.GetAtoms()) > 0
    # These two inputs have no assigned ligand stereocenters: do not invent stereo labels.
    assert not Chem.FindMolChiralCenters(ligand, includeUnassigned=False)
    assert record["SMILES_3"] == record["SMILES_1"] + "." + record["Suffix 3"]


def test_reference_prefix_is_an_explicit_distinct_path():
    reference = json.loads((EXAMPLES / "figure2/reference.json").read_text())
    record = molecular("figure2/48", base_smiles=reference["supplied_base_smiles"])
    assert record["prefix_source"] == "explicit_base_smiles"
    assert record["Suffix 3"] == CP != reference["manuscript_diagram_suffix"]
    assert record["SMILES_3"] == reference["supplied_base_smiles"] + "." + CP
    assert record["SMILES_1"] != GOLDEN["cp_star"]["SMILES_1"]


@pytest.mark.parametrize("stem,suffix", [("figure2/48", CP), ("data/405", CL)])
def test_actual_molecular_ablations(stem, suffix):
    record = molecular(stem, prefix_mode="suffix-only")
    assert record["Suffix 3"] == suffix
    if stem == "figure2/48":
        assert mask_donor_entries(suffix) == "[{Octahedral}Fe&X&X%X%X]"
        assert remove_pi_marker(suffix) == "[{Octahedral}Fe&C&C&CC&CCCCC]"
        assert remove_geometry(suffix) == "[Fe&C&C%CC%CCCCC]"
    else:
        assert mask_donor_entries(suffix) == "[{Tetrahedral}Fe&X&X&X&X]"
        assert remove_pi_marker(suffix) == CL
        assert remove_geometry(suffix) == "[Fe&Cl&Cl&Cl&Cl]"


def test_mil_oxygen_shells_from_independent_periodic_distances():
    from itertools import product
    from metsmiles import read_cif
    parsed = read_cif(EXAMPLES / "data/mil-88b.cif")
    assert len(parsed["symops"]) == 1  # this expanded P1 dump needs no symmetry inference
    cell = parsed["cell"]
    assert cell["alpha"] == cell["beta"] == 90
    assert cell["gamma"] == 120
    # Hexagonal metric, independently computed without local_molecule_for_metal.
    matrix = np.array([[cell["a"], 0, 0], [-cell["b"] / 2, cell["b"] * math.sqrt(3) / 2, 0], [0, 0, cell["c"]]])
    atoms = parsed["atoms"]
    metals = [a for a in atoms if a[1] == "Fe"]
    oxygens = [a for a in atoms if a[1] == "O"]
    assert len(metals) == 6
    shells = []
    for metal in metals:
        shell = []
        for oxygen in oxygens:
            for shift in product((-1, 0, 1), repeat=3):
                delta = np.array(oxygen[2:]) + shift - np.array(metal[2:])
                distance = np.linalg.norm(delta @ matrix)
                if distance < 2.4:  # physical first-shell window, not covalent-radii API
                    assert 1.8 < distance < 2.2
                    shell.append(oxygen[0])
        assert len(shell) == 6
        shells.append(shell)
    counts = {label: sum(label in shell for shell in shells) for label in {x for s in shells for x in s}}
    bridging = [label for label, count in counts.items() if count == 3]
    assert len(bridging) == 2
    assert all(sum(label in bridging for label in shell) == 1 for shell in shells)
    assert all(count in (1, 3) for count in counts.values())


def test_mil_core_counts_and_explicit_topology():
    args = (EXAMPLES / "data/mil-88b.cif",)
    record = generate_global_metsmiles_from_cif(*args, topology="acs", with_ligands=False, detect_symmetry=False)
    assert record["Global Suffix 3"] == MIL.replace("sg=P6_3/mmc;pg=6/mmm", "sg=P1;pg=1")
    assert record["n_metal_centers"] == len(record["centers"]) == 6
    assert len(record["clusters"]) == 2
    assert [len(c["centers"]) for c in record["clusters"]] == [3, 3]
    assert record["periodicity"] == 3
    for center in record["centers"]:
        assert center["Real Coordination Number"] == 6
        assert center["Geometry Type"] == "Octahedral"
        assert center["Suffix 3"] == "[{Octahedral}Fe&O&O&O&O&O&O:u3]"
        assert center["Shared Donors"] == 1
    automatic = generate_global_metsmiles_from_cif(*args, with_ligands=False, detect_symmetry=False)
    assert "topo=" not in automatic["Global Suffix 3"]
    assert record["space_group"] == "P1"
    assert not record["symmetry_detection_enabled"]
    assert record["SMILES_1"] == ""


def test_mil_full_golden_and_linker_charge():
    Chem = pytest.importorskip("rdkit.Chem")
    pytest.importorskip("spglib")
    record = generate_global_metsmiles_from_cif(EXAMPLES / "data/mil-88b.cif", topology="acs")
    assert record["Global Suffix 3"] == MIL
    for key, value in GOLDEN["mil88b"].items():
        if key != "variants":
            assert record[key] == value
    linker = Chem.MolFromSmiles(record["SMILES_1"])
    assert Chem.GetFormalCharge(linker) == -2
    assert sum(a.GetSymbol() == "C" for a in linker.GetAtoms()) == 8
    assert sum(a.GetSymbol() == "O" for a in linker.GetAtoms()) == 4
    assert record["ligands"][0]["n_copies"] == 6
    assert record["prefix_scope"] == "largest_finite_linker_not_full_composition"
    assert record["space_group_source"] == "spglib"
    for label, function in [("noDonor", mask_donor_entries), ("noPi", remove_pi_marker), ("noGeo", remove_geometry)]:
        assert function(record["SMILES_3"]) == GOLDEN["mil88b"]["variants"][label]
    assert mask_donor_entries(MIL) == "{cluster:Fe3;u3-O}x2.[{Octahedral}Fe&X&X&X&X&X&X:u3]x3.{periodic:3D;sg=P6_3/mmc;pg=6/mmm;topo=acs}"


@pytest.mark.parametrize("dependency,flag", [("rdkit", "--suffix-only --no-ligands"), ("spglib", "--no-symmetry-detection")])
def test_runner_missing_dependency_is_actionable(monkeypatch, dependency, flag):
    import runpy
    module = runpy.run_path(str(EXAMPLES / "run_paper_examples.py"))
    original = importlib.util.find_spec
    def find_spec(name, *args, **kwargs):
        # Simulate the other extra being installed even on a minimal test host.
        if name in ("rdkit", "spglib"):
            return None if name == dependency else object()
        return original(name, *args, **kwargs)
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    with pytest.raises(RuntimeError, match=flag):
        module["run_examples"]()


def test_core_runner_does_not_need_optional_dependencies(monkeypatch):
    import builtins
    import runpy
    module = runpy.run_path(str(EXAMPLES / "run_paper_examples.py"))
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in ("rdkit", "spglib"):
            raise ImportError("optional dependency deliberately unavailable")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    result = module["run_examples"](suffix_only=True, no_ligands=True, no_symmetry_detection=True)
    assert len(result["examples"]) == 3
    assert all(x["actual_canonical_suffix"] == x["expected_canonical_suffix"] for x in result["examples"])


@pytest.mark.parametrize("full", [False, True])
def test_runner_from_other_working_directory(tmp_path, full):
    if full:
        pytest.importorskip("rdkit")
        pytest.importorskip("spglib")
    cmd = [sys.executable, str(EXAMPLES / "run_paper_examples.py")]
    if not full:
        cmd += ["--suffix-only", "--no-ligands", "--no-symmetry-detection"]
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert len(data["examples"]) == 3
    for example in data["examples"]:
        assert example["actual_canonical_suffix"] == example["expected_canonical_suffix"]
        record = example["record"]
        assert record["method_profile"] == "manuscript-v1"
        assert record["serialization_profile"] == "canonical-sorted-v1"
        if full:
            for key, value in GOLDEN[example["id"]].items():
                assert (example["variants"] if key == "variants" else record[key]) == value
        assert not example["topology_automatic"]
        assert not any(Path(p).is_absolute() for p in example["input_files"].values())
