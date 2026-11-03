"""Focused regressions for parsing, periodic topology and scientific semantics."""
from __future__ import annotations

import csv
import itertools
import json
import math
import shutil
import subprocess
import sys

import numpy as np
import pytest

from metsmiles import (
    Atom, Molecule, expand_unit_cell, generate_global_metsmiles_from_cif,
    generate_metsmiles_records, geometry_from_metsmiles, geometry_token_only,
    load_structure, read_cif, read_xyz_file, recognize_directory,
    remove_donor_list, remove_pi_marker,
)
from metsmiles.cif import (
    _apply_symop, _bond_graph, _fraction_to_cartesian, _image_position,
    _images_within, _metal_donor_sites, _nearest_image, _orthonormalisation,
    _same_fragment, _translation_rank, fragment_to_smiles,
    ligand_fragments_from_cif, local_molecule_for_metal, periodicity_dimensionality,
)


def cell(a=10., b=10., c=10., alpha=90., beta=90., gamma=90.):
    return dict(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)


def cif_file(tmp_path, sites, metric=None, extra=""):
    metric = metric or cell()
    lines = ["data_test", "_space_group_name_H-M_alt 'P 1'", "_space_group_IT_number 1"]
    for k, v in metric.items():
        lines.append(f"_cell_{'length' if k in 'abc' else 'angle'}_{k} {v}")
    lines += [extra, "loop_", "_atom_site_label", "_atom_site_type_symbol",
              "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
    lines.extend(" ".join(map(str, site)) for site in sites)
    path = tmp_path / "test.cif"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize("columns,rows", [
    ("_space_group_symop_id\n_space_group_symop_operation_xyz", "1 'x, y, z'\n2 '-x + 1/2, -y, z'"),
    ("_space_group_symop_operation_xyz\n_space_group_symop_id", '"x, y, z" 1\n"-x + 1/2, -y, z" 2'),
    ("_symmetry_equiv_pos_site_id\n_symmetry_equiv_pos_as_xyz", "1\n'x, y, z' 2 '-x + 1/2, -y, z' # comment"),
])
def test_cif_numeric_symop_ids_spaces_and_column_order(tmp_path, columns, rows):
    path = cif_file(tmp_path, [("Fe1", "Fe", .1, .2, .3)], extra=f"loop_\n{columns}\n{rows}")
    parsed = read_cif(path)
    assert parsed["symops"] == ["x, y, z", "-x + 1/2, -y, z"]
    expanded = expand_unit_cell(parsed["atoms"], parsed["symops"])
    assert len(expanded) == 2
    assert expanded[1][2:] == pytest.approx((.4, .8, .3))


def test_cif_atom_column_order_multiline_scalar_and_text(tmp_path):
    path = cif_file(tmp_path, [("Fe1", "Fe", .1, .2, .3)], extra=(
        "_description\n;ignored\nloop_\n_atom_site_label\n;\n"
        "_space_group_symop_operation_xyz\n'x, y, z'"))
    text = path.read_text().replace(
        "_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\nFe1 Fe 0.1 0.2 0.3",
        "_atom_site_fract_z\n_atom_site_type_symbol\n_atom_site_fract_y\n_atom_site_label\n_atom_site_fract_x\n0.3 Fe 0.2\n'Fe one' 0.100(2)")
    path.write_text(text)
    assert read_cif(path)["atoms"] == [("Fe one", "Fe", .1, .2, .3)]


@pytest.mark.parametrize("bad", ["0.1garbage", "?", "nan", "1e999"])
def test_cif_malformed_numbers_rejected(tmp_path, bad):
    path = cif_file(tmp_path, [("Fe1", "Fe", bad, 0, 0)])
    with pytest.raises(ValueError):
        read_cif(path)


def test_cif_incomplete_loop_rejected(tmp_path):
    path = cif_file(tmp_path, [("Fe1", "Fe", 0, 0, 0)])
    path.write_text(path.read_text() + "O1 O .2\n")
    with pytest.raises(ValueError, match="incomplete CIF loop"):
        read_cif(path)


def test_cif_partial_occupancy_rejected(tmp_path):
    path = cif_file(tmp_path, [("Fe1", "Fe", 0, 0, 0)])
    path.write_text(path.read_text().replace("_atom_site_fract_z\n", "_atom_site_fract_z\n_atom_site_occupancy\n")
                    .replace("Fe1 Fe 0 0 0", "Fe1 Fe 0 0 0 0.5"))
    with pytest.raises(ValueError, match="occupancy"):
        read_cif(path)


def test_symmetry_deduplication_preserves_species_and_precision():
    sites = [("Fe1", "Fe", .1234567, 0, 0), ("O1", "O", .1234567, 0, 0)]
    expanded = expand_unit_cell(sites, ["x,y,z", "x+1,y,z"])
    assert len(expanded) == 2
    assert {s[1] for s in expanded} == {"Fe", "O"}
    assert expanded[0][2] == .1234567


@pytest.mark.parametrize("op", ["__import__('os'),y,z", "x*x,y,z", "x/0,y,z", "x,y", "0,y,z"])
def test_symmetry_rejects_executable_or_invalid_expressions(op):
    with pytest.raises(ValueError):
        _apply_symop(.1, .2, .3, op)


def test_symmetry_affine_fraction_arithmetic():
    assert _apply_symop(.1, .2, .3, "-x + 1/2, x-y, z+1/3") == pytest.approx((.4, -.1, .3 + 1/3))


def skew_metric():
    return cell(a=3., b=3 * math.sqrt(101), c=50., gamma=math.degrees(math.atan(.1)))


def brute_images(delta, matrix, limit, radius=None):
    values = [(t, float(np.linalg.norm(matrix @ (np.array(delta) + t))))
              for t in itertools.product(range(-limit, limit + 1), repeat=3)]
    return sorted((p for p in values if radius is None or p[1] <= radius + 1e-10), key=lambda p: (p[1], p[0]))


def test_skew_nearest_image_beyond_fixed_one_cell_box():
    metric = skew_metric()
    matrix = _orthonormalisation(metric)
    delta = np.array([.49, .49, 0])
    expected = brute_images(delta, matrix, 12)[0]
    offset, distance = _nearest_image(delta, matrix)
    assert max(abs(v) for v in offset) > 1
    assert distance == pytest.approx(expected[1])
    assert distance < np.linalg.norm(matrix @ delta)
    assert _image_position(tuple(delta), (0., 0., 0.), (0., 0., 0.), metric) == pytest.approx(matrix @ (delta + offset))
    assert set(t for t, _ in _images_within(delta, matrix, 2.5288)) == set(
        t for t, _ in brute_images(delta, matrix, 12, 2.5288))


def test_skew_graph_donor_detection_and_local_sphere_agree():
    metric = skew_metric()
    sites = [("Fe1", "Fe", 0, 0, 0), ("O1", "O", .49, .49, 0)]
    graph = _bond_graph(sites, metric, {"Fe"}, 1.16, 1.2)
    hits = _metal_donor_sites(sites, metric, [0], {"Fe"}, 1.16)
    cart = [(label, element, *_fraction_to_cartesian(*frac, metric)) for label, element, *frac in sites]
    local = local_molecule_for_metal(sites, cart, metric, 0, [1])
    edges = [(j, t) for j, t in graph[0] if j == 1]
    assert len(hits[1]) == len(edges) == local.num_atoms() - 1
    assert {a.image_offset for a in local.atoms[1:]} == {t for _, t in edges}
    assert all(0 < a.distance_to(local.atoms[0]) <= 2.5288 + 1e-10 for a in local.atoms[1:])


def diagonal_chain():
    return [(f"C{i}", "C", i / 4, i / 4, 0.) for i in range(4)]


def test_diagonal_chain_has_rank_one_not_two_axes():
    assert periodicity_dimensionality(diagonal_chain(), cell(a=4, b=4, c=20)) == 1


def test_finite_boundary_straddling_dimer_is_zero_dimensional(tmp_path):
    sites = [("C1", "C", .95, 0, 0), ("C2", "C", .05, 0, 0)]
    assert periodicity_dimensionality(sites, cell()) == 0
    path = cif_file(tmp_path, sites)
    fragments = ligand_fragments_from_cif(path)
    assert len(fragments) == 1 and not fragments[0]["infinite"]
    assert fragments[0]["bonds"] == [(0, 1)]
    assert math.dist(fragments[0]["atoms"][0][2:], fragments[0]["atoms"][1][2:]) == pytest.approx(1.)


def test_rank_invariant_to_cell_basis_wrapping_and_site_order():
    metric = cell(a=4, b=4, c=20)
    sites = diagonal_chain()
    basis = np.array([[1, 5, 0], [0, 1, 0], [0, 0, 1]])
    lattice = _orthonormalisation(metric) @ basis
    a, b, c = np.linalg.norm(lattice, axis=0)
    gamma = math.degrees(math.acos(np.dot(lattice[:, 0], lattice[:, 1]) / a / b))
    changed = cell(a=a, b=b, c=c, gamma=gamma)
    transformed = []
    for i, (label, element, *frac) in enumerate(sites):
        f = (np.linalg.inv(basis) @ frac) % 1 + np.array([i, -2 * i, 3 * i])
        transformed.append((label, element, *f))
    assert periodicity_dimensionality(transformed[::-1], changed) == 1


def test_isolated_fe_does_not_hide_periodic_host(tmp_path):
    sites = [("Fe1", "Fe", .5, .5, .5)] + diagonal_chain()
    metric = cell(a=4, b=4, c=20)
    graph = _bond_graph(sites, metric, {"Fe"}, 1.16, 1.2)
    assert graph[0] == []
    path = cif_file(tmp_path, sites, metric)
    record = generate_global_metsmiles_from_cif(path, with_ligands=False, detect_symmetry=False)
    assert record["periodicity"] == 1
    assert record["periodicity_semantics"] == "maximum_bonded_component_cycle_translation_rank"
    assert record["centers"][0]["Real Coordination Number"] == 0


def test_self_image_network_and_exact_integer_rank():
    metric = cell(a=1.5, b=1.5, c=1.5)
    sites = [("C1", "C", 0, 0, 0)]
    graph = _bond_graph(sites, metric, {"Fe"}, 1.16, 1.2)
    assert len(graph[0]) == 6
    assert all(j == 0 and any(t) for j, t in graph[0])
    assert periodicity_dimensionality(sites, metric) == 3
    assert _translation_rank([(1, 1, 0), (2, 2, 0)]) == 1
    assert _translation_rank([(10**20, 1, 0), (10**20 + 1, 1, 0), (0, 0, 1)]) == 3


def test_parallel_image_edges_count_distinct_donors_and_metal_images(tmp_path):
    metric = cell(a=4, b=20, c=20)
    sites = [("Fe1", "Fe", 0, 0, 0), ("O1", "O", .5, 0, 0)]
    path = cif_file(tmp_path, sites, metric)
    graph = _bond_graph(sites, metric, {"Fe"}, 1.16, 1.2)
    assert {t for j, t in graph[0] if j == 1} == {(-1, 0, 0), (0, 0, 0)}
    record = generate_global_metsmiles_from_cif(path, detect_symmetry=False)
    assert record["periodicity"] == 1
    assert record["centers"][0]["Suffix 2"] == "[Fe&O:u2&O:u2]"
    assert record["centers"][0]["Geometry Type"] == "Linear"


@pytest.mark.parametrize("metric", [cell(a=0), cell(gamma=0), cell(alpha=10, beta=10, gamma=170), cell(a=float('nan'))])
def test_invalid_cell_rejected(metric):
    with pytest.raises(ValueError):
        _orthonormalisation(metric)


def test_infinite_organic_fragments_do_not_get_fabricated_molecular_smiles(tmp_path):
    path = cif_file(tmp_path, [("C1", "C", 0, 0, 0)], cell(a=1.5, b=10, c=10))
    fragment = ligand_fragments_from_cif(path)[0]
    assert fragment["infinite"] and len(fragment["atoms"]) == 1
    assert fragment_to_smiles(fragment) is None


def test_fragment_deduplication_uses_graph_not_composition():
    def fragment(bonds):
        return dict(atoms=[("C", "C", 0, 0, 0)] * 4, bonds=bonds, infinite=False, truncated=False)
    chain = fragment([(0, 1), (1, 2), (2, 3)])
    star = fragment([(0, 1), (0, 2), (0, 3)])
    assert not _same_fragment(chain, star)
    assert _same_fragment(chain, fragment([(3, 2), (2, 0), (0, 1)]))


def molecule_with_donors(elements, coords, bonds):
    molecule = Molecule()
    molecule.add_atom(Atom("Fe", 0, 0, 0))
    for element, xyz in zip(elements, coords):
        molecule.add_atom(Atom(element, *xyz))
    molecule.bond_info = {(i + 2, j + 2): order for i, j, order in bonds}
    return molecule


@pytest.mark.parametrize("elements,coords,bonds", [
    (["O", "O"], [(-.7, 0, 1.8), (.7, 0, 1.8)], [(0, 1, 1)]),
    (["O", "C", "O"], [(-1.1, 0, 1.7), (0, 0, 2.1), (1.1, 0, 1.7)], [(0, 1, 1), (1, 2, 2)]),
    (["C", "C"], [(-.77, 0, 1.8), (.77, 0, 1.8)], [(0, 1, 1)]),
])
def test_connected_components_follow_manuscript_syntax_not_electronic_claim(elements, coords, bonds):
    molecule = molecule_with_donors(elements, coords, bonds)
    result = molecule.analyze_ligands(molecule.atoms[0], molecule.atoms[1:])
    assert len(result["pi_ligands"]) == 1
    assert result["normal_ligands"] == []
    assert result["real_coordination_number"] == 1
    assert result["pi_coordination_sites"][0] == pytest.approx(np.mean(coords, axis=0))


def ring_molecule(close=True):
    coords = [(1.2 * math.cos(t), 1.2 * math.sin(t), 1.6) for t in np.arange(5) * 2 * math.pi / 5]
    bonds = [(i, i + 1, 2 if i in (0, 2) else 1) for i in range(4)]
    if close:
        bonds.append((4, 0, 1))
    return molecule_with_donors(["C"] * 5, coords, bonds)


def test_five_connected_donors_are_not_automatically_cp():
    molecule = ring_molecule(False)
    result = molecule.analyze_ligands(molecule.atoms[0], molecule.atoms[1:])
    assert len(result["pi_ligands"]) == 1
    assert result["real_coordination_number"] == 1
    assert result["pi_coordination_sites"][0] == pytest.approx(np.mean([a.coordinates for a in molecule.atoms[1:]], axis=0))
    with pytest.raises(ValueError, match="cycle"):
        molecule.find_points_in_ring(molecule.atoms[1], molecule.atoms[1:])


def test_cp_virtual_sites_follow_cycle_topology_not_input_order():
    molecule = ring_molecule()
    ring = molecule.atoms[1:]
    result = molecule.analyze_ligands(molecule.atoms[0], ring)
    assert result["real_coordination_number"] == 3
    anchor, b, c = molecule.find_points_in_ring(ring[0], [ring[i] for i in (2, 0, 4, 1, 3)])
    assert anchor is ring[0]
    expected = [np.mean([ring[i].coordinates, ring[j].coordinates], axis=0) for i, j in [(1, 2), (4, 3)]]
    assert any(np.allclose(b, x) for x in expected) and any(np.allclose(c, x) for x in expected)
    assert not np.allclose(b, c)
    molecule.bond_info[(2, 4)] = 1  # chord breaks the induced cycle
    broken = molecule.analyze_ligands(molecule.atoms[0], ring)
    assert len(broken["pi_ligands"]) == 1
    assert broken["real_coordination_number"] == 1


def test_two_donor_component_uses_midpoint():
    molecule = molecule_with_donors(["C", "C"], [(-.7, 0, 1.8), (.7, 0, 1.8)], [(0, 1, 1)])
    result = molecule.analyze_ligands(molecule.atoms[0], molecule.atoms[1:])
    assert result["real_coordination_number"] == 1 and len(result["pi_ligands"]) == 1
    assert result["pi_coordination_sites"][0] == pytest.approx((0, 0, 1.8))


def test_geometry_profiles_cn_above_six_and_monotone_squared_error():
    molecule = Molecule()
    center = Atom("Fe", 0, 0, 0)
    donors = [Atom("N", math.cos(t), math.sin(t), 1) for t in np.arange(7) * 2 * math.pi / 7]
    result = molecule.analyze_coordination_geometry(center, donors)
    assert result["geometry_type"] == "unknown" and result["possible_geometries"] == []
    actual, ideal = [70., 110., 170.], [90., 90., 180.]
    paired = molecule._match_angles(sorted(actual), sorted(ideal))
    error = sum((a - b)**2 for a, b in paired)
    assert error == min(sum((a - b)**2 for a, b in zip(actual, p)) for p in itertools.permutations(ideal))
    with pytest.raises(ValueError, match="equal lengths"):
        molecule._match_angles([90], [])


@pytest.mark.parametrize("text", [
    "2\ncomment\nFe 0 0 0\n", "1\ncomment\nFe 0 0\n", "1\ncomment\nFe nan 0 0\n",
    "1\ncomment\nFe inf 0 0\n", "-1\ncomment\n", "0\ncomment\n",
    "1\ncomment\nFe 0 0 0\nO 0 0 1\n", "1\ncomment\nFe 0 0 0 junk\n",
    "1\ncomment\n\n", "1\ncomment\nNotAnElement 0 0 0\n",
])
def test_strict_xyz_rejects_malformed_truncated_and_nonfinite(tmp_path, text):
    path = tmp_path / "bad.xyz"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_xyz_file(path)


@pytest.mark.parametrize("kind", ["count", "element", "coordinates", "truncated", "bond_index", "no_counts", "bond_count", "multi_record"])
def test_sdf_xyz_mismatch_and_invalid_blocks(example_data, tmp_path, kind):
    lines = (example_data / "405.sdf").read_text().splitlines()
    if kind == "count":
        lines[3] = "  4" + lines[3][3:]
    elif kind == "element":
        lines[4] = lines[4][:31] + "Co " + lines[4][34:]
    elif kind == "coordinates":
        lines[4] = " 9999.0000" + lines[4][10:]
    elif kind == "truncated":
        lines = lines[:7]
    elif kind == "bond_index":
        n = int(lines[3][:3])
        lines[4 + n] = "999" + lines[4 + n][3:]
    elif kind == "bond_count":
        lines[3] = lines[3][:3] + f"{int(lines[3][3:6]) - 1:3}" + lines[3][6:]
    elif kind == "multi_record":
        lines += lines.copy()
    else:
        lines[3] = "bad counts"
    path = tmp_path / "bad.sdf"
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="invalid SDF"):
        load_structure(example_data / "405.xyz", path)


def test_successful_batch_rows_keep_generation_metadata(example_data, tmp_path):
    shutil.copy2(example_data / "405.xyz", tmp_path / "a.xyz")
    shutil.copy2(example_data / "405.xyz", tmp_path / "b.xyz")
    shutil.copy2(example_data / "405.sdf", tmp_path / "b.sdf")
    output = tmp_path / "out.csv"
    rows = recognize_directory(tmp_path, output, bond_tolerance=1.13, geometry_sigma=20, connectivity_tolerance=1.18, prefix_mode="suffix-only")
    assert {r["connectivity_source"] for r in rows} == {"sdf", "distance_inference"}
    for row in rows:
        assert (row["bond_tolerance"], row["geometry_sigma"], row["connectivity_tolerance"]) == (1.13, 20., 1.18)
    with output.open() as handle:
        assert all(row["bond_tolerance"] == "1.13" for row in csv.DictReader(handle))


def test_per_center_base_smiles_is_used(example_data):
    rows = generate_metsmiles_records(example_data / "405.xyz", None, base_smiles="C%10CCCCC%10", prefix_mode="suffix-only")
    assert rows[0]["SMILES_3"] == "C%10CCCCC%10." + rows[0]["Suffix 3"]
    assert rows[0]["base_smiles_scope"] == "supplied_structure_prefix"
    assert rows[0]["connectivity_source"] == "distance_inference"


def test_non_fe_multicenter_helpers_preserve_global_grammar_and_organic_rings():
    organic = "C%10CCCCC%10.[NH4+]"
    suffix = "{cluster:CoNi;u2-O}x2.[{Linear}Co&O:u2%CC]x2|[{SquarePlanar}Ni&N&N&O:u2&O].{periodic:1D;sg=P1;pg=1}"
    value = organic + "." + suffix
    assert geometry_from_metsmiles(value) == "Linear|SquarePlanar"
    stripped = remove_donor_list(value)
    assert stripped == organic + ".{cluster:CoNi;u2-O}x2.[{Linear}Co]x2|[{SquarePlanar}Ni].{periodic:1D;sg=P1;pg=1}"
    assert remove_pi_marker(value) == organic + "." + suffix.replace("%CC", "&CC")
    assert geometry_token_only(organic, value) == stripped
    assert remove_pi_marker(organic) == organic
    assert geometry_token_only(organic, organic) == organic
    assert remove_donor_list("[Co&N%CC]") == "[Co]"


def test_cif_center_indices_refer_to_expanded_unit_cell(example_data):
    path = example_data / "mil-88b.cif"
    parsed = read_cif(path)
    sites = expand_unit_cell(parsed["atoms"], parsed["symops"])
    record = generate_global_metsmiles_from_cif(path, with_ligands=False, detect_symmetry=False)
    indices = [center["metal_index"] for center in record["centers"]]
    assert indices == [i for i, site in enumerate(sites) if site[1] == "Fe"]
    assert len(set(indices)) == 6
    assert all(center["metal_label"] == sites[center["metal_index"]][0] for center in record["centers"])


def test_cli_cif_control_flags_and_invalid_combinations(example_data):
    result = subprocess.run([sys.executable, "-m", "metsmiles", str(example_data / "mil-88b.cif"),
                             "--no-ligands", "--no-symmetry-detection", "--space-group-symprec", "0.02", "--topology", "the"],
                            capture_output=True, text=True, check=True)
    record = json.loads(result.stdout)
    assert record["ligands"] == [] and record["space_group_source"] == "file"
    assert record["space_group_symprec"] == .02
    assert record["SMILES_3"].endswith("topo=the}")
    bad = subprocess.run([sys.executable, "-m", "metsmiles", str(example_data / "405.xyz"), "--no-ligands"],
                         capture_output=True, text=True)
    assert bad.returncode == 2 and "CIF-specific" in bad.stderr


def test_per_center_prefix_and_metadata_for_multiple_metals(tmp_path):
    path = tmp_path / "two.xyz"
    path.write_text("4\ntwo separated centers\nFe 0 0 0\nFe 10 0 0\nN 2 0 0\nN 12 0 0\n")
    rows = generate_metsmiles_records(path, base_smiles="N.N", prefix_mode="suffix-only")
    assert len(rows) == 2
    assert all(row["SMILES_2"] == "N.N.[Fe&N]" for row in rows)
    assert all(row["connectivity_source"] == "distance_inference" for row in rows)


def test_spglib_legacy_signature_and_exception_fallback(monkeypatch):
    from types import SimpleNamespace
    from metsmiles.cif import detect_space_group
    calls = []

    def legacy(structure, symprec):
        calls.append(symprec)
        return SimpleNamespace(international="P 1", number=1, pointgroup="1")

    monkeypatch.setitem(sys.modules, "spglib", SimpleNamespace(get_symmetry_dataset=legacy))
    result = detect_space_group([("Fe", "Fe", 0, 0, 0)], cell())
    assert result["international"] == "P1" and calls == [.05]

    def failing(*args, **kwargs):
        raise RuntimeError("no symmetry found")

    monkeypatch.setitem(sys.modules, "spglib", SimpleNamespace(get_symmetry_dataset=failing))
    assert detect_space_group([("Fe", "Fe", 0, 0, 0)], cell()) is None


def test_cif_runs_without_both_optional_dependencies(example_data, monkeypatch):
    import builtins
    original = builtins.__import__

    def without_extras(name, *args, **kwargs):
        if name == "spglib" or name.startswith("rdkit"):
            raise ImportError("optional package unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_extras)
    record = generate_global_metsmiles_from_cif(example_data / "mil-88b.cif")
    assert record["periodicity"] == 3 and record["space_group_source"] == "file"
    assert all(ligand["smiles"] is None for ligand in record["ligands"])


def test_organic_metal_ions_are_not_duplicated_by_geometry_helper():
    organic = "[Fe].C%10CCCCC%10"
    suffix = "[{Linear}Co&N&N]"
    assert geometry_from_metsmiles(organic + "." + suffix) == "Linear"
    assert geometry_token_only(organic, organic + "." + suffix) == organic + ".[{Linear}Co]"
    assert geometry_token_only(organic, organic) == organic


def test_zero_length_angles_and_nonfinite_parameters_are_rejected():
    with pytest.raises(ValueError, match="undefined"):
        Atom("Fe", 0, 0, 0).bond_angle(Atom("O", 0, 0, 0), Atom("O", 1, 0, 0))
    with pytest.raises(ValueError):
        Molecule(geometry_sigma=float("nan"))
    with pytest.raises(ValueError):
        Molecule(bond_tolerance=float("inf"))


def test_spglib_detection_is_not_ranked_by_table_number(tmp_path, monkeypatch):
    import metsmiles.cif as module
    path = cif_file(tmp_path, [("Fe1", "Fe", .1, .2, .3)])
    path.write_text(path.read_text().replace("IT_number 1", "IT_number 230"))
    monkeypatch.setattr(module, "detect_space_group", lambda *a, **k: dict(international="P1", number=1, point_group="1"))
    record = generate_global_metsmiles_from_cif(path, with_ligands=False)
    assert record["space_group"] == "P1" and record["space_group_source"] == "spglib"
