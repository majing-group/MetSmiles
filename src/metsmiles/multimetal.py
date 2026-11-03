"""Global MetSMILES annotations for multi-metal structures.

A single-center annotation only describes the local environment of one
metal. This module extends the grammar so that a whole structure - a
polynuclear complex, an inorganic cluster, or a MOF unit cell - is
described by ONE global string:

* a donor atom coordinated to ``k`` metal centers is tagged ``:uk``
  (``&O:u2`` = a donor bridging two metals, ``&O:u3`` = triply bridging;
  ``&O`` without a tag remains a terminal, single-metal donor),
* metal centers connected through shared donors are grouped into a
  ``{cluster:...}`` block summarising the inorganic core, e.g.
  ``{cluster:Fe3;u2-O6;u3-O1}`` = an Fe3 core held by six doubly-bridging
  O donors and one triply-bridging O donor,
* equivalent centers within one cluster are collapsed with ``xN``
  (``[{Octahedral}Fe&...]x3`` = three identical centers),
* a ``xN`` immediately after a ``{cluster:...}`` header counts identical
  clusters (``{cluster:Fe4;u4-O1;u2-O12}x2`` = two such clusters),
* cluster blocks are joined by ``.`` after the organic SMILES_1,
* ``Suffix 3`` additionally carries each center's geometry token,
  ``Suffix 2`` does not.

For a mononuclear structure the global string is identical to the
single-center suffix - the grammar degrades gracefully.

Example (an isolated MIL-88B-type Fe3(mu3-O)(COO)6 cluster)::

    SMILES_1.{cluster:Fe3;u2-O6;u3-O1}.[{Octahedral}Fe&O&O:u2&O:u2&O:u2&O:u2&O:u3]x3
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .generator import Atom, Molecule, load_structure
from .profiles import METHOD_PROFILE, profile_metadata
from .prefix import resolve_molecular_prefix

__all__ = [
    "sharing_map",
    "metal_clusters",
    "center_record",
    "cluster_header",
    "join_cluster_blocks",
    "global_metsmiles_for_molecule",
    "generate_metsmiles_records",
    "generate_global_metsmiles",
]


def _donor_atoms(molecule: Molecule, center: Atom) -> List[Atom]:
    """Coordination atoms of ``center`` excluding other target metal centers."""
    return [
        atom
        for atom in molecule.get_coordination_atoms(center)
        if atom.element not in molecule.target_metal_elements
    ]


def sharing_map(molecule: Molecule) -> Dict[int, List[int]]:
    """Map each donor atom index to the metal-center indices coordinating it.

    Keys and values are 0-based indices into ``molecule.atoms``.
    """
    sharing: Dict[int, List[int]] = {}
    for center in molecule.find_metal_centers():
        center_index = molecule.atoms.index(center)
        for donor in _donor_atoms(molecule, center):
            sharing.setdefault(molecule.atoms.index(donor), []).append(center_index)
    for metal_list in sharing.values():
        metal_list.sort()
    return sharing


def metal_clusters(
    molecule: Molecule, sharing: Dict[int, List[int]] | None = None
) -> List[List[int]]:
    """Group metal centers (0-based atom indices) that share donor atoms.

    Returns a list of clusters; each cluster is a sorted list of metal
    atom indices.
    """
    centers = molecule.find_metal_centers()
    metal_indices = [molecule.atoms.index(c) for c in centers]
    if sharing is None:
        sharing = sharing_map(molecule)

    parent = {i: i for i in metal_indices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for metal_list in sharing.values():
        for other in metal_list[1:]:
            union(metal_list[0], other)

    groups: Dict[int, List[int]] = {}
    for index in metal_indices:
        groups.setdefault(find(index), []).append(index)
    return [sorted(g) for g in sorted(groups.values(), key=lambda g: g[0])]


def _donor_tag(element: str, multiplicity: int) -> str:
    return element if multiplicity <= 1 else f"{element}:u{multiplicity}"


def center_record(
    molecule: Molecule,
    center_index: int,
    sharing: Dict[int, List[int]] | None = None,
    multiplicity_map: Dict[int, int] | None = None,
) -> Dict[str, Any]:
    """MetSMILES record for one metal center, with ``:uk`` shared-donor tags.

    Shared-donor multiplicities are taken from ``sharing`` (a
    :func:`sharing_map` result) unless ``multiplicity_map`` - a direct
    ``atom index -> k`` lookup - is supplied, as done by the periodic
    :mod:`metsmiles.cif` pipeline.
    """
    center = molecule.atoms[center_index]
    donors = _donor_atoms(molecule, center)
    ligand = molecule.analyze_ligands(center, donors)

    if multiplicity_map is None:
        if sharing is None:
            sharing = sharing_map(molecule)

        def multiplicity(atom: Atom) -> int:
            return len(sharing.get(molecule.atoms.index(atom), ())) or 1
    else:
        def multiplicity(atom: Atom) -> int:
            return multiplicity_map.get(molecule.atoms.index(atom), 1) or 1

    normal = sorted(
        (a.element, multiplicity(a)) for a in ligand["normal_ligands"]
    )
    normal_str = "".join(f"&{_donor_tag(el, k)}" for el, k in normal)

    pi_parts = []
    for component in ligand["pi_ligands"]:
        block = "".join(sorted(a.element for a in component))
        k = max(multiplicity(a) for a in component)
        pi_parts.append(block if k <= 1 else f"{block}:u{k}")
    pi_str = "".join(f"%{p}" for p in sorted(pi_parts))

    # Geometry is assigned on donor atoms plus virtual pi-ligand sites.
    all_ligands: List[Atom] = list(ligand["normal_ligands"])
    for site in ligand["pi_coordination_sites"]:
        all_ligands.append(Atom("[V]", site[0], site[1], site[2]))
    geometry = molecule.analyze_coordination_geometry(center, all_ligands)

    coordination = f"{center.element}{normal_str}{pi_str}"
    geom_type = geometry["geometry_type"]
    return {
        **profile_metadata(molecule.method_profile),
        "metal_index": center_index,
        "Metal Center": center.element,
        "Real Coordination Number": ligand["real_coordination_number"],
        "Geometry Type": geom_type,
        "Confidence": float(geometry["geometry_similarity"]),
        "Shared Donors": sum(1 for _, k in normal if k > 1),
        "Suffix 2": f"[{coordination}]",
        "Suffix 3": f"[{{{geom_type}}}{coordination}]",
    }


def cluster_header(
    metal_counts: Dict[str, int],
    bridges: Dict[Tuple[int, str], int],
) -> str:
    """Format the ``{cluster:Fe3;u2-O6;u3-O1}`` core descriptor.

    ``metal_counts`` maps element symbols to the number of metal centers;
    ``bridges`` maps ``(multiplicity, element)`` to the count of donor
    atoms shared by that many metals inside the cluster.
    """
    metals_str = "".join(
        f"{el}{n if n > 1 else ''}" for el, n in sorted(metal_counts.items())
    )
    bridge_str = ";".join(
        f"u{k}-{el}{n if n > 1 else ''}"
        for (k, el), n in sorted(bridges.items())
    )
    body = metals_str + (f";{bridge_str}" if bridge_str else "")
    return f"{{cluster:{body}}}"


def _cluster_header(
    molecule: Molecule,
    member_indices: Sequence[int],
    sharing: Dict[int, List[int]],
) -> str:
    metal_elements = Counter(molecule.atoms[i].element for i in member_indices)
    member_set = set(member_indices)
    bridges: Counter = Counter()
    for atom_index, metal_list in sharing.items():
        metal_set = set(metal_list)
        if len(metal_set) >= 2 and metal_set <= member_set:
            bridges[(len(metal_set), molecule.atoms[atom_index].element)] += 1
    return cluster_header(dict(metal_elements), dict(bridges))


def join_cluster_blocks(blocks: Sequence[str]) -> str:
    """Join per-cluster blocks with ``.``, collapsing identical ones.

    The repeat count ``xN`` of an identical cluster is attached to its
    ``{cluster:...}`` header (``{cluster:Fe3;u3-O}x2`` = two such clusters);
    for header-less single-metal blocks it is appended to the suffix.
    """
    counts = Counter(blocks)
    parts = []
    for block, n in sorted(counts.items()):
        if n > 1 and block.startswith("{cluster:"):
            head, _, rest = block.partition(".")
            parts.append(f"{head}x{n}.{rest}")
        elif n > 1:
            parts.append(f"{block}x{n}")
        else:
            parts.append(block)
    return ".".join(parts)


def _collapse(suffixes: List[str]) -> str:
    """Join deduplicated suffixes, appending ``xN`` for repeated entries."""
    counts = Counter(suffixes)
    return "|".join(
        f"{s}x{n}" if n > 1 else s for s, n in sorted(counts.items())
    )


def _global_block(
    molecule: Molecule,
    member_indices: Sequence[int],
    sharing: Dict[int, List[int]],
    with_geometry: bool,
) -> str:
    key = "Suffix 3" if with_geometry else "Suffix 2"
    suffixes = [center_record(molecule, i, sharing)[key] for i in member_indices]
    centers_str = _collapse(suffixes)
    if len(member_indices) > 1:
        return f"{_cluster_header(molecule, member_indices, sharing)}.{centers_str}"
    return centers_str


def _cluster_details(
    molecule: Molecule,
    member_indices: Sequence[int],
    sharing: Dict[int, List[int]],
) -> Dict[str, Any]:
    records = [center_record(molecule, i, sharing) for i in member_indices]
    member_set = set(member_indices)
    bridges: Counter = Counter()
    for atom_index, metal_list in sharing.items():
        metal_set = set(metal_list)
        if len(metal_set) >= 2 and metal_set <= member_set:
            bridges[f"u{len(metal_set)}-{molecule.atoms[atom_index].element}"] += 1
    return {
        "metal_indices": list(member_indices),
        "metal_composition": "".join(
            f"{el}{n if n > 1 else ''}"
            for el, n in sorted(
                Counter(molecule.atoms[i].element for i in member_indices).items()
            )
        ),
        "bridging_donors": dict(bridges),
        "centers": records,
    }


def global_metsmiles_for_molecule(
    molecule: Molecule,
    base_smiles: str | None = None,
) -> Dict[str, Any]:
    """Global MetSMILES record for every metal center in ``molecule``.

    Returns a dict with the global ``Suffix 2`` / ``Suffix 3`` strings, the
    ``SMILES_2`` / ``SMILES_3`` strings (``base_smiles`` + suffix when a base
    SMILES is given), a per-cluster ``clusters`` breakdown and the flat
    ``centers`` list of per-metal records.
    """
    sharing = sharing_map(molecule)
    clusters = metal_clusters(molecule, sharing)

    global_2 = join_cluster_blocks(
        [_global_block(molecule, members, sharing, with_geometry=False)
         for members in clusters]
    )
    global_3 = join_cluster_blocks(
        [_global_block(molecule, members, sharing, with_geometry=True)
         for members in clusters]
    )

    prefix = f"{base_smiles}." if base_smiles else ""
    all_centers = [
        center_record(molecule, i, sharing)
        for members in clusters
        for i in members
    ]
    return {
        **profile_metadata(molecule.method_profile),
        "SMILES_1": base_smiles or "",
        "prefix_source": "explicit_base_smiles" if base_smiles is not None else "suffix_only",
        "prefix_status": "supplied" if base_smiles is not None else "not_requested",
        "prefix_mode": "suffix-only",
        "n_metal_centers": len(all_centers),
        "Global Suffix 2": global_2,
        "Global Suffix 3": global_3,
        "SMILES_2": prefix + global_2,
        "SMILES_3": prefix + global_3,
        "clusters": [
            _cluster_details(molecule, members, sharing) for members in clusters
        ],
        "centers": all_centers,
        "bond_tolerance": molecule.bond_tolerance,
        "geometry_sigma": molecule.geometry_sigma,
    }


def generate_metsmiles_records(
    xyz_path: str | Path,
    sdf_path: str | Path | None = None,
    base_smiles: str | None = None,
    bond_tolerance: float = 1.16,
    geometry_sigma: float = 15.0,
    target_metal_elements: Sequence[str] | None = None,
    connectivity_tolerance: float = 1.20,
    method_profile: str = METHOD_PROFILE,
    prefix_mode: str = "auto",
) -> List[Dict[str, Any]]:
    """One MetSMILES record per metal center (any number of centers allowed).

    Shared donors carry ``:uk`` tags in the suffixes, so the records of
    linked centers are mutually consistent. Each record has ``SMILES_2/3``
    with an automatic SDF or explicit ``base_smiles`` prefix. That prefix describes the input
    structure, not an automatically extracted ligand local to the center;
    ``base_smiles_scope`` records this convention. Parameters are retained.
    """
    molecule = load_structure(
        xyz_path,
        sdf_path,
        bond_tolerance=bond_tolerance,
        geometry_sigma=geometry_sigma,
        target_metal_elements=target_metal_elements,
        connectivity_tolerance=connectivity_tolerance,
        method_profile=method_profile,
    )
    organic, metadata = resolve_molecular_prefix(
        sdf_path, molecule.target_metal_elements, base_smiles, prefix_mode)
    records = global_metsmiles_for_molecule(molecule, organic)["centers"]
    prefix = f"{organic}." if organic else ""
    for record in records:
        record.update({
            **metadata,
            "SMILES_1": organic,
            "SMILES_2": prefix + record["Suffix 2"],
            "SMILES_3": prefix + record["Suffix 3"],
            "base_smiles_scope": "supplied_structure_prefix" if base_smiles is not None else "whole_structure_prefix",
            "bond_tolerance": float(bond_tolerance),
            "geometry_sigma": float(geometry_sigma),
            "connectivity_source": "distance_inference" if sdf_path is None else "sdf",
            "connectivity_tolerance": float(connectivity_tolerance),
        })
    return records


def generate_global_metsmiles(
    xyz_path: str | Path,
    sdf_path: str | Path | None = None,
    base_smiles: str | None = None,
    bond_tolerance: float = 1.16,
    geometry_sigma: float = 15.0,
    target_metal_elements: Sequence[str] | None = None,
    connectivity_tolerance: float = 1.20,
    method_profile: str = METHOD_PROFILE,
    prefix_mode: str = "auto",
) -> Dict[str, Any]:
    """Global MetSMILES record for a whole (possibly multi-metal) structure.

    Accepts the same ``*.xyz``/``*.sdf`` inputs as
    :func:`metsmiles.generate_metsmiles_record` but never rejects
    multi-center structures.
    """
    molecule = load_structure(
        xyz_path,
        sdf_path,
        bond_tolerance=bond_tolerance,
        geometry_sigma=geometry_sigma,
        target_metal_elements=target_metal_elements,
        connectivity_tolerance=connectivity_tolerance,
        method_profile=method_profile,
    )
    if not molecule.find_metal_centers():
        raise ValueError("no target metal centers")
    organic, metadata = resolve_molecular_prefix(
        sdf_path, molecule.target_metal_elements, base_smiles, prefix_mode)
    record = global_metsmiles_for_molecule(molecule, organic)
    record.update(metadata)
    record["connectivity_source"] = "distance_inference" if sdf_path is None else "sdf"
    record["connectivity_tolerance"] = float(connectivity_tolerance)
    return record
