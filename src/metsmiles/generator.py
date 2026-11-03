"""Core MetSMILES generation from 3D structures of metal complexes.

MetSMILES augments a conventional organic SMILES string (SMILES_1) with a
coordination-environment suffix describing, for each metal center:

* the metal element,
* the donor atoms of classical (sigma) ligands, prefixed with ``&``,
* pi-ligand syntax blocks (connected donor components of size two or more), prefixed with ``%``,
* optionally, the assigned idealized coordination geometry, wrapped in
  curly braces at the head of the suffix.

Two suffix levels are produced:

* ``Suffix 2`` / ``SMILES_2``: metal + donor/pi-ligand composition.
* ``Suffix 3`` / ``SMILES_3``: additionally carries the geometry token.

Example
-------
An octahedral iron complex with two chlorides, two nitrogens and two
oxygens, whose organic part is ``CCO``, is annotated as::

    SMILES_1 = "CCO"
    SMILES_2 = "CCO.[Fe&Cl&Cl&N&N&O&O]"
    SMILES_3 = "CCO.[{Octahedral}Fe&Cl&Cl&N&N&O&O]"

The geometry token is assigned by matching the measured ligand-metal-ligand
angles against idealized angle profiles (e.g. 12x90 + 3x180 degrees for an
octahedron) and scoring each candidate geometry with a Gaussian kernel of
width ``geometry_sigma``.
"""

from __future__ import annotations

import csv
import glob
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .profiles import METHOD_PROFILE, profile_metadata
from .prefix import resolve_molecular_prefix

__all__ = [
    "Atom",
    "Molecule",
    "read_xyz_file",
    "load_structure",
    "perturb_and_reorder",
    "records_from_report",
    "generate_metsmiles_record",
    "recognize_directory",
]


class Atom:
    """An atom with an element symbol and Cartesian coordinates in Angstrom."""

    def __init__(self, element: str, x: float, y: float, z: float):
        self.element = element
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.coordinates = np.array([float(x), float(y), float(z)])
        if not np.all(np.isfinite(self.coordinates)):
            raise ValueError("atom coordinates must be finite")

    def distance_to(self, other_atom) -> float:
        """Euclidean distance to another atom (Angstrom)."""
        return np.linalg.norm(self.coordinates - other_atom.coordinates)

    def bond_angle(self, atom1, atom2) -> float:
        """Angle (degrees) with this atom as the vertex and ``atom1``/``atom2`` as legs."""
        v1 = atom1.coordinates - self.coordinates
        v2 = atom2.coordinates - self.coordinates

        length1, length2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if length1 == 0 or length2 == 0:
            raise ValueError("bond angle is undefined for coincident sites")
        v1_norm = v1 / length1
        v2_norm = v2 / length2

        dot_product = np.dot(v1_norm, v2_norm)
        # Guard against floating point drift slightly outside [-1, 1].
        dot_product = max(-1.0, min(1.0, dot_product))

        return math.degrees(math.acos(dot_product))


class Molecule:
    """A metal complex molecule read from an XYZ structure plus bond information."""

    # Covalent radii in Angstrom (Cordero et al., 2008).
    COVALENT_RADII = {
        'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.69, 'N': 0.71, 'O': 0.66,
        'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05,
        'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39,
        'Mn': 1.39, 'Fe': 1.52, 'Co': 1.50, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20,
        'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16, 'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75,
        'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
        'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40, 'Cs': 2.44, 'Ba': 2.15,
        'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96,
        'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75,
        'Ta': 1.70, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32,
        'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40, 'At': 1.50, 'Rn': 1.50
    }

    # All metal elements recognised by the element classifier.
    METAL_ELEMENTS = {
        'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
        'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'In', 'Sn', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
        'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
        'Bi', 'Po'
    }

    # The default behaviour keeps the Fe-only scope of the original study;
    # other metals can be enabled explicitly via ``target_metal_elements``.
    TARGET_METAL_ELEMENTS = {"Fe"}

    def __init__(
        self,
        bond_tolerance: float = 1.16,
        geometry_sigma: float = 15.0,
        target_metal_elements: Sequence[str] | None = None,
        method_profile: str = METHOD_PROFILE,
    ):
        self.method_profile = profile_metadata(method_profile)["method_profile"]
        if not math.isfinite(bond_tolerance) or bond_tolerance <= 0:
            raise ValueError("bond_tolerance must be positive")
        if not math.isfinite(geometry_sigma) or geometry_sigma <= 0:
            raise ValueError("geometry_sigma must be positive")
        self.atoms = []
        self.filename = ""
        self.bond_info: Dict[Tuple[int, int], int] = {}
        self.bond_tolerance = float(bond_tolerance)
        self.geometry_sigma = float(geometry_sigma)
        self.target_metal_elements = set(target_metal_elements or self.TARGET_METAL_ELEMENTS)

    def infer_bonds_from_distance(self, connectivity_tolerance: float = 1.20) -> None:
        """Infer ligand-internal bonds from covalent radii when no SDF is available.

        Only bonds between non-metal atoms are inferred; these are needed to
        identify connected pi-ligand donor groups. Metal-ligand contacts are
        always determined geometrically via ``get_coordination_atoms``.
        """
        if not math.isfinite(connectivity_tolerance) or connectivity_tolerance <= 0:
            raise ValueError("connectivity_tolerance must be positive")
        inferred: Dict[Tuple[int, int], int] = {}
        for left in range(self.num_atoms()):
            atom_left = self.atoms[left]
            if atom_left.element in self.target_metal_elements:
                continue
            for right in range(left + 1, self.num_atoms()):
                atom_right = self.atoms[right]
                if atom_right.element in self.target_metal_elements:
                    continue
                radius_left = self.COVALENT_RADII.get(atom_left.element, 1.0)
                radius_right = self.COVALENT_RADII.get(atom_right.element, 1.0)
                threshold = (radius_left + radius_right) * connectivity_tolerance
                if 0 < atom_left.distance_to(atom_right) <= threshold:
                    inferred[(left + 1, right + 1)] = 1
        self.bond_info = inferred

    def get_bonds_from_sdf(self, sdf_path) -> None:
        """Read the bond block of a V2000 MDL SDF file into ``bond_info``.

        Counts, element order, finite coordinates (XYZ agreement within 0.01
        Angstrom), bond indices/orders and termination are checked before use.
        V3000 and multi-record selection are not supported.
        """
        lines = Path(sdf_path).read_text(encoding="utf-8").splitlines()
        try:
            if len(lines) < 4 or "V2000" not in lines[3]:
                raise ValueError("expected a V2000 counts line")
            n_atoms, n_bonds = int(lines[3][:3]), int(lines[3][3:6])
            if n_atoms != self.num_atoms() or n_bonds < 0:
                raise ValueError("SDF/XYZ atom count mismatch or invalid bond count")
            end = 4 + n_atoms + n_bonds
            if len(lines) <= end:
                raise ValueError("truncated SDF atom/bond block")
            cursor = end
            while cursor < len(lines) and lines[cursor].startswith("M  "):
                if lines[cursor].startswith("M  END"):
                    break
                cursor += 1
            if cursor >= len(lines) or not lines[cursor].startswith("M  END"):
                raise ValueError("SDF bond count mismatch or missing M  END")
            if "$$$$" in lines[cursor + 1:]:
                delimiter = lines.index("$$$$", cursor + 1)
                if any(line.strip() for line in lines[delimiter + 1:]):
                    raise ValueError("multiple SDF records are not supported")
            for atom, line in zip(self.atoms, lines[4:4 + n_atoms]):
                coords = np.array([float(line[i:i + 10]) for i in (0, 10, 20)])
                if not np.all(np.isfinite(coords)):
                    raise ValueError("non-finite SDF coordinates")
                if line[31:34].strip() != atom.element:
                    raise ValueError("SDF/XYZ element order mismatch")
                if not np.allclose(coords, atom.coordinates, atol=0.01, rtol=0):
                    raise ValueError("SDF/XYZ coordinate mismatch (tolerance 0.01 Angstrom)")
            bonds = {}
            for line in lines[4 + n_atoms:end]:
                left, right, order = (int(line[i:i + 3]) for i in (0, 3, 6))
                if not (1 <= left <= n_atoms and 1 <= right <= n_atoms) or left == right:
                    raise ValueError("invalid SDF bond indices")
                if order not in (1, 2, 3, 4):
                    raise ValueError("unsupported SDF bond order")
                key = tuple(sorted((left, right)))
                if key in bonds:
                    raise ValueError("duplicate SDF bond")
                bonds[key] = order
            self.bond_info = bonds
        except (ValueError, IndexError) as exc:
            raise ValueError(f"invalid SDF {sdf_path}: {exc}") from exc

    def add_atom(self, atom: Atom) -> None:
        self.atoms.append(atom)

    def get_atom(self, index: int) -> Atom:
        return self.atoms[index]

    def atom_index(self, atom: Atom) -> int:
        return self.atoms.index(atom) + 1  # SDF bond block indices are 1-based.

    def num_atoms(self) -> int:
        return len(self.atoms)

    def calculate_distance(self, index1: int, index2: int) -> float:
        """Distance (Angstrom) between atoms at 0-based indices ``index1`` and ``index2``."""
        return self.atoms[index1].distance_to(self.atoms[index2])

    def calculate_angle(self, index1: int, index2: int, index3: int) -> float:
        """Bond angle (degrees) at 0-based indices, with ``index2`` as the vertex."""
        atom1 = self.atoms[index1]
        atom2 = self.atoms[index2]
        atom3 = self.atoms[index3]
        return atom2.bond_angle(atom1, atom3)

    def find_atoms_by_element(self, element: str, case_sensitive: bool = False) -> List[int]:
        """Return 0-based indices of all atoms matching ``element``."""
        indices = []

        if not case_sensitive:
            element = element.lower()

        for i, atom in enumerate(self.atoms):
            atom_element = atom.element if case_sensitive else atom.element.lower()
            if atom_element == element:
                indices.append(i)

        return indices

    def get_atoms_by_element(self, element: str, case_sensitive: bool = False) -> List[Atom]:
        """Return all atoms matching ``element``."""
        indices = self.find_atoms_by_element(element, case_sensitive)
        return [self.atoms[i] for i in indices]

    def get_element_count(self) -> Dict[str, int]:
        """Return a mapping from element symbol to atom count."""
        element_count: Dict[str, int] = {}
        for atom in self.atoms:
            element_count[atom.element] = element_count.get(atom.element, 0) + 1
        return element_count

    def get_center_by_element(self, element: str) -> Atom:
        """Return the single atom of element ``element``; raises if there is not exactly one."""
        atoms = self.get_atoms_by_element(element)
        if len(atoms) != 1:
            raise ValueError(f"Expected exactly one {element} atom, found {len(atoms)}")
        return atoms[0]

    def get_coordination_atoms(
        self,
        center: Atom,
        bond_tolerance: float | None = None,
        exclude_target_metals: bool = True,
    ) -> List[Atom]:
        """Return atoms within the coordination sphere of ``center``.

        A contact counts as coordination when the interatomic distance is at
        most ``bond_tolerance`` times the sum of the covalent radii
        (default 1.16). Atoms whose element is itself in
        ``target_metal_elements`` are skipped by default, so adjacent metal
        centers of a polynuclear core are never reported as donors; pass
        ``exclude_target_metals=False`` to keep the raw neighbour list.
        """
        coordination_atoms = []
        tolerance = self.bond_tolerance if bond_tolerance is None else bond_tolerance
        if tolerance <= 0:
            raise ValueError("bond_tolerance must be positive")

        for atom in self.atoms:
            if atom is center:
                continue
            if exclude_target_metals and atom.element in self.target_metal_elements:
                continue

            center_radius = self.COVALENT_RADII.get(center.element, 1.5)
            atom_radius = self.COVALENT_RADII.get(atom.element, 1.0)

            max_bond_length = (center_radius + atom_radius) * tolerance

            if atom.distance_to(center) <= max_bond_length:
                coordination_atoms.append(atom)

        return coordination_atoms

    def find_connected_components(self, ligand_atoms: List[Atom]) -> List[List[Atom]]:
        """Split ``ligand_atoms`` into bonded connected components (used to find pi-ligands)."""
        n = len(ligand_atoms)
        connections = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                atom_i = ligand_atoms[i]
                atom_j = ligand_atoms[j]
                global_i = self.atom_index(atom_i)
                global_j = self.atom_index(atom_j)
                if (global_i, global_j) in self.bond_info or (global_j, global_i) in self.bond_info:
                    connections[i].append(j)
                    connections[j].append(i)

        visited = [False] * n
        components: List[List[Atom]] = []

        def dfs(node: int, component: List[int]) -> None:
            visited[node] = True
            component.append(node)
            for neighbor in connections[node]:
                if not visited[neighbor]:
                    dfs(neighbor, component)

        for i in range(n):
            if not visited[i]:
                component: List[int] = []
                dfs(i, component)
                components.append([ligand_atoms[idx] for idx in component])

        return components

    def calculate_center_point(self, atoms: List[Atom]) -> np.ndarray:
        """Return the mean coordinate of ``atoms``."""
        if not atoms:
            return np.zeros(3)

        coords = np.array([atom.coordinates for atom in atoms])
        return np.mean(coords, axis=0)

    def find_nearest_point(self, center: Atom, points: List[Atom]) -> Atom:
        """Return the atom of ``points`` closest to ``center``."""
        if not points:
            raise ValueError("at least one point is required")
        return min(points, key=lambda p: (center.distance_to(p), tuple(p.coordinates)))

    def find_points_in_ring(self, center_point: Atom, ring_atoms: List[Atom]) -> Tuple[Atom, np.ndarray, np.ndarray]:
        """Pick pseudo-atom sites for an eta-5 ring ligand (e.g. cyclopentadienyl).

        Returns the ring atom closest to the metal plus the midpoints of its
        two neighbour pairs, yielding the three coordination sites contributed
        by an eta-5 ring.
        """
        if not self._five_cycle(ring_atoms) or center_point not in ring_atoms:
            raise ValueError("expected a bonded five-member cycle containing the anchor")
        neighbors = [a for a in ring_atoms if self._bond_order(center_point, a)]
        pairs = []
        for neighbor in neighbors:
            other = next(a for a in ring_atoms
                         if a is not center_point and self._bond_order(neighbor, a))
            pairs.append(self.calculate_center_point([neighbor, other]))
        return center_point, pairs[0], pairs[1]

    def _bond_order(self, left: Atom, right: Atom) -> int:
        i, j = self.atom_index(left), self.atom_index(right)
        return self.bond_info.get((i, j), self.bond_info.get((j, i), 0))

    def _five_cycle(self, atoms: List[Atom]) -> bool:
        """Require an induced, connected C5 cycle, not just five donors."""
        return (len(atoms) == 5 and all(a.element == "C" for a in atoms)
                and all(sum(bool(self._bond_order(a, b)) for b in atoms if b is not a) == 2
                        for a in atoms)
                and len(self.find_connected_components(atoms)) == 1)

    def _is_pi_group(self, atoms: List[Atom]) -> bool:
        """Conservative carbon-only eta-2 / Cp recognition, not electronic proof.

        Short C-C contacts (1.15--1.45 A) also accept coordinated olefins
        represented with a single bond in an SDF. Cp requires a C5 cycle,
        near planarity and either conjugated bond orders or short ring edges.
        O-O, carboxylates and arbitrary connected sigma donors stay classical.
        """
        if len(atoms) == 2 and all(a.element == "C" for a in atoms):
            order = self._bond_order(*atoms)
            return bool(order) and (order == 2 or 1.15 <= atoms[0].distance_to(atoms[1]) <= 1.45)
        if not self._five_cycle(atoms):
            return False
        edges = [(a, b) for i, a in enumerate(atoms) for b in atoms[i + 1:]
                 if self._bond_order(a, b)]
        orders = [self._bond_order(a, b) for a, b in edges]
        double_edges = [(a, b) for a, b in edges if self._bond_order(a, b) == 2]
        conjugated = (all(o == 4 for o in orders) or
                      (len(double_edges) == 2 and len(set(sum(([a, b] for a, b in double_edges), []))) == 4))
        short = all(1.15 <= a.distance_to(b) <= 1.48 for a, b in edges)
        coords = np.array([a.coordinates for a in atoms])
        singular = np.linalg.svd(coords - coords.mean(axis=0), compute_uv=False)
        planar = singular[1] > 0 and singular[2] / singular[1] <= 0.10
        return planar and (conjugated or short)

    def analyze_ligands(self, center: Atom, coordination_atoms: List[Atom]) -> Dict[str, Any]:
        """Classify coordination atoms into classical ligands and pi-ligands.

        Returns a dict with:

        * ``pi_ligands``: connected donor components of size two or more.
          This is a syntax convention, not proof of electronic pi bonding.
        * ``normal_ligands``: singleton donor components.
        * ``pi_coordination_sites`` / ``normal_coordination_sites``: the
          corresponding (possibly virtual) coordination site coordinates.
        * ``real_coordination_number``: number of coordination sites after
          treating a bonded C5 cycle as three sites and other groups as one.
        """
        result: Dict[str, Any] = {
            "total_ligand_atoms": len(coordination_atoms),
            "pi_ligands": [],
            "pi_coordination_sites": [],
            "normal_ligands": [],
            "normal_coordination_sites": []
        }

        # Step 1: connected components of bonded donor atoms.
        connected_components = self.find_connected_components(coordination_atoms)

        # Manuscript syntax groups connected donor components, not electronic states.
        pi_ligands = [component for component in connected_components if len(component) >= 2]
        result["pi_ligands"] = pi_ligands

        # Step 2: remaining donors are classical ligands.
        pi_atoms_set = set()
        for component in pi_ligands:
            pi_atoms_set.update(component)

        normal_ligands = [atom for atom in coordination_atoms if atom not in pi_atoms_set]
        result["normal_ligands"] = normal_ligands

        # Step 3: coordination sites.
        # 3.1 Classical ligands contribute the donor atom itself.
        for ligand in normal_ligands:
            result["normal_coordination_sites"].append(ligand.coordinates)

        # 3.2 Pi-ligands contribute virtual sites.
        for component in pi_ligands:
            if len(component) == 2:
                # Eta-2 (olefin) coordination: the C=C midpoint is the site.
                midpoint = self.calculate_center_point(component)
                result["pi_coordination_sites"].append(midpoint)
            elif self._five_cycle(component):
                # Eta-5 ring: the closest atom plus two neighbour-pair midpoints.
                nearest = self.find_nearest_point(center, component)
                point_a, point_b, point_c = self.find_points_in_ring(nearest, component)
                result["pi_coordination_sites"].append(point_a.coordinates)
                result["pi_coordination_sites"].append(point_b)
                result["pi_coordination_sites"].append(point_c)
            else:
                # Any other pi-ligand: the group centroid is the site.
                midpoint = self.calculate_center_point(component)
                result["pi_coordination_sites"].append(midpoint)

        # Step 4: real coordination number.
        real_coordination_number = len(normal_ligands)
        for component in pi_ligands:
            if self._five_cycle(component):  # Cp topology contributes three sites
                real_coordination_number += 3
            else:  # other pi-ligands contribute one site
                real_coordination_number += 1

        result["real_coordination_number"] = real_coordination_number

        return result

    def analyze_coordination_geometry(self, center: Atom, ligands: List[Atom]) -> Dict:
        """Match measured ligand angles against idealized geometry profiles.

        The similarity of each candidate geometry is the mean over matched
        angle pairs of a Gaussian score ``exp(-0.5 * (d / sigma)**2)`` where
        ``d`` is the angle deviation and ``sigma`` is ``geometry_sigma``.
        The best-scoring geometry is reported along with all candidates.
        """
        result = {
            "coordination_number": len(ligands),
            "bond_lengths": [],
            "bond_angles": [],
            "geometry_type": "unknown",
            "geometry_similarity": 0.0,
            "possible_geometries": []
        }

        for ligand in ligands:
            bond_length = center.distance_to(ligand)
            result["bond_lengths"].append((ligand.element, bond_length))

        angles = []
        if len(ligands) >= 2:
            for i in range(len(ligands)):
                for j in range(i + 1, len(ligands)):
                    angle = center.bond_angle(ligands[i], ligands[j])
                    angles.append(angle)
                    result["bond_angles"].append((
                        ligands[i].element,
                        center.element,
                        ligands[j].element,
                        angle
                    ))

        # Idealized angle profiles: coordination number -> {geometry: [angles]}.
        geometry_profiles = {
            2: {
                "Linear": [180]
            },
            3: {
                "TrigonalPlanar": [120, 120, 120]
            },
            4: {
                "Tetrahedral": [109.5] * 6,
                "SquarePlanar": [90, 90, 90, 90, 180, 180]
            },
            5: {
                "TrigonalBipyramid": [90] * 6 + [120] * 3 + [180],
                "SquarePyramid": [90] * 8 + [180] * 2
            },
            6: {
                "Octahedral": [90] * 12 + [180] * 3
            }
        }

        cn = len(ligands)

        if cn not in geometry_profiles or not angles:
            return result

        possible_geometries = []
        for geom_type, std_angles in geometry_profiles[cn].items():
            matched_angles = self._match_angles(sorted(angles), sorted(std_angles))
            similarity = self._calculate_angle_similarity(matched_angles)
            possible_geometries.append((geom_type, similarity))

        possible_geometries.sort(key=lambda x: x[1], reverse=True)
        result["possible_geometries"] = possible_geometries

        best_geom, best_similarity = possible_geometries[0]

        result["geometry_type"] = best_geom
        result["geometry_similarity"] = best_similarity

        return result

    def _match_angles(self, angles: List[float], std_angles: List[float]) -> List[Tuple[float, float]]:
        """Monotone quantile matching of equal-length angle distributions.

        Sorting minimizes total squared angular deviation (rearrangement
        inequality). The subsequent mean Gaussian score is descriptive; this
        pairing does not claim to maximize that non-convex score, nor recover
        a vertex-consistent three-dimensional ligand assignment.
        """
        if len(angles) != len(std_angles):
            raise ValueError("angle profiles must have equal lengths")
        paired = []
        min_len = min(len(angles), len(std_angles))
        for i in range(min_len):
            paired.append((angles[i], std_angles[i]))
        return paired

    def _calculate_angle_similarity(self, angle_pairs: List[Tuple[float, float]]) -> float:
        """Mean Gaussian similarity over matched angle pairs (1.0 = perfect match)."""
        if not angle_pairs:
            return 0.0

        total_score = 0.0
        for actual, std in angle_pairs:
            diff = abs(actual - std)
            score = math.exp(-0.5 * (diff / self.geometry_sigma) ** 2)
            total_score += score

        return total_score / len(angle_pairs)

    def find_metal_centers(self) -> List[Atom]:
        """Return all atoms whose element is in ``target_metal_elements``."""
        metal_centers = []

        for atom in self.atoms:
            if atom.element in self.target_metal_elements:
                metal_centers.append(atom)

        return metal_centers

    def generate_coordination_report(self) -> Dict[str, Any]:
        """Produce a structured per-metal-center coordination analysis report."""
        result: Dict[str, Any] = {
            **profile_metadata(self.method_profile),
            "has_metal": False,
            "metal_elements": [],
            "coordination_numbers": [],
            "real_coordination_numbers": [],
            "geometry_types": [],
            "geometry_similarities": [],
            "ligand_compositions": [],
            "bond_angles": [],
            "normal_coordination_sites": [],
            "pi_coordination_sites": [],
            "pi_ligands": [],
            "normal_ligands": []
        }

        metal_centers = self.find_metal_centers()
        if not metal_centers:
            return result

        result["has_metal"] = True
        result["metal_elements"] = [center.element for center in metal_centers]

        for center in metal_centers:
            coordination_atoms = self.get_coordination_atoms(center)
            if not coordination_atoms:
                result["coordination_numbers"].append(0)
                result["real_coordination_numbers"].append(0)
                result["geometry_types"].append("unknown")
                result["geometry_similarities"].append(0.0)
                result["ligand_compositions"].append("none")
                result["bond_angles"].append([])
                result["pi_ligands"].append([])
                result["normal_ligands"].append([])
                result["normal_coordination_sites"].append([])
                result["pi_coordination_sites"].append([])
                continue

            ligand_analysis = self.analyze_ligands(center, coordination_atoms)

            result["coordination_numbers"].append(len(coordination_atoms))
            result["real_coordination_numbers"].append(ligand_analysis["real_coordination_number"])
            result['normal_coordination_sites'].append(ligand_analysis["normal_coordination_sites"])
            result['pi_coordination_sites'].append(ligand_analysis["pi_coordination_sites"])

            ligand_elements = sorted({atom.element for atom in coordination_atoms})
            result["ligand_compositions"].append(",".join(ligand_elements))

            pi_ligands_info = []
            for pi_ligand in ligand_analysis["pi_ligands"]:
                atoms = [atom.element for atom in pi_ligand]
                pi_ligands_info.append({
                    "type": "Cp" if self._five_cycle(pi_ligand) else
                    "Olefin" if len(pi_ligand) == 2 else
                    f"PI-{len(pi_ligand)}",
                    "atoms": atoms,
                    "string": "".join(atoms),
                    "size": len(pi_ligand)
                })
            result["pi_ligands"].append(pi_ligands_info)

            normal_ligands_info = [{
                "element": atom.element,
                "distance": center.distance_to(atom)
            } for atom in ligand_analysis["normal_ligands"]]
            result["normal_ligands"].append(normal_ligands_info)

            # Geometry analysis uses virtual pseudo-atoms for pi-ligand sites.
            all_ligands = []
            all_ligands.extend(ligand_analysis["normal_ligands"])
            for coord_site in ligand_analysis["pi_coordination_sites"]:
                virtual_atom = Atom("[V]", coord_site[0], coord_site[1], coord_site[2])
                all_ligands.append(virtual_atom)

            analysis = self.analyze_coordination_geometry(center, all_ligands)
            result["geometry_types"].append(analysis["geometry_type"])
            result["geometry_similarities"].append(analysis["geometry_similarity"])

            angle_details = []
            for angle_data in analysis["bond_angles"]:
                angle_details.append({
                    "atom1": angle_data[0],
                    "atom2": angle_data[1],  # center atom
                    "atom3": angle_data[2],
                    "angle": round(angle_data[3], 2)
                })
            result["bond_angles"].append(angle_details)

        return result


def read_xyz_file(
    filename: str | Path,
    bond_tolerance: float = 1.16,
    geometry_sigma: float = 15.0,
    target_metal_elements: Sequence[str] | None = None,
    method_profile: str = METHOD_PROFILE,
) -> Molecule:
    """Read an XYZ file and return a :class:`Molecule`."""
    molecule = Molecule(
        bond_tolerance=bond_tolerance,
        geometry_sigma=geometry_sigma,
        target_metal_elements=target_metal_elements,
        method_profile=method_profile,
    )
    molecule.filename = os.path.basename(filename)

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

            if not lines:
                raise ValueError(f"file {filename} is empty")

            num_atoms = int(lines[0].strip())

            if num_atoms < 1 or len(lines) < num_atoms + 2:
                raise ValueError("invalid atom count or truncated XYZ file")
            if any(line.strip() for line in lines[num_atoms + 2:]):
                raise ValueError("extra XYZ records after declared atom count")
            # Exactly one frame, with one four-field record per atom.
            for i in range(2, 2 + num_atoms):
                parts = lines[i].split()
                if len(parts) != 4 or parts[0] not in Molecule.COVALENT_RADII:
                    raise ValueError(f"malformed XYZ atom line {i + 1}")
                molecule.add_atom(Atom(parts[0], *map(float, parts[1:])))

        return molecule

    except Exception as e:
        raise ValueError(f"error reading file {filename}: {e}") from e


def load_structure(
    xyz_path: str | Path,
    sdf_path: str | Path | None = None,
    bond_tolerance: float = 1.16,
    geometry_sigma: float = 15.0,
    target_metal_elements: Sequence[str] | None = None,
    connectivity_tolerance: float = 1.20,
    method_profile: str = METHOD_PROFILE,
) -> Molecule:
    """Read an XYZ file and attach connectivity from an SDF or from distance inference.

    When ``sdf_path`` is ``None``, ligand-internal bonds are inferred from
    covalent radii using ``connectivity_tolerance``.
    """
    molecule = read_xyz_file(
        xyz_path,
        bond_tolerance=bond_tolerance,
        geometry_sigma=geometry_sigma,
        target_metal_elements=target_metal_elements,
        method_profile=method_profile,
    )
    if sdf_path is None:
        molecule.infer_bonds_from_distance(connectivity_tolerance)
    else:
        molecule.get_bonds_from_sdf(str(sdf_path))
    return molecule


def perturb_and_reorder(
    molecule: Molecule,
    coordinate_noise_std: float = 0.0,
    atom_order: Sequence[int] | None = None,
    seed: int = 0,
) -> Molecule:
    """Return a copy with perturbed coordinates and/or reordered atoms.

    SDF bond indices are remapped so connectivity stays consistent. This is
    used for sensitivity analyses (atom-ordering invariance, coordinate noise).
    """
    if coordinate_noise_std < 0:
        raise ValueError("coordinate_noise_std must be non-negative")
    count = molecule.num_atoms()
    order = list(range(count)) if atom_order is None else [int(i) for i in atom_order]
    if sorted(order) != list(range(count)):
        raise ValueError("atom_order must be a zero-based permutation of all atoms")

    rng = np.random.default_rng(seed)
    transformed = Molecule(
        bond_tolerance=molecule.bond_tolerance,
        geometry_sigma=molecule.geometry_sigma,
        target_metal_elements=sorted(molecule.target_metal_elements),
        method_profile=molecule.method_profile,
    )
    transformed.filename = molecule.filename
    for old_index in order:
        atom = molecule.atoms[old_index]
        coordinates = atom.coordinates.copy()
        if coordinate_noise_std:
            coordinates += rng.normal(0.0, coordinate_noise_std, size=3)
        transformed.add_atom(Atom(atom.element, *coordinates))

    old_to_new = {old + 1: new + 1 for new, old in enumerate(order)}
    transformed.bond_info = {
        (old_to_new[left], old_to_new[right]): bond_type
        for (left, right), bond_type in molecule.bond_info.items()
    }
    return transformed


def records_from_report(
    report: Dict[str, Any], gdb_idx: int | str | None = None
) -> List[Dict[str, Any]]:
    """Convert a coordination report into one MetSMILES record per metal center."""
    rows: List[Dict[str, Any]] = []
    if not report["has_metal"]:
        return rows
    for index, metal_center in enumerate(report["metal_elements"]):
        angle_details = "\n".join(
            f"{angle['atom1']}-{angle['atom2']}-{angle['atom3']}:{angle['angle']}"
            for angle in report["bond_angles"][index]
        )
        normal_elements = sorted(
            ligand["element"] for ligand in report["normal_ligands"][index]
        )
        normal_string = "&" + "&".join(normal_elements) if normal_elements else ""
        pi_components = sorted(
            "".join(sorted(ligand["atoms"]))
            for ligand in report["pi_ligands"][index]
        )
        pi_string = "%" + "%".join(pi_components) if pi_components else ""
        coordination = f"{metal_center}{normal_string}{pi_string}"
        geometry = report["geometry_types"][index]
        rows.append(
            {
                **profile_metadata(report.get("method_profile", METHOD_PROFILE)),
                "gdb_idx": gdb_idx,
                "Metal Center": metal_center,
                "Real Coordination Number": report["real_coordination_numbers"][index],
                "Geometry Type": geometry,
                "Confidence": float(report["geometry_similarities"][index]),
                "Angle Details": angle_details,
                "Suffix 2": f"[{coordination}]",
                "Suffix 3": f"[{{{geometry}}}{coordination}]",
            }
        )
    return rows


def generate_metsmiles_record(
    xyz_path: str | Path,
    sdf_path: str | Path | None,
    base_smiles: str | None = None,
    bond_tolerance: float = 1.16,
    geometry_sigma: float = 15.0,
    coordinate_noise_std: float = 0.0,
    atom_order: Sequence[int] | None = None,
    seed: int = 0,
    target_metal_elements: Sequence[str] | None = None,
    connectivity_tolerance: float = 1.20,
    method_profile: str = METHOD_PROFILE,
    prefix_mode: str = "auto",
) -> dict[str, Any]:
    """Generate one auditable MetSMILES record for a single-metal-center complex.

    Parameters
    ----------
    xyz_path:
        Path to the XYZ structure.
    sdf_path:
        Path to the matching SDF file, or ``None`` to infer ligand-internal
        connectivity from distances (``connectivity_tolerance``).
    base_smiles:
        An explicit structure prefix overriding automatic SDF extraction. ``SMILES_2`` /
        ``SMILES_3`` are ``base_smiles`` + ``"."`` + suffix.
    bond_tolerance:
        Metal-donor contact threshold as a factor of the covalent-radii sum
        (default 1.16).
    geometry_sigma:
        Width (degrees) of the Gaussian geometry-matching kernel (default 15).
    coordinate_noise_std, atom_order, seed:
        Optional coordinate perturbation / atom reordering for sensitivity
        analyses; leave defaults for the canonical record.
    target_metal_elements:
        Elements treated as metal centers (default ``{"Fe"}``); pass e.g.
        ``["Co"]`` or ``["Ni"]`` for other metals.
    connectivity_tolerance:
        Covalent-radii factor for distance-based bond inference when no SDF
        is given (default 1.20).

    Returns
    -------
    dict
        Record with metal center, real coordination number, geometry type and
        confidence, angle details, ``Suffix 2``/``Suffix 3``,
        ``SMILES_2``/``SMILES_3`` and all generation parameters for audit.

    Raises
    ------
    ValueError
        If the structure contains no target metal center or more than one.
    """
    molecule = load_structure(
        xyz_path,
        sdf_path,
        bond_tolerance=bond_tolerance,
        geometry_sigma=geometry_sigma,
        target_metal_elements=target_metal_elements,
        method_profile=method_profile,
        connectivity_tolerance=connectivity_tolerance,
    )
    molecule = perturb_and_reorder(
        molecule,
        coordinate_noise_std=coordinate_noise_std,
        atom_order=atom_order,
        seed=seed,
    )
    rows = records_from_report(molecule.generate_coordination_report())
    if len(rows) != 1:
        expected = sorted(target_metal_elements or Molecule.TARGET_METAL_ELEMENTS)
        raise ValueError(
            f"Expected one target metal center {expected}, found {len(rows)} in {xyz_path}. "
            "For multi-metal structures use metsmiles.generate_global_metsmiles instead."
        )
    record = rows[0]
    organic, metadata = resolve_molecular_prefix(
        sdf_path, molecule.target_metal_elements, base_smiles, prefix_mode)
    record.update(metadata)
    record["SMILES_1"] = organic
    prefix = f"{organic}." if organic else ""
    record["SMILES_2"] = prefix + record["Suffix 2"]
    record["SMILES_3"] = prefix + record["Suffix 3"]
    record["bond_tolerance"] = float(bond_tolerance)
    record["geometry_sigma"] = float(geometry_sigma)
    record["coordinate_noise_std"] = float(coordinate_noise_std)
    record["connectivity_source"] = "distance_inference" if sdf_path is None else "sdf"
    record["connectivity_tolerance"] = float(connectivity_tolerance)
    return record


REPORT_COLUMNS = [
    "gdb_idx", "Metal Center", "Real Coordination Number", "Geometry Type",
    "Confidence", "Angle Details", "Suffix 2", "Suffix 3", "SMILES_1",
    "SMILES_2", "SMILES_3", "method_profile", "serialization_profile",
    "prefix_mode", "prefix_source", "prefix_status", "rdkit_version",
    "fragment_policy", "charge_policy", "connectivity_source",
    "bond_tolerance", "geometry_sigma", "connectivity_tolerance",
    "status", "error_type", "error",
]


def recognize_directory(
    directory_path: str | Path,
    output_csv: str | Path = "coordination_analysis.csv",
    base_smiles_by_index: Dict[str, str] | None = None,
    bond_tolerance: float = 1.16,
    geometry_sigma: float = 15.0,
    target_metal_elements: Sequence[str] | None = None,
    connectivity_tolerance: float = 1.20,
    method_profile: str = METHOD_PROFILE,
    prefix_mode: str = "auto",
) -> List[Dict[str, Any]]:
    """Write one row per center; failures are explicit typed rows, never fallbacks.

    A directory may mix single- and multi-center inputs. CLI exits nonzero when
    any row fails. Explicit bases are indexed by file stem, including empty bases.
    """
    from .multimetal import generate_metsmiles_records
    directory = Path(directory_path)
    if not directory.is_dir():
        raise FileNotFoundError(f"directory does not exist: {directory}")
    profile_metadata(method_profile)
    if prefix_mode not in ("auto", "suffix-only"):
        raise ValueError("prefix_mode must be 'auto' or 'suffix-only'")
    # Validate global numerical parameters even for an empty directory.
    Molecule(bond_tolerance, geometry_sigma, target_metal_elements, method_profile)
    if not math.isfinite(connectivity_tolerance) or connectivity_tolerance <= 0:
        raise ValueError("connectivity_tolerance must be positive")
    paths = list(directory.glob("*.xyz"))
    paths.sort(key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem))
    bases = base_smiles_by_index or {}
    rows = []
    for xyz in paths:
        sdf = xyz.with_suffix(".sdf")
        sdf = sdf if sdf.is_file() else None
        common = dict(gdb_idx=xyz.stem, **profile_metadata(method_profile),
                      prefix_mode=prefix_mode,
                      connectivity_source="sdf" if sdf else "distance_inference",
                      bond_tolerance=float(bond_tolerance), geometry_sigma=float(geometry_sigma),
                      connectivity_tolerance=float(connectivity_tolerance))
        try:
            records = generate_metsmiles_records(
                xyz, sdf, base_smiles=bases.get(xyz.stem),
                bond_tolerance=bond_tolerance, geometry_sigma=geometry_sigma,
                target_metal_elements=target_metal_elements,
                connectivity_tolerance=connectivity_tolerance,
                method_profile=method_profile, prefix_mode=prefix_mode)
            if not records:
                raise ValueError("no target metal centers")
            for record in records:
                row = dict(common, **{k: v for k, v in record.items() if k in REPORT_COLUMNS})
                row.update(status="ok", error_type="", error="")
                rows.append(row)
        except (ValueError, OSError) as exc:
            rows.append(dict(common, status="error", error_type=type(exc).__name__, error=str(exc)))
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows
