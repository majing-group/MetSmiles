"""CIF 1.x subset parsing and translation-labelled periodic coordination graphs.

Symmetry operations expand the ordered fractional sites without merging
species. A complete metric sphere search retains all bonded images in skew
cells, not just component-wrapped or fixed-box nearest images. Local donor
spheres use these same images. Network dimensionality is the maximum rank
of cycle translations over all distance-bonded components, not a count of
crossed cell faces. The suffix is a local descriptor, not a complete net.
"""

from __future__ import annotations

from .profiles import METHOD_PROFILE, profile_metadata

import ast
import math
import re
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .generator import Atom, Molecule
from .multimetal import (
    _collapse,
    center_record,
    cluster_header,
    join_cluster_blocks,
)

__all__ = [
    "read_cif",
    "expand_unit_cell",
    "local_molecule_for_metal",
    "detect_space_group",
    "periodicity_dimensionality",
    "periodic_block",
    "ligand_fragments_from_cif",
    "fragment_to_smiles",
    "generate_global_metsmiles_from_cif",
]

_FLOAT_RE = re.compile(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(?:\(\d+\))?$")


def _cif_tokens(text: str) -> List[Tuple[str, bool]]:
    """Tokenize CIF 1.x quotes, comments and semicolon text fields.

    The flag distinguishes quoted values from control words. Loop rows may
    span physical lines; whitespace inside quoted symmetry operations stays.
    """
    tokens = []
    lines = iter(text.splitlines())
    for line in lines:
        if line.startswith(";"):
            content = [line[1:]]
            for continuation in lines:
                if continuation.startswith(";"):
                    break
                content.append(continuation)
            else:
                raise ValueError("unterminated CIF text field")
            tokens.append(("\n".join(content), True))
            continue
        i = 0
        while i < len(line):
            if line[i].isspace():
                i += 1
                continue
            if line[i] == "#":
                break
            if line[i] in "'\"":
                quote, start = line[i], i + 1
                i = start
                while i < len(line):
                    if line[i] == quote and (i + 1 == len(line) or line[i + 1].isspace()):
                        break
                    i += 1
                if i == len(line):
                    raise ValueError("unterminated CIF quoted value")
                tokens.append((line[start:i], True))
                i += 1
            else:
                start = i
                while i < len(line) and not line[i].isspace():
                    i += 1
                tokens.append((line[start:i], False))
    return tokens


def _control(token: Tuple[str, bool]) -> bool:
    value, quoted = token
    low = value.lower()
    return not quoted and (low.startswith(("_", "data_", "save_"))
                           or low in ("loop_", "stop_", "global_"))


def _to_float(token: str) -> float:
    """Parse a CIF number, dropping the trailing ``(esd)``."""
    match = _FLOAT_RE.fullmatch(token)
    if not match:
        raise ValueError(f"not a number: {token!r}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"non-finite CIF number: {token!r}")
    return value


def read_cif(path: str | Path) -> Dict[str, Any]:
    """Parse cell parameters, the atom loop and symmetry operations.

    Returns ``{"cell": {a,b,c,alpha,beta,gamma}, "atoms": [(label, element,
    fx, fy, fz)], "symops": [str], "space_group": str|None,
    "space_group_number": int|None}``.
    Fractional coordinate and symmetry loop columns may occur in any order;
    quoted values, numeric symmetry IDs and rows spanning lines are supported.
    One ordered data block is required. Partial occupancies are rejected.
    The symmetry list falls back to identity only when absent.
    """
    cell: Dict[str, float] = {}
    atoms: List[Tuple[str, str, float, float, float]] = []
    symops: List[str] = []
    space_group: str | None = None
    space_group_number: int | None = None

    cell_keys = {
        "_cell_length_a": "a",
        "_cell_length_b": "b",
        "_cell_length_c": "c",
        "_cell_angle_alpha": "alpha",
        "_cell_angle_beta": "beta",
        "_cell_angle_gamma": "gamma",
    }

    tokens = _cif_tokens(Path(path).read_text(encoding="utf-8"))
    scalars = {}
    loops = []
    i = 0
    blocks = 0
    while i < len(tokens):
        value, quoted = tokens[i]
        low = value.lower()
        if not quoted and low.startswith("data_"):
            blocks += 1
            if blocks > 1:
                raise ValueError("multiple CIF data blocks are not supported; select one block")
        elif not quoted and low == "loop_":
            i += 1
            columns = []
            while i < len(tokens) and not tokens[i][1] and tokens[i][0].startswith("_"):
                columns.append(tokens[i][0].lower())
                i += 1
            if not columns or len(set(columns)) != len(columns):
                raise ValueError("empty or duplicate CIF loop columns")
            values = []
            while i < len(tokens) and not _control(tokens[i]):
                values.append(tokens[i][0])
                i += 1
            if len(values) % len(columns):
                raise ValueError("incomplete CIF loop row")
            loops.append((columns, [values[j:j + len(columns)]
                                    for j in range(0, len(values), len(columns))]))
            continue
        elif not quoted and low.startswith("_"):
            if i + 1 >= len(tokens) or _control(tokens[i + 1]):
                raise ValueError(f"missing CIF value for {value}")
            scalars[low] = tokens[i + 1][0]
            i += 1
        i += 1
    for key, short in cell_keys.items():
        if key in scalars:
            cell[short] = _to_float(scalars[key])
    for key in ("_space_group_name_h-m_alt", "_symmetry_space_group_name_h-m"):
        if key in scalars and scalars[key] not in ("?", "."):
            space_group = scalars[key]
            break
    for key in ("_space_group_it_number", "_symmetry_int_tables_number"):
        if key in scalars and scalars[key] not in ("?", "."):
            number = _to_float(scalars[key])
            if not number.is_integer() or not 1 <= number <= 230:
                raise ValueError("invalid space group number")
            space_group_number = int(number)
            break
    symop_keys = ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz")
    for key in symop_keys:
        if key in scalars:
            symops.append(scalars[key])
    required = ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")
    for columns, rows in loops:
        for key in symop_keys:
            if key in columns:
                symops.extend(row[columns.index(key)] for row in rows)
        if any(key in columns for key in required):
            if not all(key in columns for key in required):
                raise ValueError("incomplete fractional coordinate columns")
            for row in rows:
                data = dict(zip(columns, row))
                label = data.get("_atom_site_label", data.get("_atom_site_type_symbol", ""))
                raw_element = data.get("_atom_site_type_symbol", label)
                match = re.match(r"[A-Za-z]+", raw_element)
                element = match.group().capitalize() if match else ""
                if element not in Molecule.COVALENT_RADII:
                    raise ValueError(f"invalid atom species {raw_element!r}")
                if "_atom_site_occupancy" in data and _to_float(data["_atom_site_occupancy"]) != 1.0:
                    raise ValueError("partial/zero occupancy requires an ordered CIF model")
                atoms.append((label, element, *(_to_float(data[key]) for key in required)))

    missing = {"a", "b", "c", "alpha", "beta", "gamma"} - set(cell)
    if missing:
        raise ValueError(f"{path}: missing cell parameters {sorted(missing)}")
    _orthonormalisation(cell)  # validate the metric before using coordinates
    if not atoms:
        raise ValueError(f"{path}: no atoms parsed from the _atom_site loop")
    if not symops:
        symops = ["x,y,z"]
    if space_group is not None:
        space_group = space_group.replace(" ", "")
    return {
        "cell": cell,
        "atoms": atoms,
        "symops": symops,
        "space_group": space_group,
        "space_group_number": space_group_number,
    }


def _fraction_to_cartesian(
    fx: float, fy: float, fz: float, cell: Dict[str, float]
) -> Tuple[float, float, float]:
    return tuple(float(v) for v in _orthonormalisation(cell) @ np.array([fx, fy, fz]))


def _apply_symop(fx: float, fy: float, fz: float, op: str) -> Tuple[float, float, float]:
    def affine(node):
        if isinstance(node, ast.Name) and node.id in ("x", "y", "z"):
            result = np.zeros(4)
            result[("x", "y", "z").index(node.id)] = 1
            return result
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return np.array([0., 0., 0., float(node.value)])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return affine(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp):
            left, right = affine(node.left), affine(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                if not np.any(left[:3]):
                    return left[3] * right
                if not np.any(right[:3]):
                    return right[3] * left
            if isinstance(node.op, ast.Div) and not np.any(right[:3]) and right[3] != 0:
                return left / right[3]
        raise ValueError(f"unsupported non-affine symmetry operation: {op!r}")

    try:
        terms = op.split(",")
        if len(terms) != 3:
            raise ValueError("symmetry operation needs three coordinates")
        transform = np.array([affine(ast.parse(t.strip(), mode="eval").body) for t in terms])
        if not np.all(np.isfinite(transform)) or not np.isclose(abs(np.linalg.det(transform[:, :3])), 1):
            raise ValueError("invalid symmetry rotation")
        out = transform @ np.array([fx, fy, fz, 1.])
        return tuple(float(x) for x in out)
    except (SyntaxError, TypeError, OverflowError) as exc:
        raise ValueError(f"invalid symmetry operation {op!r}") from exc


def expand_unit_cell(
    atoms: Sequence[Tuple[str, str, float, float, float]],
    symops: Sequence[str],
) -> List[Tuple[str, str, float, float, float]]:
    """Apply every symmetry operation and deduplicate equivalent positions."""
    sites: List[Tuple[str, str, float, float, float]] = []
    seen = set()
    for label, element, fx, fy, fz in atoms:
        for op in symops:
            nx, ny, nz = _apply_symop(fx, fy, fz, op)
            position = tuple(v % 1.0 for v in (nx, ny, nz))
            key = (element, *(round(v, 8) % 1.0 for v in position))
            if key in seen:
                continue
            seen.add(key)
            sites.append((label, element, *position))
    return sites


def _images_within(delta, matrix, radius):
    """Enumerate ALL integer t with |A(delta+t)| <= radius by QR sphere search.

    Recursive bounds follow the triangular metric, not a fixed image box.
    This is complete for any numerically nonsingular skew cell, including
    images whose translation components exceed one. Returns (t, distance).
    """
    delta = np.asarray(delta, dtype=float)
    if not np.all(np.isfinite(delta)) or not math.isfinite(radius) or radius < 0:
        raise ValueError("finite coordinates and nonnegative finite radius required")
    upper = np.linalg.qr(matrix)[1]
    t = [0, 0, 0]
    out = []
    slack = 1e-12 * max(1., radius * radius)

    def visit(k, used):
        if k < 0:
            distance = float(np.linalg.norm(matrix @ (delta + t)))
            if distance <= radius + 1e-10 * max(1., radius):
                out.append((tuple(t), distance))
            return
        remaining = radius * radius + slack - used
        if remaining < 0:
            return
        tail = sum(upper[k, j] * (delta[j] + t[j]) for j in range(k + 1, 3))
        middle = -delta[k] - tail / upper[k, k]
        extent = math.sqrt(max(0., remaining)) / abs(upper[k, k])
        for value in range(math.ceil(middle - extent), math.floor(middle + extent) + 1):
            t[k] = value
            residual = upper[k, k] * (delta[k] + value) + tail
            visit(k - 1, used + residual * residual)

    visit(2, 0.)
    return sorted(out, key=lambda item: (item[1], item[0]))


def _nearest_image(delta, matrix):
    delta = np.asarray(delta, dtype=float)
    trial = -np.rint(delta).astype(int)
    radius = float(np.linalg.norm(matrix @ (delta + trial)))
    return _images_within(delta, matrix, radius)[0]


def _metal_donor_sites(frac_sites, cell, metals, targets, bond_tolerance):
    """Donor site -> [(metal site, distance)] for every bonded metal image.

    A metal index can occur more than once: these are distinct periodic
    metal images, and each contributes to the donor's sharing multiplicity.
    """
    matrix = _orthonormalisation(cell)
    site_metals = {}
    for site_index, (_, element, *frac) in enumerate(frac_sites):
        if element in targets:
            continue
        hits = []
        for m_index in metals:
            threshold = (Molecule.COVALENT_RADII[element] +
                         Molecule.COVALENT_RADII[frac_sites[m_index][1]]) * bond_tolerance
            delta = np.array(frac) - frac_sites[m_index][2:]
            hits.extend((m_index, distance) for _, distance in _images_within(delta, matrix, threshold)
                        if distance > 1e-10)
        if hits:
            site_metals[site_index] = sorted(hits, key=lambda h: (h[1], h[0]))
    return site_metals


def _image_position(
    donor_frac: Tuple[float, float, float],
    metal_cart: Tuple[float, float, float],
    metal_frac: Tuple[float, float, float],
    cell: Dict[str, float],
) -> Tuple[float, float, float]:
    """Cartesian position of the donor image closest to ``metal``."""
    matrix = _orthonormalisation(cell)
    delta = np.array(donor_frac) - metal_frac
    offset, _ = _nearest_image(delta, matrix)
    cart = np.array(metal_cart) + matrix @ (delta + offset)
    return tuple(float(v) for v in cart)


def local_molecule_for_metal(
    frac_sites: List[Tuple[str, str, float, float, float]],
    cart_sites: List[Tuple[str, str, float, float, float]],
    cell: Dict[str, float],
    metal_index: int,
    donor_site_indices: Sequence[int],
    bond_tolerance: float = 1.16,
    geometry_sigma: float = 15.0,
    target_metal_elements: Sequence[str] | None = None,
    connectivity_tolerance: float = 1.20,
) -> Molecule:
    """Non-periodic :class:`Molecule` with one metal and its donor sphere.

    Every image of each requested donor site within the metal contact
    cutoff is included, even multiple images of one unit-cell site. Atom
    attributes ``site_index`` and ``image_offset`` keep their identities.
    Internal bonds use the same Cartesian radius cutoff as the graph.
    """
    molecule = Molecule(
        bond_tolerance=bond_tolerance,
        geometry_sigma=geometry_sigma,
        target_metal_elements=target_metal_elements,
    )
    _, m_element, mx, my, mz = cart_sites[metal_index]
    molecule.add_atom(Atom(m_element, mx, my, mz))
    molecule.atoms[-1].site_index = metal_index  # type: ignore[attr-defined]
    matrix = _orthonormalisation(cell)
    metal_frac = np.array(frac_sites[metal_index][2:])
    for site_index in sorted(set(donor_site_indices)):
        _, element, fx, fy, fz = frac_sites[site_index]
        threshold = (Molecule.COVALENT_RADII[m_element] +
                     Molecule.COVALENT_RADII[element]) * bond_tolerance
        delta = np.array([fx, fy, fz]) - metal_frac
        for offset, distance in _images_within(delta, matrix, threshold):
            if distance <= 1e-10:
                continue
            position = np.array([mx, my, mz]) + matrix @ (delta + offset)
            molecule.add_atom(Atom(element, *position))
            molecule.atoms[-1].site_index = site_index
            molecule.atoms[-1].image_offset = offset
    molecule.infer_bonds_from_distance(connectivity_tolerance)
    return molecule


def _orthonormalisation(cell: Dict[str, float]) -> np.ndarray:
    """Validated 3x3 fractional-to-Cartesian matrix (columns are vectors)."""
    a, b, c = (cell[k] for k in ("a", "b", "c"))
    angles = [cell[k] for k in ("alpha", "beta", "gamma")]
    if not all(math.isfinite(v) and v > 0 for v in (a, b, c)) or not all(
            math.isfinite(v) and 0 < v < 180 for v in angles):
        raise ValueError("invalid cell lengths or angles")
    alpha, beta, gamma = map(math.radians, angles)
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 0:
        raise ValueError("cell metric must be positive definite and nonsingular")
    return np.array([[a, b * math.cos(gamma), cx],
                     [0., b * math.sin(gamma), cy], [0., 0., math.sqrt(cz2)]])


def _bond_graph(frac_sites, cell, targets, bond_tolerance, connectivity_tolerance):
    """Translation-labelled adjacency multigraph: graph[i] = [(j, t), ...].

    Edge i -> j uses the image f_j+t relative to f_i. All images within
    the radius cutoff, parallel image edges and nonzero self-image bonds
    are retained. Target-metal contacts use bond_tolerance, others the
    connectivity tolerance. This includes metal-metal distance contacts
    for dimensionality, but they are not counted as donor atoms.
    """
    if not all(math.isfinite(v) and v > 0 for v in (bond_tolerance, connectivity_tolerance)):
        raise ValueError("bond tolerances must be positive and finite")
    matrix = _orthonormalisation(cell)
    graph = [[] for _ in frac_sites]
    for i, left in enumerate(frac_sites):
        for j in range(i, len(frac_sites)):
            right = frac_sites[j]
            tolerance = bond_tolerance if left[1] in targets or right[1] in targets else connectivity_tolerance
            cutoff = (Molecule.COVALENT_RADII[left[1]] + Molecule.COVALENT_RADII[right[1]]) * tolerance
            delta = np.array(right[2:]) - left[2:]
            for offset, distance in _images_within(delta, matrix, cutoff):
                if distance <= 1e-10:
                    continue
                graph[i].append((j, offset))
                if i != j:
                    graph[j].append((i, tuple(-v for v in offset)))
    return graph


def _translation_rank(vectors):
    """Exact rank of integer cycle translations in three dimensions."""
    basis = []
    for vector in vectors:
        v = tuple(int(x) for x in vector)
        if not any(v):
            continue
        if not basis:
            basis.append(v)
        elif len(basis) == 1:
            a = basis[0]
            cross = (a[1]*v[2]-a[2]*v[1], a[2]*v[0]-a[0]*v[2], a[0]*v[1]-a[1]*v[0])
            if any(cross):
                basis.append(v)
        else:
            a, b = basis
            cross = (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
            if sum(x*y for x, y in zip(cross, v)):
                return 3
    return len(basis)


def _component_translations(graph, seed, allowed):
    """Spanning-tree image potentials and translation residuals of cycles."""
    potentials = {seed: (0, 0, 0)}
    queue = deque([seed])
    cycles = []
    while queue:
        i = queue.popleft()
        for j, offset in graph[i]:
            if j not in allowed:
                continue
            candidate = tuple(a + b for a, b in zip(potentials[i], offset))
            if j not in potentials:
                potentials[j] = candidate
                queue.append(j)
            else:
                cycles.append(tuple(a - b for a, b in zip(candidate, potentials[j])))
    return potentials, cycles


def periodicity_dimensionality(
    frac_sites, cell, target_metal_elements=None, bond_tolerance=1.16,
    connectivity_tolerance=1.20,
) -> int:
    """Maximum bonded-component cycle-translation rank (0--3).

    All components count, including a periodic host with isolated Fe.
    Tree-edge boundary crossings alone do not imply periodicity: a finite
    boundary-straddling dimer has rank zero. Integer cycle rank is invariant
    to wrapping and invertible lattice-basis changes, unlike axis counting.
    """
    targets = set(target_metal_elements or Molecule.TARGET_METAL_ELEMENTS)
    graph = _bond_graph(frac_sites, cell, targets, bond_tolerance, connectivity_tolerance)
    remaining = set(range(len(graph)))
    rank = 0
    while remaining:
        potentials, cycles = _component_translations(graph, min(remaining), remaining)
        remaining.difference_update(potentials)
        rank = max(rank, _translation_rank(cycles))
    return rank


def point_group_from_hm(space_group: str | None) -> str | None:
    """Derive the crystallographic point group from a Hermann-Mauguin symbol.

    The transform drops the lattice symbol, turns screw axes into plain
    rotations (``6_3`` -> ``6``) and glide planes into mirrors
    (``c`` -> ``m``): ``P6_3/mmc`` -> ``6/mmm``, ``P-62c`` -> ``-62m``,
    ``P2_1/c`` -> ``2/m``.
    """
    if not space_group:
        return None
    body = space_group.strip()
    if body and body[0] in "PABCIFR":
        body = body[1:]
    body = re.sub(r"(\d)_\d+", r"\1", body)   # screw axes -> rotations
    body = re.sub(r"[abcdeng]", "m", body)    # glide planes -> mirrors
    return body or None


def detect_space_group(
    frac_sites: List[Tuple[str, str, float, float, float]],
    cell: Dict[str, float],
    symprec: float = 0.05,
) -> Dict[str, Any] | None:
    """Detect the crystallographic space group of the expanded sites.

    Uses the optional ``spglib`` dependency; returns ``None`` when it is
    not installed or no symmetry is found. This matters for files exported
    as ``P1`` after symmetry expansion (e.g. by Materials Studio): the
    declared group then understates the true crystallographic symmetry.
    """
    try:
        import spglib
    except ImportError:
        return None

    lattice = _orthonormalisation(cell).T  # rows are lattice vectors
    positions = np.array([[s[2], s[3], s[4]] for s in frac_sites])
    species = sorted({s[1] for s in frac_sites})
    numbers = np.array([species.index(s[1]) + 1 for s in frac_sites])
    try:
        try:
            # New spglib versions deprecate legacy global error handling.
            dataset = spglib.get_symmetry_dataset(
                (lattice, positions, numbers), symprec=symprec, _throw=True
            )
        except TypeError:  # Compatibility with older spglib keyword signatures.
            dataset = spglib.get_symmetry_dataset(
                (lattice, positions, numbers), symprec=symprec
            )
    except Exception:
        return None
    if dataset is None:
        return None
    # spglib 2.x exposes attributes; the dict interface is deprecated.
    international = getattr(dataset, "international", None) or dataset["international"]
    number = getattr(dataset, "number", None) or dataset["number"]
    pointgroup = getattr(dataset, "pointgroup", None) or dataset.get("pointgroup")
    # Keep spglib's native ASCII form: an underscore marks a
    # crystallographic subscript (P6_3/mmc = P63/mmc conventionally).
    international = international.replace(" ", "")
    return {
        "international": international,
        "number": int(number),
        "point_group": pointgroup or point_group_from_hm(international),
        "symprec": symprec,
    }


def periodic_block(
    dimensionality: int,
    space_group: str | None = None,
    point_group: str | None = None,
    topology: str | None = None,
) -> str:
    """Format the trailing ``{periodic:3D;sg=P6_3/mmc;pg=6/mmm;topo=acs}`` block."""
    parts = [f"{dimensionality}D"]
    if space_group:
        parts.append(f"sg={space_group}")
    if point_group:
        parts.append(f"pg={point_group}")
    if topology:
        parts.append(f"topo={topology}")
    return "{periodic:" + ";".join(parts) + "}"


def ligand_fragments_from_cif(
    path: str | Path,
    bond_tolerance: float = 1.16,
    target_metal_elements: Sequence[str] | None = None,
    connectivity_tolerance: float = 1.20,
    max_states: int = 20000,
) -> List[Dict[str, Any]]:
    """Extract organic ligand molecules from a CIF.

    The periodic bond graph is built over the expanded unit cell; removing
    the metal sites cuts the framework into the individual linker
    molecules. Each returned fragment is a dict with:

    * ``atoms``: ``(label, element, x, y, z)`` in Cartesian coordinates,
      placed at a consistent set of images,
    * ``bonds``: ``(i, j)`` index pairs within the fragment,
    * ``n_copies``: how many identical copies occur in the unit cell.

    Components with nonzero cycle-translation rank are flagged ``infinite``;
    their atoms are quotient-cell representatives, not a finite molecule.
    ``max_states`` caps the number of returned representatives; exceeding it
    sets ``truncated``. Neither infinite nor truncated fragments get SMILES.
    Finite fragments are grouped by element-labelled graph isomorphism.
    """
    parsed = read_cif(path)
    cell = parsed["cell"]
    sites = expand_unit_cell(parsed["atoms"], parsed["symops"])
    targets = set(target_metal_elements or Molecule.TARGET_METAL_ELEMENTS)
    if max_states < 1:
        raise ValueError("max_states must be positive")
    graph = _bond_graph(sites, cell, targets, bond_tolerance, connectivity_tolerance)
    matrix = _orthonormalisation(cell)
    nonmetal = {i for i, site in enumerate(sites) if site[1] not in targets}
    remaining = set(nonmetal)
    fragments = []
    while remaining:
        potentials, cycles = _component_translations(graph, min(remaining), nonmetal)
        remaining.difference_update(potentials)
        infinite = bool(_translation_rank(cycles))
        # Infinite components have no molecular SMILES. Keep a bounded set
        # of quotient representatives, never a 20000-atom artificial molecule.
        selected = sorted(potentials)[:max_states]
        index = {site: k for k, site in enumerate(selected)}
        bonds = set()
        for i in selected:
            for j, offset in graph[i]:
                if j in index and tuple(a + b for a, b in zip(potentials[i], offset)) == potentials[j]:
                    bonds.add(tuple(sorted((index[i], index[j]))))
        atoms = []
        for site in selected:
            label, element, *frac = sites[site]
            cart = matrix @ (np.array(frac) + potentials[site])
            atoms.append((label, element, *(float(v) for v in cart)))
        fragments.append({
            "atoms": atoms, "bonds": sorted(bonds), "site_indices": selected,
            "infinite": infinite, "truncated": len(selected) < len(potentials),
        })

    grouped = []
    for fragment in fragments:
        for previous in grouped:
            if _same_fragment(fragment, previous):
                previous["n_copies"] += 1
                break
        else:
            fragment["n_copies"] = 1
            grouped.append(fragment)
    return sorted(grouped, key=lambda f: len(f["atoms"]), reverse=True)


def _same_fragment(left, right):
    """Exact element-labelled graph comparison, not a composition heuristic."""
    if left["infinite"] or right["infinite"] or left["truncated"] or right["truncated"]:
        return False
    if len(left["atoms"]) != len(right["atoms"]) or len(left["bonds"]) != len(right["bonds"]):
        return False
    n = len(left["atoms"])
    adjacency = []
    for fragment in (left, right):
        graph = [set() for _ in range(n)]
        for i, j in fragment["bonds"]:
            graph[i].add(j)
            graph[j].add(i)
        adjacency.append(graph)
    a, b = adjacency
    candidates = {i: [j for j in range(n) if left["atoms"][i][1] == right["atoms"][j][1]
                      and len(a[i]) == len(b[j])] for i in range(n)}
    mapping = {}
    used = set()

    def match():
        if len(mapping) == n:
            return True
        i = min((k for k in range(n) if k not in mapping),
                key=lambda k: (len(candidates[k]), -len(a[k] & mapping.keys())))
        for j in candidates[i]:
            if j not in used and all((k in a[i]) == (v in b[j]) for k, v in mapping.items()):
                mapping[i] = j
                used.add(j)
                if match():
                    return True
                used.remove(j)
                del mapping[i]
        return False

    return match()


def fragment_to_smiles(
    fragment: Dict[str, Any],
    charge: int | None = None,
    charge_candidates: Sequence[int] = (0, -1, -2, -3, -4),
) -> str | None:
    """Convert a ligand fragment to a canonical SMILES via RDKit.

    Bond orders are assigned by :func:`rdkit.Chem.rdDetermineBonds`.
    ``charge`` fixes the fragment charge; when ``None`` the candidates in
    ``charge_candidates`` are tried in order and the first valid
    assignment wins (a deprotonated linker such as a dicarboxylate
    naturally resolves at ``-2``). Returns ``None`` when RDKit is not
    installed or no assignment succeeds.
    """
    if fragment.get("infinite") or fragment.get("truncated"):
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except ImportError:
        return None

    atoms = fragment["atoms"]
    xyz = [f"{len(atoms)}", "metsmiles fragment"]
    xyz.extend(
        f"{element} {x:.5f} {y:.5f} {z:.5f}"
        for _, element, x, y, z in atoms
    )
    xyz_block = "\n".join(xyz)

    candidates = [charge] if charge is not None else list(charge_candidates)
    mol = None
    for q in candidates:
        trial = Chem.MolFromXYZBlock(xyz_block)
        if trial is None:
            return None
        try:
            rdDetermineBonds.DetermineBonds(trial, charge=q)
            mol = trial
            break
        except Exception:
            continue
    if mol is None:
        return None  # Connectivity-only output would invent unsupported bond orders.
    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol)


def generate_global_metsmiles_from_cif(
    path: str | Path,
    base_smiles: str | None = None,
    bond_tolerance: float = 1.16,
    geometry_sigma: float = 15.0,
    target_metal_elements: Sequence[str] | None = None,
    connectivity_tolerance: float = 1.20,
    topology: str | None = None,
    with_ligands: bool = True,
    space_group_symprec: float = 0.05,
    detect_symmetry: bool = True,
    method_profile: str = METHOD_PROFILE,
) -> Dict[str, Any]:
    """Global MetSMILES record for every metal center in a ``.cif`` file.

    The asymmetric unit is expanded to a full unit cell, donor sites are
    detected across all images within the radius cutoff, and each metal center is
    analysed in a local, non-periodic molecule that carries the shared-
    donor multiplicities of the whole structure. A trailing
    ``{periodic:3D;sg=...}`` block encodes the framework dimensionality
    (maximum bonded-component cycle-translation rank) and the
    space group detected by spglib, falling back to the file declaration; ``topology`` optionally records a net
    topology symbol such as ``pcu``.

    When ``with_ligands`` is true (default), the organic linkers are
    extracted from the structure and - when RDKit is installed - converted
    to canonical SMILES that occupy the ``SMILES_1`` position; an explicit
    ``base_smiles`` overrides them. ``detect_symmetry=False`` keeps the
    declared group; otherwise detection uses ``space_group_symprec`` and is
    not ranked by International Tables number. Cluster headers summarize
    quotient-cell metal groups, not necessarily isolated finite clusters.
    """
    profile_metadata(method_profile)
    Molecule(bond_tolerance, geometry_sigma, target_metal_elements, method_profile)
    parsed = read_cif(path)
    cell = parsed["cell"]
    sites = expand_unit_cell(parsed["atoms"], parsed["symops"])
    targets = set(target_metal_elements or Molecule.TARGET_METAL_ELEMENTS)

    frac_sites = list(sites)
    cart_sites = [
        (label, element, *_fraction_to_cartesian(fx, fy, fz, cell))
        for label, element, fx, fy, fz in sites
    ]
    metals = [i for i, s in enumerate(sites) if s[1] in targets]
    if not metals:
        raise ValueError(f"{path}: no {sorted(targets)} metal centers found")

    site_metals = _metal_donor_sites(
        frac_sites, cell, metals, targets, bond_tolerance
    )

    # clusters: union-find over metal sites linked by shared donor sites
    metal_position = {m: p for p, m in enumerate(metals)}
    parent = list(range(len(metals)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for site_index, hits in site_metals.items():
        for m_index, _ in hits[1:]:
            ra, rb = find(metal_position[hits[0][0]]), find(metal_position[m_index])
            if ra != rb:
                parent[rb] = ra

    cluster_groups: Dict[int, List[int]] = {}
    for m_index in metals:
        cluster_groups.setdefault(find(metal_position[m_index]), []).append(m_index)
    clusters = [sorted(g) for g in sorted(cluster_groups.values(), key=lambda g: g[0])]

    # per-center local molecules and records
    donors_by_metal: Dict[int, List[int]] = {m: [] for m in metals}
    for site_index, hits in site_metals.items():
        for m_index, _ in hits:
            donors_by_metal[m_index].append(site_index)

    all_records: Dict[int, Dict[str, Any]] = {}
    for m_index in metals:
        local = local_molecule_for_metal(
            frac_sites, cart_sites, cell, m_index, donors_by_metal[m_index],
            bond_tolerance=bond_tolerance,
            geometry_sigma=geometry_sigma,
            target_metal_elements=sorted(targets),
            connectivity_tolerance=connectivity_tolerance,
        )
        # local atom index -> shared-donor multiplicity (site-level)
        multiplicity_map = {}
        for local_index in range(1, local.num_atoms()):
            site_index = local.atoms[local_index].site_index  # type: ignore[attr-defined]
            multiplicity_map[local_index] = len(site_metals[site_index])
        all_records[m_index] = center_record(local, 0, multiplicity_map=multiplicity_map)
        all_records[m_index]["metal_index"] = m_index
        all_records[m_index]["metal_label"] = frac_sites[m_index][0]

    def block(with_geometry: bool, members: Sequence[int]) -> str:
        key = "Suffix 3" if with_geometry else "Suffix 2"
        centers_str = _collapse([all_records[m][key] for m in members])
        if len(members) <= 1:
            return centers_str
        metal_counts = Counter(frac_sites[m][1] for m in members)
        member_set = set(members)
        bridges: Dict[Tuple[int, str], int] = {}
        for site_index, hits in site_metals.items():
            shared = {m for m, _ in hits} & member_set
            if len(shared) >= 2:
                key_b = (len(shared), frac_sites[site_index][1])
                bridges[key_b] = bridges.get(key_b, 0) + 1
        return f"{cluster_header(dict(metal_counts), bridges)}.{centers_str}"

    global_2 = join_cluster_blocks([block(False, members) for members in clusters])
    global_3 = join_cluster_blocks([block(True, members) for members in clusters])

    dimensionality = periodicity_dimensionality(
        frac_sites, cell, targets, bond_tolerance, connectivity_tolerance
    )

    # Use the explicitly tolerance-dependent detection when enabled.
    # International table numbers are identifiers, not a symmetry ranking.
    space_group = parsed["space_group"]
    space_group_source = "file"
    point_group = None
    if not math.isfinite(space_group_symprec) or space_group_symprec <= 0:
        raise ValueError("space_group_symprec must be positive and finite")
    detected = detect_space_group(sites, cell, symprec=space_group_symprec) if detect_symmetry else None
    if detected is not None:
        space_group = detected["international"]
        space_group_source = "spglib"
        point_group = detected["point_group"]
    if point_group is None:
        point_group = point_group_from_hm(space_group)

    periodic = periodic_block(dimensionality, space_group, point_group, topology)
    global_2 = f"{global_2}.{periodic}"
    global_3 = f"{global_3}.{periodic}"

    ligands = []
    if with_ligands:
        for frag in ligand_fragments_from_cif(
            path,
            bond_tolerance=bond_tolerance,
            target_metal_elements=sorted(targets),
            connectivity_tolerance=connectivity_tolerance,
        ):
            ligands.append({
                "smiles": fragment_to_smiles(frag),
                "n_atoms": len(frag["atoms"]),
                "n_copies": frag["n_copies"],
                "infinite": frag["infinite"],
                "truncated": frag["truncated"],
            })

    organic = base_smiles
    if organic is None and ligands:
        # The largest fragment is the framework linker (SMILES_1 slot);
        # smaller species such as capping H2O stay in the ligands field.
        organic = ligands[0]["smiles"]

    rdkit_version = None
    if any(ligand["smiles"] is not None for ligand in ligands):
        from rdkit import rdBase
        rdkit_version = rdBase.rdkitVersion
    prefix = f"{organic}." if organic else ""
    return {
        **profile_metadata(method_profile),
        "rdkit_version": rdkit_version,
        "charge_policy": "cif_candidate_charge_search_first_success",
        "prefix_mode": "cif_largest_fragment" if with_ligands else "cif_linker_disabled",
        "SMILES_1": organic or "",
        "prefix_source": "explicit_base_smiles" if base_smiles is not None else ("cif_largest_fragment_rdkit" if organic else "cif_no_linker_prefix"),
        "prefix_status": "supplied" if base_smiles is not None else ("generated" if organic else "unavailable_or_disabled"),
        "prefix_scope": "largest_finite_linker_not_full_composition",
        "fragment_policy": "largest_fragment_only_other_species_in_ligands",
        "n_metal_centers": len(metals),
        "n_sites": len(sites),
        "space_group": space_group,
        "space_group_declared": parsed["space_group"],
        "space_group_source": space_group_source,
        "space_group_symprec": float(space_group_symprec),
        "symmetry_detection_enabled": bool(detect_symmetry),
        "point_group": point_group,
        "periodicity": dimensionality,
        "periodicity_semantics": "maximum_bonded_component_cycle_translation_rank",
        "ligands": ligands,
        "Global Suffix 2": global_2,
        "Global Suffix 3": global_3,
        "SMILES_2": prefix + global_2,
        "SMILES_3": prefix + global_3,
        "clusters": [
            {
                "metal_labels": [frac_sites[m][0] for m in members],
                "metal_composition": "".join(
                    f"{el}{n if n > 1 else ''}"
                    for el, n in sorted(Counter(frac_sites[m][1] for m in members).items())
                ),
                "centers": [all_records[m] for m in members],
            }
            for members in clusters
        ],
        "centers": [all_records[m] for m in metals],
        "bond_tolerance": float(bond_tolerance),
        "geometry_sigma": float(geometry_sigma),
        "connectivity_source": "distance_inference",
        "connectivity_tolerance": float(connectivity_tolerance),
    }
