# Method and serialization specification — 0.3.0

## Versioned contracts

`method_profile='manuscript-v1'` is the sole supported method profile. Unknown profile names fail. Records, coordination reports, per-center/global outputs and directory tables retain the profile. `serialization_profile='canonical-sorted-v1'` distinguishes output order from hand-drawn manuscript examples.

## 1. Inputs and contacts

Molecular inputs are a single strict XYZ frame and, optionally, a paired single-record V2000 SDF. Element order and atom counts must match; SDF coordinates must agree within 0.01 Å. Finite coordinates, valid element symbols, counts, bond indices/orders and block termination are required. V3000, multi-record selection, disorder and ambiguous atom mappings are not silently accepted. The SDF supplies ligand-internal connectivity and the chemical graph for molecular prefix extraction; its metal bonds do **not** define distance contacts.

The default target set is `{Fe}`. Select other target elements explicitly; unselected metals are not automatically removed. A non-target atom is a donor contact when `d(M,i) <= gamma * (r_M + r_i)`, default gamma=1.16. The executable radii table is `Molecule.COVALENT_RADII` in `generator.py`; examples are Fe 1.52, C 0.69, N 0.71, O 0.66 Å. The table follows the existing study implementation and is authoritative here; do not substitute another radius table under the same profile.

Without SDF, non-target atoms are connected when `0 < d(i,j) <= 1.20*(r_i+r_j)` by default. This route cannot infer chemically reliable charges or bond orders and does not provide an automatic molecular prefix. Contact and connectivity factors are separate parameters. Target-metal neighbors are excluded as donors. Every detected non-target contact is retained in the grouping analysis.

## 2. Donor-induced graph and geometric sites

For each center, form the **induced graph on its contacting atoms only**. An indirect path through a non-contact atom does not join two donor components.

- A singleton is an `&Element` entry with the atom position as one geometric site.
- Every connected component with at least two donors is one `%Elements` entry. This implements the SI §2.1 syntax convention for connected donors. Heteroatoms, long-bonded pairs, and connected conventional σ donors follow the same convention.
- A two-atom component contributes its midpoint, one site.
- An actual induced **all-carbon five-cycle** contributes three sites. Every vertex must have degree two, the graph must be connected, and there must be no chord. Select the atom nearest the metal (ties by Cartesian coordinate tuple); retain that anchor position and the midpoints of the two disjoint adjacent pairs remaining around the cycle. Topology, not input atom-list order, determines the pairs. Planarity/conjugation are not additional default filters.
- Every other component contributes its centroid, one site: size 3/4/6 and larger, non-Cp five-atom chains, heterocycles, or chorded five-vertex graphs. Non-Cp five-atom components contribute one centroid by convention.

`Real Coordination Number` means effective geometric-site count, not the number of donor atoms, formal valence, hapticity or coordination electron count. Donor atoms/group sizes are encoded separately. A `%CCCCC` composition alone does not certify Cp topology. The three Cp sites are **one real anchor plus two midpoints**, not three centroids. Exact nearest-anchor ties can break rotational equivariance; finite atom-permutation tests do not prove universal geometry invariance.

## 3. Geometry matching

Compute all site–metal–site angles. Monotonically pair sorted measured and ideal angle distributions. For each candidate, use the mean `exp(-0.5*((angle-ideal)/sigma)**2)`, sigma=15° by default. Select the highest mean; deterministic candidate insertion order breaks score ties.

| Effective CN | Candidates and ideal angles (degrees) |
|---|---|
| 2 | Linear: 180 |
| 3 | TrigonalPlanar: 120 ×3 |
| 4 | Tetrahedral: 109.5 ×6; SquarePlanar: 90 ×4, 180 ×2 |
| 5 | TrigonalBipyramid: 90 ×6, 120 ×3, 180; SquarePyramid: 90 ×8, 180 ×2 |
| 6 | Octahedral: 90 ×12, 180 ×3 |

CN outside 2–6 gives `unknown`; contacts are not truncated to six. Zero-contact local and global records consistently use `unknown`: this means no supported geometry, not parsing failure. There is no score rejection threshold: CN6 always selects Octahedral even for severe distortion. `Confidence` is an uncalibrated template similarity, not geometry accuracy or a probability. Sorted angle matching is not vertex-consistent 3D fitting and does not maximize the nonconvex Gaussian objective over all assignments. Coincident sites causing undefined angles raise an error instead of fabricating angles.

## 4. Molecular prefix policy

The file APIs default to `prefix_mode='auto'`. An explicit `base_smiles` overrides extraction (even `''`); it is preserved verbatim, with provenance `explicit_base_smiles`, and not chemically validated. `prefix_mode='suffix-only'` deliberately requests no automatic prefix. With neither SDF nor explicit base, `auto` raises `PrefixError`. RDKit missing raises `PrefixDependencyError`. Chemical parsing/sanitization/serialization failure raises `PrefixChemistryError`; the error suggests a validated explicit base or deliberate suffix-only use. Batch mode records these errors and the CLI returns nonzero.

Automatic extraction reads the SDF **unsanitized**, removes only target-metal atoms, disables implicit-H completion, then sanitizes the remaining graph. Existing SDF bond orders and formal charges are retained; there is no charge search, bond-order repair, electron-count heuristic or reference lookup. Standard removable explicit H atoms are folded into atom H counts by RDKit; isotope H and other non-removable H follow RDKit's rules. Stereo assignments encoded in the graph/SDF are retained when still valid after metal removal; unspecified stereo is not invented from the XYZ geometry. All resulting fragments are serialized as canonical isomeric SMILES and sorted lexicographically; duplicate fragments are **not deduplicated**. Isotopes and radical information participate in serialization.

Deletion of a metal bond may leave a radical or unsatisfied valence. This implementation does not turn neutral chlorine into chloride, carbon monoxide into a guessed charge-separated form, or ligand fragments into reference-derived charge states. Thus parseable automatic prefixes can differ substantially from familiar closed-shell ligand SMILES. Sanitization success is graph-acceptance coverage, not chemical truth or complete electronic-structure recovery. A target-metal-only input legitimately has empty prefix with `prefix_status='no_non_target_atoms'`.

Records contain `SMILES_1`, `prefix_mode`, `prefix_source`, `prefix_status`, `rdkit_version`, `fragment_policy` and `charge_policy`. Automatic molecular prefix provenance is `sdf_rdkit`; explicit or suffix-only paths do not claim an RDKit version. Perturbation/reordering affects coordination analysis, while prefix extraction uses the original matched SDF graph.

## 5. Serialization and extensions

A center block is `[Metal entries]` for Suffix 2, `[{Geometry}Metal entries]` for Suffix 3 (without spaces). Sort singleton donor entries by element/multiplicity, sort symbols inside each multi-donor entry lexicographically, and sort multi-donor entries. A nonempty prefix and suffix are joined with a dot. Strings are annotations extending SMILES, not valid whole-molecule RDKit SMILES.

```text
prefix.[{Octahedral}Fe&C&C%CC%CCCCC]
prefix.{cluster:CoNi;u2-O}.[{unknown}Co&O:u2]|[{unknown}Ni&O:u2]
prefix.{cluster:Fe3;u3-O}x2.[{Octahedral}Fe&O&O&O&O&O&O:u3]x3.{periodic:3D;sg=P6_3/mmc;pg=6/mmm;topo=acs}
```

`:uk` denotes donor sharing by k metal centers/images. A multi-donor entry uses the maximum atom sharing multiplicity within that entry; it does not encode a complete atom–center incidence matrix. Shared donors join quotient-cell cluster groups. Equivalent center blocks are collapsed with `xN` and `|` separates distinct center types. A repeat immediately after a cluster header counts repeated clusters; a repeat after a center counts those centers. Cluster/block ordering is canonical-sorted, not the diagram order.

Manuscript figures may order π entries differently from this canonical serialization; the listed composition is identical.

## 6. CIF-specific behavior

The parser accepts supported scalar/loop CIF syntax, quoted symmetry operations and finite full-occupancy sites. Partial occupancy and invalid affine operations are rejected. It is not a general disorder or all-CIF-dialect parser. Symmetry-expanded sites use a metric-aware QR search for **all** periodic images within cutoffs; a fixed neighboring-cell box or minimum image alone is insufficient for skew cells. Periodicity is the maximum integer cycle-translation rank across bonded components, including periodic hosts and distance-based metal–metal edges, not the count of crossed axes. A quotient-cell cluster is not guaranteed to be an isolated finite SBU.

Finite non-target fragments are grouped by element-labeled graph isomorphism; infinite or truncated fragments are flagged and receive no molecular SMILES. CIF has no SDF bond orders: the separate RDKit converter attempts candidate charges/bond orders and selects the first successful conversion. **This is a different policy from molecular charge retention**. The largest fragment is selected first; its SMILES is used as a main linker prefix only when finite, untruncated and convertible. A smaller convertible fragment does not silently replace an unavailable largest fragment. All fragment records/copy counts remain available. It is not a complete composition encoding. Prefix provenance explicitly identifies this CIF linker scope. RDKit missing or conversion failure leaves linker SMILES unavailable. spglib missing/unsuccessful detection retains the declared space group; detection parameters/source are recorded. Topology is caller-supplied metadata. For MIL-88B, supply `topology='acs'`; it is not discovered by this package.

## 7. Ablation definitions and limits

`mask_donor_entries` (noDonor) maps each coordination entry to **one** X: `&N -> &X`, `%CO -> %X`, retaining `:uk`, markers, metals, geometry, repeats and cluster/periodic blocks. `remove_pi_marker` (noPi) changes `%` to `&` only in center blocks. `remove_geometry` (noGeo) removes the geometry token only. Organic `%10` ring syntax is untouched. `remove_donor_list` deletes entries and is a different helper; `geometry_token_only` is a donor-deletion representation, not noDonor. DataFrame columns are specified in the README.

The suffix loses complete ligand–site mapping, many connectivity/stereochemical distinctions, Δ/Λ/trans-pair assignments, oxidation states and spin. No classifier, descriptor, tokenizer or training pipeline is included in this package.
