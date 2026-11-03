# MetSMILES 0.3.0

Coordination-environment annotations for molecular metal complexes and periodic structures. Python 3.9–3.12; method profile **`manuscript-v1`**, serialization profile **`canonical-sorted-v1`**.

[中文说明](README.zh-CN.md) · [Method specification](METHODS.md) · [Changelog](CHANGELOG.md)

## Scope

MetSMILES 0.3.0 is the reference implementation of the coordination-environment representation described in the accompanying manuscript (method profile `manuscript-v1`, serialization profile `canonical-sorted-v1`). It generates coordination suffixes and ligand prefixes from paired XYZ/SDF structures, and coordination and periodic annotations from CIF inputs. Training datasets, model checkpoints and training/evaluation pipelines are maintained separately by the authors and are not part of this generator package.

Connected donor components of two or more atoms are encoded as `%` entries following the manuscript convention; this is a representation convention for connected donors rather than an electronic-structure diagnosis. All-carbon five-membered rings contribute three geometric sites, two-atom components contribute their midpoint, and other components contribute one centroid. Geometry is scored with a Gaussian kernel (default σ=15°) against idealized CN 2–6 templates. Output strings use canonical token ordering; illustrative strings in manuscript figures may order entries differently with the same composition.

The automatic ligand prefix is extracted from the SDF with RDKit: only the selected target-metal atoms are removed, and all remaining fragments keep their SDF-encoded bond orders, formal charges and radicals. A chemically unusable input raises a typed error instead of a silently incomplete string.

## Install or use the source

From this extracted release folder:

```bash
python -m pip install .                  # numpy + pandas; suffix-only molecular and CIF graph modes
python -m pip install '.[molecular]'     # automatic molecular SDF prefixes: RDKit
python -m pip install '.[crystal]'       # finite CIF linker conversion + symmetry: RDKit + spglib
python -m pip install '.[dev,molecular,crystal]'  # tests with all extras
```

These are user-run installation instructions, not automatic downloads at import. A local wheel can be installed offline with `python -m pip install --no-index --no-deps path/to/metsmiles-0.3.0-py3-none-any.whl` if runtime dependencies are already present. Alternatively set `PYTHONPATH=src` to use the source without installation. Importing the package creates no output files. Tests require pytest; building distributions requires setuptools ≥77 and wheel (or a build frontend).

## Molecular API

```python
from metsmiles import generate_metsmiles_record, generate_global_metsmiles

# Default auto prefix requires a matching, chemically usable SDF and RDKit.
record = generate_metsmiles_record('examples/data/405.xyz', 'examples/data/405.sdf')
print(record['SMILES_1'])       # all non-target-metal fragments, including duplicates
print(record['SMILES_3'])       # prefix + coordination annotation
print(record['method_profile'], record['prefix_source'])

# Explicit structure prefix overrides automatic extraction, including an empty string.
record = generate_metsmiles_record('examples/data/405.xyz', None,
                                    base_smiles='[Cl-].[Cl-].[Cl-].[Cl-]')

# No SDF: deliberately request suffix-only instead of silently omitting the prefix.
record = generate_metsmiles_record('examples/data/405.xyz', None,
                                    prefix_mode='suffix-only')

# Any number of centers; select non-Fe targets explicitly.
record = generate_global_metsmiles('examples/data/405.xyz', 'examples/data/405.sdf',
                                     target_metal_elements=['Fe'])
```

`prefix_mode='auto'|'suffix-only'` applies to molecular file APIs and directory mode. `base_smiles is not None` always takes precedence; a supplied string is not chemically validated. All molecular automatic fragments are canonical isomeric RDKit SMILES, sorted lexicographically, with **duplicates, formal charges, isotopes, encoded stereochemistry and radicals retained**. Only selected target-metal atoms are removed; all other atoms/species remain. Implicit H completion, guessed charges, bond-order repair and known-ID replacements are not performed. Removing metal bonds may expose valence deficits/radicals. Unsupported valence or parsing raises a typed actionable error. There is no automatic fallback to a supplied answer or an incomplete suffix. See [METHODS.md](METHODS.md) for the SDF and stereochemistry boundary.

Single-center records have `Suffix 2`, `Suffix 3`, `SMILES_1/2/3`, `Real Coordination Number`, `Geometry Type`, `Confidence` (similarity, not a calibrated probability), generation parameters and profile/prefix metadata. `generate_metsmiles_records` returns one local record per center, using the **whole-structure prefix** on each row. `generate_global_metsmiles` returns a global record plus `centers` and `clusters`. The lower-level `global_metsmiles_for_molecule` has no SDF context: it assembles a supplied prefix, or explicitly reports a suffix-only record.

## CLI and directory mode

```bash
python -m metsmiles examples/data/405.xyz --sdf examples/data/405.sdf
python -m metsmiles examples/data/405.xyz --prefix-mode suffix-only
python -m metsmiles examples/data --prefix-mode suffix-only --output-csv coordination.csv
python -m metsmiles examples/data --output-csv complete.csv
python -m metsmiles examples/data/mil-88b.cif --topology acs
python -m metsmiles examples/data/mil-88b.cif --no-ligands --no-symmetry-detection
```

Defaults: Fe only, contact factor γ=1.16, Gaussian σ=15°, inferred ligand-connectivity factor 1.20. Flags `--metal` (repeatable), `--bond-tolerance`, `--geometry-sigma`, `--connectivity-tolerance`, and `--method-profile manuscript-v1` apply consistently. The CLI selects single/global molecular output by explicit center count; prefix and parsing failures are not retried as multi-center success. Directory inputs pair same-stem XYZ/SDF files; `--base-smiles-csv` accepts `gdb_idx,smiles_1`. Errors become `status=error`, `error_type`, `error` rows with no fabricated SMILES; the CLI exits nonzero if any row fails. Parameters/profiles are present even on error rows.

## CIF and global grammar

CIF analysis is independent of molecular SDF prefix generation. It expands symmetry, searches **all periodic images within contact cutoffs**, and reports the maximum bonded-component cycle-translation rank as dimensionality. Shared donor tags (`:u2`, etc.), quotient-cell cluster headers, center/cluster repeats (`xN`) and the periodic block are preserved. These are not a fully reversible chemical graph.

CIF linker conversion keeps the existing finite-fragment strategy: candidate charge/bond-order assignment is attempted by RDKit; the **largest fragment occupies the prefix only if it is finite and convertible**, not the full composition. If that fragment is infinite or cannot be converted, the automatic prefix is unavailable; the code does not silently substitute a smaller fragment. Other species and copy counts remain in `ligands`. Missing RDKit, infinite/truncated fragments or conversion failure are visible as unavailable linker SMILES and prefix provenance, rather than as molecular automatic-prefix success. Missing spglib uses the declared group. `topology='acs'` is **supplied metadata**, never automatically inferred.

## Ablations

```python
from metsmiles import mask_donor_entries, remove_pi_marker, remove_geometry
x = 'C%10CCCCC%10.[{Linear}Co&N%CO:u2]'
assert mask_donor_entries(x) == 'C%10CCCCC%10.[{Linear}Co&X%X:u2]'
assert remove_pi_marker(x) == 'C%10CCCCC%10.[{Linear}Co&N&CO:u2]'
assert remove_geometry(x) == 'C%10CCCCC%10.[Co&N%CO:u2]'
```

`add_representation_columns` maps `SMILES_NO_DONOR` to one-X-per-entry masking, `SMILES_NO_PI_MARKER` to noPi, and `SMILES_NO_GEO` to noGeo. The separate `remove_donor_list` deletion helper and `SMILES_GEOMETRY_TOKEN` are **not** the manuscript noDonor operation. All transformations are center-block scoped: organic `%10`, cluster/periodic metadata, bridges and repeats survive.

## Examples and tests

```bash
python examples/run_example.py                 # molecular extra required
python examples/run_example.py --prefix-mode suffix-only
python examples/run_mof.py                     # crystal extras for reference linker/symmetry
python examples/run_paper_examples.py          # three paper-linked cases, molecular/crystal extras
python examples/run_paper_examples.py --suffix-only --no-ligands --no-symmetry-detection
python -m pytest -q -W error
```

Three paper-linked cases are documented in [examples/README.md](examples/README.md): the Figure 2a Cp* complex, a matching Figure 2b tetrahedral four-Cl motif, and MIL-88B. The runner checks expected outputs and shows noDonor/noPi/noGeo transformations. Case definitions and reference outputs are in `examples/paper_cases.json` and `examples/paper_golden.json`.

The exact Figure 2 Cp* structure is under `examples/figure2`: ID 48 from the study reference collection, copied without coordinate changes, with provenance and input hashes. A supplied reference prefix and the automatic SDF prefix are tested separately. Its canonical suffix is `[{Octahedral}Fe&C&C%CC%CCCCC]`; the diagram uses the other π-entry order. MIL-88B needs explicit `acs` to include that topology label. CI defines minimal and extras jobs and wheel/sdist checks for Python 3.9–3.12; local execution on one Python version does not certify execution on all versions.

## Scientific scope and citation

See [METHODS.md](METHODS.md) for angle templates, grammar, input constraints and scope. No oxidation-state/spin predictor, 14-descriptor pipeline, learned model, tokenizer, graph training definition or model trainer is included in this package.

MIT license; authorship retained in [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff). Repository: https://github.com/majing-group/MetSmiles
