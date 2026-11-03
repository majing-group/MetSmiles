"""Run three mapped paper structures; print freshly computed JSON, never golden data.

Install metsmiles[molecular,crystal] for the full branch. For core-only use
--suffix-only --no-ligands --no-symmetry-detection. Paths are script-relative.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from metsmiles import (
    generate_global_metsmiles_from_cif, generate_metsmiles_record,
    mask_donor_entries, remove_geometry, remove_pi_marker,
)

HERE = Path(__file__).resolve().parent


def variants(text):
    return {"noDonor": mask_donor_entries(text),
            "noPi": remove_pi_marker(text), "noGeo": remove_geometry(text)}


def run_examples(*, suffix_only=False, no_ligands=False, no_symmetry_detection=False):
    """Compute all three cases. Optional branches are explicit, not silent fallbacks."""
    if (not suffix_only or not no_ligands) and importlib.util.find_spec("rdkit") is None:
        raise RuntimeError("Install metsmiles[molecular,crystal], or use --suffix-only --no-ligands")
    if not no_symmetry_detection and importlib.util.find_spec("spglib") is None:
        raise RuntimeError("Install metsmiles[crystal], or use --no-symmetry-detection (declared P1)")
    provenance = json.loads((HERE / "paper_cases.json").read_text(encoding="utf-8"))
    results = []
    for case in provenance["cases"]:
        if case["id"] != "mil88b":
            mode = "suffix-only" if suffix_only else "auto"
            record = generate_metsmiles_record(HERE / case["xyz"], HERE / case["sdf"], prefix_mode=mode)
            suffix = record["Suffix 3"]
            expected = case["expected_canonical_suffix"]
            expected_cn = case["expected_effective_cn"]
        else:
            mode = "cif_no_linker" if no_ligands else "cif_largest_fragment"
            record = generate_global_metsmiles_from_cif(
                HERE / case["cif"], topology="acs", with_ligands=not no_ligands,
                detect_symmetry=not no_symmetry_detection)
            suffix = record["Global Suffix 3"]
            expected = case["expected_canonical_suffix"]
            if no_symmetry_detection:
                expected = expected.replace("sg=P6_3/mmc;pg=6/mmm", "sg=P1;pg=1")
            expected_cn = [6] * 6
        results.append({"id": case["id"], "paper_mapping": case["paper_mapping"],
                        "input_files": {k: case[k] for k in ("xyz", "sdf", "cif") if k in case},
                        "expected_canonical_suffix": expected, "actual_canonical_suffix": suffix,
                        "expected_effective_cn": expected_cn,
                        "requested_prefix_mode": mode,
                        "topology_supplied": "acs" if case["id"] == "mil88b" else None,
                        "topology_automatic": False,
                        "record": record, "variants": variants(record["SMILES_3"])})
    return {"scope": "representation examples, not spin/oxidation-state predictions; scores are not probabilities",
            "examples": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix-only", action="store_true", help="Molecules: omit automatic SDF ligand prefix")
    parser.add_argument("--no-ligands", action="store_true", help="CIF: omit linker extraction")
    parser.add_argument("--no-symmetry-detection", action="store_true", help="CIF: use declared P1, not detected P6_3/mmc")
    args = parser.parse_args()
    try:
        result = run_examples(suffix_only=args.suffix_only, no_ligands=args.no_ligands,
                              no_symmetry_detection=args.no_symmetry_detection)
    except RuntimeError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
