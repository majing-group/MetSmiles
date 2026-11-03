"""Walkthrough: generate MetSMILES annotations for the bundled example structures.

Run from the repository root (after ``pip install -e .``):

    python examples/run_example.py
"""

from __future__ import annotations

from pathlib import Path

from metsmiles import generate_metsmiles_record

EXAMPLES = Path(__file__).resolve().parent / "data"
STRUCTURES = ["0", "405", "493", "498"]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-mode", choices=["auto", "suffix-only"], default="auto")
    args = parser.parse_args()
    for stem in STRUCTURES:
        xyz_path = EXAMPLES / f"{stem}.xyz"
        sdf_path = EXAMPLES / f"{stem}.sdf"

        record = generate_metsmiles_record(xyz_path, sdf_path, prefix_mode=args.prefix_mode)
        print(f"  Prefix source: {record['prefix_source']}")
        print(f"  SMILES_1: {record['SMILES_1']}")

        print(f"Structure {stem} ({record['Metal Center']}):")
        print(f"  Suffix 2    : {record['Suffix 2']}")
        print(f"  Suffix 3    : {record['Suffix 3']}")
        print(f"  Geometry    : {record['Geometry Type']} "
              f"(confidence {record['Confidence']:.2f})")
        print(f"  Coord. number: {record['Real Coordination Number']}")

        # Connectivity can also be inferred without an SDF file.
        inferred = generate_metsmiles_record(xyz_path, None, prefix_mode="suffix-only")
        assert inferred["connectivity_source"] == "distance_inference"
        print(f"  No-SDF mode : {inferred['Suffix 3']}")
        print()

    print("With a base SMILES (SMILES_1 = CCO):")
    record = generate_metsmiles_record(EXAMPLES / "0.xyz", EXAMPLES / "0.sdf",
                                       base_smiles="CCO")
    print(f"  SMILES_2: {record['SMILES_2']}")
    print(f"  SMILES_3: {record['SMILES_3']}")


if __name__ == "__main__":
    main()
