"""Command-line generation with explicit mode and prefix handling."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from .generator import load_structure, generate_metsmiles_record, recognize_directory
from .multimetal import generate_global_metsmiles
from .profiles import METHOD_PROFILE


def build_parser():
    parser = argparse.ArgumentParser(prog="metsmiles", description="Generate versioned MetSMILES annotations.")
    parser.add_argument("input", help="XYZ file, CIF file, or directory of XYZ files")
    parser.add_argument("--sdf", help="Matching V2000 SDF for molecular input")
    parser.add_argument("--base-smiles", help="Explicit structure prefix overriding automatic extraction")
    parser.add_argument("--prefix-mode", choices=["auto", "suffix-only"], default=None,
                        help="Molecular only; default auto requires SDF or an explicit base")
    parser.add_argument("--method-profile", choices=[METHOD_PROFILE], default=METHOD_PROFILE)
    parser.add_argument("--output-csv", default="coordination_analysis.csv")
    parser.add_argument("--base-smiles-csv", help="Directory bases: gdb_idx,smiles_1 columns")
    parser.add_argument("--bond-tolerance", type=float, default=1.16)
    parser.add_argument("--geometry-sigma", type=float, default=15.)
    parser.add_argument("--connectivity-tolerance", type=float, default=1.20)
    parser.add_argument("--metal", action="append", default=None, help="Target element; repeat as needed; default Fe")
    parser.add_argument("--topology", help="CIF only: supplied topology label, not inferred")
    parser.add_argument("--no-ligands", action="store_true", help="CIF only: disable finite linker extraction")
    parser.add_argument("--no-symmetry-detection", action="store_true")
    parser.add_argument("--space-group-symprec", type=float, default=.05)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    path = Path(args.input)
    directory = path.is_dir()
    cif = not directory and path.suffix.lower() == ".cif"
    if not cif and (args.topology is not None or args.no_ligands or args.no_symmetry_detection
                    or args.space_group_symprec != .05):
        parser.error("CIF-specific flags require a .cif input")
    if cif and (args.sdf is not None or args.prefix_mode is not None):
        parser.error("--sdf and --prefix-mode apply only to molecular inputs")
    if directory and (args.sdf is not None or args.base_smiles is not None):
        parser.error("directory mode uses same-stem SDFs and --base-smiles-csv")
    if not directory and args.base_smiles_csv is not None:
        parser.error("--base-smiles-csv is only valid in directory mode")
    common = dict(bond_tolerance=args.bond_tolerance, geometry_sigma=args.geometry_sigma,
                  target_metal_elements=args.metal, connectivity_tolerance=args.connectivity_tolerance,
                  method_profile=args.method_profile)
    try:
        if directory:
            bases = {}
            if args.base_smiles_csv:
                with open(args.base_smiles_csv, newline="", encoding="utf-8") as handle:
                    bases = {row["gdb_idx"]: row["smiles_1"] for row in csv.DictReader(handle)}
            rows = recognize_directory(path, args.output_csv, base_smiles_by_index=bases,
                                       prefix_mode=args.prefix_mode or "auto", **common)
            failures = sum(row["status"] == "error" for row in rows)
            print(f"Wrote {len(rows)} rows to {args.output_csv}; {failures} errors", file=sys.stderr)
            return 1 if failures else 0
        if cif:
            from .cif import generate_global_metsmiles_from_cif
            record = generate_global_metsmiles_from_cif(
                path, base_smiles=args.base_smiles, topology=args.topology,
                with_ligands=not args.no_ligands, detect_symmetry=not args.no_symmetry_detection,
                space_group_symprec=args.space_group_symprec, **common)
        else:
            # Inspect center count; never retry a chemistry error as another mode.
            molecule = load_structure(path, args.sdf, **common)
            count = len(molecule.find_metal_centers())
            if count == 0:
                raise ValueError("no target metal centers")
            generate = generate_metsmiles_record if count == 1 else generate_global_metsmiles
            record = generate(path, args.sdf, base_smiles=args.base_smiles,
                              prefix_mode=args.prefix_mode or "auto", **common)
    except (ValueError, OSError, KeyError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
