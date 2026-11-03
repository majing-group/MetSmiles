"""Walkthrough: global MetSMILES for a multi-metal MOF structure.

Run from the repository root (after ``pip install -e .``):

    python examples/run_mof.py

MIL-88B is an Fe-based metal-organic framework whose secondary building
units are Fe3(mu3-O)(carboxylate)6 clusters; its unit cell contains two
clusters (six Fe centers total).
"""

from __future__ import annotations

from pathlib import Path

from metsmiles import generate_global_metsmiles_from_cif

CIF = Path(__file__).resolve().parent / "data" / "mil-88b.cif"


def main() -> None:
    record = generate_global_metsmiles_from_cif(CIF, topology="acs")

    print(f"MIL-88B unit cell: {record['n_metal_centers']} Fe centers, "
          f"{len(record['clusters'])} inorganic clusters")
    print(f"space group {record['space_group']}, "
          f"framework dimensionality {record['periodicity']}D")
    print()
    print("Global Suffix 3:")
    print(f"  {record['Global Suffix 3']}")
    print()
    linker = record["ligands"][0] if record["ligands"] else None
    if linker:
        print("Extracted linker (SMILES_1 position):")
        print(f"  {linker['smiles']}  "
              f"({linker['n_atoms']} atoms x{linker['n_copies']})")
        print()
        print("Full MetSMILES:")
        print(f"  {record['SMILES_3']}")
        print()
    print("Reading: two {cluster:Fe3;u3-O} cores, each holding three")
    print("identical octahedral Fe centers; each Fe binds 5 terminal O")
    print("donors (&O) plus one mu3-O shared by all three metals (&O:u3).")
    print(f"The trailing {record['Global Suffix 3'].split('.')[-1]} block records")
    print("the maximum bonded-component cycle-translation rank and the")
    print(f"space group ({record['space_group_source']}, "
          f"symprec={record['space_group_symprec']} Angstrom).")
    print()
    for i, cluster in enumerate(record["clusters"], 1):
        labels = ", ".join(cluster["metal_labels"])
        center = cluster["centers"][0]
        print(f"Cluster {i} ({cluster['metal_composition']}, "
              f"sites {labels}):")
        print(f"  per-center Suffix 3: {center['Suffix 3']}  "
              f"(confidence {center['Confidence']:.2f})")


if __name__ == "__main__":
    main()
