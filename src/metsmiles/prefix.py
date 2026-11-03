"""Molecular SDF ligand-prefix extraction without charge or bond-order repair."""
from pathlib import Path


class PrefixError(ValueError):
    """An automatic molecular prefix cannot be generated faithfully."""


class PrefixDependencyError(PrefixError):
    """RDKit is required for the requested automatic prefix."""


class PrefixChemistryError(PrefixError):
    """The supplied non-target-metal graph cannot be sanitized by RDKit."""


def resolve_molecular_prefix(sdf_path, target_metal_elements, base_smiles=None,
                             prefix_mode="auto"):
    """Return a prefix and provenance; explicit bases override automatic extraction.

    Target metals only are removed. All remaining fragments, including duplicate
    ions, are retained. No charge search, bond-order inference or implicit-H
    completion is used. Valence deficits may therefore serialize as radicals.
    XYZ/SDF alignment is checked by the structure loader before this function.
    """
    if prefix_mode not in ("auto", "suffix-only"):
        raise ValueError("prefix_mode must be 'auto' or 'suffix-only'")
    metadata = dict(prefix_mode=prefix_mode, rdkit_version=None,
                    fragment_policy="all_fragments_duplicates_retained_sorted_isomeric",
                    charge_policy="retain_sdf_no_charge_search_no_implicit_H_completion")
    if base_smiles is not None:
        if not isinstance(base_smiles, str):
            raise TypeError("base_smiles must be a string or None")
        return base_smiles, dict(metadata, prefix_source="explicit_base_smiles",
                                 prefix_status="supplied")
    if prefix_mode == "suffix-only":
        return "", dict(metadata, prefix_source="suffix_only", prefix_status="not_requested")
    if sdf_path is None:
        raise PrefixError("Automatic molecular prefix requires a matching SDF; supply "
                          "base_smiles or explicitly set prefix_mode='suffix-only'.")
    try:
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise PrefixDependencyError("Automatic molecular prefix requires RDKit; install "
                                    "metsmiles[molecular], supply base_smiles, or explicitly "
                                    "set prefix_mode='suffix-only'.") from exc
    metadata["rdkit_version"] = rdBase.rdkitVersion
    try:
        mol = Chem.MolFromMolBlock(Path(sdf_path).read_text(encoding="utf-8"),
                                  sanitize=False, removeHs=False, strictParsing=True)
        if mol is None:
            raise ValueError("RDKit could not parse the SDF molecule")
        edit = Chem.RWMol(mol)
        for index in reversed(range(edit.GetNumAtoms())):
            if edit.GetAtomWithIdx(index).GetSymbol() in target_metal_elements:
                edit.RemoveAtom(index)
        mol = edit.GetMol()
        if mol.GetNumAtoms() == 0:
            return "", dict(metadata, prefix_source="sdf_rdkit", prefix_status="no_non_target_atoms")
        # SDF atom/H/charge annotations are authoritative; do not repair ligand
        # valences with guessed hydrogens after deleting metal bonds.
        for atom in mol.GetAtoms():
            atom.SetNoImplicit(True)
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        smiles = sorted(Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
                        for fragment in fragments)
        return ".".join(smiles), dict(metadata, prefix_source="sdf_rdkit", prefix_status="generated")
    except Exception as exc:
        raise PrefixChemistryError("Cannot sanitize/serialize non-target-metal SDF fragments "
                                   "without chemical repair: " + str(exc) + ". Check SDF "
                                   "bond orders/charges, supply a validated base_smiles, or "
                                   "explicitly request prefix_mode='suffix-only'.") from exc
