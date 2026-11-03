"""Suffix-scoped ablation helpers preserving organic SMILES and global grammar."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .generator import Molecule

if TYPE_CHECKING:  # pragma: no cover
    from pandas import DataFrame

_METALS = "|".join(sorted(Molecule.METAL_ELEMENTS, key=lambda s: (-len(s), s)))
CENTER_RE = re.compile(
    r"\[(?:\{(?P<geometry>[^{}]+)\})?(?P<metal>" + _METALS + r")"
    r"(?P<donors>(?:[&%][A-Z][a-z]?(?:[A-Z][a-z]?)*(?::u\d+)?)*)\]"
)


def geometry_from_metsmiles(value: object) -> str:
    """Return geometry tokens in suffix order, joined by ``|`` for many centers.

    Collapsed ``xN`` entries yield one token per encoded center block, not
    N expanded tokens. Only explicit geometry tokens are extracted; ordinary
    organic metal ions do not add tokens. With none, return ``Unknown``.
    """
    geometries = [m.group("geometry") for m in CENTER_RE.finditer(str(value))
                  if m.group("geometry") is not None]
    return "|".join(geometries) or "Unknown"


def remove_donor_list(value: object) -> str:
    """Remove donors only inside center blocks; retain metal, geometry, xN, headers."""
    def replace(match):
        geometry = match.group("geometry")
        return "[" + (f"{{{geometry}}}" if geometry else "") + match.group("metal") + "]"
    return CENTER_RE.sub(replace, str(value))


def mask_donor_entries(value: object) -> str:
    """Manuscript noDonor: one X per entry, preserving &/%, bridges and repeats."""
    def replace(match):
        geometry = match.group("geometry")
        donors = re.sub(r"([&%])[A-Z][a-z]?(?:[A-Z][a-z]?)*", r"\1X", match.group("donors"))
        return "[" + (f"{{{geometry}}}" if geometry else "") + match.group("metal") + donors + "]"
    return CENTER_RE.sub(replace, str(value))


def remove_geometry(value: object) -> str:
    """Manuscript noGeo: remove only geometry tokens inside center blocks."""
    return CENTER_RE.sub(lambda m: "[" + m.group("metal") + m.group("donors") + "]", str(value))


def remove_pi_marker(value: object) -> str:
    """Change % to & only in coordination blocks, never organic %10 ring labels."""
    return CENTER_RE.sub(lambda m: m.group().replace("%", "&"), str(value))


def geometry_token_only(smiles_1: object, smiles_3: object) -> str:
    """Use the supplied organic prefix and donor-free global annotation.

    Metals, center/cluster repeats, separators and the periodic block are
    retained. With no recognized metal block, return the organic string.
    This does not invent an Fe center for unknown or organic-only input.
    """
    text = str(smiles_3)
    prefix = str(smiles_1)
    if text == prefix:
        return prefix
    if prefix and text.startswith(prefix + "."):
        suffix = text[len(prefix) + 1:]
        return prefix + "." + remove_donor_list(suffix) if CENTER_RE.search(suffix) else prefix
    first = CENTER_RE.search(text)
    if first is None:
        return str(smiles_1)
    start = first.start()
    # A cluster header belongs to the annotation, not the organic prefix.
    cluster = text.find("{cluster:")
    if 0 <= cluster < start:
        start = cluster
    suffix = remove_donor_list(text[start:])
    prefix = str(smiles_1)
    return (prefix + "." if prefix else "") + suffix


def add_representation_columns(frame: "DataFrame") -> "DataFrame":
    """Append ablation columns to a copy with SMILES_1 and SMILES_3 columns."""
    result = frame.copy()
    result["geometry_type"] = result["SMILES_3"].map(geometry_from_metsmiles)
    result["SMILES_NO_DONOR"] = result["SMILES_3"].map(mask_donor_entries)
    result["SMILES_NO_GEO"] = result["SMILES_3"].map(remove_geometry)
    result["SMILES_NO_PI_MARKER"] = result["SMILES_3"].map(remove_pi_marker)
    result["SMILES_GEOMETRY_TOKEN"] = [
        geometry_token_only(s1, s3)
        for s1, s3 in zip(result["SMILES_1"], result["SMILES_3"])
    ]
    return result
