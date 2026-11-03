"""MetSMILES: coordination-environment-aware SMILES annotation for metal complexes."""

from __future__ import annotations

from .generator import (
    Atom,
    Molecule,
    generate_metsmiles_record,
    load_structure,
    perturb_and_reorder,
    read_xyz_file,
    recognize_directory,
    records_from_report,
)
from .multimetal import (
    center_record,
    generate_global_metsmiles,
    generate_metsmiles_records,
    global_metsmiles_for_molecule,
    metal_clusters,
    sharing_map,
)
from .cif import (
    expand_unit_cell,
    generate_global_metsmiles_from_cif,
    local_molecule_for_metal,
    read_cif,
)
from .representations import (
    add_representation_columns,
    geometry_from_metsmiles,
    geometry_token_only,
    mask_donor_entries,
    remove_geometry,
    remove_donor_list,
    remove_pi_marker,
)

__version__ = "0.3.0"

__all__ = [
    "Atom",
    "Molecule",
    "generate_metsmiles_record",
    "generate_metsmiles_records",
    "generate_global_metsmiles",
    "generate_global_metsmiles_from_cif",
    "global_metsmiles_for_molecule",
    "sharing_map",
    "metal_clusters",
    "center_record",
    "read_cif",
    "expand_unit_cell",
    "local_molecule_for_metal",
    "load_structure",
    "perturb_and_reorder",
    "read_xyz_file",
    "recognize_directory",
    "records_from_report",
    "add_representation_columns",
    "geometry_from_metsmiles",
    "geometry_token_only",
    "mask_donor_entries",
    "remove_geometry",
    "remove_donor_list",
    "remove_pi_marker",
    "__version__",
]

from .prefix import PrefixError, PrefixDependencyError, PrefixChemistryError
from .profiles import METHOD_PROFILE, SERIALIZATION_PROFILE
