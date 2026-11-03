from __future__ import annotations

import pandas as pd

from metsmiles import (
    add_representation_columns,
    geometry_from_metsmiles,
    geometry_token_only,
    remove_donor_list,
    remove_pi_marker,
)

SMILES_1 = "CCO"
SMILES_3 = "CCO.[{Octahedral}Fe&Cl&Cl&N&N&O&O]"
SMILES_3_CP = "C.[{Tetrahedral}Fe&C&N%CCCCC]"


def test_geometry_from_metsmiles():
    assert geometry_from_metsmiles(SMILES_3) == "Octahedral"
    assert geometry_from_metsmiles(SMILES_3_CP) == "Tetrahedral"
    assert geometry_from_metsmiles("CCO") == "Unknown"


def test_remove_donor_list():
    assert remove_donor_list(SMILES_3) == "CCO.[{Octahedral}Fe]"


def test_remove_pi_marker():
    assert remove_pi_marker(SMILES_3_CP) == "C.[{Tetrahedral}Fe&C&N&CCCCC]"


def test_geometry_token_only():
    assert geometry_token_only(SMILES_1, SMILES_3) == "CCO.[{Octahedral}Fe]"


def test_add_representation_columns():
    frame = pd.DataFrame({
        "SMILES_1": [SMILES_1, "C"],
        "SMILES_3": [SMILES_3, SMILES_3_CP],
    })
    result = add_representation_columns(frame)

    assert list(result["geometry_type"]) == ["Octahedral", "Tetrahedral"]
    assert list(result["SMILES_NO_DONOR"]) == [
        "CCO.[{Octahedral}Fe&X&X&X&X&X&X]",
        "C.[{Tetrahedral}Fe&X&X%X]",
    ]
    assert list(result["SMILES_NO_PI_MARKER"]) == [
        "CCO.[{Octahedral}Fe&Cl&Cl&N&N&O&O]",
        "C.[{Tetrahedral}Fe&C&N&CCCCC]",
    ]
    assert list(result["SMILES_GEOMETRY_TOKEN"]) == [
        "CCO.[{Octahedral}Fe]",
        "C.[{Tetrahedral}Fe]",
    ]
    # The input frame is not mutated.
    assert list(frame.columns) == ["SMILES_1", "SMILES_3"]
