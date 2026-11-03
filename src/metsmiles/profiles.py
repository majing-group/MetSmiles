"""Versioned method and serialization contracts."""
METHOD_PROFILE = "manuscript-v1"
SERIALIZATION_PROFILE = "canonical-sorted-v1"


def profile_metadata(method_profile=METHOD_PROFILE):
    if method_profile != METHOD_PROFILE:
        raise ValueError("method_profile must be 'manuscript-v1'")
    return {"method_profile": METHOD_PROFILE,
            "serialization_profile": SERIALIZATION_PROFILE}
