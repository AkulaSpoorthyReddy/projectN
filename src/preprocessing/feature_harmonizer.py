
# STEP 28B: FEATURE HARMONIZER - SHARED SEMANTIC VOCABULARY

import re
from difflib import SequenceMatcher


SEMANTIC_GROUPS = {
    "duration": [
        "duration", "dur", "flow_duration", "connection_duration"
    ],
    "source_bytes": [
        "source_bytes", "src_bytes", "sbytes",
        "forward_bytes", "total_forward_bytes"
    ],
    "destination_bytes": [
        "destination_bytes", "dst_bytes", "dbytes",
        "backward_bytes", "total_backward_bytes"
    ],
    "protocol": [
        "protocol", "proto", "protocol_type"
    ],
    "service": [
        "service", "network_service", "application_service"
    ],
    "source_packets": [
        "source_packets", "src_packets", "spkts"
    ],
    "destination_packets": [
        "destination_packets", "dst_packets", "dpkts"
    ]
}


ALIAS_TO_CONCEPT = {
    alias: concept
    for concept, aliases in SEMANTIC_GROUPS.items()
    for alias in aliases
}


def normalize_name(name):
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def semantic_similarity(name_a, name_b):

    a = normalize_name(name_a)
    b = normalize_name(name_b)

    if a == b:
        return 1.0

    concept_a = ALIAS_TO_CONCEPT.get(a)
    concept_b = ALIAS_TO_CONCEPT.get(b)

    if concept_a and concept_a == concept_b:
        return 1.0

    if concept_a and concept_b and concept_a != concept_b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def infer_feature_type(dtype):

    dtype = str(dtype).lower()

    if any(x in dtype for x in ["int", "float", "double"]):
        return "numeric"

    if any(x in dtype for x in ["object", "string", "category", "bool", "str"]):
        return "categorical"

    return "unknown"
