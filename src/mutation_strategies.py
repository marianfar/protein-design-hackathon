
"""Biologically defined masking strategies for the 242-aa scFv.

Team numbering is 1-based and local to VH or VL.
The concatenated construct is VH (120 aa) + linker (15 aa) + VL (107 aa).
The masking positions returned by these functions are zero-based.
"""

VH_LENGTH = 120
LINKER_LENGTH = 15
VL_LENGTH = 107
SCFV_LENGTH = 242

# Strategy 1: mask all six CDRs
STRATEGY_1 = {
    "name": "all_six_CDRs",
    "VH": [(26, 35), (50, 66), (99, 109)],
    "VL": [(24, 34), (50, 56), (89, 97)],
}

# Strategy 2: mask the light-chain non-CDR loop
STRATEGY_2 = {
    "name": "light_chain_loop",
    "VH": [],
    "VL": [(66, 71)],
}

# Strategy 3: mask the selected heavy-chain CDR loop
STRATEGY_3 = {
    "name": "heavy_chain_CDR_loop",
    "VH": [(52, 57)],
    "VL": [],
}

STRATEGIES = [STRATEGY_1, STRATEGY_2, STRATEGY_3]


def local_to_full_index(chain, position):
    """Convert 1-based local VH/VL position to 0-based full-scFv index."""
    if not 1 <= position <= (VH_LENGTH if chain == "VH" else VL_LENGTH):
        raise ValueError(f"Invalid {chain} position: {position}")

    if chain == "VH":
        return position - 1

    if chain == "VL":
        return VH_LENGTH + LINKER_LENGTH + position - 1

    raise ValueError(f"Unknown chain: {chain}")


def get_mask_positions(strategy):
    """Return sorted, unique, zero-based positions for one strategy."""
    positions = []

    for chain in ("VH", "VL"):
        for start, end in strategy[chain]:
            if start > end:
                raise ValueError(f"Invalid range: {chain} {start}-{end}")

            positions.extend(
                local_to_full_index(chain, p)
                for p in range(start, end + 1)
            )

    positions = sorted(set(positions))

    if any(p < 0 or p >= SCFV_LENGTH for p in positions):
        raise ValueError("Mask position falls outside the scFv.")

    return positions


def validate_parent_layout(parent):
    """Check expected construct lengths and linker placement."""
    if len(parent) != SCFV_LENGTH:
        raise ValueError(
            f"Expected {SCFV_LENGTH} aa parent, found {len(parent)}"
        )

    expected_linker = "GGGGSGGGGSGGGGS"
    actual_linker = parent[VH_LENGTH:VH_LENGTH + LINKER_LENGTH]

    if actual_linker != expected_linker:
        raise ValueError(
            f"Unexpected linker: {actual_linker}"
        )

    return True


def print_strategy_summary():
    """Print mask counts and full-scFv positions."""
    for strategy in STRATEGIES:
        positions = get_mask_positions(strategy)
        print(
            f"{strategy['name']}: "
            f"{len(positions)} positions masked; "
            f"full-scFv 1-based range(s): "
            f"{positions[0] + 1}-{positions[-1] + 1}"
        )
