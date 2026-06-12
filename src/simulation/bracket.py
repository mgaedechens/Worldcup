"""Official 2026 FIFA World Cup bracket structure (48-team format).

Sources: the December 5, 2025 final draw and FIFA's published knockout schedule
(matches 73-104). See docs/decisions/ADR-004.

- 12 groups (A-L). Each of our reconstructed groups is labeled by a unique "anchor" team
  (each group has exactly one of these), which lets us attach official letters while keeping
  the dataset's exact team-name spellings.
- Top 2 of each group (24) + the 8 best third-placed teams = 32 advance to the Round of 32.
- The 8 third-place slots each accept a third from a fixed set of groups; we assign the
  qualifying thirds to slots with a constraint-respecting matching (FIFA's official intent).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

# Unique anchor team -> official group letter (verified against the official draw).
GROUP_ANCHORS: dict[str, str] = {
    "Mexico": "A", "Canada": "B", "Brazil": "C", "United States": "D",
    "Germany": "E", "Netherlands": "F", "Belgium": "G", "Spain": "H",
    "France": "I", "Argentina": "J", "Portugal": "K", "England": "L",
}

# Round of 32. Each match: (match_id, source_a, source_b).
# Sources: ("W", g)=winner of group g, ("R", g)=runner-up, ("3", match_id)=third in that slot.
ROUND_OF_32: list[tuple[int, tuple, tuple]] = [
    (73, ("R", "A"), ("R", "B")),
    (74, ("W", "E"), ("3", 74)),
    (75, ("W", "F"), ("R", "C")),
    (76, ("W", "C"), ("R", "F")),
    (77, ("W", "I"), ("3", 77)),
    (78, ("R", "E"), ("R", "I")),
    (79, ("W", "A"), ("3", 79)),
    (80, ("W", "L"), ("3", 80)),
    (81, ("W", "D"), ("3", 81)),
    (82, ("W", "G"), ("3", 82)),
    (83, ("R", "K"), ("R", "L")),
    (84, ("W", "H"), ("R", "J")),
    (85, ("W", "B"), ("3", 85)),
    (86, ("W", "J"), ("R", "H")),
    (87, ("W", "K"), ("3", 87)),
    (88, ("R", "D"), ("R", "G")),
]

# Which group's third-placed team may fill each third-place slot (by R32 match id).
THIRD_SLOT_ALLOWED: dict[int, frozenset[str]] = {
    74: frozenset("ABCDF"),
    77: frozenset("CDFGH"),
    79: frozenset("CEFHI"),
    80: frozenset("EHIJK"),
    81: frozenset("BEFIJ"),
    82: frozenset("AEHIJ"),
    85: frozenset("EFGIJ"),
    87: frozenset("DEIJL"),
}

# Later rounds: match_id -> (feeder_match_a, feeder_match_b). Winners advance.
ROUND_OF_16: dict[int, tuple[int, int]] = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
}
QUARTERFINALS: dict[int, tuple[int, int]] = {
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),
}
SEMIFINALS: dict[int, tuple[int, int]] = {101: (97, 98), 102: (99, 100)}
FINAL_MATCH: tuple[int, int] = (101, 102)

# Stage labels for reporting how far a team got.
STAGES = ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final", "Champion"]


def label_groups_official(reconstructed: dict[str, list[str]]) -> dict[str, list[str]]:
    """Re-key reconstructed groups (arbitrary letters) to official letters via anchors."""
    official: dict[str, list[str]] = {}
    for teams in reconstructed.values():
        letter = next((GROUP_ANCHORS[t] for t in teams if t in GROUP_ANCHORS), None)
        if letter is None:
            raise ValueError(f"No anchor team found in group {teams}")
        official[letter] = teams
    if set(official) != set("ABCDEFGHIJKL"):
        raise ValueError(f"Expected groups A-L, got {sorted(official)}")
    return official


def assign_thirds_to_slots(qualifying_thirds: list[tuple[str, str]]) -> dict[int, str]:
    """Assign the 8 qualifying thirds to the 8 third-place slots.

    ``qualifying_thirds`` is a list of (group_letter, team). We solve a min-cost matching
    where an assignment is allowed (cost 0) only if the team's group is in the slot's
    allowed set; disallowed pairings get a prohibitive cost. FIFA's design guarantees a
    valid perfect matching exists for any set of 8 qualifying groups.
    """
    slots = list(THIRD_SLOT_ALLOWED)  # 8 R32 match ids
    big = 1e6
    cost = np.empty((8, 8))
    for i, (group, _team) in enumerate(qualifying_thirds):
        for j, slot in enumerate(slots):
            cost[i, j] = 0.0 if group in THIRD_SLOT_ALLOWED[slot] else big
    rows, cols = linear_sum_assignment(cost)
    return {slots[j]: qualifying_thirds[i][1] for i, j in zip(rows, cols)}
