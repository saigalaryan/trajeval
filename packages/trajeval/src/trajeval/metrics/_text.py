"""Tiny text-similarity helper shared by trajectory_efficiency (loop
detection) and recovery (did the agent actually reformulate its query, or
just resend the same one). Not exported publicly — internal to metrics.
"""

from __future__ import annotations


def normalized_edit_distance(a: str, b: str) -> float:
    """Levenshtein distance normalized to [0, 1] by the longer string's length.

    0.0 means identical (after trim/lowercase); 1.0 means nothing in common.
    """
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr_row[j] = min(
                prev_row[j] + 1,  # deletion
                curr_row[j - 1] + 1,  # insertion
                prev_row[j - 1] + cost,  # substitution
            )
        prev_row = curr_row

    distance = prev_row[-1]
    return distance / max(len(a), len(b))
