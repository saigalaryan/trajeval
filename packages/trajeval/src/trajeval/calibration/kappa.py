"""Cohen's kappa between judge and human labels — the honesty layer that
makes a judged metric's score credible.

Computed overall and sliced by golden-record tag, because a judge that
agrees with humans 90% of the time in aggregate can still be near-random on
one language or difficulty slice — averaging hides exactly the failure this
calibration step exists to catch.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from trajeval.calibration.labels import HumanLabel
from trajeval.calibration.verdicts import judge_verdict
from trajeval.results import CalibrationState, RunResult

# Per the project's brief: hand-label at least this many trajectories before
# a judged metric's calibration state is trusted.
MIN_LABELS_FOR_CALIBRATION = 50

KappaWeights = Literal["none", "linear", "quadratic"]

# query_quality's judge/human verdicts are "1".."5" — genuinely ordinal, so
# a judge/human gap of "3 vs 5" is a smaller miss than "1 vs 5" and should
# be weighted accordingly. recovery/faithfulness verdicts are nominal
# (answered_from_bad_context vs correctly_abstained, etc.) — there's no
# natural distance between them, so they stick with unweighted kappa, which
# is exactly what "none" weighting reduces to (see cohens_kappa docstring).
_ORDINAL_METRIC_WEIGHTS: dict[str, KappaWeights] = {"query_quality": "quadratic"}


def _category_sort_key(value: str) -> tuple[int, float | str]:
    """Numeric categories ("1".."5") sort numerically; anything else falls
    back to lexicographic, grouped after all numeric values so the two
    orderings never interleave in a way that would make weighting
    meaningless for a mixed category set."""
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def cohens_kappa(
    judge_labels: list[str], human_labels: list[str], *, weights: KappaWeights = "none"
) -> float:
    """Cohen's kappa over paired categorical labels.

    `judge_labels[i]` and `human_labels[i]` must be the same pair's two
    raters. Raises ValueError on empty input — kappa is undefined with no
    observations, and returning e.g. 0.0 would silently look like
    "measured, zero agreement" instead of "not measured at all".

    `weights="none"` (the default) is standard unweighted kappa: every
    disagreement counts the same regardless of category. `"linear"` and
    `"quadratic"` are Cohen's *weighted* kappa for ordinal categories, where
    a judge/human gap of one category should count less than a gap of four
    — categories are sorted (numerically where possible) and a disagreement
    is penalized by its distance along that ordering, squared for
    `"quadratic"`. `"none"` is mathematically the 0/1-distance special case
    of the same formula (see the derivation in this function's tests), so
    all three modes share one implementation.
    """
    n = len(judge_labels)
    if n == 0 or n != len(human_labels):
        raise ValueError("cohens_kappa needs at least one paired (judge, human) label")

    categories = sorted(set(judge_labels) | set(human_labels), key=_category_sort_key)
    k = len(categories)
    if k < 2:
        # Only one category ever appeared, for both raters — trivially
        # perfect agreement by construction (nothing to disagree about).
        return 1.0
    index = {c: i for i, c in enumerate(categories)}

    def weight(i: int, j: int) -> float:
        if weights == "none":
            return 0.0 if i == j else 1.0
        distance = abs(i - j) / (k - 1)
        return distance if weights == "linear" else distance**2

    observed = (
        sum(weight(index[j], index[h]) for j, h in zip(judge_labels, human_labels, strict=True)) / n
    )

    judge_counts = Counter(judge_labels)
    human_counts = Counter(human_labels)
    expected = sum(
        weight(index[c1], index[c2]) * (judge_counts[c1] / n) * (human_counts[c2] / n)
        for c1 in categories
        for c2 in categories
    )

    if expected == 0:
        # Every pair was zero-weight (e.g. both raters always agree, or —
        # for "none" — the same degenerate all-one-category case as above).
        return 1.0

    return 1 - observed / expected


def compute_calibration(
    run_result: RunResult,
    labels: list[HumanLabel],
    metric_name: str,
    *,
    min_labels: int = MIN_LABELS_FOR_CALIBRATION,
) -> CalibrationState:
    """Pair human labels against the judge's own verdicts on the same
    trajectories (looked up from `run_result`), and compute kappa.

    A label whose trajectory isn't in `run_result`, or whose judge verdict
    is unavailable (metric didn't apply, or — for `recovery` — the outcome
    was the deterministic non-judged branch), is silently excluded from the
    pairing: it can't be compared against a judge decision that never
    happened.
    """
    results_by_trajectory = {
        tr.trajectory_id: tr for tr in run_result.trajectory_results if tr.trajectory_id is not None
    }

    pairs: list[tuple[str, str]] = []
    tags_by_pair_index: list[list[str]] = []
    for label in labels:
        if label.metric_name != metric_name:
            continue
        tr = results_by_trajectory.get(label.trajectory_id)
        if tr is None:
            continue
        metric_result = tr.metric_results.get(metric_name)
        if metric_result is None:
            continue
        verdict = judge_verdict(metric_name, metric_result)
        if verdict is None:
            continue
        pairs.append((verdict, label.human_verdict))
        tags_by_pair_index.append(label.tags)

    if not pairs:
        return CalibrationState(is_calibrated=False, kappa=None, n_labels=0, kappa_by_tag={})

    weights = _ORDINAL_METRIC_WEIGHTS.get(metric_name, "none")
    judge_labels = [p[0] for p in pairs]
    human_labels = [p[1] for p in pairs]
    overall_kappa = cohens_kappa(judge_labels, human_labels, weights=weights)

    kappa_by_tag: dict[str, float] = {}
    tags_seen = {tag for tags in tags_by_pair_index for tag in tags}
    for tag in sorted(tags_seen):
        indices = [i for i, tags in enumerate(tags_by_pair_index) if tag in tags]
        if len(indices) < 2:
            continue  # too few labels on this slice to say anything
        kappa_by_tag[tag] = cohens_kappa(
            [judge_labels[i] for i in indices], [human_labels[i] for i in indices], weights=weights
        )

    return CalibrationState(
        is_calibrated=len(pairs) >= min_labels,
        kappa=overall_kappa,
        n_labels=len(pairs),
        kappa_by_tag=kappa_by_tag,
    )
