"""Phase 1 end-to-end verification, per the build brief:

Run the harness against a deliberately broken agent — one hardcoded to
always retrieve, regardless of the question — over the real seed dataset,
and confirm it reports a high over-retrieval rate. If it doesn't, the metric
is wrong.

This is the sanity check that the whole vertical slice (types -> adapter ->
metric -> runner -> dataset) actually works end to end, not just in
isolation per module.
"""

from __future__ import annotations

from pathlib import Path

from trajeval.adapters import CallableAdapter
from trajeval.metrics.retrieval_necessity import RetrievalNecessityMetric
from trajeval.runner import load_golden_dataset, run

SEED_DATASET = Path(__file__).parents[3] / "datasets" / "seed" / "seed.jsonl"


def _always_retrieve_agent(question: str) -> dict:
    """A deliberately broken agent: searches no matter what, then answers
    with something plausible-looking regardless of what came back."""
    return {
        "final_answer": "See the attached documentation.",
        "steps": [
            {
                "step_type": "retrieval",
                "query": question,
                "chunks": [{"doc_id": "irrelevant-doc", "text": "not actually relevant"}],
            },
            {"step_type": "answer", "text": "See the attached documentation."},
        ],
    }


def test_always_retrieve_agent_shows_high_over_retrieval_on_seed_dataset() -> None:
    goldens = load_golden_dataset(SEED_DATASET)
    num_parametric = sum(1 for g in goldens if not g.retrieval_required)
    num_retrieval_required = len(goldens) - num_parametric
    assert num_parametric >= 8, (
        "seed dataset should have at least 8 retrieval_required=false records"
    )

    adapter = CallableAdapter(_always_retrieve_agent)
    result = run(adapter, goldens, [RetrievalNecessityMetric()])

    agg = result.aggregate_scores["retrieval_necessity"]

    # Every parametric-knowledge question gets over-retrieved, since this
    # agent searches unconditionally — that's the whole point of the fixture.
    assert agg["over_retrieval"] == num_parametric
    assert agg["under_retrieval"] == 0
    # And it never fails to search when search was actually needed.
    assert agg["correct_search"] == num_retrieval_required
    assert agg["correct_skip"] == 0

    # The over-retrieval rate should be clearly nonzero and match exactly
    # the fraction of the dataset this agent had no business searching on.
    assert agg["over_retrieval_rate"] == num_parametric / len(goldens)
    assert agg["over_retrieval_rate"] > 0.2

    # necessity_score is dragged down by exactly the over-retrieval failures.
    assert agg["necessity_score"] == num_retrieval_required / len(goldens)
    assert agg["necessity_score"] < 1.0


def test_always_skip_agent_shows_high_under_retrieval_on_seed_dataset() -> None:
    """The mirror-image broken agent, for symmetry: never searches, so every
    retrieval-required question should come back as under-retrieval."""
    goldens = load_golden_dataset(SEED_DATASET)
    num_retrieval_required = sum(1 for g in goldens if g.retrieval_required)

    def never_retrieve_agent(question: str) -> dict:
        return {
            "final_answer": "I think I know this.",
            "steps": [{"step_type": "answer", "text": "I think I know this."}],
        }

    adapter = CallableAdapter(never_retrieve_agent)
    result = run(adapter, goldens, [RetrievalNecessityMetric()])
    agg = result.aggregate_scores["retrieval_necessity"]

    assert agg["under_retrieval"] == num_retrieval_required
    assert agg["over_retrieval"] == 0
    assert agg["under_retrieval_rate"] == num_retrieval_required / len(goldens)
    assert agg["under_retrieval_rate"] > 0.5
