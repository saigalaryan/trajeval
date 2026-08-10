"""Shared helpers for "does the retrieved context cover what this question
needs" — used by termination, recovery, and query_quality so the three
metrics agree on what "adequate context" means instead of each defining it
slightly differently.

`GoldenRecord.required_doc_ids` (AND-semantics) and `sufficient_doc_ids`
(OR-semantics) are mutually exclusive per the schema (see trajeval.types),
so a golden record uses exactly one mode. `is_retrieval_adequate` handles
both, plus the trivial case of a record with neither set populated.
"""

from __future__ import annotations

from trajeval.types import GoldenRecord, RetrievalStep


def relevant_doc_ids(golden: GoldenRecord) -> set[str]:
    """Every doc id this golden record cares about, regardless of AND/OR mode."""
    return set(golden.required_doc_ids) | set(golden.sufficient_doc_ids)


def step_chunk_ids(step: RetrievalStep) -> set[str]:
    return {chunk.doc_id for chunk in step.chunks}


def is_retrieval_adequate(retrieved_ids: set[str], golden: GoldenRecord) -> bool:
    """True once `retrieved_ids` satisfies this golden record's requirement.

    - OR-semantics (`sufficient_doc_ids` populated): any one id is enough.
    - AND-semantics (`required_doc_ids` populated): every required id must
      be present.
    - Neither populated (including `retrieval_required=False`): trivially
      satisfied — there was nothing that needed retrieving.
    """
    if golden.sufficient_doc_ids:
        return bool(retrieved_ids & set(golden.sufficient_doc_ids))
    if golden.required_doc_ids:
        return set(golden.required_doc_ids).issubset(retrieved_ids)
    return True
