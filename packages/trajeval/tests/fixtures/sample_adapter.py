"""A tiny adapter module used to test trajeval.config.resolve_adapter's
three supported forms: class, factory function, module-level instance."""

from __future__ import annotations

from trajeval.adapters import CallableAdapter


def _always_answers(question: str) -> dict:
    return {"final_answer": "42", "steps": [{"step_type": "answer", "text": "42"}]}


class SampleAdapterClass(CallableAdapter):
    """Class form: resolve_adapter should instantiate this with no args."""

    def __init__(self) -> None:
        super().__init__(_always_answers)


def sample_adapter_factory() -> CallableAdapter:
    """Factory form: resolve_adapter should call this with no args."""
    return CallableAdapter(_always_answers)


sample_adapter_instance = CallableAdapter(_always_answers)
"""Instance form: resolve_adapter should use this object directly."""


class NotAnAdapter:
    """Instantiable with no args but has no .run — resolve_adapter must reject it."""
