"""Tests for trajeval.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from trajeval.adapters import AgentAdapter
from trajeval.config import (
    JudgeProvider,
    MetricConfig,
    RegressionDirection,
    RegressionThreshold,
    TrajevalConfig,
    build_metrics,
    load_config,
    resolve_adapter,
)
from trajeval.judge.client import AnthropicJudgeClient, FakeJudgeClient, OpenAIJudgeClient
from trajeval.metrics.faithfulness import FaithfulnessMetric
from trajeval.metrics.recovery import RecoveryMetric
from trajeval.metrics.retrieval_necessity import RetrievalNecessityMetric

# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_minimal(tmp_path: Path) -> None:
    path = tmp_path / "trajeval.yaml"
    path.write_text(
        "adapter: my_module:MyAdapter\ndataset: datasets/seed/seed.jsonl\n", encoding="utf-8"
    )
    config = load_config(path)
    assert config.adapter == "my_module:MyAdapter"
    assert config.dataset == "datasets/seed/seed.jsonl"
    assert config.metrics == []
    assert config.judge_model == "claude-opus-5"  # default


def test_load_config_full(tmp_path: Path) -> None:
    path = tmp_path / "trajeval.yaml"
    path.write_text(
        """
adapter: my_module:MyAdapter
dataset: datasets/seed/seed.jsonl
judge_model: claude-haiku-4-5
concurrency: 8
metrics:
  - name: retrieval_necessity
  - name: recovery
    judge_model: claude-opus-5
regression_thresholds:
  - metric: retrieval_necessity
    key: necessity_score
    tolerance: 0.05
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.judge_model == "claude-haiku-4-5"
    assert config.concurrency == 8
    assert len(config.metrics) == 2
    assert config.metrics[1].judge_model == "claude-opus-5"
    assert config.regression_thresholds[0].tolerance == 0.05
    assert config.regression_thresholds[0].direction == RegressionDirection.HIGHER_IS_BETTER


def test_load_config_degenerate_not_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "trajeval.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


# ---------------------------------------------------------------------------
# resolve_adapter
# ---------------------------------------------------------------------------


def test_resolve_adapter_class_form() -> None:
    adapter = resolve_adapter("tests.fixtures.sample_adapter:SampleAdapterClass")
    assert isinstance(adapter, AgentAdapter)
    trajectory = adapter.run("q")
    assert trajectory.final_answer == "42"


def test_resolve_adapter_factory_form() -> None:
    adapter = resolve_adapter("tests.fixtures.sample_adapter:sample_adapter_factory")
    assert isinstance(adapter, AgentAdapter)


def test_resolve_adapter_instance_form() -> None:
    adapter = resolve_adapter("tests.fixtures.sample_adapter:sample_adapter_instance")
    assert isinstance(adapter, AgentAdapter)


def test_resolve_adapter_degenerate_missing_colon() -> None:
    with pytest.raises(ValueError, match="module.path:attr"):
        resolve_adapter("tests.fixtures.sample_adapter")


def test_resolve_adapter_degenerate_missing_attribute() -> None:
    with pytest.raises(ValueError, match="no attribute"):
        resolve_adapter("tests.fixtures.sample_adapter:DoesNotExist")


def test_resolve_adapter_degenerate_not_an_adapter() -> None:
    with pytest.raises(TypeError):
        resolve_adapter("tests.fixtures.sample_adapter:NotAnAdapter")


# ---------------------------------------------------------------------------
# build_metrics
# ---------------------------------------------------------------------------


def test_build_metrics_deterministic_only() -> None:
    config = TrajevalConfig(
        adapter="x:y", dataset="d", metrics=[MetricConfig(name="retrieval_necessity")]
    )
    metrics = build_metrics(config)
    assert len(metrics) == 1
    assert isinstance(metrics[0], RetrievalNecessityMetric)


def test_build_metrics_judged_uses_injected_factory() -> None:
    fake_judge = FakeJudgeClient({})
    config = TrajevalConfig(
        adapter="x:y",
        dataset="d",
        metrics=[MetricConfig(name="recovery"), MetricConfig(name="faithfulness")],
    )
    metrics = build_metrics(config, judge_factory=lambda model, label: fake_judge)
    assert len(metrics) == 2
    assert isinstance(metrics[0], RecoveryMetric)
    assert isinstance(metrics[1], FaithfulnessMetric)


def test_build_metrics_per_metric_judge_model_override() -> None:
    calls: list[tuple[str, str]] = []

    def factory(model: str, label: str):
        calls.append((model, label))
        return FakeJudgeClient({})

    config = TrajevalConfig(
        adapter="x:y",
        dataset="d",
        judge_model="claude-haiku-4-5",
        metrics=[MetricConfig(name="recovery", judge_model="claude-opus-5")],
    )
    build_metrics(config, judge_factory=factory)
    assert calls == [("claude-opus-5", "recovery")]


def test_build_metrics_degenerate_empty_list() -> None:
    config = TrajevalConfig(adapter="x:y", dataset="d", metrics=[])
    assert build_metrics(config) == []


def test_build_metrics_unknown_metric_raises() -> None:
    config = TrajevalConfig(adapter="x:y", dataset="d", metrics=[MetricConfig(name="not_a_metric")])
    with pytest.raises(ValueError, match="unknown metric"):
        build_metrics(config)


def test_load_config_judge_provider_defaults_to_anthropic(tmp_path: Path) -> None:
    path = tmp_path / "trajeval.yaml"
    path.write_text(
        "adapter: my_module:MyAdapter\ndataset: datasets/seed/seed.jsonl\n", encoding="utf-8"
    )
    assert load_config(path).judge_provider == JudgeProvider.ANTHROPIC


def test_load_config_judge_provider_openai(tmp_path: Path) -> None:
    path = tmp_path / "trajeval.yaml"
    path.write_text(
        "adapter: my_module:MyAdapter\n"
        "dataset: datasets/seed/seed.jsonl\n"
        "judge_provider: openai\n"
        "judge_model: gpt-4o-mini\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.judge_provider == JudgeProvider.OPENAI
    assert config.judge_model == "gpt-4o-mini"


def test_build_metrics_default_factory_uses_anthropic_by_default(monkeypatch) -> None:
    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: object())
    config = TrajevalConfig(adapter="x:y", dataset="d", metrics=[MetricConfig(name="recovery")])

    metrics = build_metrics(config)

    assert isinstance(metrics[0]._judge, AnthropicJudgeClient)  # noqa: SLF001


def test_build_metrics_openai_provider_uses_openai_judge_client(monkeypatch) -> None:
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: object())
    config = TrajevalConfig(
        adapter="x:y",
        dataset="d",
        judge_provider=JudgeProvider.OPENAI,
        judge_model="gpt-4o-mini",
        metrics=[MetricConfig(name="recovery")],
    )

    metrics = build_metrics(config)

    assert isinstance(metrics[0]._judge, OpenAIJudgeClient)  # noqa: SLF001
    assert metrics[0]._judge._model == "gpt-4o-mini"  # noqa: SLF001


def test_regression_threshold_defaults() -> None:
    t = RegressionThreshold(metric="retrieval_necessity", key="necessity_score")
    assert t.tolerance == 0.0
    assert t.direction == RegressionDirection.HIGHER_IS_BETTER
