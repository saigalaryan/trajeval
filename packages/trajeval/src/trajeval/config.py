"""trajeval.yaml: the single config file the CLI reads.

Deliberately dumb: it names an adapter, a dataset, and a metric list, and
resolves them into the plain Python objects `trajeval.runner.run()` already
knows how to use. No new abstraction beyond what Phase 1/2 already built —
the CLI is a thin shell around the library, per the project's core design
rule that the library is the product.
"""

from __future__ import annotations

import importlib
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from trajeval.adapters import AgentAdapter
from trajeval.cost import CostTracker
from trajeval.judge.cache import JudgeCache
from trajeval.judge.client import AnthropicJudgeClient, JudgeClient, OpenAIJudgeClient
from trajeval.metrics.base import Metric
from trajeval.metrics.faithfulness import FaithfulnessMetric
from trajeval.metrics.query_quality import QueryQualityMetric
from trajeval.metrics.recovery import RecoveryMetric
from trajeval.metrics.retrieval_necessity import RetrievalNecessityMetric
from trajeval.metrics.termination import TerminationMetric
from trajeval.metrics.trajectory_efficiency import TrajectoryEfficiencyMetric
from trajeval.types import TrajevalModel

DETERMINISTIC_METRIC_NAMES = frozenset(
    {"retrieval_necessity", "trajectory_efficiency", "termination"}
)
JUDGED_METRIC_NAMES = frozenset({"query_quality", "recovery", "faithfulness"})
ALL_METRIC_NAMES = DETERMINISTIC_METRIC_NAMES | JUDGED_METRIC_NAMES


class RegressionDirection(StrEnum):
    """Which way is worse, for a given aggregate key.

    Most metrics in this project use "higher is better" (1.0 is best). The
    one exception is `termination`'s `mean_excess_steps`, where lower is
    better — see trajeval.metrics.termination for why that metric's sign
    convention is inverted from the rest.
    """

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class RegressionThreshold(TrajevalModel):
    """A `trajeval compare` check: does `metric.key` regress beyond `tolerance`?"""

    metric: str
    key: str
    tolerance: float = 0.0
    direction: RegressionDirection = RegressionDirection.HIGHER_IS_BETTER


class JudgeProvider(StrEnum):
    """Which real JudgeClient implementation `judge_model` names a model
    for. Doesn't validate that `judge_model` actually matches — pointing
    ``judge_provider: openai`` at a Claude model name is a config mistake
    that surfaces as a clear error from the OpenAI SDK the first time a
    judge call is made, not something caught here."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class MetricConfig(TrajevalModel):
    name: str
    # Overrides config.judge_model for this metric only. Ignored for
    # deterministic metrics.
    judge_model: str | None = None


class TrajevalConfig(TrajevalModel):
    # "module.path:attr" — a class, factory function, or instance. See
    # resolve_adapter().
    adapter: str
    dataset: str
    metrics: list[MetricConfig] = Field(default_factory=list)
    judge_provider: JudgeProvider = JudgeProvider.ANTHROPIC
    judge_model: str = "claude-opus-5"
    # Plaintext prompts/responses on disk — whatever was in your dataset's
    # questions and retrieved chunks ends up here. The project's own
    # .gitignore excludes the default filename; if you change this path,
    # make sure the new one is excluded too.
    judge_cache_path: str | None = ".trajeval_judge_cache.json"
    concurrency: int = 4
    out_dir: str = "results"
    labels_path: str | None = None
    regression_thresholds: list[RegressionThreshold] = Field(default_factory=list)


def load_config(path: str | Path) -> TrajevalConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: expected a YAML mapping at the top level, got {type(raw).__name__}"
        )
    return TrajevalConfig.model_validate(raw)


def resolve_adapter(import_path: str) -> AgentAdapter:
    """Resolve `"module.path:attr"` into an `AgentAdapter` instance.

    `attr` may be a class (instantiated with no arguments), a zero-arg
    factory function returning an adapter, or an already-constructed
    module-level adapter instance.

    The current working directory is added to `sys.path` first if it isn't
    already there — running the installed `trajeval` console script does
    *not* put the cwd on `sys.path` the way `python -m` or a plain script
    invocation does, so without this, every user's project-local adapter
    module (e.g. `my_agent.py` next to `trajeval.yaml`) would 404 on import.
    """
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    module_name, sep, attr_name = import_path.partition(":")
    if not sep:
        raise ValueError(f"adapter path must be 'module.path:attr', got {import_path!r}")

    module = importlib.import_module(module_name)
    try:
        obj: Any = getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(
            f"{import_path!r}: module {module_name!r} has no attribute {attr_name!r}"
        ) from exc

    if isinstance(obj, type):
        adapter = obj()
    elif callable(obj) and not hasattr(obj, "run"):
        adapter = obj()
    else:
        adapter = obj

    if not isinstance(adapter, AgentAdapter):
        raise TypeError(
            f"{import_path!r} did not resolve to an AgentAdapter (got {type(adapter)!r})"
        )
    return adapter


def build_metrics(
    config: TrajevalConfig,
    *,
    cost_tracker: CostTracker | None = None,
    judge_factory: Any = None,
) -> list[Metric]:
    """Build the configured metric instances.

    `judge_factory(model: str, label: str) -> JudgeClient` overrides how a
    judged metric's judge client is constructed — tests pass a factory that
    returns a `FakeJudgeClient` instead of hitting the real API. Defaults to
    `AnthropicJudgeClient` or `OpenAIJudgeClient` (per `config.judge_provider`)
    wired to `config.judge_cache_path` and `cost_tracker`.
    """
    if judge_factory is None:
        judge_client_cls = (
            OpenAIJudgeClient
            if config.judge_provider == JudgeProvider.OPENAI
            else AnthropicJudgeClient
        )

        def judge_factory(model: str, label: str) -> JudgeClient:
            cache = JudgeCache(config.judge_cache_path) if config.judge_cache_path else None
            return judge_client_cls(
                model=model, cache=cache, cost_tracker=cost_tracker, label=label
            )

    metrics: list[Metric] = []
    for metric_config in config.metrics:
        name = metric_config.name
        if name not in ALL_METRIC_NAMES:
            raise ValueError(f"unknown metric {name!r}. Known metrics: {sorted(ALL_METRIC_NAMES)}")

        if name == "retrieval_necessity":
            metrics.append(RetrievalNecessityMetric())
        elif name == "trajectory_efficiency":
            metrics.append(TrajectoryEfficiencyMetric())
        elif name == "termination":
            metrics.append(TerminationMetric())
        else:
            model = metric_config.judge_model or config.judge_model
            judge = judge_factory(model, name)
            if name == "query_quality":
                metrics.append(QueryQualityMetric(judge))
            elif name == "recovery":
                metrics.append(RecoveryMetric(judge))
            elif name == "faithfulness":
                metrics.append(FaithfulnessMetric(judge))

    return metrics
