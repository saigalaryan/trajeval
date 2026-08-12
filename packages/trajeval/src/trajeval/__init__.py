"""trajeval: an evaluation harness for agentic RAG systems.

Scores the trajectory an agent takes to an answer, not just the answer.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # Reads from installed package metadata rather than duplicating the
    # version string here and in pyproject.toml (those two would drift).
    # Keyed on the PyPI *distribution* name ("trajectory-eval"), which is
    # deliberately different from this importable module name — see
    # pyproject.toml's [project] comment for why.
    __version__ = version("trajectory-eval")
except PackageNotFoundError:  # pragma: no cover - only when genuinely not installed at all
    __version__ = "0.0.0+unknown"

from trajeval.adapters import (
    AgentAdapter,
    CallableAdapter,
    OpenAIToolCallAdapter,
    TrajectoryRecorder,
    parse_openai_messages,
)
from trajeval.cost import CostTracker, estimate_cost_usd
from trajeval.results import CalibrationState, RunMetadata, RunResult, TrajectoryResult
from trajeval.runner import (
    load_golden_dataset,
    load_run_result,
    recalibrate,
    run,
    save_run_result,
)
from trajeval.types import (
    SCHEMA_VERSION,
    AnswerStep,
    GoldenRecord,
    RetrievalStep,
    RetrievedChunk,
    Step,
    StepType,
    ThoughtStep,
    ToolStep,
    Trajectory,
    TrajectoryMetadata,
)

__all__ = [
    "SCHEMA_VERSION",
    "__version__",
    "AgentAdapter",
    "AnswerStep",
    "CalibrationState",
    "CallableAdapter",
    "CostTracker",
    "GoldenRecord",
    "OpenAIToolCallAdapter",
    "RetrievalStep",
    "RetrievedChunk",
    "RunMetadata",
    "RunResult",
    "Step",
    "StepType",
    "ThoughtStep",
    "ToolStep",
    "Trajectory",
    "TrajectoryMetadata",
    "TrajectoryRecorder",
    "TrajectoryResult",
    "estimate_cost_usd",
    "load_golden_dataset",
    "load_run_result",
    "parse_openai_messages",
    "recalibrate",
    "run",
    "save_run_result",
]
