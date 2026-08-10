# Core: running, results, types

## Running an evaluation

::: trajeval.runner.run
::: trajeval.runner.load_golden_dataset
::: trajeval.runner.load_run_result
::: trajeval.runner.save_run_result
::: trajeval.runner.recalibrate

## Adapters

::: trajeval.adapters.AgentAdapter
::: trajeval.adapters.CallableAdapter
::: trajeval.adapters.OpenAIToolCallAdapter
::: trajeval.adapters.TrajectoryRecorder
::: trajeval.adapters.parse_openai_messages

## Results schema

::: trajeval.results.RunResult
::: trajeval.results.RunMetadata
::: trajeval.results.TrajectoryResult
::: trajeval.results.CalibrationState

## Trajectory/step types

::: trajeval.types.Trajectory
::: trajeval.types.TrajectoryMetadata
::: trajeval.types.GoldenRecord
::: trajeval.types.Step
::: trajeval.types.ThoughtStep
::: trajeval.types.RetrievalStep
::: trajeval.types.ToolStep
::: trajeval.types.AnswerStep
::: trajeval.types.RetrievedChunk

## Cost tracking

::: trajeval.cost.CostTracker
::: trajeval.cost.estimate_cost_usd
