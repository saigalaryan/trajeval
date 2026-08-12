# trajeval

[![CI](https://github.com/saigalaryan03/trajeval/actions/workflows/ci.yml/badge.svg)](https://github.com/saigalaryan03/trajeval/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trajectory-eval.svg)](https://pypi.org/project/trajectory-eval/)
[![Python versions](https://img.shields.io/pypi/pyversions/trajectory-eval.svg)](https://pypi.org/project/trajectory-eval/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/saigalaryan03/trajeval/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://saigalaryan03.github.io/trajeval/)

An evaluation harness for **agentic RAG systems** — retrieval systems where an
LLM agent decides when to search, what to search for, whether to search
again, and when it has enough to answer.

Single-shot RAG evaluators (RAGAS, TruLens, DeepEval, ARES) score the final
answer: one retrieval, one score. `trajeval` scores the *trajectory* — did
the agent search at the right moment, write a good query, recover from a bad
retrieval, or loop unnecessarily. It works against LangGraph, LlamaIndex, raw
tool-calling loops, or a hand-rolled agent; the core library never imports an
agent framework. Results are JSON files on disk — no database, no hosted
backend.

MIT licensed.

## Quickstart (5 minutes)

```bash
# 1. Install
pip install "trajectory-eval[cli]"

# 2. Scaffold a config, an example adapter, and a small example golden
#    dataset (30 records) into your project — enough to try the pipeline
#    end-to-end before writing your own dataset:
trajeval init --with-seed-dataset

# 3. Sanity-check the dataset before spending any time (or, with judged
#    metrics, money) running against it — catches malformed lines, missing
#    fields, and duplicate ids all at once:
trajeval validate datasets/seed/seed.jsonl

# 4. Wire my_agent.py's MyAdapter up to your real agent, then run:
trajeval run --config trajeval.yaml --out results/latest.json

# 5. Render a report you can open in a browser or paste into a PR:
trajeval report results/latest.json --html report.html
```

`trajeval init` scaffolds `my_agent.py` with a stub `MyAdapter` that answers
every question with `"TODO: call your agent here"` — running the quickstart
as-is proves the pipeline works, but the scores will be meaningless until you
replace the stub with a call to your actual agent. See `trajeval.adapters`
for the three ways to wrap an agent: `CallableAdapter` (any function
returning a trajectory-shaped dict), `OpenAIToolCallAdapter` (a function
returning raw OpenAI-style messages), or `TrajectoryRecorder` (instrument an
existing loop a few lines at a time).

## What it measures

**Deterministic** (no LLM judge, no calibration needed):

- `retrieval_necessity` — did the agent search when it should have, and skip
  when it shouldn't? Classifies every trajectory into correct-search,
  correct-skip, over-retrieval, or under-retrieval.
- `trajectory_efficiency` — `min_steps / actual_steps`, plus loop detection
  (near-identical repeated queries).
- `termination` — did the agent stop once it had enough context, or keep
  going past the point of sufficiency?

**LLM-judged** (require a `trajeval.judge.JudgeClient` and should be treated
as informative, not authoritative, until calibrated — see below):

- `query_quality` — a deterministic hit-rate (did the query's results
  contain a relevant doc?) reported *separately* from a judged 1–5
  well-formedness score, never collapsed into one number.
- `recovery` — given a retrieval that returned nothing relevant, did the
  agent reformulate and search again (deterministic), answer anyway from the
  bad context (judged — the most dangerous failure mode in production), or
  correctly abstain (judged)?
- `faithfulness` — is every claim in the final answer supported by the
  retrieved context? Decomposes the answer into atomic claims and verifies
  each one; never a single holistic score.

## Judge calibration

An LLM-judged metric is not trustworthy until you've checked it against
human judgment. `trajeval label` walks you through trajectories from a saved
run, recording your own verdict on the same category set the judge used
(without showing you the judge's answer first, to keep the comparison
honest). `trajeval.calibration.kappa` computes Cohen's kappa between judge
and human — overall, and sliced by golden-record tag, since judges often
fall apart quietly on one language or difficulty slice while looking fine in
aggregate.

Every `RunResult` carries a `CalibrationState` for each judged metric it
used. Until at least 50 labels exist for that metric, it's marked
`is_calibrated: false` — the CLI and the HTML report both surface an
**UNCALIBRATED** badge rather than letting a judged score render as a clean,
trustworthy number.

```bash
trajeval label --run results/latest.json --dataset datasets/seed/seed.jsonl \
  --metric recovery --n 50 --labels labels.jsonl
```

**No judged metric ships pre-calibrated.** The calibration *machinery*
(kappa computation, weighted kappa for ordinal metrics, per-tag slicing,
the `is_calibrated`/`UNCALIBRATED` badge) is built and tested — but
producing a real `CalibrationState` for `query_quality`, `recovery`, or
`faithfulness` requires an actual person hand-labeling ~50 real
trajectories, which hasn't been done for this repo's own example dataset.
Every judged metric you run today will show `UNCALIBRATED` until you (or
your team) run `trajeval label` yourselves against your own judge model and
dataset. Don't trust a judged score you haven't personally calibrated.

## CLI

```
trajeval init                       # scaffold a config + example adapter
trajeval run     --config trajeval.yaml --out results/latest.json
trajeval compare results/baseline.json results/candidate.json
trajeval label   --run results/latest.json --dataset datasets/seed/seed.jsonl --metric recovery
trajeval report  results/latest.json --html report.html
```

## CI

A composite GitHub Action (`.github/actions/trajeval-eval`) runs an
evaluation on every PR, compares it against a cached baseline from `main`,
comments a markdown diff table, and fails the build on a configured
regression. See `.github/workflows/trajeval.yml` for a complete example,
including the no-baseline case (first run on a repo, or a PR with no cached
baseline yet) — the action runs and uploads the candidate result either way,
it just skips the comparison.

## Development

```bash
uv venv && uv pip install -e ".[dev]" --python .venv
ruff format . && ruff check .
mypy --strict src/trajeval
pytest --cov=trajeval
```

## Status

Phases 1–3 of the project (core schema, all seven metrics, judge
calibration, CLI, CI integration, packaging) are built and tested, along
with the web viewer under `apps/web`. Two things remain open, both
deliberately not faked to look further along than they are:

- **No metric has real calibration data yet** — see "Judge calibration"
  above. This needs an actual person labeling trajectories, which is real
  work this repo hasn't had done to it.
- **No public reference benchmark exists** — a small (10–20 record),
  honestly-scored benchmark run against a real model, published alongside
  the repo. This needs real Anthropic/OpenAI API spend to produce, which
  hasn't been authorized/run yet.

Both are tracked, not silently dropped — if you're picking this project up
and want to close either gap, they're the two highest-value next steps.
