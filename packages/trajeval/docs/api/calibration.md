# Calibration

Every LLM-judged metric (`query_quality`, `recovery`, `faithfulness`) needs
a reportable judge-vs-human agreement score before you should trust it.
`trajeval label` walks you through hand-labeling a sample of trajectories;
`compute_calibration` turns judge + human labels into Cohen's kappa.

## Agreement

::: trajeval.calibration.kappa.cohens_kappa
::: trajeval.calibration.kappa.compute_calibration

`query_quality` is ordinal (rated 1–5) and uses quadratic-weighted kappa by
default, so a near-miss ("3 vs 4") counts as less disagreement than a far
miss ("1 vs 5"). `recovery` and `faithfulness` are unordered categorical
judgments and use unweighted kappa.

## Human labels

::: trajeval.calibration.labels.HumanLabel
::: trajeval.calibration.labels.load_labels
::: trajeval.calibration.labels.append_label

## Judge verdicts and the labeling session

::: trajeval.calibration.verdicts.judge_verdict
::: trajeval.calibration.cli.run_labeling_session
::: trajeval.calibration.cli.format_trajectory_for_labeling
