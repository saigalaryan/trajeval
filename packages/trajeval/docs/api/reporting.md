# Comparing runs and rendering reports

## Comparison (`trajeval compare`)

::: trajeval.compare.MetricDelta
::: trajeval.compare.diff_aggregates
::: trajeval.compare.check_regressions
::: trajeval.compare.format_comparison_table

## HTML reports (`trajeval report`)

::: trajeval.report.render_report

## Serving the web viewer (`trajeval serve`)

No Node.js required at install time — the viewer is prebuilt and bundled as
package data; this just points Python's own `http.server` at it and at your
results directory.

::: trajeval.serve.serve
::: trajeval.serve.ViewerNotBundledError
