// Mirrors the *intent* of trajeval.config.RegressionDirection on the Python
// side (packages/trajeval/src/trajeval/config.py) — there's no runtime link
// between the two, so this map has to be kept in sync by hand when a new
// metric's aggregate key uses an inverted scale. Only one exists today:
// termination.mean_excess_steps, where lower is better. Every metric not
// listed here defaults to "higher is better" (1.0 is best), which is true
// for every other metric in this project.

const LOWER_IS_BETTER_KEYS = new Set<string>(["termination.mean_excess_steps"]);

export function isLowerBetter(metricName: string, key: string): boolean {
  return LOWER_IS_BETTER_KEYS.has(`${metricName}.${key}`);
}

/** Sign-aware "did this delta make things better or worse", so UI coloring
 * (compare page, future dashboards) doesn't hardcode "positive = green". */
export function deltaSentiment(
  metricName: string,
  key: string,
  delta: number | null
): "better" | "worse" | "neutral" {
  if (delta === null || delta === 0) return "neutral";
  const improved = isLowerBetter(metricName, key) ? delta < 0 : delta > 0;
  return improved ? "better" : "worse";
}
