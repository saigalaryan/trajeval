export function fmtNumber(value: unknown, digits = 3): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(digits);
  }
  return String(value);
}

export function fmtValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return fmtNumber(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
