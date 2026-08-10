import type { NextConfig } from "next";

// Static export: the app has no API routes, server actions, or ISR — every
// page is a client component reading a RunResult JSON the user loads
// themselves (drag-and-drop or file picker). That means a static export is
// a complete, correct production build, not a reduced one — `next build`
// emits plain HTML/JS/CSS into `out/` that any static file server can host.
// This is also what `trajeval serve` should point at for "local mode": run
// the export once, then serve `out/` (e.g. `npx serve out`) pointed at a
// results directory — no Node server, no backend, per the project's
// no-hosted-backend design rule.
const nextConfig: NextConfig = {
  output: "export",
};

export default nextConfig;
