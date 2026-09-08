// Where the API lives, and the header that gets past its run guard.
//
// Empty in development: Vite proxies /runs, /subjects and /uploads to the local
// API (see vite.config.js), so relative paths stay same-origin and no CORS is
// involved. In a deployment the two are separate hosts, and VITE_API_BASE is
// set to the API's public URL at build time.
//
// This is deliberately not done with Render rewrite routes. Those would need
// the API's hostname written into render.yaml, which is a guess — Render
// appends a suffix when a service name is taken — and Render's docs say a
// rewrite to an absolute URL on another host behaves as a redirect rather than
// a proxy, so the browser ends up cross-origin either way.
const BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export function apiUrl(path) {
  return `${BASE}${path}`;
}

// Present only where the deployment sets one. It travels in the bundle, so it
// is public: it stops a crawler finding the endpoint, not a determined caller.
// The server's daily cap is what protects the search budget.
export function runHeaders(extra = {}) {
  const token = import.meta.env.VITE_RUN_TOKEN;
  return token ? { ...extra, "X-Run-Token": token } : { ...extra };
}
