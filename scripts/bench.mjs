// Run bench.html in desktop Chrome and save the results.
//
//   node scripts/bench.mjs <label> [T=8,64,256,1024]
//
// Starts the Vite dev server, opens Chrome on a dedicated profile (so the
// 248 MB weights stay in its Cache API between runs), waits for the page to
// POST its results, and writes bench-results/<label>.json.
// Override the browser with CHROME=/path/to/chrome.
import { createServer } from "vite";
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const label = process.argv[2] ?? "unlabeled";
const Ts = process.argv[3] ?? "8,64,256,1024";
const chrome = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const profileDir = path.resolve("node_modules/.cache/fovea-bench-chrome");
const outDir = path.resolve("bench-results");
const PORT = 5199;

let resolveResult;
const result = new Promise((r) => (resolveResult = r));

const server = await createServer({
  server: { port: PORT, strictPort: true },
  logLevel: "warn",
  plugins: [{
    name: "fovea-bench-collector",
    configureServer(s) {
      s.middlewares.use("/__bench", (req, res) => {
        let body = "";
        req.on("data", (c) => (body += c));
        req.on("end", () => { res.end("ok"); resolveResult(JSON.parse(body)); });
      });
    },
  }],
});
await server.listen();

const url = `http://localhost:${PORT}/bench.html?label=${encodeURIComponent(label)}&T=${Ts}`;
console.log(`bench: ${label} → ${url}`);
const browser = spawn(chrome, [
  `--user-data-dir=${profileDir}`,
  "--no-first-run",
  "--no-default-browser-check",
  "--enable-webgpu-developer-features", // full-resolution timestamp queries
  "--disable-background-timer-throttling",
  "--disable-renderer-backgrounding",
  "--disable-backgrounding-occluded-windows",
  "--new-window",
  url,
], { stdio: "ignore" });

const timeout = setTimeout(() => resolveResult({ label, error: "timed out after 15 min" }), 15 * 60 * 1000);
const data = await result;
clearTimeout(timeout);
browser.kill();
await server.close();

fs.mkdirSync(outDir, { recursive: true });
const file = path.join(outDir, `${label}.json`);
fs.writeFileSync(file, JSON.stringify(data, null, 2));
if (data.error) { console.error(data.error); process.exit(1); }
for (const [T, m] of Object.entries(data.model)) {
  console.log(`T=${T.padStart(4)}  e2e ${m.e2eMs.median.toFixed(2).padStart(8)} ms  ablated ${m.ablatedE2eMs.median.toFixed(2).padStart(8)} ms  gpu ${m.gpuMs.toFixed(2).padStart(8)} ms  logits ${m.hashes.logits}`);
}
console.log(`wrote ${path.relative(process.cwd(), file)}`);
