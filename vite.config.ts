import { defineConfig, type Plugin } from "vite";
import fs from "node:fs";
import path from "node:path";

/**
 * Serve src/kernels/*.wgsl at /kernels/*.wgsl in dev, and copy them into the
 * build output. Keeps src/kernels as the single source of truth for the WGSL
 * shaders — the engine fetches "/kernels/<name>.wgsl" and needs zero changes.
 */
function wgslKernels(outDir: string): Plugin {
  const dir = path.resolve(process.cwd(), "src/kernels");
  return {
    name: "fovea-wgsl-kernels",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? "").split("?")[0];
        if (url.startsWith("/kernels/") && url.endsWith(".wgsl")) {
          const file = path.join(dir, path.basename(url));
          if (fs.existsSync(file)) {
            res.setHeader("Content-Type", "text/plain; charset=utf-8");
            res.end(fs.readFileSync(file));
            return;
          }
        }
        next();
      });
    },
    closeBundle() {
      const out = path.resolve(process.cwd(), outDir, "kernels");
      fs.mkdirSync(out, { recursive: true });
      for (const f of fs.readdirSync(dir)) {
        if (f.endsWith(".wgsl")) {
          fs.copyFileSync(path.join(dir, f), path.join(out, f));
        }
      }
    },
  };
}

const OUT_DIR = "dist-app"; // NOT dist/ — tsc owns dist/ for the node test.

export default defineConfig({
  // esnext: top-level await + WebGPU used by the engine.
  build: { outDir: OUT_DIR, target: "esnext" },
  plugins: [wgslKernels(OUT_DIR)],
});