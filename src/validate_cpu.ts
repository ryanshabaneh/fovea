/**
 * Run: npm run build && node dist/scripts/validate_cpu.js
 * Confirms the CPU oracle matches TransformerLens golden fixtures key-for-key.
 */
import { validateCpuAgainstGolden, loadCpuWeights } from "./cpu/validate.js";
import { GPT2_SMALL } from "./engine/config.js";

const fixturesDir = "fixtures";
const manifestPath = "dist-weights-f32/manifest.json";
const binPath = "dist-weights-f32/weights.bin";

const weights = loadCpuWeights(manifestPath, binPath);
const report = validateCpuAgainstGolden(fixturesDir, GPT2_SMALL, weights, 1e-3);

console.log(`\nCompared ${report.compared} tensors.`);
if (report.failures.length === 0) {
  console.log("All passed - CPU oracle matches TransformerLens.");
  process.exitCode = 0;
} else {
  console.error(`\n${report.failures.length} failure(s):`);
  for (const f of report.failures) {
    console.error(`  ${f.prompt}/${f.hook}  maxAbs=${f.result.maxAbs.toExponential(2)}`);
  }
  process.exitCode = 1;
}