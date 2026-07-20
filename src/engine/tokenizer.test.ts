/**
 * Tokenizer parity test (Node). Verifies our GPT-2 byte-level BPE produces the
 * EXACT token IDs of the reference tokenizer (openai-community/gpt2) and that
 * decode() round-trips losslessly.
 *
 * Why this exists: the golden GPU/CPU validation feeds *pre-tokenized* fixture
 * IDs, so it never exercised this tokenizer. This test closes that gap. The
 * expected IDs below come from the real HF tokenizer - most are lifted straight
 * from the golden fixtures' own token arrays (BOS stripped).
 *
 * Run: npm test  (build then node dist/engine/tokenizer.test.js)
 */
import { readFileSync } from "node:fs";
import { GPT2Tokenizer } from "./tokenizer.js";

// Read the vendored assets from disk instead of over HTTP.
(globalThis as unknown as { fetch: unknown }).fetch = async (url: string) => {
  const name = url.split("/").pop()!;
  const text = readFileSync(`public/tokenizer/${name}`, "utf-8");
  return { json: async () => JSON.parse(text), text: async () => text };
};

// [input, expected reference IDs]. Newlines and trailing spaces are deliberate.
const CASES: Array<[string, number[]]> = [
  ["The Eiffel Tower is in the city of", [464, 412, 733, 417, 8765, 318, 287, 262, 1748, 286]],
  ["When Mary and John went to the store, John gave a drink to",
    [2215, 5335, 290, 1757, 1816, 284, 262, 3650, 11, 1757, 2921, 257, 4144, 284]],
  ["The quick brown fox jumps over the lazy", [464, 2068, 7586, 21831, 18045, 625, 262, 16931]],
  ["1 2 3 4 5 6 7 8", [16, 362, 513, 604, 642, 718, 767, 807]],
  ["import numpy as np\nimport torch", [11748, 299, 32152, 355, 45941, 198, 11748, 28034]],
  ["Q: What is the capital of France?\nA:", [48, 25, 1867, 318, 262, 3139, 286, 4881, 30, 198, 32, 25]],
  ["ABC ABC ABC ABC AB", [24694, 9738, 9738, 9738, 9564]],
  ["Hello world", [15496, 995]],
  ["My name is", [3666, 1438, 318]],
  ["My name is ", [3666, 1438, 318, 220]],   // trailing space => token 220 (" ")
  [" leading space", [3756, 2272]],
  ["Hello, world!", [15496, 11, 995, 0]],
];

let failures = 0;
function check(name: string, cond: boolean, detail = ""): void {
  if (cond) console.log(`  PASS  ${name}`);
  else { failures++; console.error(`  FAIL  ${name}\n        ${detail}`); }
}

async function main(): Promise<void> {
  const tok = await GPT2Tokenizer.load("/tokenizer");

  for (const [text, expected] of CASES) {
    const ids = Array.from(tok.encode(text));
    const idsOk = ids.length === expected.length && ids.every((v, i) => v === expected[i]);
    check(`encode ${JSON.stringify(text)}`, idsOk, `got [${ids}]  expected [${expected}]`);

    const back = tok.decode(ids);
    check(`roundtrip ${JSON.stringify(text)}`, back === text, `decoded ${JSON.stringify(back)}`);
  }

  check(
    "eot id 50256 decodes to <|endoftext|>",
    tok.decode([50256]) === "<|endoftext|>",
    JSON.stringify(tok.decode([50256])),
  );

  console.log(failures === 0 ? "\nTokenizer matches reference GPT-2." : `\n${failures} tokenizer test(s) FAILED.`);
  process.exitCode = failures === 0 ? 0 : 1;
}

main();