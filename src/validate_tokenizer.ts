/**
 * Quick sanity check for the tokenizer (Node). Mocks fetch to read the
 * vendored assets from public/tokenizer, then round-trips a few prompts.
 * Run: npm run build && node dist/validate_tokenizer.js
 */
import { readFileSync } from "node:fs";
import { GPT2Tokenizer } from "./engine/tokenizer.js";

// Mock fetch so tokenizer.load() reads local files instead of HTTP.
(globalThis as unknown as { fetch: unknown }).fetch = async (url: string) => {
  const name = url.split("/").pop()!;
  const text = readFileSync(`public/tokenizer/${name}`, "utf-8");
  return { json: async () => JSON.parse(text), text: async () => text };
};

async function main() {
  const tok = await GPT2Tokenizer.load("/tokenizer");

  const prompts = ["The quick brown fox", "Hello, world!", " leading space"];
  for (const p of prompts) {
    const ids = tok.encode(p);
    const back = tok.decode(ids);
    console.log(`prompt   : ${JSON.stringify(p)}`);
    console.log(`ids      : [${Array.from(ids).join(", ")}]`);
    console.log(`decoded  : ${JSON.stringify(back)}`);
    console.log(`roundtrip: ${back === p ? "OK" : "MISMATCH"}\n`);
  }
}

main();