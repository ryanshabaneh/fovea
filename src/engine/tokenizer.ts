/**
 * GPT-2 byte-level BPE tokenizer. Zero runtime deps: vendor OpenAI's
 * encoder.json (token → id) and vocab.bpe (merge ranks) into
 * public/tokenizer/ as static assets.
 *
 * Note: GPT-2 BPE operates on a reversible bytes→unicode remapping
 * ("bytes_to_unicode") so every byte is a printable character before merging.
 * Skipping it corrupts every token containing a space - leading-space tokens
 * like "Ġthe" are most of the vocabulary.
 */

/**
 * Reversible byte → printable-unicode map. Printable ASCII/Latin bytes map to
 * themselves; the rest map to code points 256+ so no byte is whitespace or a
 * control char during merging. This is GPT-2's "bytes_to_unicode".
 */
function bytesToUnicode(): Map<number, string> {
  const bs: number[] = [];
  const add = (from: string, to: string) => {
    for (let i = from.charCodeAt(0); i <= to.charCodeAt(0); i++) bs.push(i);
  };
  add("!", "~"); add("¡", "¬"); add("®", "ÿ");
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const map = new Map<number, string>();
  for (let i = 0; i < bs.length; i++) map.set(bs[i], String.fromCharCode(cs[i]));
  return map;
}

const BYTE_ENCODER = bytesToUnicode();                                   // byte → char
const BYTE_DECODER = new Map([...BYTE_ENCODER].map(([b, c]) => [c, b])); // char → byte

// GPT-2's contraction-aware pre-tokenization pattern (splits text into pieces).
const PATTERN =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

const TEXT_ENCODER = new TextEncoder();
const TEXT_DECODER = new TextDecoder();

/** Adjacent symbol pairs of a word, keyed "a b" (no symbol contains a space). */
function getPairs(word: string[]): Set<string> {
  const pairs = new Set<string>();
  for (let i = 0; i < word.length - 1; i++) pairs.add(`${word[i]} ${word[i + 1]}`);
  return pairs;
}

export class GPT2Tokenizer {
  private bpeCache = new Map<string, string[]>();

  private constructor(
    private encoder: Map<string, number>,
    private decoder: Map<number, string>,
    private bpeRanks: Map<string, number>,
  ) {}

  static async load(assetBase = "/tokenizer"): Promise<GPT2Tokenizer> {
    const [encJson, bpeText] = await Promise.all([
      (await fetch(`${assetBase}/encoder.json`)).json() as Promise<Record<string, number>>,
      (await fetch(`${assetBase}/vocab.bpe`)).text(),
    ]);
    const encoder = new Map(Object.entries(encJson));
    const decoder = new Map([...encoder].map(([s, i]) => [i, s]));
    const bpeRanks = new Map(
      bpeText.split("\n").slice(1).filter(Boolean).map((line, i) => [line, i] as const),
    );
    return new GPT2Tokenizer(encoder, decoder, bpeRanks);
  }

  /** Greedy lowest-rank pair merging on one pre-token's remapped-char string. */
  private bpe(token: string): string[] {
    const cached = this.bpeCache.get(token);
    if (cached) return cached;

    let word = [...token];
    let pairs = getPairs(word);
    if (pairs.size === 0) return [token];

    for (;;) {
      // pick the mergeable pair with the smallest rank
      let bestPair: string | null = null;
      let bestRank = Infinity;
      for (const p of pairs) {
        const rank = this.bpeRanks.get(p);
        if (rank !== undefined && rank < bestRank) { bestRank = rank; bestPair = p; }
      }
      if (bestPair === null) break; // no more merges apply

      const [first, second] = bestPair.split(" ");
      const merged: string[] = [];
      let i = 0;
      while (i < word.length) {
        const j = word.indexOf(first, i);
        if (j === -1) { merged.push(...word.slice(i)); break; }
        merged.push(...word.slice(i, j));
        i = j;
        if (word[i] === first && i < word.length - 1 && word[i + 1] === second) {
          merged.push(first + second);
          i += 2;
        } else {
          merged.push(word[i]);
          i += 1;
        }
      }
      word = merged;
      if (word.length === 1) break;
      pairs = getPairs(word);
    }

    this.bpeCache.set(token, word);
    return word;
  }

  encode(text: string): Uint32Array {
    const ids: number[] = [];
    for (const piece of text.match(PATTERN) ?? []) {
      // UTF-8 encode this piece, remap each byte to a printable char
      let remapped = "";
      for (const b of TEXT_ENCODER.encode(piece)) remapped += BYTE_ENCODER.get(b)!;
      // BPE-merge, then look up each resulting chunk's id
      for (const chunk of this.bpe(remapped)) {
        const id = this.encoder.get(chunk);
        if (id === undefined) throw new Error(`Unknown BPE token: ${chunk}`);
        ids.push(id);
      }
    }
    return new Uint32Array(ids);
  }

  decode(ids: ArrayLike<number>): string {
    // ids → remapped-char string
    let remapped = "";
    for (let i = 0; i < ids.length; i++) {
      const tok = this.decoder.get(ids[i]);
      if (tok !== undefined) remapped += tok;
    }
    // invert the byte remap → raw bytes → UTF-8 decode
    const bytes: number[] = [];
    for (const ch of remapped) {
      const b = BYTE_DECODER.get(ch);
      if (b !== undefined) bytes.push(b);
    }
    return TEXT_DECODER.decode(new Uint8Array(bytes));
  }

  static readonly END_OF_TEXT = 50256;
}