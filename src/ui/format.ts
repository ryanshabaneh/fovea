/** Top-k over a logits row, highest first. */
export function topK(logits: Float32Array, k: number): { id: number; logit: number }[] {
  const out: { id: number; logit: number }[] = [];
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (out.length < k) {
      out.push({ id: i, logit: v });
      out.sort((a, b) => b.logit - a.logit);
    } else if (v > out[k - 1].logit) {
      out[k - 1] = { id: i, logit: v };
      out.sort((a, b) => b.logit - a.logit);
    }
  }
  return out;
}

/** Numerically stable softmax reduction (max + sum of exp) over a logits row. */
export function softmaxStats(logits: Float32Array): { max: number; sumExp: number } {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sumExp = 0;
  for (let i = 0; i < logits.length; i++) sumExp += Math.exp(logits[i] - max);
  return { max, sumExp };
}

export function prob(logit: number, s: { max: number; sumExp: number }): number {
  return Math.exp(logit - s.max) / s.sumExp;
}

/** Full softmax over a logits row (stable). */
export function softmaxProbs(logits: Float32Array): Float32Array {
  const { max, sumExp } = softmaxStats(logits);
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) out[i] = Math.exp(logits[i] - max) / sumExp;
  return out;
}

/** KL(p‖q) over two distributions of equal length; measures how much q moved. */
export function klDiv(p: Float32Array, q: Float32Array): number {
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 1e-12) kl += p[i] * Math.log(p[i] / Math.max(q[i], 1e-12));
  }
  return Math.max(0, kl);
}

/** Clean token for display: trim GPT-2's leading space, mark blanks. */
export function display(token: string): string {
  const t = token.replace(/^ /, "");
  if (t === "" || /^\s+$/.test(token)) return "␣";
  return t.replace(/\n/g, "⏎").replace(/\t/g, "⇥");
}

export function argmax(a: Float32Array): number {
  let id = 0, best = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > best) { best = a[i]; id = i; }
  return id;
}

export function fmtProb(p: number): string {
  const pct = p * 100;
  return `${pct >= 10 ? pct.toFixed(1) : pct.toFixed(2)}%`;
}

export function fmtLogit(x: number): string {
  return (x >= 0 ? " " : "") + x.toFixed(3);
}

export function fmtMs(ms: number): string {
  return `${ms.toFixed(1)} ms`;
}