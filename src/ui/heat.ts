// Shared "inferno"-style sequential colormap: near-black → purple → magenta →
// red → orange → warm pale. Used by the boot grid and the ablation heatmap so
// low values recede into the ground and high values glow hot.
const STOPS: [number, [number, number, number]][] = [
  [0.0, [8, 5, 12]],
  [0.16, [46, 15, 66]],
  [0.36, [110, 28, 84]],
  [0.56, [184, 45, 71]],
  [0.72, [226, 74, 51]],
  [0.87, [245, 132, 42]],
  [1.0, [252, 214, 132]],
];

const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

export function inferno(t: number): string {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STOPS.length; i++) {
    if (t <= STOPS[i][0]) {
      const [t0, c0] = STOPS[i - 1];
      const [t1, c1] = STOPS[i];
      const f = (t - t0) / (t1 - t0);
      return `rgb(${Math.round(lerp(c0[0], c1[0], f))},${Math.round(lerp(c0[1], c1[1], f))},${Math.round(lerp(c0[2], c1[2], f))})`;
    }
  }
  return "rgb(252,214,132)";
}