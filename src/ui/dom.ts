/** Minimal element builder. Children are appended as text nodes (safe: model
 *  output is never interpreted as HTML) or as existing nodes. */
export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  children: (Node | string)[] = [],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) {
    node.append(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

/** Replace all children of `parent` with `nodes`. */
export function replace(parent: Element, ...nodes: (Node | string)[]): void {
  parent.replaceChildren(...nodes);
}