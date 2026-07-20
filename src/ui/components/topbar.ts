import { el } from "../dom.js";

const REPO_URL = "https://github.com/ryanshabaneh/fovea";
const GITHUB_SVG =
  '<svg viewBox="0 0 16 16" width="17" height="17" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.65 7.65 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>';

export interface TopBar {
  element: HTMLElement;
  setBusy(busy: boolean): void;
  /** Left-slot nav. null = nothing; text renders "‹ text" and calls onClick. */
  setNav(text: string | null, onClick?: () => void): void;
}

/** Persistent top bar: contextual back-nav, model identifiers, status, source. */
export function createTopBar(): TopBar {
  const nav = el("button", { class: "tb-nav", type: "button", hidden: "" }) as HTMLButtonElement;

  const dot = el("span", { class: "dot" });
  const statusText = el("span", { class: "stxt" }, ["ready"]);
  const status = el("span", { class: "status", "data-state": "ready" }, [dot, statusText]);

  const sep = (): HTMLElement => el("span", { class: "sep" }, ["·"]);
  const sysbar = el("span", { class: "sysbar" }, ["gpt2-small", sep(), "WebGPU", sep(), "f16"]);

  const source = el("a", {
    class: "iconlink", href: REPO_URL, target: "_blank", rel: "noopener noreferrer",
    "aria-label": "Source on GitHub", title: "Source on GitHub",
  });
  source.innerHTML = GITHUB_SVG;

  const element = el("header", { class: "topbar" }, [
    nav,
    el("div", { class: "topbar-spacer" }),
    sysbar,
    status,
    source,
  ]);

  return {
    element,
    setBusy(busy) {
      status.setAttribute("data-state", busy ? "running" : "ready");
      statusText.textContent = busy ? "running" : "ready";
    },
    setNav(text, onClick) {
      nav.onclick = onClick ?? null;
      if (!text) { nav.hidden = true; return; }
      nav.hidden = false;
      nav.replaceChildren(el("span", { class: "bb-mark" }, ["‹"]), el("span", {}, [text]));
    },
  };
}