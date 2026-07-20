import { el, replace } from "../dom.js";
import { fmtProb, fmtLogit, display } from "../format.js";

export interface DistTok { id: number; token: string; logit: number; prob: number; }
export interface TokenEntry { id: number; text: string; }

export interface ConsoleScene {
  element: HTMLElement;
  getPrompt(): string;
  getBos(): boolean;
  onRun(fn: () => void): void;
  onInput(fn: () => void): void;
  onAttribute(fn: () => void): void;
  setTokens(entries: TokenEntry[]): void;
  setBusy(busy: boolean): void;
  setRunInfo(text: string): void;
  renderDistribution(top: DistTok[]): void;
  focus(): void;
}

const IS_MAC = typeof navigator !== "undefined" && /Mac|iP(hone|ad|od)/.test(navigator.userAgent);
const BOS_ID = 50256;
const MAX_CHIPS = 256;

const chipText = (s: string): string => s.replace(/^ /, "␣").replace(/\n/g, "⏎").replace(/\t/g, "⇥");

/** The working hub - prompt, tokens, and the next-token distribution, all in
 *  one place so you iterate without leaving. */
export function createConsoleScene(defaultPrompt: string): ConsoleScene {
  const textarea = el("textarea", { class: "con-input", rows: "1", spellcheck: "false", autocomplete: "off", "aria-label": "Prompt" }) as HTMLTextAreaElement;
  textarea.value = defaultPrompt;

  const bos = el("input", { type: "checkbox", id: "con-bos" }) as HTMLInputElement;
  bos.checked = true;
  const bosOpt = el("label", { class: "opt", for: "con-bos", title: "Prepend <|endoftext|> (50256)" }, [bos, "prepend BOS"]);

  const tokCount = el("b", {}, ["-"]);
  const strip = el("div", { class: "con-strip", "aria-label": "Input tokens" });
  const runBtn = el("button", { class: "con-run", type: "button" }, ["Run", el("span", { class: "arr" }, ["→"])]) as HTMLButtonElement;
  const runInfo = el("span", { class: "con-runinfo" });
  const dist = el("div", { class: "con-dist" });
  const attr = el("button", { class: "con-attr", type: "button", hidden: "" }, ["head attribution", el("span", { class: "arr" }, ["→"])]) as HTMLButtonElement;

  const element = el("div", { class: "scene console-scene" }, [
    el("div", { class: "con-wrap" }, [
      el("div", { class: "con-top" }, [runInfo]),
      el("div", { class: "con-field" }, [el("span", { class: "con-caret" }, ["›"]), textarea]),
      strip,
      el("div", { class: "con-controls" }, [
        el("div", { class: "con-left" }, [runBtn, el("span", { class: "con-hint" }, [IS_MAC ? "⌘↵" : "Ctrl+↵"])]),
        el("div", { class: "con-right" }, [el("span", { class: "con-tk" }, ["tokens ", tokCount]), bosOpt]),
      ]),
      dist,
      attr,
    ]),
  ]);

  const autosize = (): void => { textarea.style.height = "auto"; textarea.style.height = `${textarea.scrollHeight}px`; };
  textarea.addEventListener("input", autosize);
  requestAnimationFrame(autosize);
  textarea.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); runBtn.click(); }
  });

  const chip = (e: TokenEntry, i: number): HTMLElement =>
    el("span", { class: e.id === BOS_ID ? "tok-chip bos" : "tok-chip", title: `pos ${i} · id ${e.id}` }, [
      el("span", { class: "t" }, [chipText(e.text)]),
      el("span", { class: "tid" }, [String(e.id)]),
    ]);

  return {
    element,
    getPrompt: () => textarea.value,
    getBos: () => bos.checked,
    onRun(fn) { runBtn.addEventListener("click", fn); },
    onInput(fn) { textarea.addEventListener("input", fn); bos.addEventListener("change", fn); },
    onAttribute(fn) { attr.addEventListener("click", fn); },
    setTokens(entries) {
      tokCount.textContent = String(entries.length);
      const chips: (Node | string)[] = entries.slice(0, MAX_CHIPS).map(chip);
      if (entries.length > MAX_CHIPS) chips.push(el("span", { class: "tok-more" }, [`+${entries.length - MAX_CHIPS}`]));
      strip.replaceChildren(...chips);
    },
    setBusy(b) { runBtn.disabled = b; textarea.disabled = b; runBtn.classList.toggle("busy", b); },
    setRunInfo(text) { replace(runInfo, text); },
    renderDistribution(top) {
      const header = el("div", { class: "con-row con-thead" }, [
        el("span", {}, [""]),
        el("span", {}, ["token"]),
        el("span", { class: "r" }, ["prob"]),
        el("span", { class: "r" }, ["id"]),
        el("span", { class: "r" }, ["logit"]),
      ]);
      const rows = top.map((t, i) =>
        el("div", { class: i === 0 ? "con-row is-top" : "con-row" }, [
          el("span", { class: "con-rank" }, [String(i + 1)]),
          el("span", { class: "con-tok" }, [display(t.token)]),
          el("span", { class: "con-prob" }, [fmtProb(t.prob)]),
          el("span", { class: "con-id" }, [String(t.id)]),
          el("span", { class: "con-logit" }, [fmtLogit(t.logit)]),
        ]),
      );
      replace(dist,
        el("div", { class: "con-dist-label" }, ["next-token distribution"]),
        el("div", { class: "con-list" }, [header, ...rows]),
      );
      attr.hidden = false;
    },
    focus() { textarea.focus(); const n = textarea.value.length; textarea.setSelectionRange(n, n); },
  };
}