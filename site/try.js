/* Qorğan try widget (landing) -- analyses ON THIS DEVICE by default (site/core); POST
   /api/analyze only on the visitor's explicit opt-in (one stateless request; the server stores
   nothing). Its words follow the page language (landing.js fires `qorgan:locale`): tactic names
   come from the reviewed taxonomy and an on-device explanation is re-rendered locally from the
   reviewed templates. A server result is never re-requested on a switch -- that would send the
   text again without a new click -- so its explanation keeps the language it was produced in. */

import { contentLocale, escapeHtml as esc, t } from "./i18n.js";
import { displayName, explain } from "./core/explain.js";
import { matchCues } from "./core/lexicon.js";

const SAMPLES = {
  scam_ru: "Здравствуйте, это служба безопасности банка. По вашему счёту зафиксирована подозрительная операция. Чтобы спасти деньги, переведите их на безопасный счёт. Никому не говорите об этом звонке и назовите код из сообщения.",
  // The Kazakh demo scene of the citizen page (core/scenarios.json, live_scam_bank_kk), as one text.
  scam_kk: "Сәлеметсіз бе. Бұл банктің қауіпсіздік қызметі. Сіздің картаңыздан күдікті операция тіркелді, жағдай шұғыл. Ешкімге айтпаңыз, бұл құпия операция. SMS-тегі кодты айтыңыз, біз операцияны тоқтатамыз. Содан кейін ақшаны қауіпсіз шотқа аударамыз, реквизиттерін айтамын.",
  legit_ru: "Здравствуйте, это банк. По вашей заявке: карта готова, можете забрать её в отделении с удостоверением. Ничего переводить и называть не нужно, коды никому не сообщайте. Хорошего дня.",
};
// The languages POST /api/analyze explains in (its AnalyzeRequest.locale, ADR D52); others ask for Russian.
const SERVER_LOCALES = ["ru", "kk", "en"];

const input = document.getElementById("twInput");
const go = document.getElementById("twGo");
const out = document.getElementById("twResult");
const where = document.getElementById("twWhere");
const serverBox = document.getElementById("twServer");

const locale = () => document.documentElement.lang || "ru";
const tr = (key, params = null) => t(locale(), key, params);

let config = null; // core/qorgan-config.json: taxonomy names + templates (from the runtime, or fetched)
let state = null; // what the result area shows: {kind: "empty"} | {kind: "error", ...} | {kind: "result", ...}
let busy = null; // the button's label key while working: {key, params}

const loadConfig = async () => {
  if (!config) {
    const res = await fetch("core/qorgan-config.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    config = await res.json();
  }
  return config;
};

// Presentation only: which evidence span is which verbatim cue. The classifier's spans carry no
// tactic id, but its cue evidence is the pure lexicon match (core/lexicon.js, as in core/score.js),
// so re-running it labels the spans. Cues are numbered in text order, like the markers on FIG. 1.
const cuesOf = (transcript, lexiconCues, cfg) =>
  matchCues(transcript, lexiconCues, cfg.taxonomy.hard_signal_ids)
    .map((m) => ({ tactic: m.tacticId, start: m.span.start, end: m.span.end }))
    .sort((a, b) => a.start - b.start)
    .map((c, i) => ({ ...c, n: i + 1 }));
// Without the lexicon (a server result when the weights cannot be fetched): a span inside a larger
// span is a cue, unnumbered by tactic.
const nestedCues = (spans) =>
  spans.filter((s) => spans.some((o) => o !== s && o.start <= s.start && o.end >= s.end && o.end - o.start > s.end - s.start))
    .sort((a, b) => a.start - b.start).map((c, i) => ({ start: c.start, end: c.end, n: i + 1 }));

// The verbatim evidence: whole utterances get the neutral wash, cue phrases the red numbered mark.
const highlight = (text, spans, cues) => {
  const cueKeys = new Set(cues.map((c) => `${c.start}:${c.end}`));
  const outers = spans.filter((s) => !cueKeys.has(`${s.start}:${s.end}`) && s.end <= text.length).sort((a, b) => a.start - b.start);
  const withCues = (from, to) => {
    let html = "";
    let cur = from;
    for (const c of cues) {
      if (c.start < cur || c.start < from || c.end > to) continue;
      const short = c.end - c.start <= 24 ? " is-short" : "";
      html += esc(text.slice(cur, c.start)) + `<mark class="q-cue${short}" data-kind="cue" data-n="${c.n}">${esc(text.slice(c.start, c.end))}</mark>`;
      cur = c.end;
    }
    return html + esc(text.slice(cur, to));
  };
  let html = "";
  let cur = 0;
  for (const o of outers) {
    if (o.start < cur) continue;
    html += withCues(cur, o.start) + `<mark>${withCues(o.start, o.end)}</mark>`;
    cur = o.end;
  }
  return html + withCues(cur, text.length);
};

let lexicon = null; // the cue lexicon, for labelling a server result's spans (fetched once, best effort)
const loadLexicon = async () => {
  if (!lexicon) {
    const res = await fetch("models/weights.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    lexicon = (await res.json()).lexicon.cues;
  }
  return lexicon;
};

const renderWhere = () => {
  if (where) where.textContent = tr(serverBox?.checked ? "landing.tw_where_server" : "landing.tw_where_device");
};
const renderButton = () => {
  go.textContent = tr(busy?.key || "landing.tw_go", busy?.params || null);
};

const renderResult = ({ r, transcript, onServer, raw, explanation, explanationLocale }) => {
  const content = config ? contentLocale(locale(), config.locales) : explanationLocale;
  let lang = explanationLocale;
  let text = explanation;
  if (!onServer && config) {
    lang = content; // pure, local: the reviewed templates in the current language
    text = explain(raw, transcript, lang, config);
  }
  const name = (id) => (config && displayName(id, content, config)) || id;
  const cues = r.cues || nestedCues(r.spans);
  const cueOf = new Map(cues.filter((c) => c.tactic).map((c) => [c.tactic, c]));
  // The tactics as Detail A's rows: verbatim cues first (numbered as in the text), then by weight.
  const sorted = [...r.tags].sort((a, b) => (cueOf.get(a.id)?.n ?? 99) - (cueOf.get(b.id)?.n ?? 99) || b.weight - a.weight);
  const rows = sorted.map((tag) => {
    const c = cueOf.get(tag.id);
    const marker = c ? `<span class="q-marker q-marker--sm" aria-hidden="true">${c.n}</span>` : `<span class="lp-ev-nomark" aria-hidden="true"></span>`;
    const phrase = c ? `<mark class="q-cue">${esc(transcript.slice(c.start, c.end))}</mark>` : `<span class="lp-ev-none">${esc(tr("landing.detail_window"))}</span>`;
    return `<div class="lp-ev-row">${marker}<span class="lp-ev-phrase">${phrase}</span>` +
      `<span class="lp-ev-tactic"><span class="lp-ev-name" lang="${content}">${esc(name(tag.id))}</span><span class="lp-ev-id">${esc(tag.id)}</span></span>` +
      `<span class="ec-bar"><i style="--w:${Math.round(tag.weight * 100)}%"></i></span><span class="ec-val">${tag.weight.toFixed(2)}</span></div>`;
  }).join("");
  // One result sheet: the verdict (the system's band label), the risk of THIS text, the verbatim
  // evidence, the tactic rows, then the folded reason and the model line.
  out.innerHTML =
    `<div class="tw-verdict"><span class="q-band ${r.flagged ? "band-critical" : "band-low"}">${esc(tr(r.flagged ? "landing.tw_verdict_scam" : "landing.tw_verdict_clear"))}</span>` +
    `<p class="tw-readout">${esc(tr("landing.tw_risk", { risk: r.risk.toFixed(2), threshold: r.threshold.toFixed(2) }))}</p></div>` +
    (r.spans.length ? `<div class="tw-transcript" lang="${esc(lang)}">${highlight(transcript, r.spans, cues)}</div>` : "") +
    (rows ? `<div class="lp-ev tw-rows">${rows}</div>` : `<p class="tw-dim">${esc(tr("landing.tw_none"))}</p>`) +
    `<details class="tw-why"><summary>${esc(tr("summary.why"))}</summary><p class="tw-reason" lang="${lang}">${esc(text.reason)}</p></details>` +
    `<p class="tw-dim" lang="${lang}">${esc(text.caveat)} ${esc(text.human_note)}</p>` +
    (lang !== locale() ? `<p class="tw-dim">${esc(tr("landing.tw_explanation_lang", { language: { $: `lang_name.${lang}` } }))}</p>` : "") +
    `<p class="tw-dim tw-model" title="${esc(r.backend)}">${esc(tr(onServer ? "landing.tw_backend_server" : "landing.tw_backend_device"))}` +
    (r.fallback ? ` ${esc(tr("landing.tw_fallback"))}` : "") +
    `</p>`;
};

const render = () => {
  if (!state) return;
  if (state.kind === "empty") out.innerHTML = `<p class="tw-dim">${esc(tr("landing.tw_empty"))}</p>`;
  else if (state.kind === "error") {
    // A failed check is not a scam alarm: ink, not the red verdict tone.
    out.innerHTML = `<div class="tw-verdict tw-error">${esc(tr(state.onServer ? "landing.tw_failed_server" : "landing.tw_failed_device", { error: state.error }))}</div>`;
  } else renderResult(state);
  out.hidden = false;
};

// On-device by default: the transcript never leaves the browser.
let runtimePromise = null;
const deviceRuntime = () => {
  if (!runtimePromise) {
    runtimePromise = import("./core/device.js").then(async ({ createDeviceRuntime }) => {
      const runtime = await createDeviceRuntime({
        onProgress: (p) => {
          if (p.type === "progress" && p.status === "progress" && p.file?.endsWith(".onnx")) {
            busy = { key: "landing.tw_downloading", params: { pct: Math.round(p.progress || 0) } };
            renderButton();
          }
        },
      });
      await runtime.warmup();
      return runtime;
    });
    runtimePromise.catch(() => { runtimePromise = null; }); // a failed download can be retried
  }
  return runtimePromise;
};

const analyzeOnDevice = async (transcript) => {
  const runtime = await deviceRuntime();
  config = runtime.config;
  const lang = contentLocale(locale(), config.locales);
  const { result, explanation, threshold } = await runtime.analyze(transcript, lang);
  return {
    onServer: false, transcript, raw: result, explanation, explanationLocale: lang,
    r: { risk: result.risk, flagged: result.risk >= threshold, threshold, backend: result.backend, fallback: false, tags: result.tags, spans: result.attributions,
      cues: cuesOf(transcript, runtime.weights.lexicon.cues, config) },
  };
};

// The explicit opt-in: one stateless request (for devices that cannot hold the model).
const analyzeOnServer = async (transcript) => {
  const lang = SERVER_LOCALES.includes(locale()) ? locale() : "ru";
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ transcript, locale: lang }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const body = await res.json();
  try { await loadConfig(); } catch { /* tactic names fall back to their ids */ }
  let cues = null;
  try { cues = cuesOf(transcript, await loadLexicon(), config); } catch { /* spans inside spans still read as cues */ }
  return {
    onServer: true, transcript, explanation: body.explanation, explanationLocale: lang,
    r: { risk: body.risk, flagged: body.flagged, threshold: body.threshold, backend: body.backend, fallback: body.fallback, tags: body.tags, spans: body.spans, cues },
  };
};

if (input && go && out) {
  document.querySelectorAll(".tw-chip[data-sample]").forEach((btn) => {
    btn.addEventListener("click", () => {
      input.value = SAMPLES[btn.dataset.sample] || "";
      input.focus();
    });
  });

  go.addEventListener("click", async () => {
    if (busy) return;
    const transcript = input.value.trim();
    if (!transcript) {
      state = { kind: "empty" };
      render();
      input.focus();
      return;
    }
    const onServer = Boolean(serverBox?.checked);
    busy = { key: "landing.tw_running" };
    go.disabled = true;
    out.setAttribute("aria-busy", "true");
    renderButton();
    try {
      state = { kind: "result", ...(await (onServer ? analyzeOnServer(transcript) : analyzeOnDevice(transcript))) };
    } catch (e) {
      state = { kind: "error", onServer, error: String(e?.message || e) };
    } finally {
      busy = null;
      go.disabled = false;
      out.removeAttribute("aria-busy");
      renderButton();
    }
    render();
  });

  serverBox?.addEventListener("change", renderWhere);
  document.addEventListener("qorgan:locale", () => {
    renderWhere();
    renderButton();
    render();
  });
  renderWhere();
  renderButton();
}
