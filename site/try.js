/* Qorğan try widget (landing) -- analyses ON THIS DEVICE by default (site/core); POST
   /api/analyze only on the visitor's explicit opt-in (one stateless request; the server stores
   nothing). Its words follow the page language (landing.js fires `qorgan:locale`): tactic names
   come from the reviewed taxonomy and an on-device explanation is re-rendered locally from the
   reviewed templates. A server result is never re-requested on a switch -- that would send the
   text again without a new click -- so its explanation keeps the language it was produced in. */

import { contentLocale, escapeHtml as esc, t } from "./i18n.js";
import { displayName, explain } from "./core/explain.js";
import { markSpans } from "./core/highlight.js";

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
  const tags = r.tags.length
    ? r.tags.map((tag) => `<span class="tw-tag" lang="${content}">${esc(name(tag.id))}&nbsp;·&nbsp;${tag.weight.toFixed(2)}</span>`).join("")
    : `<span class="tw-dim">${esc(tr("landing.tw_none"))}</span>`;
  out.innerHTML =
    `<div class="tw-verdict ${r.flagged ? "tone-oxide" : "tone-moss"}"><b>${esc(tr(r.flagged ? "landing.tw_verdict_scam" : "landing.tw_verdict_clear"))}</b>` +
    `&nbsp;·&nbsp;${esc(tr("landing.tw_risk", { risk: r.risk.toFixed(2), threshold: r.threshold.toFixed(2) }))}</div>` +
    `<div class="tw-tags">${tags}</div>` +
    (r.spans.length ? `<div class="tw-transcript">${markSpans(transcript, r.spans, esc)}</div>` : "") +
    `<p class="tw-reason" lang="${lang}">${esc(text.reason)}</p>` +
    `<p class="tw-dim" lang="${lang}">${esc(text.caveat)} ${esc(text.human_note)}</p>` +
    (lang !== locale() ? `<p class="tw-dim">${esc(tr("landing.tw_explanation_lang", { language: { $: `lang_name.${lang}` } }))}</p>` : "") +
    `<p class="tw-dim">${esc(tr(onServer ? "landing.tw_backend_server" : "landing.tw_backend_device", { backend: r.backend }))}` +
    (r.fallback ? ` ${esc(tr("landing.tw_fallback"))}` : "") +
    `</p>`;
};

const render = () => {
  if (!state) return;
  if (state.kind === "empty") out.innerHTML = `<p class="tw-dim">${esc(tr("landing.tw_empty"))}</p>`;
  else if (state.kind === "error") {
    out.innerHTML = `<div class="tw-verdict tone-oxide">${esc(tr(state.onServer ? "landing.tw_failed_server" : "landing.tw_failed_device", { error: state.error }))}</div>`;
  } else renderResult(state);
  out.hidden = false;
};

// On-device by default: the transcript never leaves the browser.
let runtimePromise = null;
let activeRuntime = null;
const deviceRuntime = () => {
  if (activeRuntime && !activeRuntime.alive()) { // its worker died: the next check starts a new one
    activeRuntime = null;
    runtimePromise = null;
  }
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
      activeRuntime = runtime;
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
    r: { risk: result.risk, flagged: result.risk >= threshold, threshold, backend: result.backend, fallback: false, tags: result.tags, spans: result.attributions },
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
  return {
    onServer: true, transcript, explanation: body.explanation, explanationLocale: lang,
    r: { risk: body.risk, flagged: body.flagged, threshold: body.threshold, backend: body.backend, fallback: body.fallback, tags: body.tags, spans: body.spans },
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
