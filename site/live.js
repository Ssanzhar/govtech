/* Qorğan live call -- the citizen page. Replay and microphone both run ON THIS DEVICE
   (site/core: the classifier in a worker, speech recognition in Vosklet/WASM -- PLAN B9);
   neither sends audio or text anywhere. Both end in a post-call summary with the
   consent-gated report review (POST /api/reports is the only content-carrying request).

   Language: one control (header) drives the whole page. Nothing on screen is a string
   frozen at the moment it was produced: chrome carries `data-i18n*` keys (site/i18n.js), and
   model content (tactic names, advice, the summary) is rendered from state -- `call`,
   `lastState` -- in the current language. So switching mid-call re-renders everything in
   place, the session keeps running, and what the citizen typed in the review is untouched.
   The model's own text (advice, tactic names, templates) comes from the reviewed YAML in
   kk / ru / en (ADR D52); `contentLocale` falls back to Russian only for a chrome-only locale.
   The chosen language is shared with the landing page (i18n-dom.js). */

import { CONSENT_VERSION, DEFAULT_LOCALE, LOCALES, contentLocale, escapeHtml as esc, hasKey, pickLocale, t } from "./i18n.js";
import { browserLanguages, localize as localizeIn, readStoredLocale, renderEl as renderElIn, storeLocale } from "./i18n-dom.js";
import { displayName, renderReason } from "./core/explain.js";
import { recommend } from "./core/recommend.js";
import { alignTokens, evidenceRanges, overlaps, runsOf } from "./core/word-runs.js";

const $ = (id) => document.getElementById(id);
const scenarioSel = $("lvScenario");
const scriptArea = $("lvScript");
const startBtn = $("lvStart");
const setupNote = $("lvSetupNote");
const setupReplay = $("lvSetupReplay");
const setupMic = $("lvSetupMic");
const modeNote = $("lvModeNote");
const micChip = $("lvMicChip");
const micStartBtn = $("lvMicStart");
const micStopBtn = $("lvMicStop");
const micNote = $("lvMicNote");
const dlReplay = $("lvDlReplay");
const dlMicSpeech = $("lvDlMicSpeech");
const dlMicModel = $("lvDlMicModel");
const dlMicEndpoint = $("lvDlMicEndpoint");
const endpointNote = $("lvEndpointNote");
const voiceModeNote = $("lvVoiceModeNote");
const liveEl = $("lvLive");
const liveWordsEl = $("lvLiveWords");
const callWrap = $("lvCallWrap");
const dot = $("lvDot");
const headEl = $("lvHead");
const scoreEl = $("lvScore");
const meterEl = $("lvMeter");
const meterFill = $("lvMeterFill");
const bandPill = $("lvBandPill");
const tagsEl = $("lvTags");
const adviceEl = $("lvAdvice");
const adviceText = $("lvAdviceText");
const transcriptEl = $("lvTranscript");
const partialEl = $("lvPartial");
const summaryWrap = $("lvSummaryWrap");
const summaryContent = $("lvSummaryContent");
const reportEl = $("lvReport");

// Long enough to watch the meter move turn by turn, short enough to stay demo-paced.
const TURN_DELAY_MS = 1200;
// One-time downloads, disclosed before they start (site/models/): e5-base int8 ONNX 279 MB +
// tokenizer 17 MB; the two Vosk tarballs 60 MB (kk) + 46 MB (ru).
const MODEL_MB = 300;
const SPEECH_MB = 147; // kk 60 + ru 46 + en 41 MB tarballs (ADR D62)
const ENDPOINT_MB = 239; // GigaAM 225 MB + its sherpa-onnx runtime 14 MB (ADR D64), downloaded once
// `live.html?debug=asr` keeps a per-utterance trace on window.__qorganVoice (evidence runs, D64).
const VOICE_TRACE = new URLSearchParams(location.search).get("debug") === "asr";
const ENDPOINT_STATUS = { loading: "mic.endpoint_loading", ready: "mic.endpoint_ready", unavailable: "mic.endpoint_unavailable" };
// The light mode (ADR D67) has no quick transcript to fall back on, so its loading notes differ.
const LIGHT_STATUS = { loading: "mic.light_loading", ready: "mic.light_ready" };
// `live.html?voice=full|light` forces a voice mode for evidence runs (ADR D67); see voiceMode().
const VOICE_OVERRIDE = new URLSearchParams(location.search).get("voice");
// The summary's "why" sentence quotes the first few flagged phrases; all of them are already
// highlighted in the transcript above, and quoting every one turns the reason into a wall.
const MAX_REASON_PHRASES = 3;
const ASR_MODELS_BASE = "models/"; // config urls are relative to site/models/
const MIC_REASON = {
  isolation: "mic.reason_browser",
  shared_memory: "mic.reason_browser",
  secure_context: "mic.reason_browser",
  audio_worklet: "mic.reason_browser",
  capture: "mic.reason_capture",
  phone: "mic.reason_phone",
  wasm: "mic.reason_browser",
};
const ASR_STATUS = { runtime: "mic.loading_runtime", model: "mic.loading_model", model_skipped: "mic.model_skipped", listening: "mic.listening", stopped: "mic.stopped" };

// ── language ─────────────────────────────────────────────────────────────────

let locale = pickLocale({ stored: readStoredLocale(), languages: browserLanguages() });
let config = null; // core/qorgan-config.json, once the device runtime exists
const cl = () => contentLocale(locale, config?.locales); // the language the model's content is shown in
const tr = (key, params = null) => t(locale, key, params);

const errText = (e) => String(e?.message || e);

/** A failure whose message is a page string, so it re-renders when the language changes. */
class UiError extends Error {
  constructor(key, params = null) {
    super(key);
    this.key = key;
    this.params = params;
  }
}
const reasonOf = (e) => (e instanceof UiError ? { $: e.key, params: e.params } : errText(e));

/** Render one element / a subtree from its data-i18n* keys (i18n-dom.js) in the current language. */
const renderEl = (el) => renderElIn(el, locale);
const localize = (root = document) => localizeIn(root, locale);

/** Bind an element to a page string (text) -- it stays bound across language switches. */
const bind = (el, key, params = null) => {
  if (!el) return;
  delete el.dataset.i18nHtml;
  if (!key) {
    delete el.dataset.i18n;
    delete el.dataset.i18nParams;
    el.textContent = "";
    return;
  }
  el.dataset.i18n = key;
  if (params) el.dataset.i18nParams = JSON.stringify(params);
  else delete el.dataset.i18nParams;
  renderEl(el);
};

const setNote = (el, key, params = null, isError = false) => {
  bind(el, key, params);
  el?.classList.toggle("is-error", Boolean(isError));
};

const tacticName = (id) => (config && displayName(id, cl(), config)) || id;

/** Tactic chips in the report review carry only their id; the name follows the language. */
const renderTactics = (root = document) => {
  root.querySelectorAll("[data-tactic]").forEach((el) => {
    el.textContent = tacticName(el.dataset.tactic);
    el.lang = cl();
  });
};

const renderFallbackNotes = () => {
  document.querySelectorAll("[data-fallback]").forEach((el) => { el.hidden = locale === cl(); });
};

const applyLocale = (next, { persist = true } = {}) => {
  locale = LOCALES.includes(next) ? next : DEFAULT_LOCALE;
  if (persist) storeLocale(locale);
  document.documentElement.lang = locale;
  document.querySelectorAll('input[name="lv-lang"]').forEach((radio) => { radio.checked = radio.value === locale; });
  // A session in progress continues in the new language: the session locale only selects
  // advice text (core/session.js); scoring and the meter never read it.
  if (liveState) liveState = { ...liveState, locale: cl() };
  if (micState) micState = { ...micState, locale: cl() };
  localize(document);
  renderTactics(document);
  renderCall();
  renderSummaryContent();
  renderFallbackNotes();
};

// ── model download disclosure ────────────────────────────────────────────────

let modelCached = false;
let voice = { mode: "full", issues: [], downgradedBy: [] }; // voiceMode() of this browser (ADR D67), set at start-up

/** Is the analysis model already stored on this device (transformers.js / service-worker cache)? */
const detectModelCache = async () => {
  try {
    if (!("caches" in globalThis)) return false;
    for (const name of await caches.keys()) {
      if (name.startsWith("qorgan-gigaam-")) continue; // the speech model's own store (ADR D64), not the analysis model
      const requests = await (await caches.open(name)).keys();
      if (requests.some((req) => /\/models\/.+\.onnx$/.test(new URL(req.url).pathname))) return true;
    }
  } catch { /* storage blocked: assume a download, which is the safe thing to disclose */ }
  return false;
};

const renderDownloadNotes = () => {
  bind(dlReplay, modelCached ? "download.model_cached" : "download.model_first", { mb: MODEL_MB });
  bind(dlMicModel, "download.model_first", { mb: MODEL_MB });
  dlMicModel.hidden = modelCached;
  if (voice.mode === "light") {
    // GigaAM is the only speech model in the light mode (no Vosk tarballs).
    bind(dlMicSpeech, "download.light_first", { mb: ENDPOINT_MB });
    dlMicEndpoint.hidden = true;
    return;
  }
  bind(dlMicSpeech, "download.speech_first", { mb: SPEECH_MB });
  bind(dlMicEndpoint, "download.endpoint_first", { mb: ENDPOINT_MB });
  dlMicEndpoint.hidden = false;
};

// ── device runtime ───────────────────────────────────────────────────────────

let runtime = null;
let runtimePromise = null;
let progressNote = setupNote; // the note of the mode that asked for the model

const deviceRuntime = (noteEl = null) => {
  if (noteEl) progressNote = noteEl;
  if (!runtimePromise) {
    runtimePromise = (async () => {
      const { createDeviceRuntime } = await import("./core/device.js");
      const rt = await createDeviceRuntime({
        onProgress: (p) => {
          if (p.type === "progress" && p.status === "progress" && p.file?.endsWith(".onnx")) {
            setNote(progressNote, "download.progress", { pct: Math.round(p.progress || 0), mb: MODEL_MB });
          }
        },
      });
      config = rt.config;
      try {
        setNote(progressNote, "model.preparing");
        await rt.warmup();
      } catch (e) {
        rt.dispose();
        throw e;
      }
      runtime = rt;
      modelCached = true;
      renderDownloadNotes();
      setNote(progressNote, "model.ready");
      return rt;
    })();
    runtimePromise.catch(() => { runtimePromise = null; }); // a failed download can be retried
  }
  return runtimePromise;
};

// ── the live panel (rendered from `call`) ─────────────────────────────────────

let running = false;
let call = null; // {turn, score, band, tags, advice: {tags, confidence} | null}
let lastState = null; // the finished on-device session: summary + report
let liveState = null; // the replay session in progress
let reportDraft = null; // the draft the review started from; edits can only remove from it

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const mode = () => document.querySelector('input[name="lv-mode"]:checked')?.value || "replay";

const bandParams = (band) => ({ band: { $: `band.${band}` }, desc: { $: `band.${band}_desc` } });

const setBand = (band) => {
  const cls = `band-${band}`;
  meterFill.className = `lv-meter-fill ${cls}`;
  bandPill.className = `lv-band-pill mono ${cls}`;
  bind(bandPill, "band.pill", bandParams(band));
  dot.className = band === "low" ? "p-dot tone-moss" : "p-dot";
};

const renderCall = () => {
  if (!call) return;
  const score = call.score.toFixed(0);
  bind(headEl, "call.head", { n: call.turn });
  scoreEl.textContent = `${score} / 100`;
  meterFill.style.width = `${Math.min(100, Math.max(0, call.score))}%`;
  meterEl.setAttribute("aria-valuenow", score);
  meterEl.setAttribute("aria-valuetext", tr("call.meter_valuetext", { score, band: { $: `band.${call.band}` } }));
  setBand(call.band);
  tagsEl.lang = cl();
  tagsEl.innerHTML = call.tags.map((tag) => `<span class="tw-tag" role="listitem">${esc(tacticName(tag.id))}</span>`).join("");
  if (call.advice && config) {
    // The same pure `recommend` the session ran, re-run in the current language.
    const [first] = recommend(call.advice.tags, cl(), call.advice.confidence, config).advices;
    adviceEl.hidden = !first;
    adviceText.lang = cl();
    adviceText.textContent = first || "";
  } else {
    adviceEl.hidden = true;
  }
};

// `evidence` is plain phrase text (verbatim substrings of `line`, per the API contract --
// no character offsets), so highlighting is a straight escaped-substring replace.
const highlightLine = (line, evidence) => {
  let html = esc(line);
  for (const phrase of evidence) {
    if (!phrase) continue;
    const escaped = esc(phrase);
    if (html.includes(escaped)) html = html.replace(escaped, `<mark>${escaped}</mark>`);
  }
  return html;
};

/** Words grouped into same-language runs, each run led by a language chip (ADR D64).
    `token(item)` renders one word's inner HTML; `cls(item)` adds per-word classes. */
const runsHtml = (items, token, cls = () => "") =>
  runsOf(items)
    .map((run) =>
      `<span class="lv-run" data-lang="${esc(run.lang)}" lang="${esc(run.lang)}">` +
      `<span class="lv-run-tag" data-i18n="lang_short.${esc(run.lang)}"></span>` +
      run.items.map((item) => `<span class="lv-w${cls(item)}">${token(item)}</span>`).join(" ") +
      `</span>`)
    .join(" ");

/** The transcript line: per-word language runs when the words line up with the text, the
    evidence phrases highlighted either way. */
const lineHtml = (line, evidence, words) => {
  const tokens = words?.length ? alignTokens(line, words) : null;
  if (!tokens) return highlightLine(line, evidence);
  const ranges = evidenceRanges(line, evidence);
  return runsHtml(tokens, (t) => (overlaps(t, ranges) ? `<mark>${esc(t.text)}</mark>` : esc(t.text)));
};

const appendLine = (line, evidence, meta = null) => {
  const div = document.createElement("div");
  div.className = "lv-line";
  div.innerHTML = lineHtml(line, evidence, meta?.words);
  if (meta) {
    div.lang = meta.language;
    div.dataset.source = meta.source || "vosk";
    const tag = document.createElement("span");
    tag.className = "lv-line-meta";
    div.append(" ", tag);
    if (meta.source === "gigaam") {
      bind(tag, "call.line_meta_src", { language: { $: `lang_short.${meta.language}` }, source: { $: "call.source_gigaam" } });
    } else {
      bind(tag, "call.line_meta", { language: { $: `lang_short.${meta.language}` }, pct: (meta.confidence * 100).toFixed(0) });
    }
  }
  localize(div); // the run chips
  transcriptEl.appendChild(div);
};

const showPartial = (text) => {
  bind(partialEl, null); // drop a light-mode activity key, if any
  partialEl.hidden = !text;
  partialEl.textContent = text || "";
};

/** Light mode (ADR D67): no words while the person speaks, only what is happening. */
const showActivity = (key) => {
  bind(partialEl, key);
  partialEl.hidden = !key;
};

/** Provisional words while the person speaks (the span vote, ADR D64): a word stays marked
    provisional until its stretch is frozen. In the fallback the committed line keeps these words
    and tags (ADR D70); GigaAM's text, when it is ready, replaces them. */
const showLiveWords = (words) => {
  liveEl.hidden = !words?.length;
  if (!words?.length) {
    liveWordsEl.innerHTML = "";
    return;
  }
  liveWordsEl.innerHTML = runsHtml(words, (w) => esc(w.word), (w) => (w.frozen ? "" : " is-provisional"));
  localize(liveWordsEl);
};

/** One committed utterance -> meter, band, tags, advice, transcript line. */
const applyUpdate = (update, state, lineText, meta = null) => {
  const latchedAdvice = update.meter.latched && update.recommendation.advices.length > 0;
  call = {
    turn: update.meter.turn_index,
    score: update.meter.score,
    band: update.band,
    tags: state.tags,
    advice: latchedAdvice ? { tags: state.tags, confidence: update.result.raw_confidence } : call?.advice ?? null,
  };
  appendLine(lineText, update.new_evidence.map((span) => span.text), meta);
  renderCall();
};

const resetCallUi = () => {
  summaryWrap.hidden = true;
  summaryContent.innerHTML = "";
  reportEl.innerHTML = "";
  transcriptEl.innerHTML = "";
  showPartial("");
  showLiveWords([]);
  lastState = null;
  reportDraft = null;
  call = { turn: 0, score: 0, band: "low", tags: [], advice: null };
  renderCall();
  callWrap.hidden = false;
  renderFallbackNotes();
};

// ── post-call summary (rendered from `lastState`) ─────────────────────────────

const renderSummaryContent = () => {
  if (!lastState || !runtime) return;
  const content = cl();
  const summary = runtime.summarize({ ...lastState, locale: content });
  const templates = config.templates[content];
  const reason = renderReason(templates, summary.tactic_names, lastState.seen_evidence.slice(0, MAX_REASON_PHRASES));
  const chips = summary.tactic_names.length
    ? summary.tactic_names.map((n) => `<span class="tw-tag" role="listitem">${esc(n)}</span>`).join("")
    : `<span class="tw-dim">${esc(tr("summary.none"))}</span>`;
  const actions = summary.recommended_actions.length
    ? `<ul class="lv-summary-actions" lang="${content}">${summary.recommended_actions.map((a) => `<li>${esc(a)}</li>`).join("")}</ul>`
    : `<p class="lv-summary-tip">${esc(tr("summary.safe_tip"))}</p>`;
  summaryContent.innerHTML =
    `<p class="eyebrow inverse lv-summary-eyebrow">${esc(tr("summary.eyebrow"))}</p>` +
    `<div class="lv-summary-head"><span class="lv-summary-score">${summary.final_score.toFixed(0)} / 100</span></div>` +
    `<span class="lv-band-pill mono lv-summary-band band-${esc(summary.band)}">${esc(tr("band.pill", bandParams(summary.band)))}</span>` +
    `<p class="lv-summary-label mono">${esc(tr("summary.signs"))}</p>` +
    `<div class="lv-tags lv-summary-tags mono" role="list" lang="${content}">${chips}</div>` +
    `<p class="lv-summary-label mono">${esc(tr("summary.why"))}</p>` +
    `<p class="lv-summary-reason" lang="${content}">${esc(reason)}</p>` +
    `<p class="lv-summary-label mono">${esc(tr("summary.actions"))}</p>` +
    actions +
    (locale === content ? "" : `<p class="lv-fallback mono">${esc(tr("call.content_fallback"))}</p>`) +
    `<p class="lv-summary-note" lang="${content}">${esc(summary.human_note)} ${esc(templates.caveat)}</p>`;
};

const showSummary = () => {
  summaryWrap.hidden = false;
  renderSummaryContent();
  reportEl.innerHTML =
    `<p class="lv-report-title" data-i18n="report.title"></p>` +
    `<p class="lv-report-note" data-i18n="report.intro"></p>` +
    `<button type="button" id="lvReportOpen" class="btn btn-ghost" data-i18n="report.open"></button>` +
    `<div id="lvReportForm" class="lv-report-form" hidden></div>`;
  localize(reportEl);
  $("lvReportOpen").addEventListener("click", () =>
    openReportReview().catch((e) => {
      const form = $("lvReportForm");
      form.hidden = false;
      form.innerHTML = `<p class="lv-report-note is-error"></p>`;
      bind(form.firstElementChild, "report.prepare_failed", { error: reasonOf(e) });
    })
  );
};

// ── report review (task.md §8: never automatic, reviewed, editable; ADR D44) ──────

const openReportReview = async () => {
  if (!lastState) return;
  const openBtn = $("lvReportOpen");
  const form = $("lvReportForm");
  const rt = await deviceRuntime();
  const { reviewDraft, scrubText } = await import("./core/report.js");
  reportDraft = rt.buildReport(lastState, { phoneNumber: null });
  const view = reviewDraft(reportDraft);
  const tacticChips = view.tactics
    .map(
      (tactic) =>
        `<label class="tw-chip lv-report-tactic"><input type="checkbox" name="lvReportTactic" value="${esc(tactic.id)}" checked>` +
        `<span data-tactic="${esc(tactic.id)}"></span></label>`
    )
    .join("");
  form.innerHTML =
    `<label class="lv-report-label" for="lvReportText" data-i18n="report.text_label"></label>` +
    `<textarea id="lvReportText" class="lv-script lv-report-text mono" rows="6" spellcheck="false"></textarea>` +
    `<p class="lv-report-label" id="lvReportPreviewLabel" data-i18n="report.preview_label"></p>` +
    `<pre id="lvReportPreview" class="lv-stored mono" aria-labelledby="lvReportPreviewLabel" aria-live="polite"></pre>` +
    (tacticChips
      ? `<fieldset class="lv-report-fieldset"><legend class="lv-report-label" data-i18n="report.tactics_label"></legend>` +
        `<div class="lv-report-tactics">${tacticChips}</div></fieldset>`
      : "") +
    `<label class="lv-report-label" for="lvReportPhone" data-i18n="report.phone_label"></label>` +
    `<div class="lv-report-row"><input id="lvReportPhone" class="lv-report-phone mono" type="tel" inputmode="tel" ` +
    `autocomplete="off" maxlength="32" aria-describedby="lvReportPhoneHelp"></div>` +
    `<p class="lv-report-help" id="lvReportPhoneHelp" data-i18n="report.phone_help"></p>` +
    `<label class="lv-report-consent"><input type="checkbox" id="lvReportConsent"> <span data-i18n="report.consent"></span></label>` +
    `<div class="lv-report-row"><button type="button" id="lvReportSend" class="btn btn-paper" disabled data-i18n="report.send"></button></div>` +
    `<div class="lv-report-note" id="lvReportNote" role="status"></div>`;
  localize(form);
  renderTactics(form);
  const text = $("lvReportText");
  const preview = $("lvReportPreview");
  const consent = $("lvReportConsent");
  const send = $("lvReportSend");
  text.value = view.transcript;
  const refresh = () => {
    preview.textContent = scrubText(text.value.trim()) || "—";
    send.disabled = !consent.checked || !text.value.trim();
  };
  text.addEventListener("input", refresh);
  consent.addEventListener("change", refresh);
  send.addEventListener("click", sendReport);
  refresh();
  form.hidden = false;
  openBtn.hidden = true;
  text.focus();
};

/** The review's status line: `data-state` (sent | deleted | error) is the stable hook for tests. */
const showReportNote = (note, state, key = null, params = null) => {
  note.dataset.state = state;
  note.className = `lv-report-note${state === "sent" ? " is-success" : state === "error" ? " is-error" : ""}`;
  bind(note, key, params); // a null key also drops any earlier binding
};

const sendReport = async () => {
  const btn = $("lvReportSend");
  const note = $("lvReportNote");
  if (!lastState || !reportDraft) return;
  btn.disabled = true;
  try {
    const { finalizeReport } = await import("./core/report.js");
    const payload = finalizeReport(reportDraft, {
      transcript: $("lvReportText").value,
      phoneNumber: $("lvReportPhone").value,
      includedTacticIds: [...document.querySelectorAll('input[name="lvReportTactic"]:checked')].map((el) => el.value),
      consent: $("lvReportConsent").checked,
    });
    const res = await fetch("/api/reports", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // The server stores which consent wording was ticked, as proof of what was agreed to.
      body: JSON.stringify({ ...payload, consent_version: CONSENT_VERSION }),
    });
    if (res.status === 422) throw new UiError("report.err_rejected", { detail: String((await res.json()).detail || res.status) });
    if (res.status === 503) throw new UiError("report.err_numbers");
    if (res.status === 429) throw new UiError("report.err_rate");
    if (!res.ok) throw new UiError("report.err_status", { status: res.status });
    const body = await res.json();
    $("lvReportForm").querySelectorAll("textarea, input").forEach((el) => { el.disabled = true; });
    showReportNote(note, "sent");
    note.innerHTML =
      `<p data-i18n-html="report.sent_html"></p>` +
      `<pre class="lv-stored mono"></pre>` +
      `<p data-i18n-html="report.queued_html"></p>` +
      `<button type="button" id="lvReportDelete" class="btn btn-ghost" data-i18n="report.delete"></button>`;
    note.firstElementChild.dataset.i18nParams = JSON.stringify({ receipt: body.receipt_id, prefix: body.number_prefix || "—" });
    note.querySelector("pre").textContent = body.stored_transcript;
    localize(note);
    $("lvReportDelete").addEventListener("click", () => deleteReport(body.receipt_id, note));
  } catch (e) {
    btn.disabled = false;
    showReportNote(note, "error", "report.send_failed", { error: reasonOf(e) });
  }
};

const deleteReport = async (receiptId, note) => {
  try {
    const res = await fetch(`/api/reports/${encodeURIComponent(receiptId)}`, { method: "DELETE" });
    if (res.status === 404) throw new UiError("report.err_gone");
    if (!res.ok) throw new UiError("report.err_status", { status: res.status });
    showReportNote(note, "deleted", "report.deleted");
  } catch (e) {
    showReportNote(note, "error", "report.delete_failed", { error: reasonOf(e) });
  }
};

// ── replay mode ──────────────────────────────────────────────────────────────

let scenarios = [];

const loadScenarios = async () => {
  try {
    const res = await fetch("core/scenarios.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    scenarios = (await res.json()).scenarios || [];
    scenarioSel.innerHTML =
      `<option value="" data-i18n="replay.custom"></option>` +
      scenarios
        .map((s) => {
          const key = `scenario.${s.id}`;
          return `<option value="${esc(s.id)}"${hasKey(key) ? ` data-i18n="${esc(key)}"` : ""}>${esc(s.label)}</option>`;
        })
        .join("");
    localize(scenarioSel);
    // Start on the scam scene in the citizen's language when there is one.
    const first = scenarios.find((s) => s.id.endsWith(`_${cl()}`)) || scenarios[0];
    if (first) {
      scenarioSel.value = first.id;
      scriptArea.value = first.lines.join("\n");
    }
  } catch (e) {
    setNote(setupNote, "replay.scenarios_failed", { error: errText(e) }, true);
  }
};

scenarioSel.addEventListener("change", () => {
  const chosen = scenarios.find((s) => s.id === scenarioSel.value);
  scriptArea.value = chosen ? chosen.lines.join("\n") : "";
});

const runReplay = async () => {
  if (running) return;
  const lines = scriptArea.value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  if (!lines.length) {
    setNote(setupNote, "replay.empty", null, true);
    return;
  }

  running = true;
  startBtn.disabled = true;
  bind(startBtn, "replay.running");
  setNote(setupNote, null);
  resetCallUi();

  try {
    const rt = await deviceRuntime(setupNote);
    liveState = rt.newSession(cl());
    for (const line of lines) {
      await sleep(TURN_DELAY_MS);
      const out = await rt.advance(liveState, line, 1);
      liveState = { ...out.state, locale: cl() }; // the language may have changed while scoring
      applyUpdate(out.update, liveState, line);
    }
    lastState = liveState;
    showSummary();
  } catch (e) {
    setNote(setupNote, "replay.failed", { error: errText(e) }, true);
  } finally {
    liveState = null;
    running = false;
    startBtn.disabled = false;
    bind(startBtn, "replay.start");
  }
};

startBtn.addEventListener("click", runReplay);

// ── microphone mode (on-device speech recognition, PLAN B9) ──────────────────

let asr = null; // the running recogniser
let micStream = null;
let micState = null; // the live session while the microphone runs
let micQueue = Promise.resolve(); // utterances are scored one at a time, in order

/** Turn the voice mode off with `key` (params) as the reason; paste and replay stay available. */
const disableVoice = (key, params, isError = false) => {
  const input = micChip.querySelector("input");
  input.disabled = true;
  micChip.classList.add("is-disabled");
  micChip.dataset.i18nAttr = `title:${key}`;
  micChip.dataset.i18nParams = JSON.stringify(params);
  renderEl(micChip);
  setNote(micNote, key, params, isError);
  modeNote.hidden = false;
  setNote(modeNote, key, params);
};

const checkMicCapability = async () => {
  // No server probe -- the server never accepts audio (ADR D12); support is a property of
  // this browser (voiceMode, ADR D67): the full word-level mode needs cross-origin isolation
  // and is still off on phones (ADR D25); without isolation -- an Android WebView, eGov
  // Mobile's included -- the light GigaAM-only mode runs; with no microphone, AudioWorklet,
  // WASM or secure context, the voice mode is off.
  let reasons;
  try {
    const { voiceMode } = await import("./core/asr.js");
    voice = voiceMode(globalThis, VOICE_OVERRIDE);
    renderDownloadNotes();
    if (voice.mode === "full") {
      setNote(micNote, "mic.ready");
      return;
    }
    if (voice.mode === "light") {
      setNote(micNote, "mic.ready_light");
      voiceModeNote.hidden = false;
      bind(voiceModeNote, "mic.light_mode");
      return;
    }
    reasons = [...new Set(voice.issues.map((code) => MIC_REASON[code] || "mic.reason_browser"))].map((key) => ({ $: key }));
  } catch {
    reasons = [{ $: "mic.reason_browser" }];
  }
  disableVoice("mic.unavailable", { reasons });
};

// In `asr.order` (kk, ru, en -- ADR D62): the JSON's keys are sorted, and the first language is
// the recogniser's initial preference for ties and the first partial shown.
const modelSpecs = (cfg) => {
  const models = cfg.asr?.models || {};
  const order = cfg.asr?.order || Object.keys(models);
  return Object.fromEntries(
    order.filter((language) => models[language]).map((language) => {
      const m = models[language];
      return [language, { id: m.id, url: new URL(ASR_MODELS_BASE + m.url, location.href).href }];
    })
  );
};

/** A final utterance: GigaAM's text (or the Vosk vote, marked as the fallback) -- the only text
    the session and the meter ever see (ADR D64). */
const onMicUtterance = (utterance, info = {}) => {
  const { text, confidence, language } = utterance;
  showLiveWords([]);
  showPartial("");
  micQueue = micQueue
    .then(async () => {
      if (!micState) return;
      const rt = await deviceRuntime();
      const out = await rt.advance(micState, text, confidence);
      micState = { ...out.state, locale: cl() };
      applyUpdate(out.update, micState, text, { language, confidence, words: utterance.words, source: utterance.source });
      if (VOICE_TRACE) {
        window.__qorganVoice.utterances.push({
          seq: utterance.seq ?? null, text, language, confidence, source: utterance.source, fallback: utterance.fallback ?? null, vosk: utterance.vosk,
          words: utterance.words, decodeMs: utterance.decodeMs ?? null, latencyMs: info.latencyMs ?? null,
          meter: out.update.meter.score, band: out.update.band, at: performance.now(),
        });
      }
    })
    .catch((e) => setNote(micNote, "mic.analysis_failed", { error: errText(e) }, true));
};

const onEndpointStatus = (status) => {
  if (VOICE_TRACE && !status.total) window.__qorganVoice.endpoint.push({ ...status, at: performance.now() });
  const light = voice.mode === "light";
  const progressKey = light ? "mic.light_progress" : "mic.endpoint_progress";
  const key = status.status === "loading" && status.total ? progressKey : (light ? LIGHT_STATUS : ENDPOINT_STATUS)[status.status];
  if (!key) return;
  endpointNote.hidden = false;
  setNote(endpointNote, key, key === progressKey ? { pct: Math.round((100 * status.loaded) / status.total) } : null, status.status === "unavailable");
};

/** The word-level models (span vote + word LID, 212 kB); without them the page shows plain
    partials and the endpoint pass is off. */
const loadWordLevel = async () => {
  try {
    const res = await fetch("core/asr-wordlevel.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch {
    return null;
  }
};

const stopTracks = () => {
  for (const track of micStream?.getTracks() || []) track.stop();
  micStream = null;
};

/** The light mode's recogniser (ADR D67): GigaAM only, phrases cut by the VAD. Null when GigaAM
    cannot load -- there is no quick transcript to fall back on, so the voice mode is turned off
    with the reason, and paste/replay stay available. */
const createLightRecogniser = async (rt) => {
  const wordLevel = await loadWordLevel();
  if (VOICE_TRACE) window.__qorganVoice = { mode: "light", utterances: [], live: [], endpoint: [], speech: [], dropped: [], wordLevel: Boolean(wordLevel) };
  const { createLightAsr } = await import("./core/voice-light.js");
  try {
    return await createLightAsr({
      endpoint: rt.config.asr?.endpoint ?? null,
      wordLevel,
      onEndpointStatus,
      onUtterance: onMicUtterance,
      onSpeech: (event) => {
        showActivity(event.type === "speech_start" ? "mic.light_hearing" : event.type === "segment" ? "mic.light_decoding" : null);
        if (VOICE_TRACE) window.__qorganVoice.speech.push({ ...event, wallAt: performance.now() });
      },
      onDropped: (dropped) => {
        showActivity(null);
        setNote(endpointNote, "mic.light_dropped");
        if (VOICE_TRACE) window.__qorganVoice.dropped.push({ ...dropped, at: performance.now() });
      },
      onStatus: (_message, info = {}) => {
        const key = ASR_STATUS[info.code];
        if (key) setNote(micNote, key);
      },
    });
  } catch (e) {
    endpointNote.hidden = true; // its "loading" line is stale now
    disableVoice("mic.light_unavailable", { error: errText(e) }, true);
    return null;
  }
};

const startMic = async () => {
  if (running) return;
  running = true;
  micStartBtn.disabled = true;
  resetCallUi();
  try {
    const rt = await deviceRuntime(micNote);
    if (!asr && voice.mode === "light") {
      asr = await createLightRecogniser(rt);
      if (!asr) { running = false; return; } // GigaAM could not load: the voice mode is now off, paste/replay stay
    }
    const { createDeviceAsr } = await import("./core/asr.js");
    if (!asr) {
      const wordLevel = await loadWordLevel();
      if (VOICE_TRACE) window.__qorganVoice = { mode: "full", utterances: [], live: [], endpoint: [], wordLevel: Boolean(wordLevel) };
      asr = await createDeviceAsr({
        models: modelSpecs(rt.config),
        lock: rt.config.asr?.lock ?? null,
        handicap: rt.config.asr?.handicap ?? null,
        optional: rt.config.asr?.optional ?? [],
        wordLevel,
        endpoint: wordLevel ? rt.config.asr?.endpoint ?? null : null,
        onEndpointStatus,
        // With the word-level models the live line is the span vote's tagged words; without them, plain partials.
        onPartial: ({ text }) => { if (!wordLevel) showPartial(text); },
        onLive: wordLevel
          ? (words) => {
            showLiveWords(words);
            if (VOICE_TRACE) window.__qorganVoice.live.push({ at: performance.now(), words });
          }
          : null,
        onUtterance: onMicUtterance,
        onStatus: (_message, info = {}) => {
          const key = ASR_STATUS[info.code];
          if (key) setNote(micNote, key, info.language ? { language: { $: `lang_name.${info.language}` } } : null);
        },
        onError: (e) => setNote(micNote, "mic.recognition_error", { error: errText(e) }, true),
      });
    }
    setNote(micNote, "mic.permission");
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    if (VOICE_TRACE) window.__qorganVoice.captureAt = performance.now();
    micState = rt.newSession(cl());
    await asr.start(micStream);
    if (VOICE_TRACE) window.__qorganVoice.listeningAt = performance.now();
    micStopBtn.hidden = false;
    micStartBtn.hidden = true;
  } catch (e) {
    running = false;
    micStartBtn.disabled = false;
    setNote(micNote, "mic.start_failed", { error: errText(e) }, true);
    stopTracks();
  }
};

const stopMic = async () => {
  micStopBtn.disabled = true;
  try {
    if (asr) await asr.stop();
    stopTracks();
    await micQueue; // let the last utterance finish scoring
    if (micState && micState.meter?.turn_index > 0) {
      lastState = micState;
      showSummary();
    } else {
      setNote(micNote, "mic.no_speech");
    }
  } catch (e) {
    setNote(micNote, "mic.stop_failed", { error: errText(e) }, true);
  } finally {
    micState = null;
    running = false;
    micStopBtn.hidden = true;
    micStopBtn.disabled = false;
    micStartBtn.hidden = false;
    micStartBtn.disabled = false;
  }
};

micStartBtn.addEventListener("click", startMic);
micStopBtn.addEventListener("click", stopMic);
window.addEventListener("pagehide", () => { stopTracks(); asr?.dispose(); });

// ── mode toggle + start ──────────────────────────────────────────────────────

const applyMode = () => {
  const current = mode();
  setupReplay.hidden = current !== "replay";
  setupMic.hidden = current !== "mic";
};

document.querySelectorAll('input[name="lv-mode"]').forEach((radio) => radio.addEventListener("change", applyMode));
document.querySelectorAll('input[name="lv-lang"]').forEach((radio) =>
  radio.addEventListener("change", () => { if (radio.checked) applyLocale(radio.value); })
);

renderDownloadNotes();
applyLocale(locale, { persist: false });
applyMode();
checkMicCapability();
loadScenarios();
detectModelCache().then((cached) => {
  if (cached && !modelCached) {
    modelCached = true;
    renderDownloadNotes();
  }
});
