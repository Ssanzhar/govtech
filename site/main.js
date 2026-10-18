/* Qorğan landing: the drawings, built from real model output.

   FIG. 1 is the demo call drawn as a blueprint:
   - every phrase, with a waveform as long as the phrase;
   - the verbatim trigger phrases, with numbered markers and leader lines to callouts that
     name the tactic;
   - the 0–100 ruler, whose needle walks the real per-turn scores;
   - the band, and the advice for the top tactic.
   It builds once on load (≤ 4 s). The toggle replays the build for the other call, and
   prefers-reduced-motion shows the final state at once.

   This module also draws:
   - the flagged phrase's path through "how it works";
   - the tactic names in Detail A;
   - the eval table's confidence intervals, as dimension lines drawn from the cell strings.

   DATA: the per-turn values below are the real output of the shipped on-device runtime
   (core/device.js). $D/design/tools/probe_runtime.mjs recorded them on 2026-10-17 into
   $D/design/probe/rt_ru_scam.json, rt_kk_scam.json and rt_ru_legit.json. The English run of
   the Russian scene has identical scores. Copied here: scores, bands, the latch, the cue spans
   and the id of the advised tactic. The tactic names and the advice text are model content:
   they are read from core/qorgan-config.json in the page's language, never copied. */

import { contentLocale, t } from "./i18n.js";
import { localize } from "./i18n-dom.js";
import { displayName } from "./core/explain.js";

const CALLS = {
  scam_ru: {
    scenario: "live_scam_bank_ru", lang: "ru",
    turns: [
      { line: "Алло, здравствуйте. Это служба безопасности вашего банка.", score: 49.68, band: "medium" },
      { line: "По вашей карте зафиксирована подозрительная операция на крупную сумму.", score: 74.83, band: "high" },
      { line: "Действовать нужно прямо сейчас, у вас есть только пять минут.", score: 87.39, band: "critical", advice: "credentials_request" },
      { line: "Никому не говорите, это секретная операция под контролем регулятора.", score: 93.7, band: "critical", advice: "secrecy", cue: { n: 1, tactic: "secrecy", text: "Никому не говорите" } },
      { line: "Продиктуйте код из SMS, чтобы мы отменили операцию.", score: 96.85, band: "critical", advice: "secrecy", cue: { n: 2, tactic: "otp_request", text: "Продиктуйте код из SMS" } },
      { line: "Теперь переведите деньги на безопасный счёт, я продиктую реквизиты.", score: 98.42, band: "critical", advice: "secrecy", cue: { n: 3, tactic: "safe_account", text: "на безопасный счёт" },
        tags: { impersonation_bank: 0.919, urgency: 0.882, fear_threat: 0.876, otp_request: 1, verification_ploy: 0.756, secrecy: 1, safe_account: 1 } },
    ],
  },
  scam_kk: {
    scenario: "live_scam_bank_kk", lang: "kk",
    turns: [
      { line: "Сәлеметсіз бе. Бұл банктің қауіпсіздік қызметі.", score: 49.28, band: "medium" },
      { line: "Сіздің картаңыздан күдікті операция тіркелді, жағдай шұғыл.", score: 74.06, band: "high" },
      { line: "Ешкімге айтпаңыз, бұл құпия операция.", score: 87.02, band: "critical", advice: "secrecy", cue: { n: 1, tactic: "secrecy", text: "Ешкімге айтпаңыз" } },
      { line: "SMS-тегі кодты айтыңыз, біз операцияны тоқтатамыз.", score: 93.51, band: "critical", advice: "secrecy", cue: { n: 2, tactic: "otp_request", text: "SMS-тегі кодты айтыңыз" } },
      { line: "Содан кейін ақшаны қауіпсіз шотқа аударамыз, реквизиттерін айтамын.", score: 96.75, band: "critical", advice: "secrecy", cue: { n: 3, tactic: "safe_account", text: "қауіпсіз шотқа аудар" },
        tags: { fear_threat: 0.846, impersonation_bank: 0.836, urgency: 0.747, otp_request: 1, verification_ploy: 0.668, secrecy: 1, safe_account: 1 } },
    ],
  },
  legit_ru: {
    scenario: "live_hard_negative_bank_ru", lang: "ru",
    turns: [
      { line: "Здравствуйте, это оператор банка, звоню по вашей заявке.", score: 37.06, band: "medium" },
      { line: "Мы подтверждаем, что перевод на 15000 тенге успешно проведён.", score: 33.14, band: "medium" },
      { line: "Никакие коды и данные карты называть не нужно.", score: 29.2, band: "low" },
      { line: "Если вы не совершали эту операцию, позвоните на официальный номер банка на обратной стороне карты.", score: 25.75, band: "low" },
    ],
  },
};
// The meter latches (the alert, and with it the advice) from turn 3 in both scam calls
// (`latched` in the probes), so `advice` is set on exactly the latched turns. The last turn of each
// scam call also carries the probe's tactic weights (`result_tags`): Detail A draws those.

const START_MS = 420;  // the frame and the ruler draw in first
const STEP_MS = 470;   // one phrase: waveform, text, needle; six phrases end before 4 s
const SETTLE_MS = 640;

const $ = (sel) => document.querySelector(sel);
const plate = $("#plate");
const lines = $("#plateLines");
const leaders = $("#plateLeaders");
const store = $(".lp-callout-store");
const head = $("#plateHead");
const scoreEl = $("#plateScore");
const meter = $("#plateMeter");
const fill = $("#plateFill");
const bandEl = $("#plateBand");
const advice = $("#plateAdvice");
const adviceText = $("#plateAdviceText");
const tip = $("#plateTip");
const toggles = [...document.querySelectorAll(".lp-toggle-btn")];
const plateBody = $(".lp-plate-body");
const releaseHeight = (key) => { if (plateBody && key.startsWith("scam")) plateBody.style.minHeight = ""; };
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const sideBySide = () => matchMedia("(min-width: 640px)").matches; // rows hold their callouts beside the text

let locale = document.documentElement.lang || "ru";
let config = null;
let which = "scam";      // the pressed toggle
let shown = null;        // the call on the plate: {key, turn}
let runId = 0;
let timers = [];
let tween = 0;

const configReady = fetch("core/qorgan-config.json")
  .then((res) => (res.ok ? res.json() : null))
  .then((json) => { config = json; return json; })
  .catch(() => null);

const cl = () => (config ? contentLocale(locale, config.locales) : locale);
const tacticName = (id) => (config && displayName(id, cl(), config)) || id;
const adviceFor = (id) => {
  const entry = config?.advice?.[cl()]?.tactic_advice?.[id];
  return Array.isArray(entry) ? entry[0] : entry || "";
};
const scamKey = () => (locale === "kk" ? "scam_kk" : "scam_ru");
const callKey = () => (which === "scam" ? scamKey() : "legit_ru");
const bandParams = (band) => ({ band: { $: `band.${band}` }, desc: { $: `band.${band}_desc` } });
const setParams = (el, params) => { el.dataset.i18nParams = JSON.stringify(params); localize(el, locale); };

// A waveform for one phrase, as long as the phrase: one burst per word with a short silence
// between words, hairline strokes (deterministic, seeded by the text).
const waveSvg = (text) => {
  let seed = 2166136261;
  for (const ch of text) seed = Math.imul(seed ^ ch.codePointAt(0), 16777619) >>> 0;
  const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
  const words = text.split(/\s+/).filter(Boolean);
  let x = 0;
  let d = "";
  words.forEach((word, w) => {
    const bars = Math.max(3, Math.round(word.length * 2.6));
    for (let k = 0; k < bars; k++, x++) {
      const env = Math.sin((Math.PI * (k + 0.5)) / bars) ** 0.7;
      const h = Math.max(1, (0.3 + 0.7 * rnd()) * env * 11);
      d += `M${x} ${(6 - h / 2).toFixed(2)}V${(6 + h / 2).toFixed(2)}`;
    }
    if (w < words.length - 1) for (let k = 0; k < 3; k++, x++) d += `M${x} 5.4V6.6`;
  });
  return `<svg viewBox="-0.5 0 ${x} 12" preserveAspectRatio="none" focusable="false"><path d="${d}"/></svg>`;
};

const escHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const phraseHtml = (line, cue) => {
  const at = cue ? line.indexOf(cue.text) : -1;
  if (at < 0) return escHtml(line);
  const after = line.slice(at + cue.text.length);
  const stem = /^\p{L}/u.test(after) ? " is-stem" : ""; // the cue matched a stem inside a word
  const short = cue.text.length <= 24 ? " is-short" : "";  // keep a short cue on one line with its marker
  return `${escHtml(line.slice(0, at))}<mark class="q-cue${stem}${short}" data-n="${cue.n}">${escHtml(cue.text)}</mark>${escHtml(after)}`;
};

const calloutOf = (n) => document.querySelector(`.lp-callout[data-n="${n}"]`);

/* ── render ─────────────────────────────────────────────────────────────────────────────── */
function renderRows(key) {
  const call = CALLS[key];
  document.querySelectorAll(".lp-callout").forEach((c) => store.append(c));
  lines.textContent = "";
  lines.lang = call.lang;
  call.turns.forEach((turn, i) => {
    const li = document.createElement("li");
    li.className = "q-line lp-line";
    const text = document.createElement("span");
    text.className = "lp-text";
    text.innerHTML = phraseHtml(turn.line, turn.cue);
    const wave = document.createElement("span");
    wave.className = "lp-wave";
    wave.setAttribute("aria-hidden", "true");
    wave.style.setProperty("--len", Math.min(1, turn.line.length / 100).toFixed(3));
    wave.innerHTML = waveSvg(turn.line);
    li.append(text, wave);
    if (turn.cue) {
      const c = calloutOf(turn.cue.n);
      setParams(c.querySelector(".lp-callout-turn"), { n: i + 1 });
      li.classList.add("has-cue");
      li.append(c);
    }
    lines.append(li);
  });
  renderNames();
}

function renderNames() {
  const lang = cl();
  document.querySelectorAll(".lp-callout").forEach((c) => {
    const turn = shown && CALLS[shown.key].turns.find((x) => x.cue && x.cue.n === Number(c.dataset.n));
    const name = c.querySelector(".lp-callout-name");
    if (turn) { name.textContent = tacticName(turn.cue.tactic); name.lang = lang; }
  });
  document.querySelectorAll("[data-tactic]").forEach((el) => { el.textContent = tacticName(el.dataset.tactic); el.lang = lang; });
}

// The instrument after turn `i` (or the empty instrument for i = -1).
function setTurn(i, { animate = false } = {}) {
  const call = CALLS[shown.key];
  shown.turn = i;
  const turn = call.turns[i];
  const score = turn ? turn.score : 0;
  const band = turn ? turn.band : "low";
  setParams(head, { n: i + 1 });
  fill.className = `q-ruler__fill band-${band}`;
  fill.style.width = `${score}%`;
  meter.setAttribute("aria-valuenow", score.toFixed(0));
  meter.setAttribute("aria-valuetext", t(locale, "call.meter_valuetext", { score: score.toFixed(0), band: { $: `band.${band}` } }));
  bandEl.className = `q-band band-${band}`;
  setParams(bandEl, bandParams(band));
  showScore(score, animate);
  advice.hidden = !turn?.advice;
  adviceText.textContent = turn?.advice ? adviceFor(turn.advice) : "";
  adviceText.lang = cl();
  // The real call ends low: the citizen page shows its safe tip there, so the figure does too.
  tip.hidden = !(turn && i === call.turns.length - 1 && band === "low");
}

function showScore(target, animate) {
  cancelAnimationFrame(tween);
  const from = Number.parseFloat(scoreEl.textContent) || 0;
  if (!animate || reduced()) { scoreEl.textContent = `${target.toFixed(0)} / 100`; return; }
  const t0 = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - t0) / 420);
    const v = from + (target - from) * (1 - (1 - k) ** 3);
    scoreEl.textContent = `${v.toFixed(0)} / 100`;
    if (k < 1) tween = requestAnimationFrame(step);
  };
  tween = requestAnimationFrame(step);
}

/* ── leader lines: marker → the gap above its phrase → down into the callout's title bar ───── */
function layoutLeaders() {
  leaders.textContent = "";
  if (!sideBySide()) return;
  const box = leaders.getBoundingClientRect();
  const ns = "http://www.w3.org/2000/svg";
  lines.querySelectorAll(".lp-line.has-cue").forEach((li) => {
    const mark = li.querySelector(".q-cue[data-n]");
    const c = li.querySelector(".lp-callout");
    if (!mark || !c) return;
    const m = mark.getClientRects()[0];
    const row = li.getBoundingClientRect();
    const cb = c.getBoundingClientRect();
    const bar = c.querySelector(".q-callout__bar").getBoundingClientRect();
    const x0 = m.left + 12 - box.left;            // marker centre (padding 3 + half of 18)
    const y0 = m.top - 1 - box.top;
    const yGap = row.top + 5 - box.top;           // inside the phrase's top padding
    const xIn = cb.left - box.left;
    const xJog = xIn - 14;
    const yBar = bar.top + bar.height / 2 - box.top;
    const path = document.createElementNS(ns, "path");
    path.setAttribute("class", "q-leader lp-leader");
    path.setAttribute("pathLength", "1");
    path.setAttribute("d", `M${x0} ${y0}V${yGap}H${xJog}V${yBar}H${xIn}`);
    const dot = document.createElementNS(ns, "circle");
    dot.setAttribute("class", "q-leader-dot lp-leader-dot");
    dot.setAttribute("cx", x0); dot.setAttribute("cy", y0); dot.setAttribute("r", "2.5");
    const on = li.classList.contains("is-cue-on");
    path.classList.toggle("is-in", on);
    dot.classList.toggle("is-in", on);
    leaders.append(path, dot);
    li._leader = [path, dot];
  });
}

/* ── the build ─────────────────────────────────────────────────────────────────────────────── */
function clearRun() { runId++; timers.forEach(clearTimeout); timers = []; cancelAnimationFrame(tween); }

function showRow(i) { lines.children[i]?.classList.add("is-in"); }
function showCue(i) {
  const li = lines.children[i];
  if (!li || !li.classList.contains("has-cue")) return;
  li.classList.add("is-cue-on");
  (li._leader || []).forEach((el) => el.classList.add("is-in"));
}

function build(key, { animate }) {
  clearRun();
  const id = runId;
  const call = CALLS[key];
  shown = { key, turn: -1 };
  plate.dataset.call = key.startsWith("scam") ? "scam" : "legit";
  plate.classList.toggle("is-instant", !animate);
  plate.classList.remove("is-building");
  renderRows(key);
  setTurn(-1);
  toggles.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.call === which)));
  if (!animate) {
    call.turns.forEach((_, i) => { showRow(i); showCue(i); });
    setTurn(call.turns.length - 1);
    layoutLeaders();
    releaseHeight(key);
    requestAnimationFrame(() => requestAnimationFrame(() => plate.classList.remove("is-instant")));
    return;
  }
  void plate.offsetWidth; // restart the draw-in keyframes
  plate.classList.add("is-building");
  layoutLeaders();
  const at = (ms, fn) => timers.push(setTimeout(() => { if (id === runId) fn(); }, ms));
  call.turns.forEach((turn, i) => {
    const t0 = START_MS + i * STEP_MS;
    at(t0, () => showRow(i));
    at(t0 + 170, () => setTurn(i, { animate: true }));
    if (turn.cue) at(t0 + 250, () => showCue(i));
  });
  at(START_MS + (call.turns.length - 1) * STEP_MS + SETTLE_MS, () => { plate.classList.remove("is-building"); releaseHeight(key); });
}

/* ── language: annotations follow the page, phrases stay in the call's language ──────────── */
// "How it works": the first trigger phrase of the page's scam call, and the score the meter showed
// right after it, labelled with its phrase (D-4). Its tactic is a [data-tactic] element (renderNames).
function renderPath() {
  const call = CALLS[scamKey()];
  const i = call.turns.findIndex((x) => x.cue && x.cue.n === 1);
  const turn = call.turns[i];
  const phrase = document.querySelector('[data-path="phrase"]');
  const band = document.querySelector('[data-path="band"]');
  if (phrase) { phrase.textContent = turn.cue.text; phrase.lang = call.lang; }
  if (band) {
    band.className = `q-band band-${turn.band}`;
    setParams(band, { n: i + 1, score: turn.score.toFixed(0), band: { $: `band.${turn.band}` } });
  }
}

// Detail A: the last turn of the same call as FIG. 1 (D-4): its trigger phrases first, numbered as
// on the plate, then the tactics the model inferred from the whole call window, by weight.
function renderDetail() {
  const box = document.querySelector("#detailRows");
  const ref = document.querySelector("#detailRef");
  if (!box) return;
  const call = CALLS[scamKey()];
  const last = call.turns.length - 1;
  const final = call.turns[last];
  const cues = call.turns.filter((x) => x.cue).map((x) => x.cue);
  const cueIds = new Set(cues.map((c) => c.tactic));
  const rest = Object.entries(final.tags).filter(([id]) => !cueIds.has(id)).sort((a, b) => b[1] - a[1]);
  const lang = cl();
  const row = (marker, phrase, id, w) =>
    `<div class="lp-ev-row">${marker}<span class="lp-ev-phrase">${phrase}</span>` +
    `<span class="lp-ev-tactic"><span class="lp-ev-name" data-tactic="${id}" lang="${lang}">${escHtml(tacticName(id))}</span><span class="lp-ev-id">${id}</span></span>` +
    `<span class="ec-bar"><i style="--w:${Math.round(w * 100)}%"></i></span><span class="ec-val">${w.toFixed(2)}</span></div>`;
  box.lang = call.lang;
  box.innerHTML =
    cues.map((c) => row(`<span class="q-marker q-marker--sm" aria-hidden="true">${c.n}</span>`, `<mark class="q-cue">${escHtml(c.text)}</mark>`, c.tactic, final.tags[c.tactic] ?? 1)).join("") +
    rest.map(([id, w]) => row(`<span class="lp-ev-nomark" aria-hidden="true"></span>`, `<span class="lp-ev-none" data-i18n="landing.detail_window" lang="${locale}">${escHtml(t(locale, "landing.detail_window"))}</span>`, id, w)).join("");
  if (ref) ref.textContent = `${call.scenario} · ${t(locale, "landing.plate_phrase", { n: last + 1 })} · ${final.score.toFixed(0)} / 100`;
}

function relabel() {
  const scamBtn = toggles.find((b) => b.dataset.call === "scam");
  if (scamBtn) { scamBtn.dataset.i18n = `scenario.${CALLS[scamKey()].scenario}`; localize(scamBtn, locale); }
  renderNames();
  if (shown && shown.turn >= 0) setTurn(shown.turn);
  renderPath();
  renderDetail();
  layoutLeaders();
}

let started = false;
async function onLocale(next) {
  const before = shown?.key;
  locale = next || document.documentElement.lang || "ru";
  if (!started) {
    started = true;
    await Promise.race([configReady, new Promise((r) => setTimeout(r, 900))]);
    relabel();
    build(callKey(), { animate: !reduced() });
    return;
  }
  if (before && before !== callKey()) build(callKey(), { animate: false }); // ru/en ↔ kk swaps the scam call
  relabel();
}

if (plate && lines) {
  document.addEventListener("qorgan:locale", (e) => onLocale(e.detail?.locale));
  // landing.js fires qorgan:locale on load; if it ran before this module, start on our own.
  setTimeout(() => { if (!started) onLocale(document.documentElement.lang); }, 400);
  // The real-bank call has fewer phrases: while it is shown the plate body keeps the scam call's
  // height, so the toggle (and the statement under the plate) never move under the pointer. The
  // reserve is released once a scam build has finished (releaseHeight).
  toggles.forEach((btn) => btn.addEventListener("click", () => {
    if (btn.dataset.call === "legit" && plateBody) {
      plateBody.style.minHeight = `${Math.max(plateBody.offsetHeight, parseFloat(plateBody.style.minHeight) || 0)}px`;
    }
    which = btn.dataset.call;
    build(callKey(), { animate: !reduced() });
  }));
  configReady.then(() => { if (started) relabel(); });
  let raf = 0;
  new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(layoutLeaders); }).observe(lines);
  document.fonts?.ready.then(() => layoutLeaders());
}

/* ── the eval table: each 95 % interval as a dimension line on a 0–1 scale, from the cell text ─ */
for (const td of document.querySelectorAll(".lp-ci-table tbody td")) {
  const m = td.textContent.match(/([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]/);
  if (!m) continue;
  const [p, lo, hi] = m.slice(1).map(Number).map((v) => Math.max(0, Math.min(1, v)) * 100);
  td.insertAdjacentHTML("beforeend",
    `<svg class="lp-ci" viewBox="-2 0 104 12" aria-hidden="true" focusable="false">` +
    `<path class="lp-ci-base" d="M0 6H100M0 3V9M50 4V8M100 3V9"/>` +
    `<path class="lp-ci-dim" d="M${lo} 6H${hi}M${lo} 2.5V9.5M${hi} 2.5V9.5"/>` +
    `<rect class="lp-ci-pt" x="${p - 2}" y="4" width="4" height="4"/></svg>`);
}
