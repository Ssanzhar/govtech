/* Qorğan live call — replay runs ON THIS DEVICE (site/core, no server round-trip); only an
   microphone (on-device, pending) drive the same meter UI; both end in a
   post-call summary with the consent-gated report button (POST .../report). */
(() => {
  "use strict";

  const scenarioSel = document.getElementById("lvScenario");
  const scriptArea = document.getElementById("lvScript");
  const startBtn = document.getElementById("lvStart");
  const setupNote = document.getElementById("lvSetupNote");
  const setupReplay = document.getElementById("lvSetupReplay");
  const setupMic = document.getElementById("lvSetupMic");
  const micChip = document.getElementById("lvMicChip");
  const micStartBtn = document.getElementById("lvMicStart");
  const micStopBtn = document.getElementById("lvMicStop");
  const micNote = document.getElementById("lvMicNote");
  const callWrap = document.getElementById("lvCallWrap");
  const dot = document.getElementById("lvDot");
  const turnEl = document.getElementById("lvTurn");
  const scoreEl = document.getElementById("lvScore");
  const meterFill = document.getElementById("lvMeterFill");
  const bandPill = document.getElementById("lvBandPill");
  const tagsEl = document.getElementById("lvTags");
  const adviceEl = document.getElementById("lvAdvice");
  const transcriptEl = document.getElementById("lvTranscript");
  const partialEl = document.getElementById("lvPartial");
  const summaryWrap = document.getElementById("lvSummaryWrap");
  if (!scenarioSel || !scriptArea || !startBtn) return;

  // Long enough to watch the meter move turn by turn, short enough to stay demo-paced.
  const TURN_DELAY_MS = 1200;
  const BAND_LABELS = {
    low: "LOW — no obvious scam indicators",
    medium: "MEDIUM — caution, warning signs detected",
    high: "HIGH — strong warning",
    critical: "CRITICAL — immediate warning",
  };

  let scenarios = [];
  let running = false;
  let lastSessionId = null; // legacy (server sessions); kept for renderSummary's signature
  let lastState = null; // the finished on-device session; consumed by the report button

  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const locale = () => document.querySelector('input[name="lv-loc"]:checked')?.value || "ru";
  const mode = () => document.querySelector('input[name="lv-mode"]:checked')?.value || "replay";

  const setNote = (el, msg, isError) => {
    el.textContent = msg || "";
    el.classList.toggle("is-error", Boolean(isError));
  };

  // `evidence` is plain phrase text (verbatim substrings of `line`, per the API contract —
  // no character offsets), so highlighting is a straight escaped-substring replace.
  const highlightLine = (line, evidence) => {
    let html = esc(line);
    for (const phrase of evidence) {
      if (!phrase) continue;
      const escaped = esc(phrase);
      if (html.includes(escaped)) {
        html = html.replace(escaped, `<mark>${escaped}</mark>`);
      }
    }
    return html;
  };

  const setBand = (band) => {
    const cls = `band-${band}`;
    meterFill.className = `lv-meter-fill ${cls}`;
    bandPill.className = `lv-band-pill mono ${cls}`;
    bandPill.textContent = BAND_LABELS[band] || band.toUpperCase();
    dot.className = band === "low" ? "p-dot tone-moss" : "p-dot";
  };

  const appendLine = (line, evidence) => {
    const div = document.createElement("div");
    div.className = "lv-line";
    div.innerHTML = highlightLine(line, evidence);
    transcriptEl.appendChild(div);
  };

  const showPartial = (text) => {
    partialEl.hidden = !text;
    partialEl.textContent = text || "";
  };

  // One committed utterance → meter, band, tags, advice, transcript line.
  // Adapt the on-device core's update to the fields this page renders.
  const toUpdate = (update, state) => ({
    turn: update.meter.turn_index,
    meter: update.meter.score,
    band: update.band,
    latched: update.meter.latched,
    risk: update.result.risk,
    advice: update.recommendation.advices,
    note: update.recommendation.note,
    new_evidence: update.new_evidence,
    tactics: state.tags,
  });

  let runtimePromise = null;
  const deviceRuntime = () => {
    if (!runtimePromise) {
      runtimePromise = import("./core/device.js").then(async ({ createDeviceRuntime }) => {
        const runtime = await createDeviceRuntime({
          onProgress: (p) => {
            if (p.type === "progress" && p.status === "progress" && p.file?.endsWith(".onnx")) {
              setNote(setupNote, `downloading the on-device model… ${Math.round(p.progress || 0)}% (278 MB, cached after the first time)`);
            }
          },
        });
        setNote(setupNote, "preparing the on-device model…");
        await runtime.warmup();
        setNote(setupNote, "on-device model ready — nothing leaves this browser");
        return runtime;
      });
    }
    return runtimePromise;
  };

  const applyUpdate = (update, lineText) => {
    turnEl.textContent = String(update.turn);
    scoreEl.textContent = `${update.meter.toFixed(0)} / 100`;
    meterFill.style.width = `${Math.min(100, Math.max(0, update.meter))}%`;
    setBand(update.band);
    tagsEl.innerHTML = update.tactics
      .map((t) => `<span class="tw-tag">${esc(t.id)}&nbsp;&middot;&nbsp;${t.weight.toFixed(2)}</span>`)
      .join("");
    appendLine(lineText, update.new_evidence.map((span) => span.text));
    if (update.latched && update.advice.length) {
      adviceEl.hidden = false;
      adviceEl.textContent = update.advice[0];
    }
  };

  const renderSummary = (summary, sessionId) => {
    lastSessionId = sessionId;
    const tactics = summary.tactic_names.length
      ? summary.tactic_names.map((n) => `<span class="tw-tag">${esc(n)}</span>`).join("")
      : '<span class="tw-dim">none</span>';
    const actions = summary.recommended_actions.length
      ? `<ul class="lv-summary-actions">${summary.recommended_actions
          .map((a) => `<li>${esc(a)}</li>`)
          .join("")}</ul>`
      : "";
    summaryWrap.hidden = false;
    summaryWrap.innerHTML =
      `<div class="lv-summary">` +
      `<p class="eyebrow inverse lv-summary-eyebrow">Post-call summary</p>` +
      `<div class="lv-summary-head"><span class="lv-summary-score">${summary.final_score.toFixed(0)} / 100</span></div>` +
      `<span class="lv-band-pill mono lv-summary-band band-${esc(summary.band)}">${esc(
        BAND_LABELS[summary.band] || summary.band
      )}</span>` +
      `<div class="lv-tags lv-summary-tags mono">${tactics}</div>` +
      actions +
      `<p class="lv-summary-note">${esc(summary.human_note)}</p>` +
      `<div class="lv-report mono" id="lvReport">` +
      `<div class="lv-report-title">Report this call to the analyst service?</div>` +
      `<div class="lv-report-row">` +
      `<input id="lvReportPhone" class="lv-report-phone mono" type="text" maxlength="32" placeholder="caller number (optional)">` +
      `<button type="button" id="lvReportSend" class="btn btn-paper">Send report</button>` +
      `</div>` +
      `<p class="lv-report-note" id="lvReportNote">Consent-gated: nothing is sent until you click. ` +
      `What is kept: the transcript with numbers/IDs redacted, the caller number only as a keyed hash ` +
      `plus its prefix (e.g. +7 700 ***), and the detected tactics. You get a receipt to delete it.</p>` +
      `</div>` +
      `</div>`;
    document.getElementById("lvReportSend").addEventListener("click", sendReport);
  };

  const sendReport = async () => {
    const btn = document.getElementById("lvReportSend");
    const note = document.getElementById("lvReportNote");
    const phone = document.getElementById("lvReportPhone").value.trim();
    if (!lastState) return;
    btn.disabled = true;
    try {
      const runtime = await deviceRuntime();
      const draft = runtime.buildReport(lastState, { phoneNumber: phone || null });
      const res = await fetch("/api/reports", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...draft, consent: true }),
      });
      if (res.status === 422) throw new Error((await res.json()).detail || "the report was rejected");
      if (res.status === 503) throw new Error("this server is not configured to accept caller numbers");
      if (res.status === 429) throw new Error("too many reports from this device — try again in a minute");
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      const body = await res.json();
      note.className = "lv-report-note is-success";
      note.innerHTML =
        `Stored — receipt <b>${esc(body.receipt_id)}</b>. Number kept as <b>${esc(body.number_prefix || "—")}</b>; ` +
        `transcript as stored (redacted):` +
        `<pre class="lv-stored mono">${esc(body.stored_transcript)}</pre>` +
        `It is now a pending report on the <a href="admin.html">analyst dashboard</a>. ` +
        `<button type="button" id="lvReportDelete" class="btn btn-ghost">Delete my report</button>`;
      document.getElementById("lvReportDelete").addEventListener("click", () => deleteReport(body.receipt_id, note));
    } catch (e) {
      btn.disabled = false;
      note.className = "lv-report-note is-error";
      note.textContent = `Could not submit — ${e.message || e}`;
    }
  };

  const deleteReport = async (receiptId, note) => {
    try {
      const res = await fetch(`/api/reports/${encodeURIComponent(receiptId)}`, { method: "DELETE" });
      if (res.status === 404) throw new Error("already deleted");
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      note.className = "lv-report-note";
      note.textContent = "Report deleted — removed from the pending queue and from the analysis if it had been ingested.";
    } catch (e) {
      note.className = "lv-report-note is-error";
      note.textContent = `Could not delete — ${e.message || e}`;
    }
  };

  const resetCallUi = () => {
    summaryWrap.hidden = true;
    summaryWrap.innerHTML = "";
    transcriptEl.innerHTML = "";
    tagsEl.innerHTML = "";
    adviceEl.hidden = true;
    showPartial("");
    lastSessionId = null;
    lastState = null;
    turnEl.textContent = "0";
    scoreEl.textContent = "0 / 100";
    meterFill.style.width = "0%";
    setBand("low");
    callWrap.hidden = false;
  };

  // ── replay mode ──────────────────────────────────────────────────────────────

  const loadScenarios = async () => {
    try {
      const res = await fetch("core/scenarios.json");
      if (!res.ok) throw new Error(`scenarios returned ${res.status}`);
      const body = await res.json();
      scenarios = body.scenarios || [];
      scenarioSel.innerHTML =
        `<option value="">custom — paste your own</option>` +
        scenarios.map((s) => `<option value="${esc(s.id)}">${esc(s.label)}</option>`).join("");
      if (scenarios.length) {
        scenarioSel.value = scenarios[0].id;
        scriptArea.value = scenarios[0].lines.join("\n");
      }
    } catch (e) {
      setNote(setupNote, `Could not load scenarios — ${e.message || e}`, true);
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
      setNote(setupNote, "Paste or pick a script first.", true);
      return;
    }

    running = true;
    startBtn.disabled = true;
    startBtn.textContent = "Running…";
    setNote(setupNote, "");
    resetCallUi();

    try {
      const runtime = await deviceRuntime();
      let state = runtime.newSession(locale());
      for (const line of lines) {
        await sleep(TURN_DELAY_MS);
        const out = await runtime.advance(state, line, 1);
        state = out.state;
        applyUpdate(toUpdate(out.update, state), line);
      }
      lastState = state;
      renderSummary(runtime.summarize(state), null);
    } catch (e) {
      setNote(
        setupNote,
        `On-device analysis failed — ${e.message || e}. The model files under /models/ may still be downloading.`,
        true
      );
    } finally {
      running = false;
      startBtn.disabled = false;
      startBtn.textContent = "Start live analysis";
    }
  };

  startBtn.addEventListener("click", runReplay);

  // ── microphone mode ──────────────────────────────────────────────────────────

  let micSessionId = null;

  const disableMicChip = (reason) => {
    const input = micChip?.querySelector("input");
    if (!micChip || !input) return;
    input.disabled = true;
    micChip.classList.add("is-disabled");
    micChip.title = reason;
  };

  const checkMicCapability = async () => {
    // No server probe: microphone mode arrives as on-device speech recognition (PLAN B7).
    const reason = "microphone mode is coming as an on-device feature — this server never accepts audio";
    disableMicChip(reason);
    setNote(micNote, reason);
  };

  // Microphone capture returns as an on-device feature (no audio leaves the browser);
  // until then the chip is disabled with the server's stated reason.

  // ── mode toggle ──────────────────────────────────────────────────────────────

  const applyMode = () => {
    const current = mode();
    setupReplay.hidden = current !== "replay";
    setupMic.hidden = current !== "mic";
  };

  document
    .querySelectorAll('input[name="lv-mode"]')
    .forEach((radio) => radio.addEventListener("change", applyMode));

  applyMode();
  checkMicCapability();
  loadScenarios();
})();
