/* Qorğan live call — replay (POST /api/live/session + /utterance + /end) and
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
  let lastSessionId = null; // set once a call finishes; consumed by the report button

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
      `The transcript and detected tactics join the analyst dashboard's pending queue.</p>` +
      `</div>` +
      `</div>`;
    document.getElementById("lvReportSend").addEventListener("click", sendReport);
  };

  const sendReport = async () => {
    const btn = document.getElementById("lvReportSend");
    const note = document.getElementById("lvReportNote");
    const phone = document.getElementById("lvReportPhone").value.trim();
    if (!lastSessionId) return;
    btn.disabled = true;
    try {
      const res = await fetch(`/api/live/session/${encodeURIComponent(lastSessionId)}/report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone_number: phone || null }),
      });
      if (res.status === 404) throw new Error("this call was already reported");
      if (res.status === 422) throw new Error("nothing to report — no utterances were scored");
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      const body = await res.json();
      note.className = "lv-report-note is-success";
      note.innerHTML =
        `Report <b>${esc(body.report_id)}</b> submitted — it is now a pending report on the ` +
        `<a href="admin.html">analyst dashboard</a>, where “ingest reports” folds it into the cluster analysis.`;
    } catch (e) {
      btn.disabled = false;
      note.className = "lv-report-note is-error";
      note.textContent = `Could not submit — ${e.message || e}`;
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
    turnEl.textContent = "0";
    scoreEl.textContent = "0 / 100";
    meterFill.style.width = "0%";
    setBand("low");
    callWrap.hidden = false;
  };

  // ── replay mode ──────────────────────────────────────────────────────────────

  const loadScenarios = async () => {
    try {
      const res = await fetch("/api/live/scenarios");
      if (!res.ok) throw new Error(`API returned ${res.status}`);
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
      const createRes = await fetch("/api/live/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ locale: locale(), backend: null }),
      });
      if (!createRes.ok) throw new Error(`session create returned ${createRes.status}`);
      const { session_id: sessionId } = await createRes.json();

      for (const line of lines) {
        await sleep(TURN_DELAY_MS);
        const res = await fetch(`/api/live/session/${sessionId}/utterance`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: line }),
        });
        if (!res.ok) throw new Error(`utterance returned ${res.status}`);
        applyUpdate(await res.json(), line);
      }

      const endRes = await fetch(`/api/live/session/${sessionId}/end`, { method: "POST" });
      if (!endRes.ok) throw new Error(`end returned ${endRes.status}`);
      renderSummary(await endRes.json(), sessionId);
    } catch (e) {
      setNote(
        setupNote,
        `Live analysis offline — ${e.message || e}. Serve the page through the API: python -m qorgan.api`,
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
    try {
      const res = await fetch("/api/live/capabilities");
      const body = await res.json();
      if (!body.microphone) {
        disableMicChip(body.reason || "microphone mode is not available");
        setNote(micNote, body.reason || "");
        return;
      }
    } catch {
      disableMicChip("could not reach /api/live/capabilities");
    }
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
