/* Qorğan try-widget — the real connection: POST /api/analyze on the qorgan API. */
(() => {
  "use strict";

  const SAMPLES = {
    scam: "Здравствуйте, это служба безопасности банка. По вашему счёту зафиксирована подозрительная операция. Чтобы спасти деньги, переведите их на безопасный счёт. Никому не говорите об этом звонке и назовите код из сообщения.",
    legit: "Здравствуйте, это банк. По вашей заявке: карта готова, можете забрать её в отделении с удостоверением. Ничего переводить и называть не нужно, коды никому не сообщайте. Хорошего дня.",
  };

  const input = document.getElementById("twInput");
  const go = document.getElementById("twGo");
  const out = document.getElementById("twResult");
  if (!input || !go || !out) return;

  document.querySelectorAll(".tw-chip[data-sample]").forEach((btn) => {
    btn.addEventListener("click", () => {
      input.value = SAMPLES[btn.dataset.sample] || "";
      input.focus();
    });
  });

  const esc = (s) => s.replace(/[&<>"']/g, (ch) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

  // transcript with attributed spans wrapped in <mark>, offsets are character-based
  const highlight = (text, spans) => {
    const sorted = [...spans].sort((a, b) => a.start - b.start);
    let html = "", cur = 0;
    for (const s of sorted) {
      if (s.start < cur || s.end > text.length) continue;
      html += esc(text.slice(cur, s.start)) + "<mark>" + esc(text.slice(s.start, s.end)) + "</mark>";
      cur = s.end;
    }
    return html + esc(text.slice(cur));
  };

  const render = (r, transcript) => {
    const verdictClass = r.flagged ? "tone-oxide" : "tone-moss";
    const verdictWord = r.flagged ? "scam patterns detected" : "no scam patterns";
    const tags = r.tags.length
      ? r.tags.map((t) => `<span class="tw-tag">${esc(t.id)}&nbsp;·&nbsp;${t.weight.toFixed(2)}</span>`).join("")
      : '<span class="tw-dim">none</span>';
    out.innerHTML =
      `<div class="tw-verdict ${verdictClass}">label: <b>${r.flagged ? '"scam_risk"' : '"clear"'}</b>` +
      `&nbsp;·&nbsp;risk ${r.risk.toFixed(2)} / threshold ${r.threshold.toFixed(2)}&nbsp;·&nbsp;${verdictWord}</div>` +
      `<div class="tw-tags">${tags}</div>` +
      (r.spans.length ? `<div class="tw-transcript">${highlight(transcript, r.spans)}</div>` : "") +
      `<p class="tw-reason">${esc(r.explanation.reason)}</p>` +
      `<p class="tw-dim">${esc(r.explanation.caveat)} ${esc(r.explanation.human_note)}</p>` +
      `<p class="tw-dim">backend: ${esc(r.backend)}${r.fallback ? " (fallback — configured backend unavailable)" : ""}</p>`;
    out.hidden = false;
  };

  const fail = (msg) => {
    out.innerHTML = `<div class="tw-verdict tone-oxide">${esc(msg)}</div>` +
      `<p class="tw-dim">serve the page through the API:&nbsp;python -m qorgan.api&nbsp;→&nbsp;http://localhost:8000</p>`;
    out.hidden = false;
  };

  go.addEventListener("click", async () => {
    const transcript = input.value.trim();
    if (!transcript) { input.focus(); return; }
    const locale = document.querySelector('input[name="loc"]:checked')?.value || "ru";
    go.disabled = true;
    go.textContent = "Analyzing…";
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript, locale }),
      });
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      render(await res.json(), transcript);
    } catch (e) {
      fail(`analyzer offline — ${e.message || e}`);
    } finally {
      go.disabled = false;
      go.textContent = "Analyze";
    }
  });
})();
