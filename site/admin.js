/* Qorğan admin dashboard — overview, priority queue with sort/search/filter, drill-down
   with a searchable calls table, and per-call on-demand model analysis (the rank graph:
   GET /api/admin/incidents/{id}/analysis — verdict, tags, trigger phrases, an excerpt).
   The full transcript is shown only after an explicit "open case" (POST .../open) by an
   investigator with a stated purpose, which the server writes to its tamper-evident audit
   log before answering (PLAN C4). Every request carries the analyst key (X-Analyst-Key);
   the key lives in sessionStorage (this tab only) and a 401 returns to the sign-in panel.
   All list management is client-side. */
(() => {
  "use strict";

  const kpisEl = document.getElementById("admKpis");
  const novelEl = document.getElementById("admNovelWrap");
  const queueWrap = document.getElementById("admQueueWrap");
  const ddEl = document.getElementById("admDrilldown");
  const refreshBtn = document.getElementById("admRefresh");
  const ingestBtn = document.getElementById("admIngest");
  const ingestNote = document.getElementById("admIngestNote");
  const searchInput = document.getElementById("admSearch");
  const queueCount = document.getElementById("admQueueCount");
  const statsSection = document.getElementById("admStatsSection");
  const statsEl = document.getElementById("admStats");
  const modal = document.getElementById("admModal");
  const modalTitle = document.getElementById("admModalTitle");
  const modalDot = document.getElementById("admModalDot");
  const modalClose = document.getElementById("admModalClose");
  const modalBackdrop = document.getElementById("admModalBackdrop");
  const consoleEl = document.getElementById("admConsole");
  const toolbarEl = document.getElementById("admToolbar");
  const sessionEl = document.getElementById("admSession");
  const whoEl = document.getElementById("admWho");
  const signOutBtn = document.getElementById("admSignOut");
  const signinSection = document.getElementById("admSignin");
  const signinForm = document.getElementById("admSigninForm");
  const keyInput = document.getElementById("admKey");
  const signinBtn = document.getElementById("admSigninBtn");
  const signinNote = document.getElementById("admSigninNote");
  if (!kpisEl || !queueWrap || !ddEl || !modal || !consoleEl || !signinForm) return;

  const SEED_HINT =
    "Run <code>python scripts/demo_seed.py</code> then " +
    "<code>python -m qorgan.analytics.pipeline</code>.";

  const KPI_LABELS = [
    ["incidents", "incidents analyzed"],
    ["organizations", "organizations"],
    ["novel_schemes", "new schemes"],
    ["pending_reports", "pending reports"],
  ];

  // dir: 1 ascending, -1 descending. Text columns default ascending, numeric descending.
  const QUEUE_COLUMNS = [
    { key: "name", label: "organization", value: (o) => o.name.toLowerCase(), dir: 1 },
    { key: "priority", label: "priority", value: (o) => o.priority, dir: -1 },
    { key: "incidents", label: "incidents", value: (o) => o.incidents, dir: -1 },
    { key: "numbers", label: "numbers", value: (o) => o.numbers.length, dir: -1 },
    { key: "last_activity", label: "last activity", value: (o) => o.last_activity || "", dir: -1 },
  ];
  const CALL_COLUMNS = [
    { key: "date", label: "date", value: (c) => c.date || "", dir: -1 },
    { key: "number", label: "number", value: (c) => c.number || "", dir: 1 },
    { key: "risk", label: "risk", value: (c) => c.risk, dir: -1 },
    { key: "excerpt", label: "excerpt" }, // not sortable
  ];

  let orgs = [];
  let selectedId = null;
  let queueSort = { key: "priority", dir: -1 };
  let queueFilter = "all";
  let queueQuery = "";
  let detail = null;
  let callSort = { key: "date", dir: -1 };
  let callQuery = "";
  let expandedCall = null;
  const analysisCache = new Map(); // `${incidentId}|${locale}` → analysis payload
  const openNotes = new Map(); // incidentId → {tone, text} shown under the open-case control

  // ── session: the analyst key lives in sessionStorage (this tab only), never localStorage ──

  const KEY_STORE = "qorgan.analystKey";
  const PURPOSE_LABELS = {
    pattern_review: "pattern review — confirm or dismiss this scheme or verdict",
    citizen_request: "citizen request — the reporter asked about their report",
    partner_request: "partner request — a bank or Anti-Fraud Center case query",
  };
  const readStoredKey = () => {
    try { return sessionStorage.getItem(KEY_STORE) || ""; } catch { return ""; }
  };
  const storeKey = (k) => {
    try { sessionStorage.setItem(KEY_STORE, k); } catch { /* storage blocked: memory only */ }
  };
  const forgetKey = () => {
    try { sessionStorage.removeItem(KEY_STORE); } catch { /* nothing was stored */ }
  };
  let analystKey = readStoredKey();
  let me = null; // {id, role, can_open_cases, open_purposes} from GET /api/admin/session

  class AuthError extends Error {}

  // Every console request carries the key; a 401 ends the session (key revoked or rotated).
  const api = async (path, opts = {}) => {
    const headers = { ...(opts.headers || {}), "X-Analyst-Key": analystKey };
    const res = await fetch(path, { ...opts, headers, cache: "no-store" });
    if (res.status === 401) {
      endSession("Your key is no longer accepted — sign in again.", "error");
      throw new AuthError("signed out");
    }
    return res;
  };

  const errorDetail = async (res) => {
    try {
      const body = await res.json();
      return typeof body.detail === "string" ? body.detail : `API returned ${res.status}`;
    } catch {
      return `API returned ${res.status}`;
    }
  };

  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

  const locale = () => document.querySelector('input[name="adm-loc"]:checked')?.value || "ru";

  const showEmpty = (target, html) => {
    target.className = "data-empty";
    target.innerHTML = html;
  };

  // Offset-based span highlighting (spans are validated verbatim server-side).
  const highlightSpans = (text, spans) => {
    const sorted = [...spans].sort((a, b) => a.start - b.start);
    let html = "";
    let cur = 0;
    for (const s of sorted) {
      if (s.start < cur || s.end > text.length) continue;
      html += esc(text.slice(cur, s.start)) + "<mark>" + esc(text.slice(s.start, s.end)) + "</mark>";
      cur = s.end;
    }
    return html + esc(text.slice(cur));
  };

  const sortedBy = (items, columns, sort) => {
    const col = columns.find((c) => c.key === sort.key);
    if (!col || !col.value) return [...items];
    return [...items].sort((a, b) => {
      const va = col.value(a);
      const vb = col.value(b);
      return (va < vb ? -1 : va > vb ? 1 : 0) * sort.dir;
    });
  };

  const headerRow = (columns, sort) =>
    columns
      .map((c) => {
        if (!c.value) return `<th scope="col">${esc(c.label)}</th>`;
        const active = c.key === sort.key;
        const ind = active ? (sort.dir === 1 ? "▲" : "▼") : "▼";
        return (
          `<th scope="col" class="sortable${active ? " is-sorted" : ""}" data-key="${esc(c.key)}">` +
          `${esc(c.label)}<span class="sort-ind">${ind}</span></th>`
        );
      })
      .join("");

  const toggleSort = (sort, columns, key) => {
    const col = columns.find((c) => c.key === key);
    if (!col || !col.value) return sort;
    if (sort.key === key) return { key, dir: -sort.dir };
    return { key, dir: col.dir };
  };

  // ── KPI row + novel banner ───────────────────────────────────────────────────

  const renderKpis = (kpis) => {
    kpisEl.innerHTML = KPI_LABELS.map(
      ([key, label]) =>
        `<div class="kpi-card"><div class="kpi-value">${esc(kpis[key])}</div>` +
        `<div class="kpi-label">${esc(label)}</div></div>`
    ).join("");
  };

  const renderNovel = (allOrgs) => {
    const novel = allOrgs.filter((o) => o.is_novel);
    if (!novel.length) {
      novelEl.innerHTML = "";
      return;
    }
    const items = novel
      .map((o) => `<li>${esc(o.name)} &middot; ${o.incidents} incidents</li>`)
      .join("");
    novelEl.innerHTML =
      `<div class="novel-banner mono"><b>NEW SCHEME${novel.length > 1 ? "S" : ""} DETECTED</b>` +
      `<ul>${items}</ul></div>`;
  };

  // ── priority queue (sort + search + filter) ──────────────────────────────────

  const visibleOrgs = () => {
    const q = queueQuery.trim().toLowerCase();
    const filtered = orgs.filter(
      (o) =>
        (queueFilter !== "novel" || o.is_novel) &&
        (!q ||
          o.name.toLowerCase().includes(q) ||
          o.id.toLowerCase().includes(q) ||
          o.numbers.some((n) => n.toLowerCase().includes(q)))
    );
    return sortedBy(filtered, QUEUE_COLUMNS, queueSort);
  };

  const renderQueue = () => {
    if (!orgs.length) {
      queueCount.textContent = "";
      showEmpty(queueWrap, `No organizations to display yet. ${SEED_HINT}`);
      return;
    }
    const visible = visibleOrgs();
    queueCount.textContent =
      visible.length === orgs.length
        ? `${orgs.length} organizations`
        : `${visible.length} of ${orgs.length} organizations`;
    if (!visible.length) {
      showEmpty(queueWrap, "No organizations match — clear the search or filter.");
      return;
    }
    const rows = visible
      .map(
        (o) => `<tr data-org="${esc(o.id)}" class="${o.id === selectedId ? "is-selected" : ""}">` +
          `<td>${esc(o.name)}${o.is_novel ? '<span class="queue-badge">NEW</span>' : ""}${feedbackBadge(o.feedback)}</td>` +
          `<td>${o.priority.toFixed(2)}</td>` +
          `<td>${o.incidents}</td>` +
          `<td>${o.numbers.length}</td>` +
          `<td>${esc(o.last_activity || "—")}</td></tr>`
      )
      .join("");
    queueWrap.className = "";
    queueWrap.innerHTML =
      `<table class="eval-table queue-table mono"><thead><tr>` +
      headerRow(QUEUE_COLUMNS, queueSort) +
      `</tr></thead><tbody>${rows}</tbody></table>`;
    queueWrap.querySelectorAll("th.sortable").forEach((th) => {
      th.addEventListener("click", () => {
        queueSort = toggleSort(queueSort, QUEUE_COLUMNS, th.dataset.key);
        renderQueue();
      });
    });
    queueWrap.querySelectorAll("tr[data-org]").forEach((tr) => {
      tr.addEventListener("click", () => selectOrg(tr.dataset.org));
    });
  };

  // ── modal shell ──────────────────────────────────────────────────────────────

  let lastFocus = null;

  const openModal = () => {
    if (!modal.hidden) return;
    lastFocus = document.activeElement;
    modal.hidden = false;
    document.body.style.overflow = "hidden";
    modalClose.focus();
  };

  const closeModal = () => {
    if (modal.hidden) return;
    modal.hidden = true;
    document.body.style.overflow = "";
    expandedCall = null;
    if (lastFocus?.focus) lastFocus.focus();
  };

  // ── drill-down (inside the modal): org header + searchable calls table ───────

  const renderDrilldown = () => {
    const badge = detail.is_novel ? '<span class="queue-badge">NEW SCHEME</span>' : "";
    const numbers = detail.numbers.length
      ? detail.numbers.map((n) => `<span class="tw-tag">${esc(n)}</span>`).join("")
      : '<span class="tw-dim">none linked</span>';
    const tactics = detail.tactics.length
      ? detail.tactics
          .map((t) => `<span class="tw-tag">${esc(t.name)}&nbsp;&middot;&nbsp;${t.count}</span>`)
          .join("")
      : '<span class="tw-dim">none</span>';
    const script = detail.representative_script
      ? `<div class="dd-section-label mono">representative script</div>` +
        `<div class="dd-script">${esc(detail.representative_script)}</div>`
      : "";

    const others = orgs.filter((o) => o.id !== detail.id);
    const mergeOptions = others.map((o) => `<option value="${esc(o.id)}">${esc(o.name)} (${o.incidents})</option>`).join("");
    const feedback =
      `<div class="dd-feedback mono">` +
      `<span class="dd-feedback-label">analyst verdict${detail.feedback ? ` · <b>${esc(detail.feedback)}</b>` : ""}</span>` +
      `<button type="button" class="btn dd-fb-btn" data-fb="confirm">Confirm</button>` +
      `<button type="button" class="btn dd-fb-btn" data-fb="dismiss">Dismiss</button>` +
      (others.length
        ? `<label class="dd-fb-merge">merge into <select id="ddMergeTarget">${mergeOptions}</select>` +
          `<button type="button" class="btn dd-fb-btn" data-fb="merge">Merge</button></label>`
        : "") +
      `<span class="dd-fb-note tw-dim">logged as ${esc(me ? me.id : "")}; a dismissed operation drops to 20 % priority</span>` +
      `</div>`;

    modalTitle.textContent = `organization · ${detail.id}`;
    modalDot.className = `p-dot${detail.is_novel ? "" : " tone-moss"}`;
    ddEl.innerHTML =
      `<div class="dd-title">${esc(detail.name)} ${badge}${feedbackBadge(detail.feedback)}</div>` +
      `<div class="dd-meta mono">priority ${detail.priority.toFixed(2)}</div>` +
      feedback +
      `<div class="dd-section-label mono">linked numbers</div><div class="dd-chips">${numbers}</div>` +
      `<div class="dd-section-label mono">tactic profile</div><div class="dd-chips">${tactics}</div>` +
      script +
      `<div class="dd-section-label mono">calls</div>` +
      `<div class="dd-call-tools mono">` +
      `<input id="ddCallSearch" class="dd-call-search" type="search" ` +
      `placeholder="search excerpt or number…" aria-label="Search calls" value="${esc(callQuery)}">` +
      `<span class="dd-call-note" id="ddCallNote"></span>` +
      `</div>` +
      `<div id="ddCalls"></div>`;
    ddEl.querySelectorAll("button[data-fb]").forEach((btn) => {
      btn.addEventListener("click", () => sendFeedback(detail.id, btn.dataset.fb));
    });
    document.getElementById("ddCallSearch").addEventListener("input", (ev) => {
      callQuery = ev.target.value;
      renderCalls();
    });
    renderCalls();
  };

  const visibleCalls = () => {
    const q = callQuery.trim().toLowerCase();
    const filtered = detail.sample_incidents.filter(
      (c) =>
        !q ||
        c.excerpt.toLowerCase().includes(q) ||
        (c.number || "").toLowerCase().includes(q) ||
        (c.date || "").includes(q)
    );
    return sortedBy(filtered, CALL_COLUMNS, callSort);
  };

  const analysisBlock = (incidentId) => {
    const cached = analysisCache.get(`${incidentId}|${locale()}`);
    let inner;
    if (!cached) {
      inner = `<div class="dd-loading">scoring with the model&hellip;</div>`;
    } else if (cached.error) {
      inner = `<div class="dd-loading">could not analyze — ${esc(cached.error)}</div>`;
    } else {
      const a = cached;
      const maxWeight = Math.max(...a.tags.map((t) => t.weight), 0.0001);
      const rank = a.tags.length
        ? a.tags
            .map(
              (t) =>
                `<div class="ec-row dd-rank-row"><span class="ec-name">${esc(t.name)}</span>` +
                `<span class="ec-bar"><i style="--w:${((t.weight / maxWeight) * 100).toFixed(0)}%"></i></span>` +
                `<span class="ec-val">${t.weight.toFixed(2)}</span></div>`
            )
            .join("")
        : '<p class="tw-dim">no tactic signals detected</p>';
      const verdict = a.flagged
        ? '<span class="tone-oxide">"scam_risk"</span>'
        : '<span class="tone-moss">"clear"</span>';
      inner =
        `<div class="dd-analysis-head mono">model verdict ${verdict}` +
        `&nbsp;&middot;&nbsp;risk ${a.risk.toFixed(2)} / threshold ${a.threshold.toFixed(2)}` +
        `&nbsp;&middot;&nbsp;backend ${esc(a.backend)}${a.fallback ? " (fallback)" : ""}</div>` +
        `<div class="dd-rank">${rank}</div>` +
        (a.transcript
          ? `<div class="dd-section-label mono">full transcript &middot; trigger phrases &middot; ` +
            `<span class="tone-moss">opened by ${esc(a.opened_by || "")} &middot; ${esc(a.opened_for || "")} &middot; logged</span></div>` +
            `<div class="dd-script dd-analysis-script">${highlightSpans(a.transcript, a.spans)}</div>`
          : `<div class="dd-section-label mono">excerpt &middot; trigger phrases</div>` +
            `<div class="dd-script dd-analysis-script">${esc(a.excerpt)}</div>` +
            `<div class="dd-spans">${a.spans.map((sp) => `<mark>${esc(sp.text)}</mark>`).join(" ") || '<span class="tw-dim">no trigger phrases in the excerpt</span>'}</div>` +
            (a.withheld_spans
              ? `<p class="dd-withheld mono">${a.withheld_spans} more trigger phrase${a.withheld_spans > 1 ? "s" : ""} beyond the excerpt &mdash; counted, not quoted, until the case is opened</p>`
              : "") +
            openControls(a.incident_id)) +
        `<p class="dd-analysis-reason">${esc(a.reason)}</p>` +
        `<p class="dd-analysis-caveat mono">${esc(a.caveat)}</p>`;
    }
    return `<tr class="dd-analysis-tr"><td colspan="4"><div class="dd-analysis">${inner}</div></td></tr>`;
  };

  // Investigator-only, purpose first: the button stays disabled until a purpose is chosen,
  // and for the analyst role the whole control is visibly locked with the reason.
  const openControls = (incidentId) => {
    const allowed = Boolean(me && me.can_open_cases);
    const who = me ? `<b>${esc(me.id)}</b> (${esc(me.role)})` : "";
    const purposes = ((me && me.open_purposes) || [])
      .map((p) => `<option value="${esc(p)}">${esc(PURPOSE_LABELS[p] || p)}</option>`)
      .join("");
    const locked = allowed ? "" : " disabled";
    const fine = allowed
      ? `Logged as ${who} with this incident and the purpose you choose. The note must not carry numbers or call text.`
      : `Opening a full transcript needs the <b>investigator</b> role. You are signed in as ${who}; ` +
        `the excerpt and trigger phrases above are what your role sees.`;
    const msg = openNotes.get(incidentId);
    return (
      `<div class="dd-open${allowed ? "" : " is-locked"}">` +
      `<div class="dd-section-label mono">open the full transcript &middot; audited</div>` +
      `<div class="dd-open-row">` +
      `<select class="dd-open-purpose mono" aria-label="Purpose for opening the full transcript"${locked}>` +
      `<option value="">choose a purpose…</option>${purposes}</select>` +
      `<input class="dd-open-note mono" type="text" maxlength="160" aria-label="Note for the audit log (optional)" ` +
      `placeholder="note (optional) — e.g. a ticket reference"${locked}>` +
      `<button type="button" class="btn dd-open-btn" data-open="${esc(incidentId)}" disabled>Open full transcript</button>` +
      `</div>` +
      `<p class="dd-open-fine mono">${fine}</p>` +
      (msg ? `<p class="dd-open-msg mono${msg.tone === "error" ? " is-error" : ""}" role="status">${esc(msg.text)}</p>` : "") +
      `</div>`
    );
  };

  const renderCalls = () => {
    const callsEl = document.getElementById("ddCalls");
    const noteEl = document.getElementById("ddCallNote");
    if (!callsEl) return;
    const total = detail.sample_incidents.length;
    const visible = visibleCalls();
    noteEl.textContent =
      (visible.length === total ? `${total} calls` : `${visible.length} of ${total} calls`) +
      " · click a call for the model analysis";
    if (!visible.length) {
      callsEl.innerHTML = '<p class="tw-dim mono">no calls match</p>';
      return;
    }
    const rows = visible
      .map((c) => {
        const expanded = c.id === expandedCall;
        return (
          `<tr data-incident="${esc(c.id)}" data-has-transcript="${c.has_transcript === false ? "0" : "1"}" class="${expanded ? "is-expanded" : ""}">` +
          `<td>${esc(c.date || "—")}</td><td>${esc(c.number || "—")}</td>` +
          `<td>${(c.risk * 100).toFixed(0)}%</td><td>${c.has_transcript === false ? '<span class="tw-dim">signals only — partner report without a transcript</span>' : esc(c.excerpt)}</td></tr>` +
          (expanded ? analysisBlock(c.id) : "")
        );
      })
      .join("");
    callsEl.innerHTML =
      `<table class="dd-samples mono"><thead><tr>` +
      headerRow(CALL_COLUMNS, callSort) +
      `</tr></thead><tbody>${rows}</tbody></table>`;
    callsEl.querySelectorAll("th.sortable").forEach((th) => {
      th.addEventListener("click", () => {
        callSort = toggleSort(callSort, CALL_COLUMNS, th.dataset.key);
        renderCalls();
      });
    });
    callsEl.querySelectorAll("tr[data-incident]").forEach((tr) => {
      if (tr.dataset.hasTranscript === "0") return; // nothing to analyse or open
      tr.addEventListener("click", () => expandCall(tr.dataset.incident));
    });
    if (!(me && me.can_open_cases)) return; // locked control: nothing to wire
    callsEl.querySelectorAll(".dd-open").forEach((box) => {
      const purpose = box.querySelector(".dd-open-purpose");
      const note = box.querySelector(".dd-open-note");
      const btn = box.querySelector("button[data-open]");
      purpose.addEventListener("change", () => {
        btn.disabled = !purpose.value;
      });
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        openCase(btn.dataset.open, purpose.value, note.value);
      });
    });
  };

  // The explicit, audited action (PLAN C4): investigator role + a stated purpose. The server
  // writes the audit line (who, which incident, why) before it answers with the transcript;
  // the identity comes from the key, never from anything this page claims.
  const openCase = async (incidentId, purpose, note) => {
    if (!purpose) return;
    const key = `${incidentId}|${locale()}`;
    const body = { purpose };
    if (note && note.trim()) body.note = note.trim();
    try {
      const res = await api(
        `/api/admin/incidents/${encodeURIComponent(incidentId)}/open?locale=${locale()}`,
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      );
      if (!res.ok) {
        openNotes.set(incidentId, { tone: "error", text: `not opened — ${await errorDetail(res)}` });
      } else {
        openNotes.delete(incidentId);
        analysisCache.set(key, { ...(await res.json()), opened_by: me ? me.id : "", opened_for: purpose });
      }
    } catch (e) {
      if (e instanceof AuthError) return;
      openNotes.set(incidentId, { tone: "error", text: `not opened — ${e.message || e}` });
    }
    if (expandedCall === incidentId) renderCalls();
  };

  // ── per-call model analysis (the rank graph) ─────────────────────────────────

  const expandCall = async (incidentId) => {
    expandedCall = expandedCall === incidentId ? null : incidentId;
    if (!expandedCall) {
      renderCalls();
      return;
    }
    const key = `${incidentId}|${locale()}`;
    if (analysisCache.get(key)?.error) analysisCache.delete(key); // retry failed ones
    renderCalls(); // shows the scoring placeholder if not cached yet
    if (analysisCache.has(key)) return;
    try {
      const res = await api(
        `/api/admin/incidents/${encodeURIComponent(incidentId)}/analysis?locale=${locale()}`
      );
      if (!res.ok) throw new Error(await errorDetail(res));
      analysisCache.set(key, await res.json());
    } catch (e) {
      if (e instanceof AuthError) return;
      analysisCache.set(key, { error: String(e.message || e) });
    }
    if (expandedCall === incidentId) renderCalls();
  };

  // ── analyst feedback (PLAN C6): confirm / dismiss / merge, applied server-side ──

  const feedbackBadge = (state) =>
    state ? `<span class="queue-badge fb-${esc(state)}">${esc(state.toUpperCase())}</span>` : "";

  const sendFeedback = async (orgId, action) => {
    const body = { action };
    if (action === "merge") {
      const target = document.getElementById("ddMergeTarget")?.value;
      if (!target) return;
      body.target_org_id = target;
    }
    try {
      const res = await api(`/api/admin/organizations/${encodeURIComponent(orgId)}/feedback?locale=${locale()}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(await errorDetail(res));
      const shown = await res.json();
      await load(); // the queue, KPIs and stats all change; re-read everything
      selectOrg(shown.id);
    } catch (e) {
      if (e instanceof AuthError) return;
      ddEl.insertAdjacentHTML("afterbegin", `<div class="dd-loading">feedback failed — ${esc(e.message || e)}</div>`);
    }
  };

  // ── data loading ─────────────────────────────────────────────────────────────

  const selectOrg = async (orgId) => {
    selectedId = orgId;
    expandedCall = null;
    queueWrap.querySelectorAll("tr[data-org]").forEach((tr) => {
      tr.classList.toggle("is-selected", tr.dataset.org === orgId);
    });
    openModal();
    modalTitle.textContent = `organization · ${orgId}`;
    modalDot.className = "p-dot";
    ddEl.innerHTML = '<div class="dd-loading">loading&hellip;</div>';
    try {
      const res = await api(
        `/api/admin/organizations/${encodeURIComponent(orgId)}?locale=${locale()}`
      );
      if (!res.ok) throw new Error(await errorDetail(res));
      const body = await res.json();
      if (selectedId !== orgId || modal.hidden) return; // stale response — user moved on
      detail = body;
      renderDrilldown();
    } catch (e) {
      if (e instanceof AuthError) return;
      ddEl.innerHTML =
        `<div class="dd-loading">Could not load this organization — ${esc(e.message || e)}</div>`;
    }
  };

  const load = async () => {
    kpisEl.setAttribute("aria-busy", "true");
    analysisCache.clear(); // locale-dependent names; cheap to re-score on demand
    openNotes.clear();
    closeModal();
    try {
      const res = await api(`/api/admin/overview?locale=${locale()}`);
      if (!res.ok) throw new Error(await errorDetail(res));
      const body = await res.json();

      if (!body.available) {
        kpisEl.innerHTML = "";
        novelEl.innerHTML = "";
        queueCount.textContent = "";
        statsSection.hidden = true;
        showEmpty(queueWrap, `No Level-2 analysis yet. ${SEED_HINT}`);
        return;
      }

      orgs = body.organizations;
      renderKpis(body.kpis);
      renderNovel(orgs);
      renderQueue();
      await loadStats();
    } catch (e) {
      if (e instanceof AuthError) return;
      kpisEl.innerHTML = "";
      novelEl.innerHTML = "";
      statsSection.hidden = true;
      showEmpty(
        queueWrap,
        `Dashboard offline &mdash; ${esc(e.message || e)}. Serve the page through the API: ` +
          "<code>python -m qorgan.api</code>."
      );
    } finally {
      kpisEl.removeAttribute("aria-busy");
    }
  };

  // Statistics are additive — a failure here must never take the dashboard down.
  const loadStats = async () => {
    try {
      const res = await api(`/api/admin/stats?locale=${locale()}`);
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      renderStats(await res.json());
    } catch {
      statsSection.hidden = true;
    }
  };

  // ── results / statistics (single data hue; status colors only with labels) ───

  let tipEl = null;
  const ensureTip = () => {
    if (!tipEl) {
      tipEl = document.createElement("div");
      tipEl.className = "st-tip mono";
      tipEl.hidden = true;
      document.body.appendChild(tipEl);
    }
    return tipEl;
  };

  const renderStats = (s) => {
    if (!s.available) {
      statsSection.hidden = true;
      return;
    }
    statsSection.hidden = false;

    const maxCount = Math.max(...s.activity.map((p) => p.count), 1);
    const bars = s.activity
      .map((p) => {
        const height = p.count ? Math.max(3, (p.count / maxCount) * 100).toFixed(1) : 0;
        return (
          `<div class="st-bar-slot" data-tip="${esc(p.date)} · ${p.count} calls">` +
          `<div class="st-bar" style="height:${height}%"></div></div>`
        );
      })
      .join("");
    const t = s.trend;
    const trendHtml =
      t.delta_pct === null
        ? `<span class="st-delta">${t.this_week} this week</span>`
        : `<span class="st-delta">${t.this_week} this week&nbsp;&middot;&nbsp;` +
          `${t.delta_pct >= 0 ? "▲" : "▼"}&nbsp;${Math.abs(t.delta_pct).toFixed(0)}% vs prior week</span>`;
    const first = s.activity[0]?.date || "";
    const last = s.activity[s.activity.length - 1]?.date || "";
    const activityCard =
      `<div class="st-card st-card--wide">` +
      `<div class="st-card-head mono"><span>incidents recorded · last 30 days</span>${trendHtml}</div>` +
      `<div class="st-bars" role="img" aria-label="Incidents per day over the last 30 days">${bars}</div>` +
      `<div class="st-axis mono"><span>${esc(first)}</span><span>${esc(last)}</span></div>` +
      `</div>`;

    const r = s.reports;
    const rate = r.submitted ? Math.round((r.ingested / r.submitted) * 100) : null;
    const strip = r.submitted
      ? `<div class="st-strip" role="img" aria-label="${r.ingested} ingested, ${r.pending} pending">` +
        (r.ingested ? `<span class="st-seg st-seg--done" style="flex-grow:${r.ingested}"></span>` : "") +
        (r.pending ? `<span class="st-seg st-seg--pending" style="flex-grow:${r.pending}"></span>` : "") +
        `</div>` +
        `<div class="st-legend mono">` +
        `<span><i class="st-dot st-dot--done"></i>ingested ${r.ingested}</span>` +
        `<span><i class="st-dot st-dot--pending"></i>pending ${r.pending}</span></div>`
      : `<p class="st-empty mono">no citizen reports yet — they arrive from the live-call page</p>`;
    const reportsCard =
      `<div class="st-card">` +
      `<div class="st-card-head mono"><span>citizen reports processed</span></div>` +
      `<div class="st-hero">${r.ingested}` +
      `<span class="st-hero-sub">&nbsp;of ${r.submitted} ingested${rate === null ? "" : ` · ${rate}%`}</span></div>` +
      strip +
      `</div>`;

    const maxIncidents = Math.max(...s.top_organizations.map((o) => o.incidents), 1);
    const orgRows = s.top_organizations
      .map(
        (o) =>
          `<div class="st-org-row">` +
          `<span class="st-org-name">${esc(o.name)}${o.is_novel ? '<span class="queue-badge">NEW</span>' : ""}</span>` +
          `<span class="st-org-bar"><i style="--w:${((o.incidents / maxIncidents) * 100).toFixed(0)}%"></i></span>` +
          `<span class="st-org-val">${o.incidents}</span></div>`
      )
      .join("");
    const topCard =
      `<div class="st-card">` +
      `<div class="st-card-head mono"><span>largest organizations · incidents</span></div>` +
      `<div class="st-orgs">${orgRows}</div>` +
      `</div>`;

    statsEl.innerHTML = activityCard + reportsCard + topCard;

    const tip = ensureTip();
    statsEl.querySelectorAll(".st-bar-slot").forEach((slot) => {
      slot.addEventListener("mouseenter", () => {
        tip.textContent = slot.dataset.tip;
        tip.hidden = false;
      });
      slot.addEventListener("mousemove", (ev) => {
        tip.style.left = `${ev.clientX + 12}px`;
        tip.style.top = `${ev.clientY - 32}px`;
      });
      slot.addEventListener("mouseleave", () => {
        tip.hidden = true;
      });
    });
  };

  // "Ingest reports": the analyst's explicit click that folds pending citizen reports
  // (from the live-call page) into the cluster analysis — never automatic.
  const runIngest = async () => {
    if (!ingestBtn) return;
    ingestBtn.disabled = true;
    ingestBtn.textContent = "ingesting…";
    ingestNote.className = "adm-note";
    ingestNote.textContent = "embedding new reports…";
    try {
      const res = await api(`/api/admin/ingest?locale=${locale()}`, { method: "POST" });
      if (!res.ok) throw new Error(await errorDetail(res));
      const body = await res.json();
      ingestNote.className = "adm-note is-success";
      if (!body.ingested) {
        ingestNote.textContent = "no pending reports to ingest";
      } else {
        const placed = body.placements
          .map((p) => `${p.org_name}${p.org_is_novel ? " (new scheme)" : ""}`)
          .join("; ");
        ingestNote.textContent = `ingested ${body.ingested} · ${placed}`;
      }
      await load();
    } catch (e) {
      if (e instanceof AuthError) return;
      ingestNote.className = "adm-note is-error";
      ingestNote.textContent = `ingest failed — ${e.message || e}`;
    } finally {
      ingestBtn.disabled = false;
      ingestBtn.textContent = "ingest reports";
    }
  };

  // ── sign-in / sign-out ─────────────────────────────────────────────────────────

  const showSignin = (message = "", tone = "") => {
    consoleEl.hidden = true;
    toolbarEl.hidden = true;
    sessionEl.hidden = true;
    signinSection.hidden = false;
    signinNote.className = `adm-note${tone ? ` is-${tone}` : ""}`;
    signinNote.textContent = message;
    keyInput.value = "";
    keyInput.focus();
  };

  const showConsole = () => {
    signinSection.hidden = true;
    consoleEl.hidden = false;
    toolbarEl.hidden = false;
    sessionEl.hidden = false;
    whoEl.innerHTML =
      `signed in as <b>${esc(me.id)}</b> <span class="adm-role adm-role--${esc(me.role)}">${esc(me.role)}</span>` +
      (me.can_open_cases ? "" : ' <span class="adm-who-note">aggregates &amp; excerpts; opening a transcript needs an investigator</span>');
  };

  // Forget the key and everything rendered with it.
  const endSession = (message, tone) => {
    analystKey = "";
    me = null;
    forgetKey();
    analysisCache.clear();
    openNotes.clear();
    orgs = [];
    detail = null;
    closeModal();
    kpisEl.innerHTML = "";
    novelEl.innerHTML = "";
    queueWrap.innerHTML = "";
    statsEl.innerHTML = "";
    statsSection.hidden = true;
    queueCount.textContent = "";
    ingestNote.textContent = "";
    showSignin(message, tone);
  };

  // GET /api/admin/session checks the key and names who is signed in (an audited session start).
  const startSession = async (key) => {
    signinBtn.disabled = true;
    signinNote.className = "adm-note";
    signinNote.textContent = "checking the key…";
    try {
      const res = await fetch("/api/admin/session", { headers: { "X-Analyst-Key": key }, cache: "no-store" });
      if (!res.ok) {
        const why =
          res.status === 401 ? "That key was not accepted."
          : res.status === 429 ? "Too many failed attempts from this address — wait a minute."
          : await errorDetail(res);
        analystKey = "";
        forgetKey();
        showSignin(why, "error");
        return;
      }
      me = await res.json();
      analystKey = key;
      storeKey(key);
      showConsole();
      await load();
    } catch (e) {
      analystKey = "";
      showSignin(`Could not reach the server — ${e.message || e}. Serve the page through the API: python -m qorgan.api`, "error");
    } finally {
      signinBtn.disabled = false;
    }
  };

  // ── wiring ───────────────────────────────────────────────────────────────────

  signinForm.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const key = keyInput.value.trim();
    if (!key) {
      showSignin("Paste the analyst key issued to you.", "error");
      return;
    }
    startSession(key);
  });
  signOutBtn?.addEventListener("click", () =>
    endSession("Signed out — the key was removed from this tab.", "success"));

  searchInput?.addEventListener("input", (ev) => {
    queueQuery = ev.target.value;
    renderQueue();
  });
  document.querySelectorAll('input[name="adm-filter"]').forEach((radio) =>
    radio.addEventListener("change", () => {
      queueFilter = document.querySelector('input[name="adm-filter"]:checked')?.value || "all";
      renderQueue();
    })
  );
  document
    .querySelectorAll('input[name="adm-loc"]')
    .forEach((radio) => radio.addEventListener("change", load));
  refreshBtn?.addEventListener("click", load);
  ingestBtn?.addEventListener("click", runIngest);
  modalClose.addEventListener("click", closeModal);
  modalBackdrop.addEventListener("click", closeModal);
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") closeModal();
  });

  if (analystKey) startSession(analystKey);
  else showSignin();
})();
