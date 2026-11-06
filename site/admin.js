/* Qorğan admin dashboard — overview, priority queue with sort/search/filter, drill-down
   with a searchable calls table, and per-call on-demand model analysis (the rank graph:
   GET /api/admin/incidents/{id}/analysis — verdict, tags, trigger phrases, an excerpt).
   The full transcript is shown only after an explicit "open case" (POST .../open) by an
   investigator with a stated purpose, which the server writes to its tamper-evident audit
   log before answering (PLAN C4). Every request carries the analyst key (X-Analyst-Key);
   the key lives in sessionStorage (this tab only) and a 401 returns to the sign-in panel.
   All list management is client-side. The console speaks kk / ru / en (site/i18n-admin.js), from
   one control shared with the other pages; what the server returns (tactic names, reasons) is
   asked for in the same language. */

import { DEFAULT_LOCALE, LOCALES, pickLocale } from "./i18n.js";
import { markSpans } from "./core/highlight.js";
import { adminText } from "./i18n-admin.js";
import { browserLanguages, localize, readStoredLocale, renderEl, storeLocale } from "./i18n-dom.js";

(() => {
  const demoNoteEl = document.getElementById("admDemoNote");
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

  let lang = pickLocale({ stored: readStoredLocale(), languages: browserLanguages() });
  const locale = () => lang;
  const tr = (key, params = null) => adminText.t(lang, key, params);
  const trHtml = (key, params = null) => adminText.tHtml(lang, key, params);

  /** Bind an element to a string, so it follows a language switch; a null key clears it. */
  const bind = (el, key, params = null) => {
    if (!key) {
      delete el.dataset.i18n;
      delete el.dataset.i18nParams;
      el.textContent = "";
      return;
    }
    el.dataset.i18n = key;
    if (params) el.dataset.i18nParams = JSON.stringify(params);
    else delete el.dataset.i18nParams;
    renderEl(el, lang, adminText);
  };

  const KPI_LABELS = [
    ["incidents", "admin.kpi_incidents"],
    ["organizations", "admin.kpi_organizations"],
    ["novel_schemes", "admin.kpi_novel"],
    ["pending_reports", "admin.kpi_pending"],
  ];

  // dir: 1 ascending, -1 descending. Text columns default ascending, numeric descending.
  const QUEUE_COLUMNS = [
    { key: "name", label: "admin.col_organization", value: (o) => o.name.toLowerCase(), dir: 1 },
    { key: "priority", label: "admin.col_priority", value: (o) => o.priority, dir: -1 },
    { key: "incidents", label: "admin.col_incidents", value: (o) => o.incidents, dir: -1 },
    { key: "numbers", label: "admin.col_numbers", value: (o) => o.numbers.length, dir: -1 },
    { key: "last_activity", label: "admin.col_last_activity", value: (o) => o.last_activity || "", dir: -1 },
  ];
  const CALL_COLUMNS = [
    { key: "date", label: "admin.col_date", value: (c) => c.date || "", dir: -1 },
    { key: "number", label: "admin.col_number", value: (c) => c.number || "", dir: 1 },
    { key: "risk", label: "admin.col_risk", value: (c) => c.risk, dir: -1 },
    { key: "excerpt", label: "admin.col_excerpt" }, // not sortable
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
  const purposeLabel = (code) => {
    const label = tr(`admin.purpose_${code}`);
    return label === `admin.purpose_${code}` ? code : label; // a code this page does not know yet
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
    // Keyless only on a server with open demo access (ADR D59): it answers as `public-demo`.
    const headers = { ...(opts.headers || {}), ...(analystKey ? { "X-Analyst-Key": analystKey } : {}) };
    const res = await fetch(path, { ...opts, headers, cache: "no-store" });
    if (res.status === 401) {
      endSession("admin.err_key_revoked", "error");
      throw new AuthError("signed out");
    }
    return res;
  };

  const errorDetail = async (res) => {
    try {
      const body = await res.json();
      return typeof body.detail === "string" ? body.detail : tr("admin.err_api", { status: res.status });
    } catch {
      return tr("admin.err_api", { status: res.status });
    }
  };

  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

  const showEmpty = (target, html) => {
    target.className = "data-empty";
    target.innerHTML = html;
  };

  const highlightSpans = (text, spans) => markSpans(text, spans, esc);

  const sortedBy = (items, columns, sort) => {
    const col = columns.find((c) => c.key === sort.key);
    if (!col || !col.value) return [...items];
    return [...items].sort((a, b) => {
      const va = col.value(a);
      const vb = col.value(b);
      return (va < vb ? -1 : va > vb ? 1 : 0) * sort.dir;
    });
  };

  // A real <button> in each sortable header (keyboard + screen readers); the header carries
  // aria-sort for the active column.
  const headerRow = (columns, sort) =>
    columns
      .map((c) => {
        if (!c.value) return `<th scope="col">${esc(tr(c.label))}</th>`;
        const active = c.key === sort.key;
        const ind = active ? (sort.dir === 1 ? "▲" : "▼") : "▼";
        const ariaSort = active ? ` aria-sort="${sort.dir === 1 ? "ascending" : "descending"}"` : "";
        return (
          `<th scope="col" class="sortable${active ? " is-sorted" : ""}" data-key="${esc(c.key)}"${ariaSort}>` +
          `<button type="button" class="adm-sortbtn" data-sort="${esc(c.key)}">${esc(tr(c.label))}` +
          `<span class="sort-ind" aria-hidden="true">${ind}</span></button></th>`
        );
      })
      .join("");

  // Re-rendering a table replaces its buttons; keep keyboard focus on the equivalent control.
  const focusedControl = (root) => {
    const el = document.activeElement;
    if (!el || !root.contains(el)) return null;
    for (const attr of ["sort", "openOrg", "expand"]) if (el.dataset[attr]) return [attr, el.dataset[attr]];
    return null;
  };
  const restoreFocus = (root, control) => {
    if (!control) return;
    const [attr, value] = control;
    const name = attr.replace(/[A-Z]/g, (ch) => `-${ch.toLowerCase()}`);
    root.querySelector(`[data-${name}="${CSS.escape(value)}"]`)?.focus();
  };

  const roleName = (role) => {
    const name = tr(`admin.role_${role}`);
    return name === `admin.role_${role}` ? role : name;
  };

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
        `<div class="kpi-label">${esc(tr(label))}</div></div>`
    ).join("");
  };

  const renderNovel = (allOrgs) => {
    const novel = allOrgs.filter((o) => o.is_novel);
    if (!novel.length) {
      novelEl.innerHTML = "";
      return;
    }
    const items = novel
      .map((o) => `<li>${esc(tr("admin.novel_item", { name: o.name, n: o.incidents }))}</li>`)
      .join("");
    novelEl.innerHTML =
      `<div class="novel-banner mono"><b>${esc(tr("admin.novel_title"))}</b>` +
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
      showEmpty(queueWrap, `${esc(tr("admin.queue_empty"))} ${trHtml("admin.seed_hint_html")}`);
      return;
    }
    const visible = visibleOrgs();
    queueCount.textContent =
      visible.length === orgs.length
        ? tr("admin.queue_count", { n: orgs.length })
        : tr("admin.queue_count_filtered", { shown: visible.length, n: orgs.length });
    if (!visible.length) {
      showEmpty(queueWrap, esc(tr("admin.queue_no_match")));
      return;
    }
    const focused = focusedControl(queueWrap);
    // The whole row opens the organization for a pointer; the name is a real button, so the
    // queue works from the keyboard too.
    const rows = visible
      .map(
        (o) => `<tr data-org="${esc(o.id)}" class="${o.id === selectedId ? "is-selected" : ""}">` +
          `<td><button type="button" class="adm-rowbtn" data-open-org="${esc(o.id)}">${esc(o.name)}</button>` +
          `${o.is_novel ? `<span class="queue-badge">${esc(tr("admin.badge_new"))}</span>` : ""}${feedbackBadge(o.feedback)}</td>` +
          `<td>${o.priority.toFixed(2)}</td>` +
          `<td>${o.incidents}</td>` +
          `<td>${o.numbers.length}</td>` +
          `<td>${esc(o.last_activity || "—")}</td></tr>`
      )
      .join("");
    queueWrap.className = "";
    queueWrap.innerHTML =
      `<div class="table-scroll" tabindex="0" role="region" aria-label="${esc(tr("admin.queue_table_aria"))}"><table class="eval-table queue-table mono"><thead><tr>` +
      headerRow(QUEUE_COLUMNS, queueSort) +
      `</tr></thead><tbody>${rows}</tbody></table></div>`;
    restoreFocus(queueWrap, focused);
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
  // While the dialog is open the page behind it is inert: Tab stays in the dialog and a screen
  // reader cannot wander into the queue underneath.
  const background = [document.querySelector("header.nav"), document.querySelector("main"), document.querySelector("footer")].filter(Boolean);

  const openModal = () => {
    if (!modal.hidden) return;
    lastFocus = document.activeElement;
    modal.hidden = false;
    background.forEach((el) => { el.inert = true; });
    document.body.style.overflow = "hidden";
    modalClose.focus();
  };

  const closeModal = () => {
    if (modal.hidden) return;
    modal.hidden = true;
    background.forEach((el) => { el.inert = false; });
    document.body.style.overflow = "";
    expandedCall = null;
    // The queue may have been re-rendered meanwhile: fall back to the same organization's button.
    const target = lastFocus?.isConnected
      ? lastFocus
      : selectedId && queueWrap.querySelector(`[data-open-org="${CSS.escape(selectedId)}"]`);
    target?.focus?.();
  };

  // ── drill-down (inside the modal): org header + searchable calls table ───────

  const renderDrilldown = () => {
    const badge = detail.is_novel ? `<span class="queue-badge">${esc(tr("admin.badge_new_scheme"))}</span>` : "";
    const numbers = detail.numbers.length
      ? detail.numbers.map((n) => `<span class="tw-tag">${esc(n)}</span>`).join("")
      : `<span class="tw-dim">${esc(tr("admin.none_linked"))}</span>`;
    const tactics = detail.tactics.length
      ? detail.tactics
          .map((t) => `<span class="tw-tag" lang="${locale()}">${esc(t.name)}&nbsp;&middot;&nbsp;${t.count}</span>`)
          .join("")
      : `<span class="tw-dim">${esc(tr("admin.none"))}</span>`;
    const script = detail.representative_script
      ? `<div class="dd-section-label mono">${esc(tr("admin.dd_script"))}</div>` +
        `<div class="dd-script">${esc(detail.representative_script)}</div>`
      : "";

    const others = orgs.filter((o) => o.id !== detail.id);
    const mergeOptions = others.map((o) => `<option value="${esc(o.id)}">${esc(o.name)} (${o.incidents})</option>`).join("");
    const feedback =
      `<div class="dd-feedback mono">` +
      `<span class="dd-feedback-label">${esc(tr("admin.dd_verdict"))}${detail.feedback ? ` · <b>${esc(feedbackName(detail.feedback))}</b>` : ""}</span>` +
      `<button type="button" class="btn dd-fb-btn" data-fb="confirm">${esc(tr("admin.fb_confirm"))}</button>` +
      `<button type="button" class="btn dd-fb-btn" data-fb="dismiss">${esc(tr("admin.fb_dismiss"))}</button>` +
      (others.length
        ? `<label class="dd-fb-merge">${esc(tr("admin.fb_merge_into"))} <select id="ddMergeTarget">${mergeOptions}</select>` +
          `<button type="button" class="btn dd-fb-btn" data-fb="merge">${esc(tr("admin.fb_merge"))}</button></label>`
        : "") +
      `<span class="dd-fb-note tw-dim">${esc(tr("admin.fb_note", { id: me ? me.id : "" }))}</span>` +
      `</div>`;

    modalTitle.textContent = tr("admin.dd_title", { id: detail.id });
    modalDot.className = `p-dot${detail.is_novel ? "" : " tone-moss"}`;
    ddEl.innerHTML =
      `<div class="dd-title">${esc(detail.name)} ${badge}${feedbackBadge(detail.feedback)}</div>` +
      `<div class="dd-meta mono">${esc(tr("admin.dd_priority", { p: detail.priority.toFixed(2) }))}</div>` +
      feedback +
      `<div class="dd-section-label mono">${esc(tr("admin.dd_numbers"))}</div><div class="dd-chips">${numbers}</div>` +
      `<div class="dd-section-label mono">${esc(tr("admin.dd_tactics"))}</div><div class="dd-chips">${tactics}</div>` +
      script +
      `<div class="dd-section-label mono">${esc(tr("admin.dd_calls"))}</div>` +
      `<div class="dd-call-tools mono">` +
      `<input id="ddCallSearch" class="dd-call-search" type="search" ` +
      `placeholder="${esc(tr("admin.calls_search_placeholder"))}" aria-label="${esc(tr("admin.calls_search_aria"))}" value="${esc(callQuery)}">` +
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
      inner = `<div class="dd-loading">${esc(tr("admin.scoring"))}</div>`;
    } else if (cached.error) {
      inner = `<div class="dd-loading">${esc(tr("admin.analysis_failed", { error: cached.error }))}</div>`;
    } else {
      const a = cached;
      const maxWeight = Math.max(...a.tags.map((t) => t.weight), 0.0001);
      const rank = a.tags.length
        ? a.tags
            .map(
              (t) =>
                `<div class="ec-row dd-rank-row"><span class="ec-name" lang="${locale()}">${esc(t.name)}</span>` +
                `<span class="ec-bar"><i style="--w:${((t.weight / maxWeight) * 100).toFixed(0)}%"></i></span>` +
                `<span class="ec-val">${t.weight.toFixed(2)}</span></div>`
            )
            .join("")
        : `<p class="tw-dim">${esc(tr("admin.no_signals"))}</p>`;
      const verdict = a.flagged
        ? '<span class="tone-oxide">"scam_risk"</span>'
        : '<span class="tone-moss">"clear"</span>';
      inner =
        `<div class="dd-analysis-head mono">${esc(tr("admin.model_verdict"))} ${verdict}` +
        `&nbsp;&middot;&nbsp;${esc(tr("admin.risk_threshold", { risk: a.risk.toFixed(2), threshold: a.threshold.toFixed(2) }))}` +
        `&nbsp;&middot;&nbsp;${esc(tr("admin.backend", { backend: a.backend }))}${a.fallback ? ` ${esc(tr("admin.backend_fallback"))}` : ""}</div>` +
        `<div class="dd-rank">${rank}</div>` +
        (a.transcript
          ? `<div class="dd-section-label mono">${esc(tr("admin.transcript_label"))} &middot; ` +
            `<span class="tone-moss">${esc(tr("admin.opened_by", { by: a.opened_by || "", purpose: a.opened_for || "" }))}</span></div>` +
            `<div class="dd-script dd-analysis-script">${highlightSpans(a.transcript, a.spans)}</div>`
          : `<div class="dd-section-label mono">${esc(tr("admin.excerpt_label"))}</div>` +
            `<div class="dd-script dd-analysis-script">${esc(a.excerpt)}</div>` +
            `<div class="dd-spans">${a.spans.map((sp) => `<mark>${esc(sp.text)}</mark>`).join(" ") || `<span class="tw-dim">${esc(tr("admin.no_spans"))}</span>`}</div>` +
            (a.withheld_spans
              ? `<p class="dd-withheld mono">${esc(tr("admin.withheld", { n: a.withheld_spans }))}</p>`
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
    const who = { id: me ? me.id : "", role: me ? roleName(me.role) : "" };
    const purposes = ((me && me.open_purposes) || [])
      .map((p) => `<option value="${esc(p)}">${esc(purposeLabel(p))}</option>`)
      .join("");
    const locked = allowed ? "" : " disabled";
    const fine = trHtml(allowed ? "admin.open_fine_html" : "admin.open_locked_html", who);
    const msg = openNotes.get(incidentId);
    return (
      `<div class="dd-open${allowed ? "" : " is-locked"}">` +
      `<div class="dd-section-label mono">${esc(tr("admin.open_title"))}</div>` +
      `<div class="dd-open-row">` +
      `<select class="dd-open-purpose mono" aria-label="${esc(tr("admin.open_purpose_aria"))}"${locked}>` +
      `<option value="">${esc(tr("admin.open_choose"))}</option>${purposes}</select>` +
      `<input class="dd-open-note mono" type="text" maxlength="160" aria-label="${esc(tr("admin.open_note_aria"))}" ` +
      `placeholder="${esc(tr("admin.open_note_placeholder"))}"${locked}>` +
      `<button type="button" class="btn dd-open-btn" data-open="${esc(incidentId)}" disabled>${esc(tr("admin.open_button"))}</button>` +
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
      (visible.length === total ? tr("admin.calls_count", { n: total }) : tr("admin.calls_count_filtered", { shown: visible.length, n: total })) +
      ` · ${tr("admin.calls_hint")}`;
    if (!visible.length) {
      callsEl.innerHTML = `<p class="tw-dim mono">${esc(tr("admin.calls_no_match"))}</p>`;
      return;
    }
    const focused = focusedControl(callsEl);
    // As in the queue: the row is the pointer target, the date is the keyboard one.
    const rows = visible
      .map((c) => {
        const expanded = c.id === expandedCall;
        const date = esc(c.date || "—");
        return (
          `<tr data-incident="${esc(c.id)}" data-has-transcript="${c.has_transcript === false ? "0" : "1"}" class="${expanded ? "is-expanded" : ""}">` +
          `<td>${c.has_transcript === false ? date : `<button type="button" class="adm-rowbtn" data-expand="${esc(c.id)}" aria-expanded="${expanded}">${date}</button>`}</td>` +
          `<td>${esc(c.number || "—")}</td>` +
          `<td>${(c.risk * 100).toFixed(0)}%</td><td>${c.has_transcript === false ? `<span class="tw-dim">${esc(tr("admin.signals_only"))}</span>` : esc(c.excerpt)}</td></tr>` +
          (expanded ? analysisBlock(c.id) : "")
        );
      })
      .join("");
    callsEl.innerHTML =
      `<div class="table-scroll" tabindex="0" role="region" aria-label="${esc(tr("admin.calls_table_aria"))}"><table class="dd-samples mono"><thead><tr>` +
      headerRow(CALL_COLUMNS, callSort) +
      `</tr></thead><tbody>${rows}</tbody></table></div>`;
    restoreFocus(callsEl, focused);
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
        openNotes.set(incidentId, { tone: "error", text: tr("admin.open_failed", { error: await errorDetail(res) }) });
      } else {
        openNotes.delete(incidentId);
        analysisCache.set(key, { ...(await res.json()), opened_by: me ? me.id : "", opened_for: purpose });
      }
    } catch (e) {
      if (e instanceof AuthError) return;
      openNotes.set(incidentId, { tone: "error", text: tr("admin.open_failed", { error: e.message || e }) });
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

  const feedbackName = (state) => {
    const name = tr(`admin.fb_state_${state}`);
    return name === `admin.fb_state_${state}` ? state : name;
  };
  const feedbackBadge = (state) =>
    state ? `<span class="queue-badge fb-${esc(state)}">${esc(feedbackName(state).toUpperCase())}</span>` : "";

  const sendFeedback = async (orgId, action) => {
    const body = { action, fingerprint: detail?.fingerprint };
    if (action === "merge") {
      const target = document.getElementById("ddMergeTarget")?.value;
      if (!target) return;
      body.target_org_id = target;
      body.target_fingerprint = orgs.find((o) => o.id === target)?.fingerprint;
    }
    try {
      const res = await api(`/api/admin/organizations/${encodeURIComponent(orgId)}/feedback?locale=${locale()}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      if (res.status === 409) {
        // Reports arrived or left while this was open: nothing was saved; show the queue as it is now.
        await load();
        ingestNote.className = "adm-note is-error";
        bind(ingestNote, "admin.fb_stale");
        return;
      }
      if (!res.ok) throw new Error(await errorDetail(res));
      const shown = await res.json();
      await load(); // the queue, KPIs and stats all change; re-read everything
      selectOrg(shown.id);
    } catch (e) {
      if (e instanceof AuthError) return;
      ddEl.insertAdjacentHTML("afterbegin", `<div class="dd-loading">${esc(tr("admin.fb_failed", { error: e.message || e }))}</div>`);
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
    modalTitle.textContent = tr("admin.dd_title", { id: orgId });
    modalDot.className = "p-dot";
    ddEl.innerHTML = `<div class="dd-loading">${esc(tr("admin.loading"))}</div>`;
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
      ddEl.innerHTML = `<div class="dd-loading">${esc(tr("admin.org_failed", { error: e.message || e }))}</div>`;
    }
  };

  // Loads overlap (a sign-in completing while the language switch applies the stored locale,
  // a quick language change): only the newest may render, or a slower answer in the previous
  // language overwrites it.
  let loadSeq = 0;
  const load = async () => {
    const seq = ++loadSeq;
    const stale = () => seq !== loadSeq;
    kpisEl.setAttribute("aria-busy", "true");
    analysisCache.clear(); // locale-dependent names; cheap to re-score on demand
    openNotes.clear();
    closeModal();
    try {
      const res = await api(`/api/admin/overview?locale=${locale()}`);
      if (!res.ok) throw new Error(await errorDetail(res));
      const body = await res.json();
      if (stale()) return;

      if (!body.available) {
        kpisEl.innerHTML = "";
        novelEl.innerHTML = "";
        queueCount.textContent = "";
        statsSection.hidden = true;
        showEmpty(queueWrap, `${esc(tr("admin.analysis_empty"))} ${trHtml("admin.seed_hint_html")}`);
        return;
      }

      orgs = body.organizations;
      renderKpis(body.kpis);
      renderNovel(orgs);
      renderQueue();
      await loadStats(stale);
    } catch (e) {
      if (e instanceof AuthError || stale()) return;
      kpisEl.innerHTML = "";
      novelEl.innerHTML = "";
      statsSection.hidden = true;
      showEmpty(queueWrap, trHtml("admin.offline_html", { error: e.message || e }));
    } finally {
      if (!stale()) kpisEl.removeAttribute("aria-busy");
    }
  };

  // Statistics are additive — a failure here must never take the dashboard down.
  const loadStats = async (stale = () => false) => {
    try {
      const res = await api(`/api/admin/stats?locale=${locale()}`);
      if (!res.ok) throw new Error(tr("admin.err_api", { status: res.status }));
      const body = await res.json();
      if (!stale()) renderStats(body);
    } catch {
      if (!stale()) statsSection.hidden = true;
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
          `<div class="st-bar-slot" data-tip="${esc(tr("admin.stats_tip", { date: p.date, n: p.count }))}">` +
          `<div class="st-bar" style="height:${height}%"></div></div>`
        );
      })
      .join("");
    const t = s.trend;
    const trendHtml =
      t.delta_pct === null
        ? `<span class="st-delta">${esc(tr("admin.trend_week", { n: t.this_week }))}</span>`
        : `<span class="st-delta">${esc(tr("admin.trend_week", { n: t.this_week }))}&nbsp;&middot;&nbsp;` +
          `${esc(tr("admin.trend_delta", { arrow: t.delta_pct >= 0 ? "▲" : "▼", pct: Math.abs(t.delta_pct).toFixed(0) }))}</span>`;
    const first = s.activity[0]?.date || "";
    const last = s.activity[s.activity.length - 1]?.date || "";
    const activityCard =
      `<div class="st-card st-card--wide">` +
      `<div class="st-card-head mono"><span>${esc(tr("admin.stats_activity"))}</span>${trendHtml}</div>` +
      `<div class="st-bars" role="img" aria-label="${esc(tr("admin.stats_activity_aria"))}">${bars}</div>` +
      `<div class="st-axis mono"><span>${esc(first)}</span><span>${esc(last)}</span></div>` +
      `</div>`;

    const r = s.reports;
    const rate = r.submitted ? Math.round((r.ingested / r.submitted) * 100) : null;
    const strip = r.submitted
      ? `<div class="st-strip" role="img" aria-label="${esc(tr("admin.stats_strip_aria", { ingested: r.ingested, pending: r.pending }))}">` +
        (r.ingested ? `<span class="st-seg st-seg--done" style="flex-grow:${r.ingested}"></span>` : "") +
        (r.pending ? `<span class="st-seg st-seg--pending" style="flex-grow:${r.pending}"></span>` : "") +
        `</div>` +
        `<div class="st-legend mono">` +
        `<span><i class="st-dot st-dot--done"></i>${esc(tr("admin.stats_ingested", { n: r.ingested }))}</span>` +
        `<span><i class="st-dot st-dot--pending"></i>${esc(tr("admin.stats_pending", { n: r.pending }))}</span></div>`
      : `<p class="st-empty mono">${esc(tr("admin.stats_no_reports"))}</p>`;
    const reportsCard =
      `<div class="st-card">` +
      `<div class="st-card-head mono"><span>${esc(tr("admin.stats_reports"))}</span></div>` +
      `<div class="st-hero">${r.ingested}` +
      `<span class="st-hero-sub">&nbsp;${esc(tr("admin.stats_of", { n: r.submitted }))}${rate === null ? "" : ` · ${rate}%`}</span></div>` +
      strip +
      `</div>`;

    const maxIncidents = Math.max(...s.top_organizations.map((o) => o.incidents), 1);
    const orgRows = s.top_organizations
      .map(
        (o) =>
          `<div class="st-org-row">` +
          `<span class="st-org-name" title="${esc(o.name)}">${esc(o.name)}${o.is_novel ? `<span class="queue-badge">${esc(tr("admin.badge_new"))}</span>` : ""}</span>` +
          `<span class="st-org-bar"><i style="--w:${((o.incidents / maxIncidents) * 100).toFixed(0)}%"></i></span>` +
          `<span class="st-org-val">${o.incidents}</span></div>`
      )
      .join("");
    const topCard =
      `<div class="st-card">` +
      `<div class="st-card-head mono"><span>${esc(tr("admin.stats_largest"))}</span></div>` +
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
    bind(ingestBtn, "admin.ingesting");
    ingestNote.className = "adm-note";
    bind(ingestNote, "admin.ingest_embedding");
    try {
      const res = await api(`/api/admin/ingest?locale=${locale()}`, { method: "POST" });
      if (!res.ok) throw new Error(await errorDetail(res));
      const body = await res.json();
      ingestNote.className = "adm-note is-success";
      if (body.skipped) {
        ingestNote.className = "adm-note is-error";
        bind(ingestNote, "admin.ingest_skipped", { n: body.ingested, skipped: body.skipped });
      } else if (!body.ingested) {
        bind(ingestNote, "admin.ingest_none");
      } else {
        const placed = body.placements
          .map((p) => `${p.org_name}${p.org_is_novel ? ` (${tr("admin.ingest_new_scheme")})` : ""}`)
          .join("; ");
        bind(ingestNote, "admin.ingest_done", { n: body.ingested, placed });
      }
      await load();
    } catch (e) {
      if (e instanceof AuthError) return;
      ingestNote.className = "adm-note is-error";
      bind(ingestNote, "admin.ingest_failed", { error: String(e.message || e) });
    } finally {
      ingestBtn.disabled = false;
      bind(ingestBtn, "admin.ingest");
    }
  };

  // ── sign-in / sign-out ─────────────────────────────────────────────────────────

  const showSignin = (key = null, tone = "", params = null) => {
    consoleEl.hidden = true;
    toolbarEl.hidden = true;
    sessionEl.hidden = true;
    demoNoteEl.hidden = true;
    signinSection.hidden = false;
    signinNote.className = `adm-note${tone ? ` is-${tone}` : ""}`;
    bind(signinNote, key, params);
    keyInput.value = "";
    keyInput.focus();
  };

  const showConsole = () => {
    signinSection.hidden = true;
    consoleEl.hidden = false;
    toolbarEl.hidden = false;
    sessionEl.hidden = false;
    const keyless = !analystKey;
    if (signOutBtn) signOutBtn.hidden = keyless;
    // Keyless means the public demo (ADR D59): say plainly that nothing here is a real person's.
    demoNoteEl.hidden = !keyless;
    const role = `<span class="adm-role adm-role--${esc(me.role)}">${esc(roleName(me.role))}</span>`;
    whoEl.innerHTML =
      (keyless
        ? `${esc(tr("admin.who_open"))} ${role} <span class="adm-who-note">${trHtml("admin.who_open_note_html", { id: me.id })}</span>`
        : `${trHtml("admin.who_signed_in_html", { id: me.id })} ${role}`) +
      (me.can_open_cases ? "" : ` <span class="adm-who-note">${esc(tr("admin.who_limited"))}</span>`);
  };

  const RATE_LIMIT_RETRY_MS = 60_000;
  // A server with open demo access (QORGAN_ADMIN_OPEN_ACCESS, ADR D59) admits a keyless visitor
  // as `public-demo`; anywhere else this answers 401 and the sign-in stays.
  const tryOpenAccess = async () => {
    try {
      const res = await fetch("/api/admin/session", { cache: "no-store" });
      if (res.status === 429) {
        // The demo's request budget, not a missing key: say so, and come back by itself.
        showSignin("admin.too_many_requests", "error");
        setTimeout(() => { if (!me && !analystKey) tryOpenAccess(); }, RATE_LIMIT_RETRY_MS);
        return;
      }
      const body = res.ok ? await res.json() : null;
      if (!body?.open_access) {
        showSignin();
        return;
      }
      me = body;
      showConsole();
      await load();
    } catch {
      showSignin();
    }
  };

  // Forget the key and everything rendered with it.
  const endSession = (key, tone) => {
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
    bind(ingestNote, null);
    showSignin(key, tone);
  };

  // GET /api/admin/session checks the key and names who is signed in (an audited session start).
  const startSession = async (key) => {
    signinBtn.disabled = true;
    signinNote.className = "adm-note";
    bind(signinNote, "admin.checking");
    try {
      const res = await fetch("/api/admin/session", { headers: { "X-Analyst-Key": key }, cache: "no-store" });
      if (!res.ok) {
        analystKey = "";
        forgetKey();
        if (res.status === 401) showSignin("admin.key_rejected", "error");
        else if (res.status === 429) showSignin("admin.too_many", "error");
        else {
          // The server's own reason (a console closed for want of keys): shown as it was written.
          const detail = await errorDetail(res);
          showSignin(null, "error");
          signinNote.textContent = detail;
        }
        return;
      }
      me = await res.json();
      analystKey = key;
      storeKey(key);
      showConsole();
      await load();
    } catch (e) {
      analystKey = "";
      showSignin("admin.unreachable", "error", { error: String(e.message || e) });
    } finally {
      signinBtn.disabled = false;
    }
  };

  // ── wiring ───────────────────────────────────────────────────────────────────

  signinForm.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const key = keyInput.value.trim();
    if (!key) {
      showSignin("admin.key_missing", "error");
      return;
    }
    startSession(key);
  });
  signOutBtn?.addEventListener("click", () =>
    endSession("admin.signed_out", "success"));

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
  // One language control for the page, the same stored choice as the other pages (i18n-dom.js).
  // Names and reasons come from the server in the language asked for, so a session reloads; before
  // sign-in there is nothing to reload, and a keyless request would read as a revoked key.
  const applyLanguage = (next, { persist = true } = {}) => {
    lang = LOCALES.includes(next) ? next : DEFAULT_LOCALE;
    if (persist) storeLocale(lang);
    document.documentElement.lang = lang;
    document.querySelectorAll('input[name="adm-lang"]').forEach((radio) => { radio.checked = radio.value === lang; });
    localize(document, lang, adminText);
    if (me) {
      showConsole();
      load();
    }
  };
  document.querySelectorAll('input[name="adm-lang"]').forEach((radio) =>
    radio.addEventListener("change", () => { if (radio.checked) applyLanguage(radio.value); })
  );

  refreshBtn?.addEventListener("click", load);
  ingestBtn?.addEventListener("click", runIngest);
  modalClose.addEventListener("click", closeModal);
  modalBackdrop.addEventListener("click", closeModal);
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") closeModal();
  });

  applyLanguage(lang, { persist: false });
  if (analystKey) startSession(analystKey);
  else tryOpenAccess();
})();
