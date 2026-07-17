/* Qorğan admin dashboard — overview, priority queue with sort/search/filter, drill-down
   with a searchable calls table, and per-call on-demand model analysis (the rank graph:
   GET /api/admin/incidents/{id}/analysis). All list management is client-side. */
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
  if (!kpisEl || !queueWrap || !ddEl || !modal) return;

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
          `<td>${esc(o.name)}${o.is_novel ? '<span class="queue-badge">NEW</span>' : ""}</td>` +
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

    modalTitle.textContent = `organization · ${detail.id}`;
    modalDot.className = `p-dot${detail.is_novel ? "" : " tone-moss"}`;
    ddEl.innerHTML =
      `<div class="dd-title">${esc(detail.name)} ${badge}</div>` +
      `<div class="dd-meta mono">priority ${detail.priority.toFixed(2)}</div>` +
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
        `<div class="dd-section-label mono">transcript &middot; trigger phrases</div>` +
        `<div class="dd-script dd-analysis-script">${highlightSpans(a.transcript, a.spans)}</div>` +
        `<p class="dd-analysis-reason">${esc(a.reason)}</p>` +
        `<p class="dd-analysis-caveat mono">${esc(a.caveat)}</p>`;
    }
    return `<tr class="dd-analysis-tr"><td colspan="4"><div class="dd-analysis">${inner}</div></td></tr>`;
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
          `<tr data-incident="${esc(c.id)}" class="${expanded ? "is-expanded" : ""}">` +
          `<td>${esc(c.date || "—")}</td><td>${esc(c.number || "—")}</td>` +
          `<td>${(c.risk * 100).toFixed(0)}%</td><td>${esc(c.excerpt)}</td></tr>` +
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
      tr.addEventListener("click", () => expandCall(tr.dataset.incident));
    });
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
      const res = await fetch(
        `/api/admin/incidents/${encodeURIComponent(incidentId)}/analysis?locale=${locale()}`
      );
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      analysisCache.set(key, await res.json());
    } catch (e) {
      analysisCache.set(key, { error: String(e.message || e) });
    }
    if (expandedCall === incidentId) renderCalls();
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
      const res = await fetch(
        `/api/admin/organizations/${encodeURIComponent(orgId)}?locale=${locale()}`
      );
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      const body = await res.json();
      if (selectedId !== orgId || modal.hidden) return; // stale response — user moved on
      detail = body;
      renderDrilldown();
    } catch (e) {
      ddEl.innerHTML =
        `<div class="dd-loading">Could not load this organization — ${esc(e.message || e)}</div>`;
    }
  };

  const load = async () => {
    kpisEl.setAttribute("aria-busy", "true");
    analysisCache.clear(); // locale-dependent names; cheap to re-score on demand
    closeModal();
    try {
      const res = await fetch(`/api/admin/overview?locale=${locale()}`);
      if (!res.ok) throw new Error(`API returned ${res.status}`);
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
      const res = await fetch(`/api/admin/stats?locale=${locale()}`);
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
      const res = await fetch(`/api/admin/ingest?locale=${locale()}`, { method: "POST" });
      if (!res.ok) throw new Error(`API returned ${res.status}`);
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
      ingestNote.className = "adm-note is-error";
      ingestNote.textContent = `ingest failed — ${e.message || e}`;
    } finally {
      ingestBtn.disabled = false;
      ingestBtn.textContent = "ingest reports";
    }
  };

  // ── wiring ───────────────────────────────────────────────────────────────────

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

  load();
})();
