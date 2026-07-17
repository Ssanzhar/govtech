/* Qorğan analyzer — draws a seeded waveform + mel spectrogram and loops two cases:
   a "spoof" verdict and a "bonafide" one. Pure canvas, no assets, no network. */
(() => {
  "use strict";

  const CASES = [
    {
      seed: 7,
      bursts: [[0.05, 0.32], [0.40, 0.57], [0.65, 0.96]],
      drift: 0.6, breaths: [], hif: 0.05,
      log: ["// stream in", "$ analyzing 4.2 s clip", "formants: drifting",
            "hi-freq energy: low", "breath markers: 2", "// reading the signal…"],
      verdict: { label: '"spoof"', p: "0.97", reason: "kk · ru", time: "1.3 s", tone: "oxide" },
    },
    {
      seed: 21,
      bursts: [[0.04, 0.21], [0.28, 0.46], [0.53, 0.60], [0.68, 0.78], [0.85, 0.97]],
      drift: 0.08, breaths: [0.24, 0.49, 0.63, 0.81], hif: 0.3,
      log: ["// stream in", "$ analyzing 5.1 s clip", "formants: stable",
            "hi-freq energy: natural", "breath markers: 6", "// reading the signal…"],
      verdict: { label: '"bonafide"', p: "0.04", reason: "kk", time: "1.2 s", tone: "moss" },
    },
  ];

  const SWEEP_MS = 5600;   // playhead travel = one analysis pass
  const HOLD_MS = 2600;    // rest on the verdict before the next case
  const LINE_MS = 560;     // log line cadence

  const waveCv = document.getElementById("waveCv");
  const specCv = document.getElementById("specCv");
  const pBody = document.getElementById("pBody");
  const pDot = document.getElementById("pDot");
  const playhead = document.getElementById("playhead");
  const logEl = document.getElementById("aLog");
  const verdictEl = document.getElementById("aVerdict");
  if (!waveCv || !specCv) return;

  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ── deterministic helpers ────────────────────────────
  const rng = (seed) => {
    let s = seed || 1;
    return () => ((s = Math.imul(48271, s) & 0x7fffffff)) / 0x7fffffff;
  };
  const smoothstep = (a, b, x) => {
    const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
    return t * t * (3 - 2 * t);
  };
  const envelope = (t, bursts) => {
    let e = 0;
    for (const [a, b] of bursts) {
      e = Math.max(e, smoothstep(a, a + 0.03, t) * (1 - smoothstep(b - 0.03, b, t)));
    }
    return e;
  };
  // layered sines seeded per case — smooth organic noise in [0,1]
  const makeNoise = (rand) => {
    const ph = [rand() * 9, rand() * 9, rand() * 9];
    return (x, y = 0) => 0.5 + 0.5 * (
      Math.sin(x * 1.7 + ph[0] + y * 2.3) * 0.55 +
      Math.sin(x * 4.3 + ph[1] - y * 1.1) * 0.3 +
      Math.sin(x * 9.1 + ph[2] + y * 5.7) * 0.15);
  };

  // inferno-ish ramp, bottom (low freq) hot → top cold
  const STOPS = [
    [0.00, [247, 161, 43]], [0.16, [226, 113, 47]], [0.32, [194, 67, 155]],
    [0.52, [139, 47, 174]], [0.74, [74, 35, 130]], [1.00, [30, 17, 60]],
  ];
  const rampColor = (f) => {
    for (let i = 1; i < STOPS.length; i++) {
      if (f <= STOPS[i][0]) {
        const [f0, c0] = STOPS[i - 1], [f1, c1] = STOPS[i];
        const t = (f - f0) / (f1 - f0);
        return c0.map((v, k) => Math.round(v + (c1[k] - v) * t));
      }
    }
    return STOPS[STOPS.length - 1][1];
  };

  const fitCanvas = (cv) => {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const r = cv.getBoundingClientRect();
    cv.width = Math.round(r.width * dpr);
    cv.height = Math.round(r.height * dpr);
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [ctx, r.width, r.height];
  };

  // ── drawing ──────────────────────────────────────────
  const drawWave = (c) => {
    const [ctx, W, H] = fitCanvas(waveCv);
    const rand = rng(c.seed);
    const noise = makeNoise(rand);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "rgba(226,155,46,0.5)";
    ctx.fillRect(0, H / 2 - 0.5, W, 1);            // carrier line through silences
    const step = 3, block = H * 0.05;
    for (let x = 0; x < W; x += step) {
      const t = x / W;
      const e = envelope(t, c.bursts);
      if (e < 0.03) continue;
      let a = e * (0.28 + 0.72 * noise(t * 40)) * H * 0.44;
      a = Math.max(2, Math.round(a / block) * block); // quantized, blocky bins
      ctx.fillStyle = `rgba(226,155,46,${0.5 + 0.5 * noise(t * 23, 1)})`;
      ctx.fillRect(x, H / 2 - a, step - 1, a * 2);
    }
  };

  const drawSpec = (c) => {
    const [ctx, W, H] = fitCanvas(specCv);
    const rand = rng(c.seed * 31 + 5);
    const noise = makeNoise(rand);
    ctx.clearRect(0, 0, W, H);
    const cw = 5, ch = 4;
    const centers = [0.10, 0.24, 0.40, 0.58];       // formant bands (0 = bottom)
    for (let x = 0; x < W; x += cw) {
      const t = x / W;
      const e = envelope(t, c.bursts);
      const breath = c.breaths.some((b) => Math.abs(t - b) < 0.012);
      if (e < 0.03 && !breath) continue;
      for (let y = 0; y < H; y += ch) {
        const f = 1 - y / H;                        // 0 bottom → 1 top
        let b;
        if (breath && e < 0.03) {
          b = 0.16 * noise(t * 60, f * 12);         // faint wideband breath stripe
        } else {
          let boost = 0;
          for (let i = 0; i < centers.length; i++) {
            const drift = c.drift * 0.05 * Math.sin(t * 9 + i * 2.1);
            const d = (f - (centers[i] + drift)) / 0.042;
            boost += Math.exp(-d * d) * (1 - i * 0.2);
          }
          b = e * Math.pow(1 - f, 1.15) * (0.24 + 1.3 * boost) *
              (0.72 + 0.55 * noise(t * 30, f * 20));
          if (f > 0.45) b += e * c.hif * Math.pow(f, 2) * noise(t * 44, f * 16); // upper-band air
        }
        if (b < 0.05) continue;
        const [r, g, bl] = rampColor(f);
        ctx.fillStyle = `rgba(${r},${g},${bl},${Math.min(1, b)})`;
        ctx.fillRect(x, y, cw - 1, ch - 1);
      }
    }
  };

  // ── annotations ──────────────────────────────────────
  const renderLog = (c) => {
    logEl.replaceChildren();
    return c.log.map((text) => {
      const div = document.createElement("div");
      div.className = "ln";
      div.textContent = text;
      logEl.appendChild(div);
      return div;
    });
  };

  const renderVerdict = (c) => {
    const v = c.verdict;
    verdictEl.className = `a-side a-verdict mono tone-${v.tone}`;
    verdictEl.innerHTML =
      `<div>label: <b>${v.label}</b></div>` +
      `<div>spoof_probability: ${v.p}</div>` +
      `<div>threshold: 0.50</div>` +
      `<div>reason → ${v.reason}</div>` +
      `<div class="v-dim">// verdict in ${v.time}</div>`;
    pDot.className = `p-dot tone-${v.tone}`;
  };

  const sweepPlayhead = () => {
    const pad = 20; // panel body padding, px
    playhead.style.transition = "none";
    playhead.style.left = pad + "px";
    void playhead.offsetWidth;
    playhead.style.transition = `left ${SWEEP_MS}ms linear`;
    playhead.style.left = `calc(100% - ${pad}px)`;
  };

  // ── the loop ─────────────────────────────────────────
  let idx = 0;
  const timers = new Set();
  const later = (fn, ms) => {
    const id = setTimeout(() => { timers.delete(id); fn(); }, ms);
    timers.add(id);
  };
  const clearTimers = () => { for (const id of timers) clearTimeout(id); timers.clear(); };

  const showCase = (c, animate) => {
    drawWave(c);
    drawSpec(c);
    renderVerdict(c);
    const lines = renderLog(c);
    if (!animate) {
      lines.forEach((l) => l.classList.add("on"));
      verdictEl.classList.add("on");
      return;
    }
    sweepPlayhead();
    lines.forEach((l, i) => later(() => l.classList.add("on"), 250 + i * LINE_MS));
    later(() => verdictEl.classList.add("on"), SWEEP_MS * 0.74);
  };

  const runLoop = () => {
    const c = CASES[idx % CASES.length];
    pBody.classList.add("fading");
    later(() => {
      pBody.classList.remove("fading");
      showCase(c, true);
      idx += 1;
      later(runLoop, SWEEP_MS + HOLD_MS);
    }, 320);
  };

  // static first paint (also the reduced-motion final state)
  showCase(CASES[0], false);

  if (!reduceMotion) {
    let started = false;
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting && !started) {
          started = true;
          idx = 1;                    // first animated pass shows the bonafide case
          later(runLoop, 900);
          io.disconnect();
        }
      }
    }, { threshold: 0.35 });
    io.observe(pBody);
  }

  let resizeT;
  addEventListener("resize", () => {
    clearTimeout(resizeT);
    resizeT = setTimeout(() => {
      const c = CASES[Math.max(0, idx - 1) % CASES.length];
      drawWave(c); drawSpec(c);
    }, 150);
  });

  addEventListener("pagehide", clearTimers);
})();
