# Qorğan tooling shortlist — verified 2026-10-07

Versions below were checked with `pip index versions`, `npm view`, `gh api`, `curl` against the HF Hub/MCP endpoints, and a scratch venv (nothing installed into the project). "3P" = third-party source only; "UNVERIFIED" = could not confirm.

## 0. Claude Code under Max — what actually applies

| Question | Verified answer | Source |
|---|---|---|
| Limits | 5-hour rolling window + rolling 7-day weekly cap (weekly caps since 28 Aug 2025), shared between claude.ai, Cowork and Claude Code. 2026 (3P): +25 % weekly made permanent 14 Sep 2026; 5-hour limits raised with Opus 5.5; peak-hour (05–11 PT weekdays) faster drain since 26 Mar 2026. `/usage` shows attribution per subagent/MCP server; `/usage-credits` enables pay-as-you-go past the cap. | code.claude.com/docs/en/costs (official); support article 11014257 returned 404; limits detail is 3P |
| Does `claude -p` / Agent SDK run on Max without API credits? | **Yes, today.** Official: "We're pausing the changes to Claude Agent SDK usage… For now, nothing has changed: Claude Agent SDK, `claude -p`, and third-party app usage still draw from your subscription's usage limits." The announced 15 Jun 2026 separate credit ($100 Max 5x / $200 Max 20x) is **paused**. Caveat 1: `claude --bare -p` "doesn't use your subscription login" and needs `ANTHROPIC_API_KEY` — so do **not** use `--bare`. Caveat 2: the SDK note forbids *third-party products* offering claude.ai login; your own scripts are the allowed case. | support.claude.com/en/articles/15036540; code.claude.com/docs/en/headless; /docs/en/agent-sdk/overview |
| Headless flags | `claude -p "..." --output-format json --json-schema '{...}'` → `structured_output`; `--allowedTools "Read,Write,Bash(python *)"`, `--permission-mode acceptEdits`, `--permission-prompts none` (≥2.1.259), `--resume <id>`, `--max-turns`, `--append-system-prompt-file`, `--forward-subagent-text`. Exit code ≠0 on failure; stdin capped at 10 MB. Installed: Claude Code 2.1.292; `claude-agent-sdk` PyPI 0.2.164 (2026-10-06), npm `@anthropic-ai/claude-agent-sdk` 0.3.292. | headless docs |
| Orchestration options | (a) **Subagents** (Agent tool, background, fork) — results summarised back. (b) **Dynamic Workflows** — Claude writes a JS script (`agent()`, `pipeline()`, `parallel()`, `phase()`, `schema:` for JSON output); 16 concurrent agents default (`CLAUDE_CODE_WORKFLOW_MAX_CONCURRENT_AGENTS` 1–256), ≤4,096 items per `pipeline()`, ≤1,000 agents/run, **resumable in-session** (completed agents replay from cache), pauses on usage limit instead of failing (interactive, ≥2.1.271). Trigger with "use a workflow"/`ultracode`; save to `.claude/workflows/<name>.js`; `/workflow-authoring` skill. (c) **Agent Teams** — experimental, `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`, ~7× tokens, interactive only. | /docs/en/workflows, /docs/en/agent-teams |

**Recommended data-generation pattern (need 1):** a saved workflow `gen-dialogues.js` that `pipeline()`s over a seed manifest (scenario × language × persona × label) in chunks of 20–30 dialogues; each agent gets a JSON `schema`, writes `data/synthetic/gen/<run>/<chunk>.jsonl` itself, and returns only `{chunk, n, sha256}`. Keep the lexicon **out** of agent context (leak check afterwards). Resumability comes for free in-session; across sessions, make chunks idempotent (skip if file hash exists). Spend: run one directory-sized slice first; workflows draw on the same 5-h/weekly pool, so schedule big runs right after a reset.

## 1. Synthetic data QC (need 1)

| Item | Role | Install / use | Cost · licence | Maturity | Caveats | Pri |
|---|---|---|---|---|---|---|
| pydantic 2.13.5 + jsonschema 4.26.0 | Schema validation of every JSONL row; same schema fed to `--json-schema`/workflow `schema` | already a dep; `pip install jsonschema==4.26.0` | MIT | — | `format` keywords are not enforced by Claude Code | P0 |
| **semhash 0.5.0** | Embedding near-dup removal + train/held-out leakage check (`self_deduplicate()`, `deduplicate(records=test)`), outlier/representative filters | `pip install semhash==0.5.0`; `SemHash.from_records(texts, model=SentenceTransformer("intfloat/multilingual-e5-base"))` | MIT · 973★ · 2026-09-30 | Default model2vec `potion` is English-only — **pass the multilingual e5 encoder you already ship** | P0 |
| datasketch 2.0.0 | MinHash-LSH exact/near-dup on char-shingles (cheap first pass, catches template copies) | `pip install datasketch==2.0.0` | MIT · 3.0k★ · 2026-07 | — | P1 |
| Diversity metrics | distinct-n, self-BLEU (`sacrebleu==2.6.0`), Vendi score | `vendi-score` 0.0.3 is from 2022 — hand-roll (eigenvalues of normalised e5 kernel, ~10 lines) | — | — | P1 |
| cleanlab 2.9.0 | Label-noise ranking from out-of-fold LR probabilities (`find_label_issues`) → review queue | `pip install cleanlab==2.9.0` | AGPL-3.0 (OSS core is AGPL — fine for internal use) · 11.7k★ · 2026-01 | Licence matters only if you redistribute | P1 |
| text-dedup 0.4.1 | Alternative MinHash/SimHash CLI | — | Apache-2.0 · 2025-12 | Redundant with semhash+datasketch | P2 |
| great_expectations 1.24 | — | — | — | Overkill: pydantic + pytest already gate the corpus | skip |

## 2. Annotation / review UI (need 2)

| Item | Role | Install | Cost · licence | Maturity | Caveats | Pri |
|---|---|---|---|---|---|---|
| **Label Studio 1.23.2** | Per-utterance tactic tags via `<Paragraphs layout="dialogue" audioUrl="$audio">` + `<ParagraphLabels>`; exports JSON with paragraph index + char offsets; audio sync | `pip install label-studio==1.23.2 && label-studio start` (or `uvx label-studio`); import via `label-studio-sdk` | Free Community Edition · Apache-2.0 · 28.4k★ · 2026-09-29 | Fabricated data only on a hosted instance; keep it local | P0 |
| Label Studio MCP (HumanSignal/label-studio-mcp-server) | Create projects / import tasks / add predictions from Claude | `uvx --from git+https://github.com/HumanSignal/label-studio-mcp-server mcp-label-studio` (env `LABEL_STUDIO_URL`, `LABEL_STUDIO_API_KEY`) | MIT | 37★, last push 2025-05-30, not on PyPI — dormant | Use the SDK from a script instead | P2 |
| Argilla 2.8.0 | Rich text-classification/span UI, HF-native | docker (needs Elasticsearch/OpenSearch) | Apache-2.0 · 5.1k★ · PyPI 2025-03 | Heavy for 2 people; **no official Argilla MCP exists** | P2 |
| doccano 1.8.4 / Prodigy / Lilac | — | — | doccano PyPI 2023; Prodigy paid; Lilac PyPI 2024-02, repo defunct | | skip |

## 3. STT evaluation, TTS, telephone channel (need 3)

| Item | Role | Install / use | Cost · licence | Maturity | Caveats | Pri |
|---|---|---|---|---|---|---|
| **jiwer 4.0.0** | WER/CER with normalisation pipeline | `pip install jiwer==4.0.0` | Apache-2.0 · 933★ · 2025-06 | — | P0 |
| **edge-tts 7.2.8** | KK/RU test audio. Verified voices: `kk-KZ-AigulNeural` (F), `kk-KZ-DauletNeural` (M), `ru-RU-DmitryNeural`, `ru-RU-SvetlanaNeural` | `uvx --from edge-tts edge-tts --voice kk-KZ-AigulNeural --text "…" --write-media out.mp3` | Free · unofficial MS Edge endpoint (ToS grey) · 12.2k★ · 2026-03 | Cloud: only fabricated text; never real reports | P0 |
| **piper-tts 1.8.0** | Offline KK TTS: `rhasspy/piper-voices` → `kk/kk_KZ/issai/high/kk_KZ-issai-high.onnx` (128 MB, 6 ISSAI speakers, verified on Hub); RU voices irina/denis/dmitri/ruslan | `pip install piper-tts==1.8.0`; `hf download rhasspy/piper-voices kk/kk_KZ/issai/high/kk_KZ-issai-high.onnx kk/kk_KZ/issai/high/kk_KZ-issai-high.onnx.json` | GPL-3.0 (tool; voices CC-BY) · OHF-Voice/piper1-gpl active 2026-10; rhasspy/piper archived | GPL only matters if you ship it; test tooling is fine | P0 |
| ISSAI KazakhTTS2 | 271 h corpus; ESPnet models at IS2AI/Kazakh_TTS (CC-BY-4.0, pushed 2025-08) | — | | Already distilled into the piper voice above — use piper | P2 |
| **audiomentations 0.43.1** | Channel sim: `BandPassFilter(300,3400)`, `Mp3Compression`, `AddBackgroundNoise`, `Gain`, `ApplyImpulseResponse` | `pip install audiomentations==0.43.1` | MIT · 2.3k★ · 2025-09 | — | P0 |
| **ffmpeg (installed) for codecs** | G.711 `-ar 8000 -c:a pcm_mulaw`; Opus `-c:a libopus -b:a 12k`; AMR-NB `-c:a libopencore_amrnb -ar 8000 -b:a 7.4k` (check `ffmpeg -encoders`) | subprocess from pytest fixtures | LGPL/GPL | **torchaudio 2.11 removed `apply_codec`/`AudioEffector`** (maintenance mode since 2.9) — do not plan on it | P0 |
| pyroomacoustics 0.10.1 | Synthetic RIRs for speakerphone rooms | `pip install pyroomacoustics==0.10.1` | MIT · 2026-05 | Nice-to-have | P2 |
| sherpa-onnx 1.13.8 (py + `sherpa-onnx-node`) | Offline engine comparison vs Vosk; runs Whisper ONNX exports | `pip install sherpa-onnx==1.13.8` | Apache-2.0 · 15.1k★ · 2026-09 | **No Kazakh model in its zoo**; KK candidates are HF Whisper fine-tunes (`akuzdeuov/whisper-base.kk` 15.4 % WER KSC2; `shyngys879/kazakh-whisper-large-v3-turbo` 11.8 % FLEURS) — run those with `faster-whisper==1.2.1` on MPS first | P1 |
| Datasets | `google/fleurs` ungated, `data/kk_kz` parquet → `load_dataset("google/fleurs","kk_kz",streaming=True)` with `datasets==5.1.0`. `issai/Kazakh_Speech_Corpus_2` on Hub ungated (7-part tar.gz, MIT card). **Common Voice is gone from HF**: CV17 repo holds only a README ("Effective October 2025… exclusively available through Mozilla Data Collective", CC0, account + ToS required; Kazakh hours UNVERIFIED — page is JS-rendered). | | | | P0 |

## 4. Experiment tracking (need 4)

| Item | Verdict | Why | Pri |
|---|---|---|---|
| **MLflow 3.17.0** (local) | **Use** | `mlflow server --backend-store-uri sqlite:///.mlflow/mlruns.db --port 5000`; `mlflow.sklearn.autolog()`; log `git_sha`, HF dataset revision, lexicon hash as params; artefacts stay on disk (no transcripts leave the laptop). SQLite is the default backend since 3.7. Optional MCP: `pip install 'mlflow[mcp]>=3.5.1'` → `mlflow mcp run` (experiments/runs/models in `ml` mode; experimental). Also an official `mlflow` Claude plugin (skills). Apache-2.0, 28.3k★, released today. | P1 |
| Reproducibility glue | **Use** | `hf repos tag create sanzh-ts/govtech_ds v2026.10.07 --repo-type dataset` per corpus snapshot; MLflow param = tag. `hf` CLI 2.1.1 (project venv has 1.23.0 — `pip install -U huggingface_hub`); `huggingface-cli` is dead ("deprecated and no longer works"). | P0 |
| W&B 0.30.0 | skip | Cloud; free plan 5 seats/5 GB (3P); sends metrics abroad — avoidable | — |
| aim 3.29.1 | skip | Last release 2025-05 | — |
| DVC 3.67.1 + dvclive | skip | Second data store next to HF; no gain | — |

## 5. Android (need 5)

| Item | Role | Install | Maturity | Caveats | Pri |
|---|---|---|---|---|---|
| **Bubblewrap 1.25.0 (TWA)** | Ship the PWA as a Play-store APK/AAB running in Chrome Custom Tab → **COOP/COEP honoured, SharedArrayBuffer and WASM threads work exactly as in Chrome** | `npx @bubblewrap/cli@1.25.0 init --manifest https://<host>/manifest.webmanifest`; host `/.well-known/assetlinks.json` | Apache-2.0 · 3.1k★ · 2026-10 | Needs HTTPS origin + Digital Asset Links; Chrome ≥72 on device | P0 |
| Capacitor 8.5.3 | — | — | MIT · 16.8k★ | **Do not use for voice mode**: Android WebView "does not support cross-origin isolation at all" (single renderer per app; Chromium issue 40914606, Capacitor #4520/#6182) → `crossOriginIsolated=false`, no SAB; your `site/core/asr.js` already gates on exactly that. Same applies to an **eGov mobile WebView**: only paste/replay mode (single-thread ORT) would work there. | skip |
| `adb` | Device bench | `brew install --cask android-platform-tools` (37.0.1) | — | not installed yet | P0 |
| **mobile-mcp 1.0.8** (`@mobilenext/mobile-mcp`) | Claude drives the phone (accessibility tree + screenshots): launch TWA, tap, read UI | `claude mcp add mobile-mcp --scope user -- npx -y @mobilenext/mobile-mcp@latest` | Apache-2.0 · 8.7k★ · 2026-10-02 | Native UI only; no in-page JS | P1 |
| Playwright `_android` (1.63.0) | In-page numbers on device: `android.devices()` → `device.launchBrowser()`/`device.webView()`, `page.evaluate(() => performance…)` | already installed; Chrome ≥87, adb, "Enable command line on non-rooted devices" flag | experimental | Local smoke bench, not CI | P1 |
| Device clouds | Firebase Test Lab Spark: **5 physical + 10 virtual runs/day free** (official); BrowserStack trial ≈100 automated min (3P) | — | — | Instrumentation tests only, not interactive | P2 |

## 6. Research workflow inside Claude Code (need 6)

| Item | Capabilities | Install | Cost | Maturity | Pri |
|---|---|---|---|---|---|
| **Hugging Face MCP (official)** `https://huggingface.co/mcp` | Server v0.4.28 probed live: anonymous tools `hub_repo_search` (models/datasets/spaces/papers), `hub_repo_details`, `hf_fs`, `hf_whoami`; after login you enable more in hf.co/settings/mcp (dataset viewer/SQL, repo create/upload, Spaces as tools, Jobs). Also official plugin `huggingface-skills`. | `claude mcp add --transport http hf "https://huggingface.co/mcp?login" --scope user` then `/mcp` → OAuth | free (HF token) · MIT · evalstate/hf-mcp-server 302★, pushed today | P0 |
| **Context7** 4.2.0 | Version-pinned library docs (your ECC `docs`/`docs-lookup` skills expect it) | `claude plugin install context7@claude-plugins-official`, or `claude mcp add --transport http context7 https://mcp.context7.com/mcp --header "CONTEXT7_API_KEY: $KEY"` | free key ≈1,000 req/month (3P, Jan 2026) · MIT · 62.8k★ | P1 |
| Built-in WebSearch/WebFetch + `/deep-research` workflow | Already there; `/deep-research` fans out, cross-checks, cites | nothing | included | P0 |
| Exa 3.4.2 | Neural search, `web_fetch_exa`, `agent_run` deep research; ECC `exa-search`/`deep-research` skills use it | `claude plugin install exa@claude-plugins-official` (hosted, anonymous rate-limited; $10 onboarding credit 3P) | MIT · 5.1k★ | P2 |
| Tavily 0.2.22 / Firecrawl 3.28.2 | Tavily 1,000 credits/mo free; Firecrawl keyless `https://mcp.firecrawl.dev/v2/mcp` (scrape/search/parse) | plugins `tavily`, `firecrawl` in the official marketplace | MIT | pick at most one of Exa/Tavily/Firecrawl | P2 |
| GitHub MCP v2.0.1 (2026-10-07) | Remote `https://api.githubcopilot.com/mcp/`, OAuth via `/mcp` or PAT header; plugin `github` | `claude mcp add --transport http github https://api.githubcopilot.com/mcp/` | free · MIT · 33.4k★ | `gh` CLI already covers you and the docs recommend CLIs for context economy | P2 |
| Railway MCP (official) | Logs, redeploy, variables for the Railway deployment | `railway setup agent` (CLI ≥5.44) or `railway mcp install --local`; remote `mcp.railway.com` (OAuth); plugin `railway`. Old npm `@railway/mcp-server` 0.1.12 — repo archived 2026-05, ignore | free | P1 |
| Brave Search MCP 2.1.4 | — | — | free tier removed for new sign-ups Feb 2026 ($5/mo credit) | skip |
| `mcp-server-fetch` 2026.8.18, SQLite MCP | — | — | redundant with WebFetch; SQLite reference server archived 2025-05 — use `sqlite3`/`duckdb-skills` plugin | skip |
| Sentry MCP (`https://mcp.sentry.dev/mcp`, 0.42.0) | only if you adopt Sentry on Railway | plugin `sentry` | free tier | P2 |
| data.egov.kz | REST v4: `https://data.egov.kz/api/v4/{dataset}/{version}?apiKey=…&source={"size":100,"from":0,"query":{…}}` (Elasticsearch-style JSON, HTTPS GET). Browsing/download needs no account; API key from the developer cabinet `/profile/apikeylist` after eGov login (ЭЦП or eGov Mobile QR — UNVERIFIED end-to-end). Rate limits unpublished. | — | free | P2 |

## 7. Security & privacy (need 7)

| Item | Role | Install / command | Licence · maturity | Caveats | Pri |
|---|---|---|---|---|---|
| **pip-audit 2.10.1**, `npm audit` | Dependency CVEs (onnxruntime pin will show up — document the exception) | `pip install pip-audit==2.10.1 && pip-audit`; `npm audit --omit=dev` | Apache-2.0 · 2026-06 | — | P0 |
| **gitleaks 8.30.1** | Secret scan (pre-commit + CI); you already found a key in `.env` once | `brew install gitleaks && gitleaks git --redact -v` | MIT · 29.8k★ · 2026-03 | trufflehog 3.99 (AGPL) adds live-key verification — not needed | P0 |
| bandit 1.9.4 / **semgrep 1.179.0** | Python SAST / multi-language rules | `pip install bandit==1.9.4; bandit -r src`; `brew install semgrep && semgrep scan --config p/python --config p/javascript` (or plugin `semgrep`) | Apache-2.0 / LGPL-2.1 | semgrep anonymous registry OK; no login needed for `p/*` | P1 |
| presidio-analyzer 2.2.364 | **Independent oracle** for the scrubber's RU recall: English-only by default; add RU via NlpEngine YAML + spaCy `ru_core_news_lg` 3.8.0; regex recognisers (phone/email/card/IBAN) are language-agnostic, context words are not. **No Kazakh.** | `pip install presidio-analyzer==2.2.364 && python -m spacy download ru_core_news_lg` | MIT · 11.2k★ · 2026-07 | For KK names use HF KazNERD models (`yeshpanovrustem/xlm-roberta-large-kaznerd`; ISSAI `issai/kaznerd*` datasets) as a test-time oracle | P1 |
| natasha 1.6.0 | Lighter RU PER/LOC/ORG (slovnet) oracle; spaCy's RU model is trained on Natasha's Nerus | `pip install natasha==1.6.0` | MIT · PyPI 2023-07, repo active 2026-10 | RU only | P1 |
| Faker 40.41.0 | Fabricated PII in tests: `Faker("ru_RU")` works; **`kk_KZ` does not exist** (verified: `Invalid configuration for faker locale kk_KZ`) | write a 50-line Kazakh provider (names, 12-digit ИИН with checksum, `+7 7xx` numbers, Astana/Almaty addresses) | MIT | — | P1 |

## Install today (≤10)

```bash
claude mcp add --transport http hf "https://huggingface.co/mcp?login" --scope user     # then /mcp → log in
claude plugin install context7@claude-plugins-official                                   # add key from context7.com/dashboard
claude mcp add mobile-mcp --scope user -- npx -y @mobilenext/mobile-mcp@latest
brew install gitleaks semgrep && brew install --cask android-platform-tools
pip install semhash==0.5.0 datasketch==2.0.0 cleanlab==2.9.0 jiwer==4.0.0 audiomentations==0.43.1 pip-audit==2.10.1 bandit==1.9.4 mlflow==3.17.0 -U huggingface_hub
pip install presidio-analyzer==2.2.364 natasha==1.6.0 && python -m spacy download ru_core_news_lg
pip install piper-tts==1.8.0 && hf download rhasspy/piper-voices kk/kk_KZ/issai/high/kk_KZ-issai-high.onnx kk/kk_KZ/issai/high/kk_KZ-issai-high.onnx.json --local-dir data/tts
uvx --from edge-tts edge-tts --voice kk-KZ-DauletNeural --text "Сәлеметсіз бе" --write-media /tmp/kk.mp3
uvx label-studio start                                                                    # Paragraphs/ParagraphLabels template
npm i -g @bubblewrap/cli@1.25.0 && railway setup agent                                    # TWA + Railway MCP (CLI ≥5.44)
```
(Pin these in a `tooling` extra of `pyproject.toml` rather than ad hoc; none touch the runtime pins — keep `onnxruntime` 1.21 untouched.)

## Skip, and why

- **Capacitor / any WebView shell for voice mode** — WebView has no cross-origin isolation, so no SharedArrayBuffer; Vosklet and threaded ORT die. TWA (Bubblewrap) is the only Android path that keeps the live meter.
- **`claude --bare -p` and the Agent SDK for batch generation** — bare mode and third-party-style SDK apps need an API key; plain `claude -p` and in-session workflows run on the Max pool (the June-2026 split is paused).
- **Agent Teams for data generation** — ~7× tokens, interactive-only, no resume; Dynamic Workflows are the right primitive.
- **Brave Search MCP** — free tier gone for new accounts. **mcp-server-fetch**, **SQLite MCP** — redundant/archived. **GitHub MCP** — `gh` is cheaper in context. **Label Studio MCP** — dormant since May 2025; use the SDK. **Argilla** — no MCP, Elasticsearch dependency.
- **doccano, Prodigy, Lilac, great_expectations, vendi-score (2022), text-dedup** — stale, paid, or overkill.
- **torchaudio codec path** — removed in 2.9+; ffmpeg does it. **W&B, aim, DVC** — cloud, stale, or a second data store you don't need.
- **Common Voice via HF** — the mirrors are README-only now; register on Mozilla Data Collective (CC0, ToS) or lean on FLEURS + KSC2 which stream from the Hub ungated.
