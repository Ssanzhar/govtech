# DATA_SOURCES.md — external dataset survey (2026-09-26)

_Why this exists: the government stakeholder has said real data will come "later" (PLAN A3 is
still open, `docs/STATUS.md`). This is the survey of what already exists publicly, what we can
legally use, and what we should stop looking for. Every entry was **fetched**, not recalled:
Hugging Face metadata + `datasets-server` row samples, raw file downloads, and licence pages.
Claims I could not verify are marked as such._

**Method.** Programmatic sweep of the HF Hub — ~100 query terms (English, Russian, Kazakh)
plus `language:kk` / `language:ru` filters → 2,275 unique datasets; then HF **full-text card
search** (which matches card bodies, not just names — this is what surfaced the useful ones);
then per-candidate verification of licence, row counts, real row content, and whether a data
file actually downloads today. Four parallel research lanes covered non-HF sources
(speech/telephony catalogues, the fraud-detection literature, Kazakhstan institutional and
legal sources, legitimate-call and register corpora).

---

## 1. The bottom line

**There is no public Kazakh or Russian scam-call corpus. That conclusion is now evidence-based,
not an assumption** — `data/README.md` has asserted it since July; this survey confirms it
against ~2,300 HF datasets and the non-HF catalogues. The stakeholder ask (A3) remains the
only route to real KZ call data, and `docs/DATA_INTAKE.md` is still the right instrument.

What *did* turn up is more useful than another synthetic corpus: **real-world Russian scam
phrasing**, **real Kazakh–Russian code-switched speech**, **Kazakh institutional register**,
and a **permissively licensed Chinese fraud-call corpus** that is a genuine second generator.

And the survey paid for itself twice over:

- probing our own pipeline against real-world phrasing exposed a **measured, reproducible bug in
  the shipped cue lexicon** (§6);
- and the Kazakhstan legal frame (§4.4) turns out to contain a **hard boundary we were about to
  walk into**: the Gemini API, the Hugging Face uploads and non-KZ storage are all lawful today
  *only because the corpus is synthetic*. Each one breaks on the first real transcript. Worth
  knowing before the stakeholder delivers data, not after.

A parallel GovTech entrant (`daurenoralbek/qalqan`, created 2026-09-21) independently reached
the same "no public KZ scam-call data" conclusion in the same week — see §4.1.

---

## 2. Usable now — ranked

Access status verified by attempting a real data-file download as `sanzh-ts` on 2026-09-26.

| # | Dataset | What it is | Licence | Access | Use for |
|---|---|---|---|---|---|
| 1 | **`Abdurohman/fraudlens-ru-v1`** | 6,327 Russian anti-fraud reports from 8 Telegram channels (incl. official `kasperskylab_ru`, F6). Structured `fraud_type / method / target / severity / platform`. **1,844 phone-related rows containing 1,771 deduped verbatim quotes of what scammers actually say.** | **CC-BY-4.0** | ✅ open | Real scam phrasing → cue-lexicon hardening; scheme seeds for a third generator; taxonomy gap-finding. **Not** transcripts. |
| 2 | **`BorisovMaksim/kk_ru_csw`** | 619 authentic KK/RU **code-switched** social-media sentences, triple-aligned: original code-switch + clean Kazakh + clean Russian. Two native bilingual linguist annotators. | **Apache-2.0** | ✅ open | The `mixed` register — our weakest split (shift recall 0.091). Better licensed than the alternatives. |
| 3 | **`Tim2190/kazakh-codeswitch-asr`** | 31 clips / ~3.7 min of **real** KK–RU code-switched speech from CC-BY YouTube. Two independent reference transcripts (verbatim + normalised) and flags for contractions, dialect slang, barbarisms. Claims to be the first KK–RU code-switch speech resource. | **CC-BY-4.0** | ✅ open | ASR cue-survival testing (ADR D39) on genuine code-switch — something synthesised audio cannot produce (the exact gap that kept D40 language-locking shipped OFF). |
| 4 | **`mangi-llm/kazakh-customer-support-qa`** | 406 Kazakh banking/customer-support Q&A with `domain / subdomain / intent / slots`. | **CC-BY-4.0** | ✅ open | **Kazakh legitimate institutional register** — our scarcest class, and FPR is the primary metric. |
| 5 | **`JimmyMa99/TeleAntiFraud`** | 28,511 Chinese fraud-call pairs / 307 h / 12.7 GB: 4,400 binary-classification + 34,153 SFT reasoning samples, plus TeleAntiFraud-Bench (7,021). The methodology `data/README.md` already cites (arXiv:2503.24115). | **Apache-2.0** | ⚠️ **gated** — data download returns `GatedRepoError`; accept the terms on the dataset page while signed in | A genuinely independent generator for the cross-generator gap (ADR D35), and a taxonomy cross-check. Honest caveat: it is ASR-of-real-calls **regenerated through TTS** plus LLM expansion — a hybrid, not raw real calls. |
| 6 | **`kz-transformers/multidomain-kazakh-dataset`** | Large Kazakh multi-domain corpus (news, books, CC-100, Leipzig, OSCAR under one roof). | **Apache-2.0** | ✅ open | Register widening (ADR D42 continued). Subsumes three separate provenance lines into one permissive entry for ТЗ §9. Card admits it is **not anonymised** → scrub first. |
| 7 | **`Shirali/ISSAI_KSC_335RS_v_1_1`** · **`issai/Kazakh_Speech_Corpus_2`** | KSC: 147k Kazakh utterances with text, already **lowercase and unpunctuated** — the same normalisation our ASR path produces. KSC2: ~1.2k h / 600k+ utterances incl. TV, radio, senate, podcasts. | CC-BY-4.0 · **MIT** | ✅ open | KSC text pulls cheaply via the rows API without the audio. KSC2 is the widest spontaneous Kazakh register but ships as a 10-part split tarball — expensive. |
| 8 | **`DeepPavlov/Multi2WOZ`** (`ru*` configs) | Human-translated MultiWOZ in Russian, ~14.7k turns. Agent turns legitimately hand out phone numbers, booking refs and time pressure. | ⚠️ **no licence tag** on the mirror (upstream MIT) | ✅ open | Hard negatives — the surface features our lexicon over-fires on. **Only `validation`/`test` splits exist for `ru`**, and entities are Cambridge UK → must be re-localised to KZ. |
| 9 | **`mteb/sib200`** (`kaz_Cyrl`) · **`issai/kazqad-retrieval`** | Kazakh classification (899 rows) and native Kazakh retrieval. | CC-BY-SA-4.0 | ✅ / ⚠️ gated | Answers a question we have never measured: **is e5-base's Kazakh the bottleneck?** Mirrors our own embeddings+LR setup. Cheap, and it is the missing evidence behind keeping e5-base. |
| 10 | **`BothBosu/scam-dialogue`** + siblings | 1,600 + 1,600 + 1,000 English synthetic scam-call dialogues in `caller:` / `receiver:` form — structurally identical to our `Dialogue` schema, from a different generator. | Apache-2.0 | ✅ open | Scenario coverage and a non-Gemini register, via translation. English. |

---

## 3. Speech & telephony corpora — and the one that actually matches our channel

Our ASR numbers (ADRs D39/D40/D41) are all measured on **read/studio** Kazakh speech or on
`say`-synthesised audio. Real **telephone-channel** Kazakh exists, but only behind LDC
licensing:

| Corpus | What | Licence / cost | Status |
|---|---|---|---|
| **IARPA Babel Kazakh, `LDC2018S13`** | **~203 h recorded / ~64 h transcribed Kazakh conversational + scripted TELEPHONE speech** (2013–14, NE + S dialects). The only real Kazakh telephone corpus in existence. | Babel Kazakh Agreement (For-Profit / Non-Member / Not-For-Profit). **Price login-gated.** LDC's rule: non-members "cannot use LDC data to develop or test products for commercialization"; government members are limited to "noncommercial linguistic research and education only". | ⚠️ licence is the blocker, not the money. Won't land quickly. |
| **MATERIAL Kazakh-English, `LDC2025S03`** | ~57 h Kazakh conversational telephone speech, varied handsets and environments; **only ~17% transcribed** (≈10 h). | as above | ⚠️ thinner than Babel |
| **CALLFRIEND Russian, `LDC2023S08`** + text `LDC2023T09` | ~48 h / 100 recordings of long-form **8 kHz two-channel** conversational Russian, with 97 transcript files. Closest thing to our real deployment channel. | LDC User Agreement for Non-Members | ⚠️ 1999 vintage, US-diaspora accents |
| **OpenSTT `asr_calls_2_val`** | **7.7 h / 12,950 utterances of real Russian phone-call audio, manually annotated, 99% "crisp"** — 0.8 GB, download verified live on Azure Open Datasets. | **CC-BY-NC** ("commercial usage available after agreement with the authors") | ✅ **downloadable today** |
| OpenSTT `asr_public_phone_calls_1/2` | 812 h of phone calls — but **ASR-generated labels at 70% "noisy" quality**, and ~3.6 s isolated utterances with no call-level grouping or turn structure. | CC-BY-NC | ❌ not a label source |
| **MCSKL Module 1** (OSF `osf.io/6zjdq`) | **12 h / 33 speech events / 78 speakers of naturally-occurring conversational Kazakh**, time-aligned transcription + English translations. Every other free Kazakh corpus is read, broadcast or TTS. Incidental Russian code-switching preserved. | **CC-BY-NC-SA-4.0 per the OSF deposit** (the paper abstract says CC-BY-4.0 — the deposit wins) | ✅ 7.18 GB present; internal eval only |
| **KSD, OpenSLR SLR140** | 554 h / 204,250 Kazakh utterances, recorded on **iOS/Android mobile mics** — the closest acoustic match to our on-device PWA capture among free Kazakh corpora. | **CC-BY-SA-3.0** — ShareAlike obligation on derivatives | ✅ live |
| KSC, OpenSLR SLR102 · Golos SLR114 · RuLS SLR96 | The standard read-speech benchmarks. OpenSLR has **exactly four** kk/ru entries — none telephone. | CC-BY-4.0 · custom (probably BY-SA-equivalent, PDF only — read it) · US public domain | ✅ live |
| Common Voice v27 | **kk: 60.24 validated hours**, 1,082 speakers, but 60.5% male and 52% in their twenties. ru: 2,290 validated hours. | CC0 | ✅ live |

**A cross-lane correction worth recording.** The KRCS paper (arXiv:2503.20007 / NAACL 2025 SRW)
presents "the first code-switching Kazakh-Russian parallel corpus" and contains **no release
link**, so it reads as unreleased. It *is* released — as `BorisovMaksim/kk_ru_csw` under
**Apache-2.0** (§2 #2); the card carries the paper's own citation. Found by card full-text
search, not by following the paper.

**`naadgob/KazNLP` — the largest KK/RU code-switch text resource, and it is unusable as-is.**
331,468 deduplicated Telegram / Kaspi / 2GIS documents (4.9% mixed), including a
**document-level gold LID set of 3,076 samples (1,000 ru / 999 kk / 1,077 mixed)**. The
Kaspi/2GIS consumer-finance register is unusually close to the legitimate-bank-call negative we
lack. **But the repo has no LICENSE file — GitHub reports `license: null`, i.e. all rights
reserved.** "It was on GitHub" is not a defence for a government deliverable. One email to the
author could unblock the single most useful code-switch resource found; until then, do not
touch it. Its companion `liminovna/KazRusCSW` (claimed token-level code-switch annotation)
contains **only notebooks — no data at all** — and is likewise unlicensed.

---

## 4. Kazakhstan: what exists domestically, and the legal boundary

### 4.1 No public KZ scam-call corpus — independently corroborated

Official bodies publish **text** warnings and **scripted** reconstructions, never call audio.
Checked: National Bank news (`nationalbank.kz/.../16279`, `/16486`), eGov's citizen guide
(`egov.kz/cms/ru/articles/legal_relations/pass_scammers`, RU+KK), polisia.kz, KZ-CERT.

A **parallel project** reached the same conclusion in the same week:
`github.com/daurenoralbek/qalqan` (verified via the GitHub API — created 2026-09-21, pushed
2026-09-23, **no licence**), "Қалқан — incremental phone-scam detector for Kazakhstan (kk/ru)
… XLM-R+LoRA, synthetic corpus, real-call evaluation, federated learning". Almost certainly
another Decentrathon/GovTech Camp entrant. Its `data/external/README.md` states that no public
Kazakh-language scam-call recordings exist and that no public RU *legitimate* calls exist
either — so it can measure recall but not FPR. Read for intelligence; **copy nothing** (no
licence). The competitive implication is worth knowing about.

### 4.2 Usable domestic material

| Source | What | Legal footing |
|---|---|---|
| **Kaspi.kz × ARDFM "против мошенников"** — `youtube.com/playlist?list=PLRLZbDT_OenoBGJ-P5ojw4e-NaVAiW2Wq` (19 videos, `@kaspikz`, verified) | Officially sanctioned awareness series, **actors performing realistic RU scam dialogue**. Not real calls. | Cleanest available: scripted, no real personal data. Best source for new `authored_heldout` cases; cite each video URL. Check for KK variants. |
| **MVD-sourced dialogue quoted in KZ news** (e.g. `nur.kz/incident/crime/2093117-…`) | nur.kz / zakon.kz / tengrinews routinely **transcribe MVD-released scam-call video into article text** — short, real, attributable RU fragments. | Art. 7(5) re-use with source citation (see §4.4). Ideal for validating cue coverage and the D39 matcher. |
| **4 verified real YouTube recordings** of scam calls to Kazakhstanis (`2vNm2U_AzNo`, `Hhu6yEzfxOU`, `z0xJJvniKXM`, `4NkRpo_mrcM`) | Genuine calls, all **Russian**, all **scam-baiter style** (victim trolls the caller) — behaviourally unlike a real victim. | No consent chain → **local smoke test only, never redistributed.** Expect depressed recall; that is itself a reportable finding. |
| **data.egov.kz — Prosecutor-General / КПСиСУ form №1-М** `data.egov.kz/datasets/view?index=gp_od_service_1m` | Registered criminal offences with a documented JSON API (`/proxy/gp_od_service_1m?apiKey=…&date=yyyy-MM-dd`; field spec at `data.egov.kz/pages/portalpage?id=122085422` §16) — includes a **`mowenichestvo`** field. Portal search only works in Kazakh («қылмыс»), not «мошен». | Free API key at `data.egov.kz/profile/apikeylist`. Cheapest real citable KZ number for the repo. |
| **National Bank Antifraud Centre results** `nationalbank.kz/file/download/119314` (verified: 200, PDF, 596 KB) | 80,871 incidents with fraud indicators as of 01.01.2026; 2.8 bn ₸ blocked. Scheme mix (>111k incidents, Jul 2024–Feb 2026): fictitious shops 24,042 · **false phone calls 21,027** · fake investments 16,220. | NB's data-usage terms permit free use incl. building software products, **with source attribution**. |
| **qamqor.gov.kz legal statistics** (use **without** `www.`) | A public Grafana: `/stats/api/search?query=&limit=500&type=dash-db` returns 33 dashboards anonymously, with region/period/agency filters. | ⚠️ The dashboard JSON exposes a live MySQL datasource. Read dashboards; **do not push arbitrary SQL at `/stats/api/ds/query`** on a Prosecutor-General system. |
| **ARDFM list of 121 organisations with signs of illegal activity** `gov.kz/memleket/entities/ardfm/documents/details/318075` | 56 pyramid-like + 65 unlicensed brokers, regularly updated. Page is an 836-byte SPA shell → needs a browser/Playwright. | Real named seed entities for the **L2 analyst panel** — one real cluster in the demo is worth rubric points. |

**Dead end:** `stat.gov.kz` / `taldau.stat.gov.kz` have **no crime rubric** — fraud counts live
with КПСиСУ (qamqor / data.egov.kz), not the statistics bureau. The Antifraud Centre publishes
no dataset or API; its statistics go to participants only.

### 4.3 Who actually holds real KZ call data

- **Antifraud Centre** — owned by the **National Bank**, operated by **НПЦК/NPCK**
  (`npck.kz/en/anti-fraud-center/`, `Support.anti-fraud@npck.kz`, +7 727 297 9100). Legal basis:
  Правление НБ РК № 54 of 25.08.2025 (MinJust № 36742), restated by № 58 of 10.06.2026; full
  text at `zakon.uchet.kz/rus/docs/V2500036742`. It holds **transactions, IINs, accounts and
  compromised subscriber numbers — not call audio**, and п.4(5) requires it to hold personal
  data **«в обезличенном виде»**. Its п.3(9) «иные лица, определяемые решением Национального
  Банка» is the legal door for a tool like Qorğan.
  → **Pitch line, and it is true: nobody in the official contour owns a labelled scam-call
  transcript corpus.** That is exactly the gap this project fills.
- **Анти Call-центр, Astana** — launched 31.07.2026 by **Kazakhtelecom + Astana Prosecutor's
  Office**; operators keep scammers talking while cyber-police warn the victim; ~1.5 M
  intercepted calls, Kcell has redirected >30,000. **Whether calls are recorded is not stated
  in any source found — unverified.** Nonetheless the most likely holder of real bilingual
  scam-call audio in the country, and the strongest ask to route through the government
  representative.

### 4.4 The legal boundary — this is the part that changes how we work

> **Read `docs/LEGAL_ASSESSMENT.md` first** (added 2026-09-26 on `dev/loop`, dated 2026-09-25).
> It is the authoritative engineering-side legal review — statute-graded (**[S]** / **[I]** /
> **[U]**), read against the consolidated text as of 24.09.2026, and more current than this
> section: it covers the new Constitution (in force 2026-07-01), Law 326-VIII's notification
> duty and breach register, the AI Law's risk/autonomy classification, the amended Criminal
> Code Art. 148, the Art. 3(3)(1) personal-needs exemption for Level 1, and the genuinely hard
> problem this survey does not address — **the caller never consented** (Art. 9(5), untested).
>
> This section keeps only what that review does not cover, both of which bear on *data
> acquisition* rather than on the running product:
> 1. **Art. 7(5)** — the re-use basis for public material (below). Not in the legal assessment.
> 2. **The Hugging Face publication path** as a personal-data *distribution* channel. ADR D45
>    already makes `hf_upload.py` refuse a split that is not a scrub fixed point, so the code is
>    hardened; the statutory reason for that guard is still unwritten.

Quotes verified against the primary text (`zakon.uchet.kz/rus/docs/Z1300000094`, ЗРК
21.05.2013 № 94-V). Adilet serves a JS shell; use that mirror.

- **Art. 7(5)** — third parties **may** re-collect, process and redistribute lawfully published
  personal data **«при условии наличия ссылки на источник информации»**. This is what makes the
  news-quote and awareness-video route legitimate: cite every source URL.
- **Art. 17(1)** — for **statistical, sociological, scientific or marketing research**, the
  party *transferring* personal data **«обязаны их обезличить»**. So a bank must anonymise
  *before* handing anything over; `DATA_INTAKE.md` §6 already matches this, and the on-prem
  ingest option is the right shape.
- **Art. 12(2)** — **«Хранение персональных данных осуществляется … в базе, находящейся на
  территории Республики Казахстан.»** Data localization.
- **Art. 16** — cross-border transfer only to states ensuring protection, or with consent.
- **Art. 9 has no general research exemption** — a research project gets no consent-free basis.
- **Art. 19-1 (new, added by ЗРК 17.11.2025 № 231-VIII)** — verified verbatim:
  **«Запрещается автоматизированная обработка персональных данных, в результате которой у
  субъекта возникают, изменяются или прекращаются права, законные интересы»** without consent;
  the operator must explain the automated processing, accept an objection, and **respond within
  three working days**.
  → Qorğan's locked "decision-support, a human always decides" stance plus grounded, templated
  explanations map onto 19-1 almost line for line. **This is a compliance argument, not just a
  design preference — say it in the README and the deck.** The cheap missing piece is an
  explicit "this was wrong" objection affordance in the UI.
- **Цифровой кодекс РК № 255-VIII** (in force 2026) and the **AI Law № 230-VIII** (in force
  18.01.2026) — both are covered properly in `docs/LEGAL_ASSESSMENT.md`. **Correction to an
  earlier draft of this section:** the AI Law is *not* "still pending" — it is adopted and in
  force, and under it Qorğan classifies as a **low-autonomy** system (a human makes the final
  choice), which is the legal value of our "human always decides" invariant.
- **УК РК ст. 147 / 148** (privacy and secrecy of telephone conversations) are the criminal
  backstop. **Unverified and left open on purpose:** whether recording by a *participant* in a
  call falls outside Art. 148. Do not assert either way — this needs a lawyer.

**The boundary, stated plainly.** Three things in this repo are fine **today only because the
corpus is synthetic**, and all three break the moment one real transcript enters the pipeline:

1. the **Gemini API** (cross-border processing — Arts. 7(6), 16),
2. the **Hugging Face uploads** `sanzh-ts/govtech` and `sanzh-ts/govtech_ds` via `hf_upload.py`
   (distribution in a public source **and** cross-border — Art. 7(3), 7(6), 16),
3. **non-KZ storage** (Art. 12(2) localization).

`DATA_INTAKE.md` §8 already says `data/real/` is gitignored and excluded from the Hub
allow-patterns — that instinct was right, and this is the statutory reason for it. The boundary
deserves a named section in `data/README.md` and an ADR, because the ТЗ §9 provenance grading
will reward it and the government partner will ask.

### 4.5 Court verdicts — collected 2026-10-09 as `court_quotes` (ADR D68)

| Source | Reachable? | Allowed? | Used |
|---|---|---|---|
| **office.sud.kz** «Сот актілерінің банкі» (KZ, RU + KK verdicts 2009–2025) | yes | no robots.txt, but **reCAPTCHA v2 on every search** | no: automating it is CAPTCHA circumvention. A person must search by hand; the only route to Kazakh verdicts. |
| **sudrf.ru** court sites (RU, official) | yes | robots.txt **disallows** `name=sud_delo` / `name=bsr` (the judgment pages) | no |
| **sud.gov.kz** press releases (KZ Supreme Court) | browser only (curl gets an empty reply) | robots allows news (Crawl-delay 10); re-use with a hyperlink | 2 passages read, 1 row |
| **sudact.ru** (RU aggregator of sudrf texts) | yes | robots allows `/regular/doc/<id>/`; its search (`doc_ajax`) is disallowed; no prohibiting terms found on its About / PD-policy pages | 135 verdict pages via web-search discovery, ≥ 12 s apart → 46 rows |

Result: 47 rows, all Russian, 42 reported speech + 5 direct quotes, from 7 sources; 224 rule
candidates, 177 excluded on review, 2 dropped for residual PII. Text stays in
`data/raw/court_quotes/` (gitignored, never published); `data/manifests/court_quotes.manifest.json`
holds counts and every citation. Held out: never tune on it, never extend the lexicon from it.
Yield is the bottleneck (6 of 135 pages gave a kept row); sudact's own search, with permission, or
a hand search of office.sud.kz would fix it.

---

## 5. Verified dead ends — stop looking

- **`MTSAIR/MWS-Antifraud-Bench`** — the name is misleading. I checked all three configs
  (`default`/`en`/`zh`, 100 rows each): it is **document-forgery image classification**
  (original / edited / ai_gen scans), not call data. Irrelevant to us.
- **Machine-translated SMS spam** (`dbarbedillo/SMS_Spam_Multilingual_...`) — has a `text_ru`
  column, and it is visibly broken (`"ОК ЛАР... ЖУРЬ ВИФ У ОНИ..."`). MT register collapse is
  the exact pathology we are fighting. **Actively harmful as RU training data.**
- **MASSIVE, XNLI, Banking77 have no Kazakh** — verified in the loader source / language
  lists, correcting an assumption in my own research brief. Banking77's value is its
  **77-intent list as a specification for authored hard negatives**, not as data.
- **`esimijoq/Kazakh-Russian-Child-Directed-Speech-Corpus`** — not code-switched (separate
  `Kazakh/` and `Russian/` trees), licence unspecified, copyrighted sources, wrong register.
- **`MaratDV/russian-call-center-speech-ru`** — 832 h of *real* Russian call-centre telephony,
  the only such corpus found, but **"Commercial license only. Redistribution not allowed"**,
  **no transcripts** ("Metadata: Not available"). Buy-and-transcribe only.
- **`kurumikz/telegram-corpus-russian-kazakh`** — 1.49M lines of genuine KK/RU code-switch,
  but **CC-BY-NC-SA** (non-commercial + viral), pervasive profanity, and PII at scale
  (~12k `t.me/` links, ~5k usernames extrapolated). Given commit `b194707` scrubbed a
  fabricated phone number out of an augmentation, ingesting this raw repeats that failure at
  1000× volume. Mine patterns, do not ingest rows. `kk_ru_csw` (#2 above) is the clean
  alternative.
- **`BothBosu/youtube-scam-conversations`** — 20 real scambait transcripts, but **all
  label=1** (no negatives), all **outgoing** calls where the baiter phoned the scammer (register
  inverted vs a victim receiving a call), and tagged Apache-2.0 over third-party YouTube
  content the uploader cannot license. Treat as reference only.
- **Licence-blocked elsewhere**: KorCCVi (Korean vishing — real transcripts, but FSS + AI Hub
  forbid redistribution, and source/label are perfectly correlated so its ~98 F1 is channel
  recognition), `Telecom_Fraud_Texts_5` (Chinese — research institutions only, commercial use
  forbidden), Korea FSS perpetrator audio (all rights reserved), Japanese police awareness
  audio (no licence).
- **Kaggle "1,000-hour Russian call-centre" datasets are vendor teasers.** Verified by reading
  the actual download sizes from each page's embedded schema.org JSON:
  `axondata/russian-call-center-audio-transcription` claims 1,000+ h → **2.5 MB**;
  `unidpro/russian-speech-recognition-dataset` claims 338 h → **295 bytes**;
  `axondata/call-center-speech-dataset` claims 10,000 h → **12.8 MB** (and no Kazakh). All are
  CC-BY-NC or NC-ND previews with a "buy the dataset" call to action. **Do not put the headline
  hours into `data/README.md`.** Axon Labs / Unidata are the vendors to quote if budget ever
  appears.
- **No Kazakh speech corpus from any Kazakhstani government body or language agency** — two
  targeted sweeps found nothing. The Kazakh Language Corpus (KLC, ~135M words) is text and has
  no working download. Treat as non-existent rather than citing it.
- **ELRA is not a viable route**: the Russian mobile-phone corpus `ELRA-S0443` lists at
  **€180,861**, and the catalogue search endpoint returned HTTP 429 on every attempt across
  both mirrors — so "no Kazakh in ELRA" is probable, not proven.
- **CLARIN/VLO** — nothing for kk/ru telephony; Kazakhstan is not a member country.
- **Common Voice code-switching** — the `code-switching` directory exists but contains only a
  README saying "Alpha test phase -- no releases yet". No kk/ru code-switched CV data exists.
- **Paper-only, no download**: the scam-baiting corpus (arXiv:2307.01965, which
  `data/README.md` cites — its release URL is an unfilled `to.be.released.on.acceptance`
  placeholder), the Turkish scam-call set (arXiv:2606.24523, IEEE S&P'26), TeleAntiFraud 2.0
  (anonymous review repo, 401/403).

---

## 6. What the survey found in **our** pipeline (highest-value output)

Probing the shipped cue lexicon against real Russian scam phrasing exposed a script-level gap
that our own corpus hides. All numbers below are reproducible with the repo's own
`cue_lexicon` + `cue_match`.

**The bug.** Every SMS cue in `data/lexicon/hard_signal_cues.yaml` spells SMS in **Latin**
(`"код из SMS"`, `"Продиктуйте код из SMS"`, `"SMS-тегі кодты айтыңыз"`). Real Russian writes
and real ASR decodes **Cyrillic "СМС"**. `normalize()` correctly keeps the scripts distinct
(`кодизсмс` ≠ `кодизsms`), so the bounded-edit matcher cannot bridge it:

```
MISS  text=«Продиктуйте код из СМС»  cue=«Продиктуйте код из SMS»
HIT   text=«Продиктуйте код из SMS»  cue=«Продиктуйте код из SMS»
```

**It is not hypothetical — it fires on real recogniser output.** In `data/asr_capture/`,
`secrecy_mixed_3#6` decoded cleanly except for the script, and the hard signal died:

| | |
|---|---|
| reference | «…нужно отправить **код из SMS**. Только нам, и никому другому.» → cue `otp_request: код из SMS` **hits** |
| real Vosk hypothesis | «…нужно отправить **код из смс** только нам и никому другому» → **no cue at all** |

**Scale, measured on the built corpus.** 76 positive rows contain Cyrillic "смс"; of those,
**53 voice a request-shaped ask and fire zero `otp_request` cues** — 46 in `train`, 3 in
`test`, 2 in `val`, 2 in `ood`. Canonical OTP-phishing lines like
`«…необходимо немедленно сообщить нам код из СМС»` currently carry no hard signal, so the
61/81 meter floor never arms for them and they rest entirely on the embedding head.

**The fix is constrained by the lexicon's own invariant.** A bare `"смс"` cue would be unsafe:
96 **negative** rows mention it benignly, always as a *notification*
(`«вам придёт смс-уведомление о блокировке»`, `«СМС-хабарлама келеді»`), never as a request.
So: add **request-shaped** Cyrillic variants only. Per the July sprint lesson recorded in
`docs/STATUS.md` — *pair lexicon edits with training data or the joint-LR refit moves the KK
boundary* — this must be a lexicon **+** data change, retrained, with every FPR gate and
sentinel re-run. `push`/`пуш` is the same class of bug at much smaller scale (1 positive row).

**Two taxonomy gaps, grounded in real 2026 reporting** (`fraudlens-ru-v1`, mention counts):
- **Voice cloning / deepfake relatives** — 77 mentions; **no tactic id covers it.** Note that
  `family_money_request` exists only as a hard *negative* ("a relative genuinely asking for a
  transfer"), while the real-world relative-in-trouble scam has no positive tactic. This is a
  plausible mechanism for the documented `shift` miss where relative-in-trouble scams score
  ≤ 0.10 (ADR D35); 6 of the 66 `shift` calls are framed that way.
- **Minor-targeted schemes** — 417 mentions of children/teenagers, including "self-kidnapping"
  («Уйди из дома и никому не говори») and recruiting teenagers into "tasks" (mule recruitment
  of minors). `mule_recruitment` does not cover the child angle, and the secrecy cues are
  formal (`«Никому не говорите»`) while the real phrasing is informal
  (`«не говори родителям»`, `«ни с кем не говорить»`).

**Mandatory caveat on `fraudlens-ru-v1`: it is 97% Russia-grounded.** I counted institution
markers: Russia 4,750 (Госуслуги 709, Сбер/ВТБ/Тинькофф 927, рубль 2,122, ФСБ/МВД 375) vs
Kazakhstan 147 (Казахстан 58, eGov 54, тенге 2, Kaspi/Halyk 12). Use it for **mechanism and
phrasing**; every institution, currency and agency must be re-localised to KZ
(eGov.kz, Нацбанк РК / АРРФР, КНБ, Kaspi/Halyk/BCC, тенге, ЖСН) before anything enters the
corpus. Do **not** mine cues from it into an eval split's blind spot.

---

## 7. External validation worth citing

- **arXiv:2609.29528** (voice-agent honeypot data descriptor) reports that detectors trained on
  published *synthetic* scam corpora "collapse in precision on real traffic (F1 0.02–0.40)",
  and that TF-IDF baselines scoring ROC-AUC 1.00 on their synthetic source fall apart on real
  calls. **This is our generator-shift result (ADR D35) reproduced independently** — a citable
  external justification for the FPR-first framing and for keeping the `shift` split, whether
  or not we ever obtain their data. Their open subset (1,000 real calls, 500 scam / 500 spam,
  CC BY-NC) is **paper-only today** — no persistent identifier, absent from the Hub; the route
  is an email to the authors.
- **arXiv:2606.24523** (Turkish scam calls, IEEE S&P'26) found that **transcript input beat raw
  audio**, and that **raw ASR ≈ human-corrected ASR**. Turkish is the closest well-resourced
  agglutinative Turkic relative of Kazakh, so this externally supports our transcript-first
  architecture and our tolerance for ASR noise.
- **TeleAntiFraud 2.0** (arXiv:2609.18748) builds its non-fraud class deliberately from
  *near-domain* lawful calls, because unrelated-topic negatives let models key on the channel
  instead of the fraud. That is our hard-negatives-are-first-class principle with an external
  citation — and the KorCCVi channel-leak artifact is the cautionary case.

---

## 8. Recommended next actions

Ordered by value per day. None of these is started.

1. **Fix the Cyrillic-СМС cue gap** (§6) — lexicon + paired training data + retrain + full
   gate re-run. This is a live hard-signal loss on real ASR output affecting 53 corpus
   positives, found by measurement, with a bounded and safe fix. Highest value, lowest risk.
2. **Accept the TeleAntiFraud terms** (2 minutes, signed in as `sanzh-ts`) so the option
   exists, then decide on a translated slice as a third training generator.
3. **Download OpenSTT `asr_calls_2_val`** (0.8 GB, verified live) and run the existing ASR
   harness on it. 7.7 h of *real* Russian telephone audio with human transcripts gives us the
   first honest telephone-channel WER to sit beside the `say`-synthesised numbers behind ADRs
   D39–D41 — best value per hour of effort in this survey. CC-BY-NC → eval harness only, never
   shipped.
4. **Run `mteb/sib200:kaz_Cyrl` + `issai/kazqad-retrieval` through e5-base** — half a day, and
   it finally answers whether Kazakh embedding quality is the KK-boundary bottleneck.
5. **Pull the four open CC-BY/Apache Kazakh + code-switch sets** (#2, #3, #4 above) into a
   `data/external/` staging area with per-source provenance for ТЗ §9. Small, clean, licensed.
6. **Replicate the Turkish recipe for kk/ru** — harvest real scam calls from public KZ
   awareness material through the existing ADR D39 ASR harness. This is the only route to an
   independently-authored real KZ eval set that does not depend on the stakeholder, and the
   legal basis for each source must be recorded per §8 of `DATA_INTAKE.md`.
7. **Decide the two taxonomy gaps** (voice-clone impersonation; minor-targeted schemes) as an
   ADR before adding data, since tactic ids are the classifier's label space.

**Kazakhstan-side, in parallel (§4):**

8. **Write the legal boundary into `data/README.md` + an ADR** (§4.4). Cheap, graded by ТЗ §9,
   and it prevents a real transcript reaching Gemini or the Hub by accident.
9. **Add an Art. 19-1 objection affordance** ("this was wrong") to the PWA, and put the
   compliance mapping on a slide. We already do the hard parts; this closes it.
10. **Hand the government representative a one-paragraph ask** naming the three actual holders:
    NPCK as antifraud-centre operator (`Support.anti-fraud@npck.kz`) — including whether п.3(9)
    «иные лица» could cover a pilot; the Astana Анти Call-центр (Kazakhtelecom + Prosecutor's
    Office) — whether intercepted calls are recorded at all; and the MVD press service, for the
    source audio behind scam-call videos they already publish.
11. **Get a `data.egov.kz` API key** and pull the `mowenichestvo` series — the cheapest real,
    citable Kazakhstani number we can put in the repo.
12. **Transcribe the Kaspi×ARDFM series** (19 videos) into new `authored_heldout` cases, citing
    each URL. Scripted and officially sanctioned, so it is the safest realistic RU dialogue
    available — and it is *not* Gemini's register, which is the point.

_Licence triage for the pending legal review (`DATA_INTAKE.md` §9): **safe** — Apache-2.0 and
CC-BY-4.0 items above, FTC robocall audio (US public domain), PersuasionForGood (Apache-2.0).
**Needs a ruling on whether a public-interest government tool qualifies** — CC-BY-NC items
(honeypot subset, International Robocalls, `farabi-lab/Code_switching`, the Telegram corpus).
**Do not ship** — KorCCVi, `Telecom_Fraud_Texts_5`, FSS/Japanese police audio._
