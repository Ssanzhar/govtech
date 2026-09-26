# LEGAL_ASSESSMENT.md — Qorğan under Kazakhstan law (draft, 2026-09-25)

> **This is not legal advice.** It is an engineering-side legal review. Its job is to map the
> code's data flows onto the statutes, list the gaps, and brief a qualified Kazakhstan advocate.
> **No pilot** (bank, Anti-Fraud Center, inDrive, state body) should start until a KZ advocate
> has signed off on §2 (legal bases), §3 (call secrecy) and the texts in §7.
>
> **Markers used below.** **[S]** means the statute says this. The text was read on
> adilet.zan.kz on 2026-09-25 and is the consolidated text "as of 24.09.2026". **[I]** means
> our interpretation. **[U]** means unverified: we could not confirm it from a primary source.
>
> **Scope.** Code on branch `dev/loop` (off `sanzh-ts`) as of 2026-09-25. Two other agents are
> changing this tree during the same iteration:
> - **Analyst console:** analyst authentication (`src/qorgan/analysts.py`,
>   `api_admin_auth.py`), the roles `analyst` and `investigator`, closed purpose codes for
>   opening a transcript, and an HMAC-chained audit log (`audit.py`).
> - **Citizen page:** localization.
>
> This document assesses the design with that work included. §5.2 lists what it must satisfy.

## 0. Summary

1. **Level 1 on the device is the legally strongest shape available.** Audio never leaves the
   device and the transcript stays in memory. A citizen who analyses their own call for their
   own protection is arguably outside the Personal Data Law altogether, under the
   personal-needs exemption (Art. 3(3)(1)) **[I]**. Keep it that way.
2. **The weakest legal point is the caller.** A report carries the *other party's* speech and
   number. The citizen cannot consent for the caller. The most plausible basis is PD Law
   Art. 9(5) ("protection of constitutional rights … if consent is impossible"), but that
   reading is untested **[I]**. Most such callers are scammers, but some are real bank
   employees.
3. **The code still has paths the product story says do not exist:**
   - `backend=llm` on public and admin routes sends raw text to Google (outside KZ) and caches
     verbatim phrases on disk.
   - `/api/live/session/*` holds transcripts in server memory, and its `/report` route is a
     second citizen ingress with no consent field.
   - Retention is keyed on a client-supplied timestamp, and the purge is never scheduled.
   - Deletion does not reach `org_feedback.jsonl`.
4. **Kazakh law changed a lot in 2026.** The precursor notes in CLAUDE.md §8 predate these
   changes:
   - A **new Constitution** has been in force since 2026-07-01 (Art. 21 now names personal
     data and digital communications explicitly).
   - **Law 326-VIII** (24.06.2026) added a *notification* duty (PD Law Art. 10-1), size classes
     for operators (Art. 25-1) and a breach register.
   - The **Digital Code** (No. 255-VIII) came into force in July 2026.
   - **Criminal Code Art. 148** was amended from 2026-07-01.
5. **Under the AI Law, Qorğan is a "low-autonomy" system.** A human makes the final choice
   (Art. 17(2)(1)). Level 1 is plausibly **medium risk** and Level 2 at a state body is
   possibly **high risk** **[I]**. Two duties apply now: a user agreement shown before use
   (Art. 15(2)(5)) and a disclosure that AI is used (Art. 21(1)). Neither exists on the site
   today.

## 1. Data-flow inventory (from the code)

| # | Data | Where it lives | Retention | Who can access | Legal basis claimed in code |
|---|---|---|---|---|---|
| F1 | Call audio (mic mode) | Browser only: Vosklet WASM in `site/core/asr.js`. No route accepts audio (`tests/test_architecture.py`). | RAM for the length of the call | The user | None needed: the user's own processing |
| F2 | Live transcript, meter, advice | Browser only (`site/live.js`, `site/core/*`). `localStorage` holds only a UI preference (`live.js:72`). | RAM; cleared on reset or when the page closes | The user | Same as F1 |
| F3 | Model download | Own origin `site/models/`. **Third-party requests:** transformers.js from jsDelivr (`site/core/embed-worker.js:11`) and Google Fonts (`site/*.html:9-11`). These expose the user's IP and page visits to foreign hosts. | n/a | CDN and Google | None stated |
| F4 | `POST /api/analyze`, opt-in server scoring (`src/qorgan/api.py`, `site/try.js:89`) | Server RAM; nothing is written for `linear`/`mock` (architecture test 2) | Request lifetime | Server operator | The user ticks "score on server" |
| F4b | Same route with `backend=llm` (the client controls this field) | Raw, unscrubbed text sent to **Google Gemini**. `llm_classifier._write_cache` writes the result, including **verbatim trigger phrases**, to `data/cache/llm_predictions/`. | Cache has no expiry | Google; server operator | **None.** There is no notice and no consent. It breaks the "persists nothing" invariant for this backend. |
| F5 | `/api/live/session/*` (`api_live.py`, `live/session_store.py`) | Server RAM, up to 200 sessions, oldest-evicted, **no time limit (TTL)**. `/session/{id}/report` writes a report through `live/summary.submit_report`. | Until evicted | Server operator | The report is filed on the click alone: no `consent` field, no review, and `received_at` is not set |
| F6 | Citizen report `POST /api/reports` (`api_reports.py`, `reports/store.py`) | `data/processed/citizen_reports.jsonl` (plain JSONL). Contents: scrubbed transcript, HMAC digest + `+7 700 ***` prefix, tactic ids, flagged phrases, risk score, client `timestamp`, server `received_at`. | 180 days, but only through the manual CLI `reports.purge`. The purge compares against the **client** `timestamp` (`store.purge_expired`, `purge.expired_receipts`). | Analysts after ingest; the citizen by receipt (delete only) | `consent_basis="citizen_explicit_submit"`, plus the D44 review with the consent checkbox |
| F7 | Incidents, organizations, embeddings (`analytics/intake.py`, `data/processed/*.jsonl`, `.npz`) | Derived from F6 and F9 | Removed along with the report by `forget_report` | Analysts | Same basis as F6 |
| F8 | Analyst feedback `org_feedback.jsonl` (`analytics/feedback.py`) | Snapshots of **number digests** and incident ids, plus the analyst id | **Never deleted.** Deleting or purging a report leaves the digest here. | Analysts | None stated |
| F9 | Partner report `POST /api/v1/reports` (`api_partner.py`) | Same file as F6 with `source=partner`. The transcript must already be scrubbed, and the number is hashed. | Same as F6 | Analysts; the partner can delete its own reports | A `consent_basis` code supplied by the partner. It is not checked against any agreement. |
| F10 | Partner export `GET /api/v1/organizations` (`api_partner_export.py`) | Aggregates only: no digests, no text | n/a | The partner | No personal data leaves |
| F11 | Audit log `audit_log.jsonl` (`audit.py`, being rewritten) | Actor id, action, subject id, outcome, purpose. HMAC chain in progress. | **No retention rule** | Operator | Security logging |
| F12 | Server access logs (uvicorn defaults) | Client IP plus path. The path includes the **receipt id** on `DELETE /api/reports/{id}`. Timing plus IP can be correlated with `received_at`. | Whatever the host keeps | Host operator | None stated |
| F13 | Cloud "second opinion" (`classifier/llm_classifier.py`, `llm_tools.py`, `gemini-2.5-pro`) | No citizen UI exists. Reachable through `backend=llm` on `/api/analyze`, `/api/live/session`, `/api/admin/incidents/{id}/analysis` and `/open`. | Google keeps logs for abuse monitoring (Paid tier) | Google | ADR D11/D38 intend "on explicit request"; the code does not enforce it |
| F14 | Training/eval data (`data/`, public HF `sanzh-ts/govtech_ds`) | Synthetic (Gemini, Claude). HF `ood.jsonl` still holds one synthetic IIN plus a full name (ADR D45). Real calls: `data/real/`, gitignored (`docs/DATA_INTAKE.md`). | Indefinite | Public (HF); intake operator (real) | Synthetic data; the real-call basis is open (DATA_INTAKE §9) |
| F15 | Secrets `QORGAN_NUMBER_HMAC_KEY`, `QORGAN_PARTNER_API_KEYS`, `GEMINI_API_KEY`, and the incoming analyst and audit-chain keys | Environment / `.env` | n/a | Server | n/a |

**Scrubbing limits that affect every row.** `data/scrub.py` redacts only digit-pattern phones,
cards, IINs and e-mails. It does not catch:
- names, addresses or KZ IBANs;
- numbers spoken as words. The on-device recogniser writes digits as words ("восемь семьсот …"),
  so a mic-mode report can carry a full phone number or card number unredacted.

## 2. Legal bases and consent

### 2.1 Statutory frame (PD Law No. 94-V as amended, including 231-VIII, 256-VIII and 326-VIII)

**Personal data and identifiers**
- **[S]** Personal data is "information about a subject … supplemented by one or more
  identifiers" (Art. 1(2)).
- **[S]** An identifier is information that identifies the subject *or links separate data
  sets about them* (Art. 1(14-1)).
- **[S]** Art. 6 lists identifiers: full name taken together, IIN, face image, and face
  biometric vector.

**Legal bases**
- **[S]** Processing needs the subject's consent "in the order set by the authorized body",
  except in the Art. 9 cases (Art. 7(1)).
- **[S]** Transfer to third parties and cross-border transfer need consent, subject to the
  Art. 16 exceptions (Art. 7(6)).
- **[S]** Purposes must be specific and determined in advance (Art. 7(8)), and processing must
  not be excessive (Art. 7(9)).
- **[S]** Consent is given in writing, through the state or non-state consent service, "or by
  another method allowing receipt of consent to be confirmed" (Art. 8(1)).
- **[S]** When the data sits in digital objects of state bodies, consent goes through the
  **state service** (Art. 8(1), second paragraph).
- **[S]** Withdrawing consent stops processing within 15 working days (Art. 8(7)).
- **[S]** Consent without the subject is allowed "to protect constitutional rights and freedoms
  … if obtaining consent is impossible" (Art. 9(5)).

**Content of a consent (Art. 8(4))**
- **[S]** The operator's name and BIN.
- **[S]** The subject's full name.
- **[S]** The term.
- **[S]** Whether data goes to third parties.
- **[S]** Whether data goes abroad.
- **[S]** Whether data is published.
- **[S]** The list of data collected.

**Storage, transfer, deletion and automated decisions**
- **[S]** Storage must be in a database or digital object located **in Kazakhstan**
  (Art. 12(2); in force since 2016). The retention term ends when the purpose is achieved.
- **[S]** Cross-border transfer is allowed to states that protect personal data. For other
  states it is allowed with consent, or when consent is impossible and constitutional rights
  must be protected (Art. 16(2)–(3)).
- **[S]** Deletion is required when the retention term expires or when data was processed
  without a basis (Art. 18).
- **[S]** Automated processing that creates, changes or ends a subject's rights is
  **prohibited** without consent or a law. The subject must be told how it works, must be able
  to object, and the objection must be answered within 3 working days (Art. 19-1, added by
  231-VIII).

**Operator duties**
- **[S]** Approve a personal-data policy and a list of the data processed (Art. 25(2)(1),
  (1-1)).
- **[S]** Keep proof of consent (Art. 25(2)(5)).
- **[S]** Notify a breach within one working day (Art. 25(2)(8)).
- **[S]** Appoint a person responsible for processing (Art. 25(2)(10)).
- **[S]** Log transfers to third parties and abroad (Art. 22(1)(5)).
- **[S]** Hashing and masking are named as *protection methods* (Art. 1(18), Art. 23(2)).
- **[S]** Anonymization requires an *irreversible* transformation (Art. 1(2-5)).

**Notification and size class (both new under 326-VIII, in force late August 2026, which is 60
days after publication on 25.06.2026; the date is computed, confirm it)**
- **[S]** An operator must notify the authorized body before starting and before stopping
  processing (Art. 10-1). Small and medium operators are exempt.
- **[S]** Size classes: small is ≤ 10,000 unique subjects, medium is up to 500,000, large is
  ≥ 500,000 (Art. 25-1).
- **[S]** Restricted-access personal data moves the operator up one class.

### 2.2 Per-flow analysis

| Flow | Whose data | Basis we propose | Assessment |
|---|---|---|---|
| F1–F2 on the device | The citizen; the caller's voice and words | Personal/family-needs exemption, Art. 3(3)(1) **[I]** | Strong, *as long as* nothing reaches Qorğan. The vendor never receives the data. |
| F3 CDN and fonts | The citizen (IP address) | None | Self-host these files. This removes a foreign transfer and keeps "nothing leaves your machine" (`live.html:152`) true. |
| F4 server scoring | The citizen; the caller | Consent by ticking a box (transient, KZ server) | Acceptable once there is a privacy notice. The server must be in KZ even though nothing is stored **[I]**. |
| F4b / F13 cloud tier | The citizen; **the caller** | Art. 16(3)(1) consent is possible for the citizen's own data only | **Weakest point.** The citizen cannot consent for the caller's data going abroad, and Art. 16(3)(4) is untested **[I]**. Close the ungated paths now (§6 M2). Before offering the tier to the public, choose one: (a) consent per request (§7.2), scrubbing before sending, Google *Paid Services* plus Google's DPA, and an advocate's opinion on the caller's data; or (b) a model hosted in KZ ("local AI system", AI Law Art. 17(4)). |
| F5 live session routes | Both parties | None | Turn these off in any deployment. It is a server-side live-transcript store and a second ingress (§6 M2). |
| F6 citizen report: the citizen's own data | The citizen (possibly their name, voice or IIN in the text) | Consent (Art. 7(1), 8(1) "another method allowing confirmation"). Proof of consent = consent-text version + timestamp stored per report. | Mostly fine. **Gaps:** the text in §7.1 must list the Art. 8(4) items. The report is anonymous, so the subject's full name (8(4)(2)) cannot be captured; the advocate must confirm that anonymous consent is valid when no identifiers of the citizen are collected **[U]**. |
| F6 citizen report: **the caller's** data | The caller (speech; number as a digest) | Art. 9(5) (protection of the citizen's constitutional rights, including property, where the caller's consent is impossible) **[I]** | **The open item in DATA_INTAKE §9.** The basis is plausible for scammers. It fails for *legitimate* callers wrongly flagged: their speech and number sit in a "scam" store, which cuts against Art. 7(8), second paragraph (no processing that causes harm). **Mitigations:** make the transcript optional and default to tactics plus flagged phrases; require analyst review before any external use; honour "I am the caller, delete my data" requests. The advocate decides whether Art. 9(5) holds. |
| F7–F8 derived data | Both | Follows F6 | F8 breaks deletion: digests outlive the report (§6 M4). |
| F9 partner reports | The partner's customers and callers | The partner's basis plus a data-processing agreement (DPA) | The partner transfers to a third party (Qorğan), which needs the subject's consent or a law (Art. 7(6)). For banks, **bank secrecy** also applies (new Constitution Art. 21(2) protects "тайна банковских операций"; the article in the Law on Banks is **[U]**). `consent_basis` should be checked against the codes each partner's DPA allows. |
| F10 aggregates | None | n/a | Outside the PD Law **[I]**. Keep it that way: no digests, no excerpts. |
| F11–F12 logs | Analysts and partners (employee identity); citizens (IP) | Security logging, which is a duty under Art. 22 | Set a retention period, host in KZ, and drop or truncate IPs in access logs. |
| F14 training data | Synthetic; real calls later | Synthetic data needs no basis; for real calls see DATA_INTAKE | Republish `ood.jsonl` (D45). **Never** run real calls through Gemini or Claude for labeling (D38 names this as the next experiment), because it is a cross-border transfer of third parties' data. |

### 2.3 Is the HMAC'd phone number still personal data? Yes: treat it as pseudonymised PD **[I]**

- The digest's *purpose* is to link reports about the same number. That fits Art. 1(14-1)
  ("link separate data sets about the subject").
- Mobile numbers are registered to a person. The Anti-Fraud Center gets a subscriber's full
  name and IIN by number from operators (Communications Law Art. 15-4(1) **[S]**).
- The KZ number space is small. Whoever holds `QORGAN_NUMBER_HMAC_KEY` can hash every
  plausible number and reverse any digest in minutes. The digest is therefore *not*
  "anonymised" under Art. 1(2-5).
- The statute frames hashing as a protection measure (Art. 23(2)), not as an exit from the law.

Consequences:
- Key custody must be separate from analysts: a secret store, with no analyst access.
- There must be no "look up a number" feature.
- Digests fall under retention and deletion like any other personal data.
- Only F10 (no digests) is outside the law.

Art. 6's closed-looking identifier list (name, IIN, face) might support a narrower reading.
We do not rely on it.

### 2.4 Localization, notification and deployment-dependent duties

- **Hosting.** Everything in F6–F12 must be hosted in KZ (Art. 12(2)). HF and GitHub may hold
  only synthetic data and code.
- **Size class.** Call content is very likely *restricted-access* data (private life and the
  secrecy of communications) **[I]**, which moves Qorğan up one class.
  - At ≤ 10,000 unique subjects (citizens plus callers) Qorğan is "medium" and needs no
    notification.
  - Above 10,000 it is "large", so **notification under Art. 10-1 is required** before
    processing starts.
- **Security measures.** News reports on the 2026 orders of the Ministry of AI and Digital
  Development describe per-class measures: MFA and role-based access for medium operators,
  independent audit and penetration tests for large ones, and servers in KZ (zakon.kz,
  30.06.2026 and 14.09.2026). **[U]** The order texts were not read.
- **Deployment by a state body.** If a state body operates Level 2, its store becomes a state
  digital object. Consent may then have to go through the **state consent service**
  (Art. 8(1), second paragraph) instead of a checkbox **[I]**, and the Law on Languages
  applies (§4.4).

## 3. Secrecy of call content

**[S] What the statutes say**

- *New Constitution (referendum 15.03.2026, in force 01.07.2026):* Art. 21(1) guarantees
  privacy and protection of personal data "including with the use of digital technologies".
  Art. 21(2) protects the secrecy of telephone conversations "transmitted by means of
  communication, including with the use of digital technologies", and allows limits only by
  law. *The 1995 Constitution (old Art. 18) lost force on 2026-07-01.*
- *Law "On Communications" Art. 36(2):* **operators** keep the secrecy of telephone
  conversations sent over their networks.
- *Criminal Code Art. 148:* criminalises the "unlawful violation of the secrecy of … telephone
  conversations" of individuals. Part 2 (up to 5 years) covers acts done with "special
  technical means intended for covert obtaining of information", unlawful interception, "or
  with the use of digital technologies". It was amended by 307-VIII from 01.07.2026; which
  words were added is **[U]**.
- *Criminal Code Art. 147(2):* criminalises unlawful collection of private-life information
  without consent.
- *Criminal Procedure Code Art. 231–232, 243:* covert monitoring and interception of
  telecommunications are **covert investigative actions**. They need the sanction of an
  **investigating judge** (Art. 232(3)). When a person's life, health or property is
  threatened, that person may give **written consent**, and the investigation body may then
  act by its own decision, notifying the prosecutor within 24 h (Art. 232(6)). This is the
  lawful path for "a victim consents to interception", and it belongs to the state.

**[I] Why the Level 1 design is different from interception**

1. **The user is a party to the call**, not a third party. The secrecy right protects a
   communication *from outsiders*. The user already hears every word on speakerphone, and
   on-device transcription is closer to taking notes than to interception.
2. **Nothing reaches a third party in real time.** No server, operator or state body receives
   audio or text during the call (invariants 1–2, ADR D12). A third party receiving the
   content is what turns "listening" into "interception".
3. **The user starts it, sees it and ends it for each call.** There is no background capture
   and no operator hook, and nothing is stored after the call.
4. **Only a deliberate, reviewed, after-call report leaves the device.** That report is a
   citizen's own disclosure (§2.2 F6), not a tap.

KZ has no statute that settles one-party recording. Practitioner commentary calls it "not
categorically prohibited", advises notifying the other party, and notes that courts disagree on
whether such recordings are admissible (yuristy.kz; zakon.kz 2015) **[U]**. Qorğan does not
*record*. Keep it that way.

**Never build these (each one loses at least one of points 1–4 above):**

| Do not build | Why |
|---|---|
| Operator-level or network capture (SIP/SS7 taps, an operator API that streams calls) | Covert interception of telecommunications is reserved to authorized bodies with judicial sanction (CPC Art. 232–243). For anyone else it is Criminal Code Art. 148(2). |
| Background or automatic listening; auto-start on incoming calls without a per-call user action; Accessibility-API call capture | The user stops being a knowing party, so it becomes "covert obtaining". Google Play also bans Accessibility call recording. |
| Storing call audio, anywhere | Audio is not needed for the purpose (Art. 7(9)). Voice is biometric-capable (Digital Code Art. 48; D7). A store of audio becomes a wiretap archive. |
| Streaming text or audio to a server during the call (including the old `/api/live/session` pattern) | A third party receiving content in real time is the council's deal-breaker. It is indistinguishable from interception. |
| Installing on, or capturing, *someone else's* phone or calls | Criminal Code Art. 148(2) ("special technical means … digital technologies") and Art. 147. |
| Voice or emotion analysis of either party | AI Law Art. 17(3)(6) (see §4.3). |

**Should:** add a short, optional prompt on the live page along the lines of "you may tell the
caller the call is being checked by an assistant". It costs nothing with a legitimate caller and
deters a scammer.

## 4. AI Law (No. 230-VIII, signed 17.11.2025, in force 18.01.2026; amended by 326-VIII)

### 4.1 Classification

- **[S] Risk levels (Art. 17(1)).** Minimal: failure has minimal impact. Medium: failure may
  reduce users' effectiveness or cause moral or material damage. High: failure may cause an
  emergency or significant negative consequences for defence, security, the economy, users,
  infrastructure or life.
- **[S] Who classifies.** The owner or possessor classifies the system under "rules for
  classifying digital objects" (rules text **[U]**).
- **[S] Autonomy levels (Art. 17(2)).** *Low autonomy* means the system gives recommendations
  and "the final choice and actions are always made by a human".

**[I] What this means for Qorğan**

- **Autonomy.** Both levels are low-autonomy (no auto-block, report or hang-up). This must stay
  a product invariant: it is the legal classification.
- **Level 1 risk.** Probably **medium**. A missed scam can cause material damage, and a false
  alarm can make someone hang up on a real bank.
- **Level 2 risk.** At a state body, or feeding the Anti-Fraud Center, it is plausibly
  **high**: its output can lead to number suspension (Communications Law Art. 15-4(3)) and
  payment blocks (Payments Law Art. 25-1(3-1)). If it is also designated a critically
  important digital object, state cybersecurity requirements apply (Art. 17(1), last
  paragraph).
- **Paperwork.** Write down the classification and its reasons now. It costs one page.

### 4.2 Transparency and user rights

- **[S]** Users must be told that a service is provided using AI (Art. 21(1)).
- **[S]** Users must get full information about the system's characteristics and limits
  (Art. 7(1)).
- **[S]** A person affected by an AI-assisted decision may be informed of the automated
  processing, may object, and has remedies (Art. 7(2)).
- **[S]** The owner must let users read a **user agreement before use** (Art. 15(2)(5)) and
  keep documentation by risk level (Art. 15(2)(3)).
- **[S]** Users may ask for explanations and for the data a result was based on, and may refuse
  to interact with the system (Art. 16(1)(4)–(6)).

**What exists today:**
- Templated grounded reasons, verbatim spans, the caveat, and "a human always decides". This is
  a good fit for Art. 7 and 16(1)(4)–(5).

**What is missing:**
- A **user agreement** (Пользовательское соглашение).
- An explicit **"this verdict is produced by an AI system and can be wrong"** line on the live
  page and in the report flow.
- The model's limitations in plain words: `shift` recall is 0.364, weakest in Kazakh. That
  honesty belongs to Art. 7(1).

### 4.3 Prohibitions (Art. 17(3)) and whether the framing holds

- **(6) Emotion detection without consent.** The framing holds. The `fear_threat` and
  `urgency` tactics are defined as the **caller's script content** ("Threats … to induce
  panic", `data/taxonomy/tactics.yaml`). No acoustic, prosody or emotion features exist
  anywhere in `src/` or `site/`. To keep it that way:
  - labels must describe what the caller *says*, never how anyone *feels*;
  - no voice features, ever;
  - the "emotional pressure" wording in `task.md` should read "pressure tactics in the caller's
    words".
- **(3) Scoring or classifying people over time by social behaviour.** Level 2 ranks
  "organizations" (clusters of numbers and scripts) by priority. **[I]** This is
  fraud-signal analysis of *calls*, not scoring of persons. It stays defensible only if:
  - there are no person-level profiles;
  - nothing is attached to identified individuals;
  - a human reviews every output;
  - analysts cannot look up a number.
- **(4) Personal-data processing that breaks the PD Law** is itself a prohibited AI function.
  Every PD gap in §6 is therefore also an AI Law gap.
- **(1) Manipulation.** Advice must stay transparent: no dark patterns, and no fear copy
  beyond the evidence.

### 4.4 Human oversight, audit and deployment by a state body

- **[S]** Continuous risk management across the whole lifecycle, updated at least once a year
  (Art. 18).
- **[S]** A system that becomes prohibited must be suspended (Art. 18(2)).
- **[S]** Audit is required to be included in a sector's **list of trusted high-risk
  systems**. The audit assesses the quality and *lawfulness* of the training data and the
  absence of prohibited functions (Art. 19–20).

**[I] What this means for Qorğan**
- `data/README.md` provenance, the inspection ledger and the eval report are real assets for
  such an audit.
- The Gemini and Claude terms for synthetic data must be checked. The Gemini API terms forbid
  using the service "to develop models that compete with" Gemini; a scam classifier arguably
  does not **[I]**.

**Language (Law "On Languages" Art. 22, as amended by 256-VIII) [S].** Digital objects of state
and quasi-state bodies that serve state functions must be released in **Kazakh and Russian**.
Replies to citizens go in the state language or the language of the request (Art. 11). The new
Constitution's Art. 9 has Kazakh as the state language, with Russian used officially alongside
it. The citizen page, the consent texts and the analyst console must therefore be KK plus RU
before any state deployment. The page chrome is English-only today (D44, "Not done").

## 5. Level 2

### 5.1 Purpose limitation, retention and rights

- **Purpose** (Art. 7(8), Art. 14 **[S]**): "detect scam schemes from consented reports and
  warn the public and partners with aggregates". Write this into the privacy notice and a
  one-page **Level-2 charter**. Every access purpose code must map to it.
- **Retention:** 180 days (`QORGAN_REPORT_RETENTION_DAYS`) is defensible if it is tied to the
  scheme-detection window (Art. 12(2)). **Code bugs:**
  - Retention is keyed on the client `timestamp`, which the citizen route does not bound.
    A future date means the report never expires; a past date means it is purged at once.
  - The purge is a CLI command that nothing schedules.
  - Feedback digests (F8) and the audit log have no retention.
- **Right of access (Art. 24(1)(1), 25(2)(9))** is only partly met. The receipt shows what was
  stored at the moment of submission, but there is no `GET /api/reports/{receipt}`, and the
  receipt is not saved anywhere. Once the page closes, the citizen can neither see nor
  **delete** their report.
- **Deletion and withdrawal (Art. 8(7), 18)** are immediate and cascade (`forget_report`),
  apart from F8.
- **Objection to automated processing (Art. 19-1)** does not arise in Level 1, which has no
  legal effect. It *does* arise the moment a partner acts on a Qorğan signal: the DPA must
  require human review, or the partner's own Art. 19-1 process.

### 5.2 What the analyst-auth, roles, purpose-code and audit-chain work (in progress) must satisfy

The current design is sound:
- per-analyst credentials, and the identity comes only from the credential;
- the roles `analyst` and `investigator`;
- the console fails closed without keys;
- purpose codes `pattern_review` / `citizen_request` / `partner_request`;
- an HMAC-chained audit log with external anchoring.

It still has to satisfy the following.

1. **Authentication strength.** Shared-secret API keys are fine for a demo. A pilot needs SSO
   or MFA, reportedly required from the "medium" class upward **[U]**, and key rotation.
2. **Separation of duties.** Whoever holds `QORGAN_AUDIT_CHAIN_KEY`, `QORGAN_NUMBER_HMAC_KEY`
   or server access must not hold the `investigator` role. Audit review must be a separate role
   or person: the "person responsible for processing" of Art. 25(2)(10).
3. **Purpose codes.**
   - `partner_request` must **not** lead to showing transcript content to a partner unless the
     consent text (§7.1) says so. That would be a transfer to a third party (Art. 7(6), 15);
     today's text promises "aggregates only".
   - Add a code for law-enforcement requests. It must carry a content-free reference to the
     formal request (a request under the Criminal Procedure Code); otherwise such requests
     will be logged as something else.
   - `citizen_request` should require the receipt as proof.
4. **Audit coverage.** Log content-free lines for:
   - content reads by the `analyst` role (org detail with excerpts, `/analysis`);
   - purge runs;
   - deletions (citizen and partner);
   - configuration and key changes;
   - every cross-border call (Art. 22(1)(5) asks for transfers to be logged).
5. **Audit integrity.** Anchor the head outside the server's trust domain, as `audit.py` itself
   says. Keep the log in KZ, set a retention period (for example 3 years, which the legal owner
   decides), and make the verifier's output shareable with the authorized body on request
   (Art. 25(2)(3-1)).
6. **Model backend.** The `backend` query parameter on `/analysis` and `/open` must be removed
   or restricted to local backends. Otherwise an investigator can send a citizen's report
   abroad with one URL parameter.

### 5.3 Answering the council's "surveillance infrastructure" objection

**What already makes Level 2 defensible *by construction*:**
- no audio;
- no automatic ingress;
- one reviewed and consented citizen ingress;
- numbers hashed;
- scrubbed text;
- partners get aggregates only;
- deletion by receipt;
- a human decides.

**Point 1: the lexicon cannot widen collection.** The council feared a surveillance switch "one
lexicon line away". The lexicon cannot widen collection, because collection happens only when
a citizen chooses to report. Say this explicitly in the pitch.

**Point 2: what is still needed.** Collection must be impossible to widen *remotely*:
- no server-controlled auto-report threshold;
- no remote flag that turns on capture;
- a signed, public device bundle.

**Point 3: publish a transparency report.** Per quarter, publish counts of:
- reports received;
- reports deleted;
- cases opened, by purpose;
- law-enforcement requests;
- partner exports.

**Point 4: the key and the lookup feature.** Keep the number key out of analysts' reach, and
never build a "check this number" lookup.

**Point 5: where signals go.** Route signals to the **existing legal channel**:
- *Payments Law Art. 25-1* **[S]**: the Anti-Fraud Center (NBK; operated by the National
  Payment Corporation) takes fraud events from financial organizations, mobile operators and
  criminal-prosecution bodies. Its participants are listed in the Law, and others can join
  "by decision of the National Bank" (25-1(3)(9)).
- *NBK Resolution No. 54 of 25.08.2025* **[S]**: under its Rules the Center processes personal
  data "in depersonalized form" (п. 4(5)), and "compromised subscriber number" is a defined
  term there.
- Qorğan is not a participant. The lawful paths are **through a participating bank or
  operator** under a DPA, or **designation by the NBK**. Qorğan must not become a parallel
  number-blacklist.

## 6. Gap list (prioritized)

`M` = must before pilot · `S` = should · `N` = nice. Owner: C = code, L = legal, B = business.

| ID | Change | Tag | Owner | Basis |
|---|---|---|---|---|
| M1 | Advocate opinion on the legal basis for the **caller's** data in reports and the cloud tier (Art. 9(5), 16(3)(4)); decide whether reports default to "tactics + flagged phrases, transcript optional" | M | L | §2.2 |
| M2 | Close the unintended paths. (a) Remove the client/admin-controlled `backend` parameter, or allow only `linear`/`mock` on public and admin routes. (b) Disable `/api/live/session/*` in deployments (or delete it; it remains the Streamlit harness's concern). (c) Stop the `llm` cache from writing verbatim phrases to disk (hash-only, or no cache in serving). (d) Add an architecture test for each. | M | C | F4b, F5, F13 |
| M3 | **Privacy notice** plus **user agreement** (AI Law Art. 15(2)(5)) in KK and RU; the §7 consent texts in the report flow; store `consent_text_version` with each report as proof (Art. 25(2)(5)); an "AI produced this verdict" line (Art. 21(1)) | M | L+C | §2, §4 |
| M4 | Retention and deletion correctness. Purge by `received_at` and bound the citizen `timestamp` like the partner route does. Schedule the purge. Extend `forget_report` and the purge to `org_feedback.jsonl` digests. Set retention for the audit and access logs. | M | C | Art. 12(2), 18 |
| M5 | KZ hosting (Art. 12(2)) with TLS. Put the HMAC and audit keys in a secret store with no analyst access. Configure access logs without IPs or with short retention. Put a named person responsible for processing (Art. 25(2)(10)) and a written **breach procedure** (one working day, Art. 25(2)(8)) in place. | M | B+C+L | §2.4 |
| M6 | Determine the Art. 25-1 size class, with the restricted-data bump. **Notify the authorized body under Art. 10-1** if it lands in "large" (> 10,000 subjects) | M | L | §2.4 |
| M7 | Scrubbing: redact numbers spoken as words (RU/KK), KZ IBAN, and cards and IINs split by spaces or words; keep Python/JS parity (D45 pattern). Until then the report preview must warn "check for spoken numbers". | M | C | F6, Art. 7(9) |
| M8 | Cloud tier: until M1 is answered, no citizen-facing access. When it ships: per-request cross-border consent (§7.2), scrub before sending, Gemini **Paid Services** with Google's DPA, a content-free audit line per call. Otherwise a KZ-hosted model. | M | L+C+B | Art. 16, 7(6) |
| M9 | Analyst console per §5.2 (MFA/SSO, separation of duties, a law-enforcement purpose code, `partner_request` semantics, audit coverage) | M | C+L | §5.2 |
| M10 | Partner **DPA template**: roles (the partner as owner and operator of its customers' data, Qorğan as third party); the allowed `consent_basis` codes (validated in code for each partner); bank-secrecy warranty; no automated adverse decisions on Qorğan signals (Art. 19-1); deletion and return; KZ storage; breach notice | M | L+C | F9, §5.1 |
| M11 | Republish HF `ood.jsonl` without the IIN (D45); add a LICENSE plus HF cards that state "synthetic, no real persons" | M | B | F14 |
| S1 | Self-host fonts and transformers.js (removes foreign requests; keeps the on-device claim true) | S | C | F3 |
| S2 | `GET /api/reports/{receipt}` (what is stored, status, purge date) plus an optional local save or print of the receipt | S | C | Art. 24(1)(1) |
| S3 | AI Law pack: risk classification memo, risk register with yearly review (Art. 18), documentation list (Art. 15(2)(3)), limitations statement including `shift` recall | S | L+C | §4 |
| S4 | A "Delete my data" contact for callers who say they were wrongly reported | S | L+B | §2.2 |
| S5 | Optional "tell the caller" hint on the live page; KK/RU chrome (the localization agent) | S | C | §3, §4.4 |
| S6 | Level-2 charter plus quarterly transparency report; state in the pitch that the lexicon cannot widen collection | S | B | §5.3 |
| S7 | Check that the synthetic-data generation complies with the Gemini and Anthropic terms; record the result in `data/README.md` | S | L | §4.4 |
| N1 | Anti-Fraud Center path: talk to a participating bank; explore NBK designation (Payments Law 25-1(3)(9)); privacy-preserving matching of digests with a partner (shared-key DPA) | N | B | §5.3 |
| N2 | Data-transfer register (Art. 22(1)(5)) generated from the audit log | N | C | §2.1 |

## 7. Draft user-facing texts

Placeholders in `[…]` must be filled in before use. **The KK drafts need a native
legal-translation review.** The terms follow the KK text of the PD Law as closely as we could:
- "дербес деректер" (personal data);
- "иесіздендіру" (depersonalization);
- "трансшекаралық беру" (cross-border transfer).

### 7.1 Consent to send a report

**RU**
> **Отправка сообщения о мошенничестве.** Нажимая «Отправить», я соглашаюсь, что
> [оператор, БИН] получит и будет обрабатывать этот отчёт: отредактированный мной текст
> разговора (номера телефонов и карт, ИИН и e-mail скрываются автоматически, но
> **произнесённые словами номера — нет, проверьте текст**); признаки мошенничества и оценку
> риска, найденные системой искусственного интеллекта; номер звонившего, если я его указал(а)
> — хранится только как защищённый хеш и первые цифры (+7 700 ***).
> **Зачем:** выявлять мошеннические схемы и предупреждать людей. Отчёт не используется, чтобы
> установить мою личность. **Кто увидит:** уполномоченные аналитики [оператора]; каждый
> просмотр полного текста записывается в журнал. Партнёры (банки, операторы связи) получают
> только обезличенную статистику — без текста и номеров. **Где и сколько:** на серверах в
> Казахстане, не дольше 180 дней; за границу не передаётся.
> **Мои права:** удалить отчёт в любой момент по коду квитанции (сохраните его), узнать, что
> хранится, отозвать согласие. **Важно:** в тексте есть слова другого человека — отправляйте,
> только если считаете звонок мошенническим, и удалите лишние личные сведения. Оценку сделала
> система ИИ, она может ошибаться; решение принимаю я. [Политика конфиденциальности] ·
> версия согласия [v1]
> ☐ Я проверил(а) отчёт и согласен(на) его отправить.

**KK**
> **Алаяқтық туралы хабарлама жіберу.** «Жіберу» түймесін басу арқылы мен [оператор, БСН]
> осы есепті алып, өңдеуіне келісемін: мен өңдеген әңгіме мәтіні (телефон және карта
> нөмірлері, ЖСН және e-mail автоматты түрде жасырылады, бірақ **сөзбен айтылған нөмірлер
> жасырылмайды — мәтінді тексеріңіз**); жасанды интеллект жүйесі тапқан алаяқтық белгілері
> мен тәуекел бағасы; егер көрсетсем, қоңырау шалушының нөмірі — тек қорғалған хеш және
> алғашқы цифрлар (+7 700 ***) түрінде сақталады.
> **Мақсаты:** алаяқтық схемаларды анықтау және адамдарды ескерту. Есеп менің жеке басымды
> анықтау үшін пайдаланылмайды. **Кім көреді:** [оператордың] уәкілетті талдаушылары; толық
> мәтінді әрбір ашу журналға жазылады. Серіктестер (банктер, байланыс операторлары) тек
> иесіздендірілген статистиканы алады — мәтінсіз және нөмірсіз. **Қайда және қанша уақыт:**
> Қазақстандағы серверлерде, 180 күннен аспайды; шетелге берілмейді.
> **Менің құқықтарым:** түбіртек коды арқылы есепті кез келген уақытта жою (оны сақтап
> қойыңыз), не сақталғанын білу, келісімді кері қайтарып алу. **Маңызды:** мәтінде басқа
> адамның сөздері бар — қоңырауды алаяқтық деп санасаңыз ғана жіберіңіз және артық жеке
> мәліметтерді өшіріңіз. Бағаны ЖИ жүйесі берді, ол қателесуі мүмкін; шешімді мен қабылдаймын.
> [Құпиялылық саясаты] · келісім нұсқасы [v1]
> ☐ Есепті тексердім және оны жіберуге келісемін.

### 7.2 Cross-border notice for the cloud second opinion (shown on every request, never remembered)

**RU**
> **Второе мнение в облаке (необязательно).** Проверка на вашем устройстве уже выполнена.
> Если нажать «Отправить на проверку», этот текст будет передан компании Google (сервис
> Gemini) и обработан на серверах **за пределами Казахстана** (в том числе в США) — это
> трансграничная передача персональных данных. Передаётся только этот текст и только один
> раз; [оператор] его не сохраняет. Google может временно хранить запрос для защиты от
> злоупотреблений и не использует его для обучения моделей. В тексте могут быть слова и
> сведения другого человека — перед отправкой удалите имена, адреса и номера. Результат —
> мнение системы ИИ, а не решение.
> ☐ Я согласен(на) на однократную передачу этого текста за пределы Казахстана.
> [Отправить на проверку] [Отмена]

**KK**
> **Бұлттағы екінші пікір (міндетті емес).** Құрылғыңыздағы тексеру аяқталды. «Тексеруге
> жіберу» түймесін бассаңыз, бұл мәтін Google компаниясына (Gemini қызметі) беріліп,
> **Қазақстаннан тыс жерлердегі** (соның ішінде АҚШ-тағы) серверлерде өңделеді — бұл дербес
> деректерді трансшекаралық беру. Тек осы мәтін және тек бір рет беріледі; [оператор] оны
> сақтамайды. Google сұранысты теріс пайдаланудан қорғау үшін уақытша сақтауы мүмкін және
> модельдерді оқытуға пайдаланбайды. Мәтінде басқа адамның сөздері мен мәліметтері болуы
> мүмкін — жібермес бұрын аты-жөндерді, мекенжайларды және нөмірлерді өшіріңіз. Нәтиже — ЖИ
> жүйесінің пікірі, шешім емес.
> ☐ Осы мәтінді Қазақстаннан тыс жерге бір рет беруге келісемін.
> [Тексеруге жіберу] [Болдырмау]

*Both notices describe Google's **Paid Services** terms (no training on prompts; logs kept for
a limited time to detect abuse). On the unpaid tier Google may have humans review the data and
may use it to improve its products, so these texts would be false there. Do not ship this
notice on an unpaid key.*

## 8. Sources

All accessed 2026-09-25. Adilet statute texts were read through the adilet mirror
`old.adilet.zan.kz`. The consolidated texts are "as of 24.09.2026". The canonical URLs are
below.

**Statutes and regulations**
- PD Law No. 94-V (Art. 1, 3, 6–10-1, 12, 16–19-1, 22–25-1): https://adilet.zan.kz/rus/docs/Z1300000094
- AI Law No. 230-VIII (Art. 7–11, 15–22, 31): https://adilet.zan.kz/rus/docs/Z2500000230 · EN: https://adilet.zan.kz/eng/docs/Z2500000230
- Law No. 231-VIII (AI and digitalization amendments; published 18.11.2025): https://adilet.zan.kz/rus/docs/Z2500000231
- Law No. 256-VIII (09.01.2026; published 10.01.2026): https://adilet.zan.kz/rus/docs/Z2600000256
- Law No. 326-VIII (24.06.2026; published 25.06.2026): https://adilet.zan.kz/rus/docs/Z2600000326
- Digital Code No. 255-VIII (Art. 48, 106): https://adilet.zan.kz/rus/docs/K2600000255
- Constitution adopted 15.03.2026, in force 01.07.2026 (Art. 9, 21): https://adilet.zan.kz/rus/docs/K2600000000
- The 1995 Constitution (lost force 01.07.2026): https://adilet.zan.kz/rus/docs/K950001000_
- Criminal Code (Art. 147, 148, 190, 232-1 [added by 210-VIII of 16.07.2025]): https://adilet.zan.kz/rus/docs/K1400000226
- Criminal Procedure Code (Art. 231, 232, 243): https://adilet.zan.kz/rus/docs/K1400000231
- Law "On Communications" (Art. 36; Art. 15-4 [added by 205-VIII of 30.06.2025]): https://adilet.zan.kz/rus/docs/Z040000567_
- Law "On Payments and Payment Systems" (Art. 25-1): https://adilet.zan.kz/rus/docs/Z1600000011
- NBK Board Resolution No. 54 of 25.08.2025 (MoJ reg. 29.08.2025 No. 36742; amended by No. 58 of 10.06.2026): https://adilet.zan.kz/rus/docs/V2500036742
- Law "On Languages" (Art. 11, 22): https://adilet.zan.kz/rus/docs/Z970000151_
- Rules on personal-data protection measures (the base order; the 2026 edits were not read): https://adilet.zan.kz/rus/docs/V2300032810

**Secondary sources (news and law-firm briefings)**
- NBK news, Anti-Fraud Center, 04.06.2024 (full launch planned for 22.07.2024): https://nationalbank.kz/ru/news/informacionnye-soobshcheniya/16782
- National Payment Corporation, Anti-Fraud Center: https://npck.kz/en/anti-fraud-center/
- zakon.kz on the 2026 PD-protection rule updates: https://www.zakon.kz/pravo/6523127-obnovleny-pravila-zashchity-personalnykh-dannykh.html · https://www.zakon.kz/pravo/6531379-pravila-zashchity-personalnykh-dannykh-obnovili-v-kazakhstane.html
- EY alert on the AI Law: https://www.ey.com/en_kz/technical/tax-alerts/2025/12/law-on-artificial-intelligence-kazakhstan · Library of Congress: https://www.loc.gov/item/global-legal-monitor/2026-01-12/kazakhstan-new-law-introduces-rules-for-ai-systems-operating-in-the-country
- On recording calls (practitioner commentary): https://yuristy.kz/news/polesnoe/zakonno-li-zapisyvat-telefonnyy-razgovor-2381

**Provider terms and licenses**
- Gemini API Additional Terms (paid vs unpaid data use; the competing-models clause): https://ai.google.dev/gemini-api/terms
- Vosklet license (MIT on `main`; check the 1.2.1 tag): https://github.com/msqr1/Vosklet/blob/main/LICENSE

**Could not verify (do not cite as fact)**
- The text of the June and September 2026 protection-rule orders.
- The "rules for classifying digital objects".
- Whether the US or other countries count as "ensuring protection" under Art. 16(2).
- The bank-secrecy article in the Law on Banks.
- The "250+ organizations" count at the Anti-Fraud Center.
- The claim that Law 231-VIII allows special-category data in the cloud only when a KZ entity
  holds the encryption keys. We did not find it in the texts of Law 94-V, 231-VIII or the
  Digital Code.
- A rule on one-party recording.
- The exact dates when 326-VIII and the Digital Code came into force (both computed from
  their publication dates).
