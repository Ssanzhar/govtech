# Dataset gap-check for Qorğan — what the 2026-09-26 survey missed

Date: 2026-10-07. Method: HF API (`/api/datasets?search=`, `filter=language:kk` → 632 kk datasets, datasets-server `first-rows`), Zenodo API, Kaggle Croissant JSON, GitHub API, arXiv abs/html pages, YouTube watch-page `captionTracks`, direct curl of KZ/RU portals. "Verified" = I fetched the artefact named. Items already in the brief are skipped.

## A. VERIFIED — real scam-call transcripts (none in RU/KK; all English/other)

**A1. Kaggle `rivalcults/youtube-scam-phone-call-transcripts`** — 243 English transcripts of the *beginning* of scam calls from YouTube scam-baiting videos + some auto-dialler messages; "manually verified and amended, based on YouTube auto-transcripts"; names/addresses/numbers removed; source URLs given. Licence: **"CC0: Public Domain"** (Croissant JSON). Verified: Croissant metadata (146 KB zip, `FullTranscriptData.csv`, fields ID/Source/Content/Char_Len, published 2024-08-05); download needs a free Kaggle login (302 unauthenticated). Use: EN eval only (openers → good for the "arms from 3rd utterance" meter check). Risk: scam-baiter register, tech-support/refund heavy, US. Confidence: high.

**A2. Kaggle `mealss/call-transcripts-scam-determinations`** — 60 English calls, scam + legitimate, turn-level rows (`CONVERSATION_ID, CONVERSATION_STEP, TEXT, CONTEXT, LABEL, FEATURES, ANNOTATIONS`), **CC0**. Verified via Croissant (43.7 KB, 2024-02). Provenance of the transcripts is not stated (likely hand-written) — treat as authored, not real. Use: tiny EN sanity set. Confidence: medium.

**A3. scam.ai voice-agent honeypot corpus — arXiv 2609.29528 (descriptor) + 2608.24127 (analysis)**, submitted 2026-08-25. 10,015 real inbound US calls to an LLM "elderly widow" persona, 328,869 transcribed turns, labels scam 949 / spam 3,949 / legitimate 380 / too brief 1,096. Paper text: "The public release is distributed under a **CC BY-NC** license with a datasheet; the full corpus and audio are available under a data-use agreement" — public subset = **1,000 de-identified transcripts, no audio**; "A persistent identifier will be assigned on publication". Verified: arXiv HTML; corresponding author Simiao Ren, benren@scam.ai. **No HF/GitHub/Zenodo record exists yet** (searched all three). Use: the best future EN independent-generator eval (real scammers, real legit/spam negatives). Risk: not yet downloadable; NC; US register. Confidence: high that it exists, low that it is obtainable this month → email the author.

**A4. Korean FSS "보이스피싱 지킴이 / 그놈 목소리"** — https://www.fss.or.kr/fss/main/sub1voice.do?menuNo=200012 (HTTP 200, verified). The Korean financial regulator publishes real recorded vishing calls (audio, Korean) as a public-awareness board; KorCCVi was transcribed from it. HF `HyaDoo/ko-voicephishing-binary-classification` (8,725 rows, ASR-style Korean text, label 1 = vishing, uploader-declared apache-2.0) is almost certainly this material re-labelled — provenance/licence washed, do not redistribute. Use: translate-adapt *scripts* only (bank-security, "safe account", police impersonation — same families as KZ). Risk: licence unclear, Korean. Confidence: medium.

**A5. Japanese prefectural police real "アポ電" audio** — Nagano (pref.nagano.lg.jp/police/jikenjiko/tokushu/onsei.html), Ibaraki (`furikome_gallery`), Osaka, Nara post real recorded scam calls on official pages/YouTube. Audio only, Japanese, no licence beyond government site terms. Use: script reference only. Confidence: medium (verified via news + page URLs, not fetched).

## B. VERIFIED — real scam *phrasing* sources for RU/KK (text, not datasets)

**B1. Criminal judgments, Russia — sudact.ru** (HTTP 200; doc `sudact.ru/regular/doc/qwB6WXlsxpv/` = Butyrsky district court Moscow, 2025-11-27, ч.3 ст.159 УК РФ) reproduces the script in reported speech: "представился сотрудником ГБУ «Жилищник»… сообщил, что… происходит замена дверных замков"; "представившись сотрудником ФСБ… сообщил, что для проведения оперативно-розыскных мероприятий… необходимо следовать инструкциям"; "собрать все имеющиеся дома денежные средства и передать курьеру". Bulk tooling exists: **github.com/dataout-org/sudrfparser** (scrapes first-instance criminal cases from 2,311 sudrf.ru court sites; updated 2026-03; licence NOASSERTION) and `courtandrey/SUDRFScraper`. Judgments are public by law (ФЗ-262), names already masked ("фио"). Use: real multi-stage scripts (pretext → escalation → cash-out) to *author* KK/RU test calls and to audit tactic taxonomy; **not** for lexicon extension without paired training data (ADR D35/D43). Risk: reported speech, Russian register (ФСБ, Госуслуги, ЦБ) ≠ KZ; scraping sudact.ru itself violates its ToS — use sudrf.ru via sudrfparser. Confidence: high.

**B2. Criminal judgments, Kazakhstan — "Сот кабинеті" court-acts search** https://office.sud.kz/courtActs/index.xhtml (HTTP 200; form verified with "Негізгі сөз бойынша іздеу" keyword search, case category, region, court, instance; KK/RU interface). Verdicts under ст.190 УК РК are published in Russian *and Kazakh* and describe the call in reported speech (press on such cases: inbusiness.kz "мошенник обещал помочь с делом о наркотиках", kazpravda "мужчина заплатил мошенникам, чтобы избежать ареста"). No API / bulk download (sud.gov.kz open-data list exists but has no judgments feed; data.egov.kz API returns 403 without a key). Use: the only *Kazakh-language* source of real scam scripts found; manual harvest of ~50–100 verdicts is feasible. Risk: JSF portal (session-based, manual), PII of defendants, reported speech. Confidence: high (portal), medium (yield).

**B3. KZ press with verbatim lines** — finratings.kz 2025-10-23 "Я менеджер Kaspi" (RU, two verbatim scammer lines: "Вам сейчас на телефон придет разовый номер, продиктуйте его, пожалуйста", "Неправильный номер. Посмотрите, может, ошиблись цифрой"; Instagram video exists); nur.kz KK article 2020-12-20 (reported speech only: "операция жасап жатқан шынымен де сіз бе деп сұрады"); 24.kz 2026-09-28 "Сәлемдемеңіз келді" (narrative only); fingramota.kz/kk/post/vishing (ARDFM site; KK description of the "Иә"-voice-biometry trick, CVV/SMS-code asks — no dialogue). NBK press 2026-09-28 (courier/SMS-code scheme) and Antifraud-centre reports are narrative. Verdict: KZ public sources yield **phrases, not transcripts**. Confidence: high.

**B4. RU bank/press phrase lists** — t-j.ru/short/govoryat-moshenniki (2024-03-12; 8 opener phrases, e.g. "Наконец-то я до вас дозвонилась", "Завтра отключим ваш номер телефона"); Habr 970892 (Femida Search, 100 victims; verbatim: "по так называемому основному счёту в системе ЦБ от вашего имени были инициированы переводы", "Сейчас важно действовать очень быстро"); mobile-review 2024-08-21 (audio discussed, no transcripts); cbr.ru/faq/information_security ("безопасный/защищённый счёт" vocabulary). T-Bank **Фрод-рулетка** (фрод-рулетка.рф) routes live scam calls to volunteers — real RU scam audio exists at scale, **nothing published**; institutional ask only. Confidence: high.

## C. VERIFIED — legitimate-call negatives

**C1. HF `apptek-com/apptek_callcenter_dialogues`** — 128.6 h, 873 role-played EN agent–customer dialogues, 14 accents, 16 domains incl. **banking, delivery service, telecom, government, insurance, financial services, utilities**; fully manual transcription (85 annotators); turn-level `segments` with role. Licence **CC BY-SA 4.0** (card + arXiv 2604.27543). Verified: HF API (2,910 downloads), README. Use: EN FPR gate on acted-but-realistic bank/delivery/telecom calls — directly targets the D54 "legit sales call" FP class. Risk: acted; SA licence irrelevant for eval. Confidence: high.

**C2. HF `AIxBlock/92k-real-world-call-center-scripts-english`** — 91,706 **real** BPO call transcripts (~10,500 h; AssemblyAI ASR; PII-redacted; inbound/outbound; insurance, Medicare, auto, home-service, **telecom outbound**); Indian/US/Filipino accents; **CC BY-NC 4.0**; arXiv 2507.02958. No speaker labels (viewer copy `yevgeniy03/home-telecom-callcenter-transcripts-viewer` adds LLM-inferred roles for 3,239 telecom/home-service calls). Verified: README, file list. Use: large EN *real* legit negative pool for FPR; the outbound insurance/telecom sales calls are exactly what trips the EN head. Risk: NC (eval OK, no redistribution), ASR noise, no roles. Confidence: high.

**C3. Gridspace-Stanford HarperValleyBank** — github.com/cricketclub/gridspace-stanford-harper-valley (`data/` dir in repo, 23 h, 1,446 simulated bank calls, human transcripts + intents + dialog acts), **CC-BY-4.0** (repo licence field). Verified: GitHub API, README. Use: EN bank-register negatives (balance, checkbook, hours). Confidence: high.

**C4. Russian call-centre speech — samples only.** `InfoBayAI/Russian_Call_Center_Audio_Dataset_Dual_Channel` and `…Single-Channel` (declared **cc-by-4.0**, gated-auto, lastModified 2026-09-16) hold **3 sample calls each**; card: 1,025 h claimed, "sample data only", full set via datareq@infobay.ai, no transcripts. `AxonData/russian-speech-recognition-dataset` (CC BY-NC 4.0): 2 sample mp3+docx ("RU Account manager Finance", "RU Pharma Sale"), full 1,000 h commercial. Kaggle `axondata/russian-call-center-audio-transcription` same 2.46 MB. Use: a handful of real RU legit calls for a smoke test; nothing more. Confidence: high.

**C5. LDC CALLFRIEND Russian — LDC2023S08 (speech, 48 h, 100 calls, 8 kHz) + LDC2023T09 (transcripts, tab-delimited)**. Real 1999 US-domestic personal calls between Russian speakers. Fee: "Login for the applicable fee" — not visible publicly (LDC non-member corpora are typically low thousands USD each). Use: real-telephone RU negatives + 8 kHz ASR eval. Risk: 1999 émigré register, cost. Confidence: high (exists), low (price).

**C6. HF `basis-ai/basis-conversations-1500`** — real multi-party casual voice-app conversations; **kk-KZ 32.1 h / 60 conversations / 109 speakers; ru-KZ+ru-BY 28.5 h / 61 conversations**; transcripts "noisy… do not recommend supervising on them"; gated, custom "basis-data-license-1.0" (no re-identification, no impersonation). Verified: card. Use: the only sizeable *real conversational Kazakh* speech found — ASR/cue-survival eval and casual-negative stress test; not call-centre register. Confidence: medium.

**C7. Text-only RU/KK negatives** — `DeepPavlov/banking77_ru` (10,003 train / 3,080 test RU utterances, 77 bank intents; translated banking77 whose original is CC BY 4.0; verified rows: "Пожалуйста, помогите мне с моей картой. Она не активируется."); `TilQazyna/til-kk-intent-v1` (10,982 KK utterances, generic intent classes, partly synthetic, Apache-2.0, gated-manual); `kukunechka/russian-everyday-dialogues` (20 dialogues, CC BY 4.0); `kolestv108/Russian-Banking-Dialogues-Dataset` (LLM-synthetic, gated). Use: single-utterance FPR probes for the bank-intent class. Confidence: high.

## D. VERIFIED — independent-generator synthetic sets (second-generator eval, not training)

**D1. ICFD-31k** (IJCAI-ECAI 2026; HF `rishia2220/icfd-31k`, github SPELLAILab/ICFD-31k). 31k+ LLM-generated EN/Hinglish calls, 10 fraud umbrellas + 5 unseen, 3-s streaming chunks (712k train / 174k val / 180k test chunk rows), slow-thinking rationales. Data licence: custom "restricted, defensive-use" (research/education OK, "Commercial deployment without explicit written permission" prohibited; all PII synthetic). Verified: HF rows, LICENSE-DATA.md. Use: EN streaming-detection eval from a different generator + Indian fraud taxonomy; strip Hinglish. Risk: synthetic, licence. Confidence: high.

**D2. ASLC-448** (Mendeley 10.17632/p384bgyzz3.4, 2026-06-01) — 448 human-authored Arabic scam/legit telephone conversations, 9 dialects, 23 fine categories, urgency/threat/impersonation scores, TTS audio; **CC BY 4.0**. Use: taxonomy cross-check; the risk-scoring columns mirror Qorğan's tactic tags. Confidence: high.

**D3. Chinese 2026 benchmarks** — TeleAntiFraud 2.0 (arXiv 2609.18748; 900 calls/month frozen sets, synthetic from real case abstracts, 4open.science anon repo) and Vishing-Tactics-Bench (arXiv 2609.07151; 35,340 utterances / 5,645 synthetic calls, 6-tactic MITRE-style taxonomy, no URL). Use: taxonomy only. Confidence: medium.

**D4. Hinglish/Vietnamese** — `ysangam/Indian_Cyber_Scam_PhoneCall_Hinglish_Dataset` (10k single utterances, synthetic fields like `urgency_level`), `adamtc/scam_dialogues` (vi, tagged synthetic), `jencyyy/fraudcall` (91k rows = Chinese dialogues machine-translated into Bengali). Skip.

## E. Speech side (brief)

No new KK/RU **telephone-channel** corpus since KSC2. `Flamme-VRM/kazakh-speech-dataset` = KSC2 re-upload (726 h, 16 kHz, CC BY 4.0). ISSAI's 2026 releases (KazParaD, KazMix-3, PersonaMix, kaznerd2…) are text/eval sets, not telephone speech. Golos (OpenSLR 114, 1,240 h RU) is crowd/far-field, custom licence. Only C5 (LDC, 8 kHz) and C6 (Basis, app audio) are real conversational KK/RU speech.

## F. YouTube (scam-baiting) — access facts

RU: channel **Опа_Горностай** publishes ~36-min compilations ("Разговоры с мошенниками", "Перехват мошеннических звонков", "Сборник…", ids UkR82Z_cvQ4, -dYvWv_L-54, NHExEUwMw7w) — each has a **`ru` ASR caption track** (verified in `ytInitialPlayerResponse`); АНТИМОШЕННИК (UCAocR1_8rHDsszSCvAPJ8QA) and @moshennikzvonit exist (JS-rendered listings; not enumerated). `yt-dlp --write-auto-subs --skip-download` would fetch them; yt-dlp is not installed here. **KK: YouTube ASR does not support Kazakh** — KZ scam videos carry `uz`/`ru`/`tr` mislabelled auto tracks (A7Ghf9-KMlM → uz; 24kz/Astana TV news → ru); one manual `kk` track found (ZAgfKK6gC4M). Kazakh captions would have to come from our own Vosk KK. Legal: YouTube ToS forbids downloading; Standard YouTube Licence = all rights reserved; internal evaluation is defensible as research use in KZ (Art. 7(5) citation re-use per docs/LEGAL_ASSESSMENT.md) but **never redistribute**; scam-baiter speech is one party's recording of a criminal — PII of scammers only. EN: Kitboga/Scammer Payback already covered by BothBosu + A1.

## COULD NOT VERIFY

- scam.ai honeypot public subset URL/DOI (A3) — pending publication.
- LREC 2026 "Towards Safer Calls for Everyone" (aclanthology 2026.lrec-1.585; Kang, Choi, …, Hansaem Kim) — benchmark with 4 tasks (topic classification, dialogue-structure recognition…); language (presumably Korean), size, release URL, licence not in abstract; PDF not parseable here.
- Zenodo 15212527 "Scam Conversation Corpus" (NTNU/TU Wien, GPT-4o honeypot vs real scammers) — **restricted access, no licence, email/Telegram/Instagram not calls**.
- IEEE DataPort: KorCCVi v2 augmented (CC BY-NC-SA 4.0, subscription) and Vispute "real-time fraud detection telecom NLP" (882 KB CSV, subscription, provenance unstated).
- `BaekRok/vishing_data_3` (16,496 audio items with segment labels, 80 GB, no card) — likely AI-Hub Korean vishing audio; unusable without provenance.
- `DevangDalvi/fraud-call-dataset`, `Shouninger/ScamBench` — gated, no card.
- Robocalls international honeypot (arXiv 2606.31790; 839 transcripts, 65 countries) — countries/languages and release not in abstract.
- Kazakhstan Anti Call-центр (Astana prosecutor + Kazakhtelecom) and NBK Antifraud-centre hold intercepted-call data; no researcher access path published.

## Ranked top-10 (value for Qorğan ÷ effort)

1. **C2 AIxBlock 92k real EN call-centre transcripts** — immediate EN FPR pool (NC).
2. **C1 AppTek dialogues** — bank/delivery/telecom/government EN negatives with roles (CC BY-SA).
3. **B2 KZ court-acts portal** — only real Kazakh-language scam scripts; manual harvest.
4. **B1 sudrf.ru judgments via sudrfparser** — bulk real RU scam scripts (reported speech) for authoring eval calls.
5. **A3 scam.ai honeypot corpus** — email benren@scam.ai for the 1,000-call CC BY-NC subset + DUA.
6. **A1 Kaggle 243 YouTube transcripts (CC0)** — cheap EN opener eval.
7. **F RU scam-baiting auto-captions** — real RU scammer speech through ASR; eval-only, legal caveats.
8. **C6 Basis kk/ru conversations** — real Kazakh conversational speech for ASR/cue-survival (gated).
9. **D1 ICFD-31k** — second-generator EN streaming eval.
10. **C7 banking77_ru / til-kk-intent-v1** — single-utterance bank-intent FPR probes; C3 HarperValleyBank as EN bank baseline.

## Nothing exists for — stop looking

- Any public RU or KK **scam-call transcript** dataset (HF 632 kk datasets, all "мошенн/алаяқ/scam/vishing/fraud call" searches, Zenodo, Figshare, Kaggle, IEEE DataPort, Mendeley): none.
- Kazakh call-centre or 8 kHz telephone speech beyond KSC2; Kazakh YouTube auto-captions.
- data.egov.kz datasets on fraud (search is JS-only, API 403 without key; only crime statistics would exist, no text).
- Ukrainian, Polish, Spanish, Brazilian-Portuguese scam-*call* corpora (only PT-BR WhatsApp/Telegram message sets FraudWhatsApp.Br/FraudTelegram.Br).
- Open Russian bank/telecom call-centre corpora with transcripts: all commercial (InfoBay, AxonData, MaratDV, UniData, FutureBee) with ≤3 public samples.
- Published recordings/transcripts from T-Bank Фрод-рулетка, Sber Кибрарий, NBK Antifraud-centre, Astana Anti Call-центр — institutional requests only.
