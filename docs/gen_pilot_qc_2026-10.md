# Generation pilot — QC report (2026-10-08, ADR D63)

The first Claude-written batch from the taxonomy-v2 grid. **Not in the corpus, not trained on.**
Its job is to show that the pipeline works and what its gates measure before phase 2
(mass generation, retrain, gates) starts. Every number below was printed by
`python -m qorgan.data.gen qc --run pilot` (full JSON: `data/synthetic/gen/pilot/qc_report.json`,
gitignored like all of `data/synthetic/`) or by the one-off v1 scoring in the last section.

## How it was made
- `python -m qorgan.data.gen manifest --run pilot --size 120` then `chunks --chunk-size 20` →
  6 chunk specs. Each spec has the instructions, the output schema, an English glossary
  (tactic / scheme / role names and descriptions, no example phrases, no cue lists) and 20 cells.
  Chunks are interleaved, so each one mixes classes and registers.
- 3 sub-agents (Claude Opus) wrote 2 chunks each. Each agent read only its two spec files, never
  `data/lexicon`, the corpus or any eval split, and wrote `data/synthetic/gen/pilot/out/chunk_NN.jsonl`.
  Wall time was about 10–11 minutes per agent.
- **Grid coverage:** 120 cells, 72 scam / 48 negative (40 %).
  - Languages: kk 36 · ru 36 · mixed 30 · en 18.
  - All 21 schemes are covered, with 3–4 cells each. The 9 low-arousal / new schemes are weighted 1.4×.
  - Negatives: the 6 new guardrail classes have 5 each, legit_bank_call 5, legit_gov_service 4,
    family_money_request 4, general_conversation 3, delivery_notification 2.
  - 24 confusable hard negatives. Scam phrasing: 36 direct, 36 indirect.
  - Length targets: 5 ×13, 7 ×27, 9 ×28, 12 ×28, 15 ×24.

## Gates

| Gate | Result | Note |
|---|---|---|
| 1. Schema + cell agreement + verbatim phrases + label rules | **120 / 120 valid**, 0 warnings | Warnings would cover length off target, script mismatch and single speaker. |
| 2. Scrub fixed point (D45) | **0** rows changed by `scrub_text` | |
| 3. e5 near-duplicate | **0 flagged** | See below. |
| 4. ROUGE-L novelty (< 0.7) | **0 flagged**. Max 0.307, mean 0.162, p95 0.223 | Compared with every same-language train row (940, ASR copies excluded) and every other pilot row. |
| 5. Disjoint from every processed split (hash only) | **0 overlaps** | Checked train, val, test, authored_heldout, ood, shift, adversarial and adversarial_legit. |
| 6. Lexicon leak: cue-free scams (target ≥ 0.30 per scheme) | **64 / 72 = 0.889 [0.793, 0.951]**. No scheme is below target. | Direct 31/36, indirect 33/36. Lowest schemes: bank_card_phish 1/3, bank_safe_account 1/3. |
| 6b. Negatives on which a hard-signal cue fires | **3 / 48** | pilot_0087, _0092, _0095. These are confusable negatives, as designed. |
| 7. Generator fingerprint (5-fold CV LR, pilot vs Gemini train matched on language × label) | **char n-gram AUC 0.975 [0.956, 0.989]**, **e5 AUC 0.948 [0.922, 0.972]** | See below. |
| 7b. Fingerprint restricted to v1 topics (36 rows) | **char 0.911 [0.837, 0.969]**, **e5 0.909 [0.834, 0.966]** | See below. |
| Layer 3: scheme rules vs cell scheme | **72 / 72** | |
| Layer 2: role slot vs cell role | **43 / 102 = 0.42**. On 18 role-less rows it fired once. | See below. |

**Near-duplicate threshold.** The research doc proposed 0.92, but e5 cosine is compressed. Train's own
nearest neighbour within a label has a median of 0.937, a p99 of 0.968 and a max of 0.974, and 798 train
rows sit above 0.92. The threshold is therefore calibrated as "no train pair exceeds it" = **0.974**.
The pilot's nearest train neighbour has a median of 0.904 and a max of 0.942. Its nearest pilot neighbour
has a median of 0.903 and a max of 0.945. So the pilot is *less* self-similar than train is.

**Generator fingerprint.** A classifier separates Claude rows from Gemini rows almost perfectly. Part
of that is topic: the v2 grid adds new schemes on purpose. Restricting to the topics both generators
share still gives about 0.91, so the style leak is real. Consequences are below.

**Role slot.** Most misses return no role at all (employer 8, telecom 7, relative 7, bank 7, utility 5).
The callers identify themselves by name or by context, with no claim-shaped phrase the lexicon knows.
Six misses are wrong-role picks, e.g. a "telecom" call later hands over to the "police".
This is display-only, so a miss costs nothing but the line. It agrees with the corpus migration's
`no_role` rate (below).

## What the shipped v1 model does on the pilot (diagnostic, not an evaluation)
The served `site/models/weights.json` was used: v1 heads, ONNX embedder as the server proxy, whole
transcript, threshold 0.59. The pilot is future *training* data, so this is a diagnostic of the gap
taxonomy v2 targets. It is not a held-out number and nothing is tuned on it.

| | hits / n | rate [95 % CP] |
|---|---|---|
| Recall, all 72 scams | 15 / 72 | **0.208 [0.122, 0.320]** |
| Recall, 7 low-arousal / new pretexts (parcel, clinic/benefits, eGov EDS, telecom, utility, erroneous transfer, job task) | **0 / 27** | 0.000 [0.000, 0.128] |
| Recall on the v1-covered bank schemes (bank_safe_account + bank_card_phish) | 5 / 6 | |
| FPR, all 48 negatives | 1 / 48 | **0.021 [0.001, 0.111]** (one legit_telecom_offer, ru) |

By language, scams: kk 4/22 · ru 5/21 · mixed 3/18 · en 3/11. Negatives: 0 on kk/mixed/en, 1/15 on ru.

## Findings that change phase 2
1. **The fingerprint is ≈ 0.91–0.97.** Generator identity is learnable, so in a mixed corpus it must not
   correlate with the label. Phase 2 writes Claude negatives at ≥ 40 % in every register, so that within
   the Claude rows the style carries no label. The honest held-out also has to move off Claude (§3.2):
   `shift` becomes `shift_v1`, a validation split only.
2. **Cue-free share 0.89 is far above the 0.30 floor.** The lexicon misses most of Claude's requests,
   even the direct ones (31/36). The embedding head therefore carries these rows: good for
   generalisation, but the hard-signal floor (meter latch) will rarely help on v2 schemes. Lexicon
   additions for v2 (e.g. claim-shaped OTP asks) are possible only from **train** rows with paired
   negatives, never from `shift` / held-out (D35/D43). They are a separate measured step.
3. **Grid fixes found by the writers:**
   - Persona axes are drawn independently, so the grid produced a "pensioner 18-25" and a "student 61+".
     Constrain occupation × age.
   - Some optional tactics contradict the cell's role, e.g. `imp_regulator` drawn for a bank caller. Draw
     optional `imp_*` tactics only when they agree with the role.
   - `stage=hook_only` can collide with a scheme whose core tactics are extraction tactics. Derive the
     allowed stages from the core.

   These go into `grid.py` before phase 2. The pilot manifest stays as generated, for the record.
4. **Role lexicon recall 0.42.** It may be widened from train rows and the phase-2 batch: it is
   display-only, so this is not the retrain rule. It must then be re-measured on a batch it was not
   widened from.
