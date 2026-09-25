# DATA_INTAKE.md — Real-call data policy and intake protocol (PLAN_2026-09 A8)

_Status: draft for review by whoever owns legal (PLAN §9 q5). Tooling:
`scripts/ingest_partner_calls.py`, `src/qorgan/data/real_intake.py`, `real_allocation.py`;
tests `tests/data/test_real_intake.py`, `tests/data/test_heldout_lock.py`._

## 1. Why this exists

Every number in `docs/eval_report.md` is measured on synthetic or author-written calls.
The first real calls change the evaluation; the first few hundred change the model. This
protocol says how real calls enter the project so that (a) nothing that could identify a
person is stored, (b) the held-out set stays honest (locked, never read, never tuned on),
and (c) each batch's provenance is graded-quality (ТЗ §9).

## 2. Roles

| Role | Does | Must not |
|---|---|---|
| **Partner** (bank fraud desk, telecom, hotline, Antifraud centre) | Selects calls, obtains/holds consent, labels or confirms labels, delivers the batch | Send audio or transcripts by e-mail / messengers |
| **Intake operator** (project) | Runs the ingest, verifies counts, deletes the raw delivery | Read transcripts of the locked set |
| **Labeler** | Assigns `scam` / `legit` (+ tactic ids) | Have edited the lexicons or anchors (`labeler_edited_lexicons: false` is declared per batch) |
| **Legal owner** | Confirms the consent basis and retention for each partner | — |

## 3. What we ask partners for (ranked)

1. **Both classes from the same domains**: confirmed scam calls **and** legitimate calls
   (bank, telecom, delivery, e-gov). The legitimate half drives the false-positive rate and
   is the scarce one. Target for the locked set: **≥ 60 legit / ≥ 40 scam**, RU / KK / mixed.
2. Outcome labels (confirmed fraud or not) and, if available, the partner's tactic taxonomy.
3. Kazakh telephone-speech audio (for ASR evaluation only; audio is never stored by us).
4. Confirmed scam-number lists — hashed is fine.

## 4. Transfer

- **Encrypted, out of band**: SFTP to the project host, or an `age`/GPG-encrypted archive
  on a USB device; the passphrase travels separately. Never e-mail, never chat apps.
- **On-prem option**: the ingest script runs on the partner's machine; only the output
  under `data/real/` (scrubbed text, hashed numbers, manifests) leaves the premises.
- The partner records a sha256 of the archive; the intake operator confirms it on receipt.
- The **raw delivery is deleted** (secure erase) once the ingest is verified; the batch
  manifest keeps the input hashes, so what was ingested remains provable.

## 5. Batch format

```
<batch_dir>/
  batch.yaml            provenance + consent record + labeler declaration
  calls.csv             one row per call (columns below, in this order)
  transcripts/*.txt     "speaker: text" per line; bare lines allowed
  audio/*               optional; transcribed only with --transcribe (faster-whisper)
```

`batch.yaml`

| field | example | note |
|---|---|---|
| `batch_id` | `bank-a-2026-09-a` | `^[a-z0-9][a-z0-9_-]{2,39}$`, unique across batches |
| `partner_id` | `bank_a` | same id as the partner API key |
| `delivered_on` | `2026-09-16` | |
| `transfer_method` | `sftp_encrypted` / `on_prem` | |
| `consent_basis` | `customer_consent` | machine-readable code from the agreement |
| `legal_reference` | `DPA-2026-014` | the agreement / DPIA reference |
| `labeler_id` | `fraud-desk-analyst-2` | pseudonymous is fine |
| `labeler_edited_lexicons` | `false` | must be `false`; the ingest refuses `true` |
| `notes` | free text | no call content |

`calls.csv` columns: `call_id, language, label, tactic_ids, caller_number, transcript_file,
audio_file, consent_ref` — `language` ∈ `ru|kk|mixed`; `label` ∈ `scam|legit`;
`tactic_ids` semicolon-separated taxonomy ids (optional, scam rows only);
`caller_number` optional (hashed on ingest, needs `QORGAN_NUMBER_HMAC_KEY`);
one of `transcript_file` / `audio_file`; `consent_ref` = the partner's consent record id.

## 6. What the ingest does — and never does

```bash
python scripts/ingest_partner_calls.py <batch_dir> --dry-run     # validate, print counts only
python scripts/ingest_partner_calls.py <batch_dir>               # write + allocate + materialise
```

Stores under `data/real/` (**gitignored** — real calls never enter git, even scrubbed):
`batches/<batch_id>.jsonl` (scrubbed `Dialogue`s), `batches/<batch_id>.linkage.jsonl`
(dialogue id → number digest + `+7 700 ***`), `batches/<batch_id>.manifest.json`
(provenance, counts, input sha256s), `allocation.json`, `manifest.json` (the lock).
Materialises `data/processed/real_heldout_v2.jsonl` and `real_train.jsonl` for the harness:
`python -m qorgan.eval.run --split real_heldout_v2`.

Never: raw numbers (the schema refuses them), audio, unscrubbed text, transcript content
in logs or terminal output. Emails, cards, IINs and numbers inside transcripts become
`[EMAIL]` / `[CARD]` / `[IIN]` / `[PHONE]`.

## 7. Split rule and the lock

- **First come**: the first 60 legitimate and 40 scam calls ever ingested form
  **`real_heldout_v2`**; every later call goes to **`real_train`**. `allocate()` never
  moves an id.
- When both targets are met the set is **locked**: its sha256 is written to
  `data/real/manifest.json`; any later change to the file or the allocation is a
  `LockViolation` (ingest refuses; `tests/data/test_heldout_lock.py` fails).
- **Scored, never read.** Nobody on the modelling side opens locked transcripts. If one
  must be inspected (a suspected label error), it is moved to `dev_anchors` and recorded
  in `data/anchors/inspection_ledger.yaml`; it never returns to the locked set.
- The locked set is scored **only at release points, at most once per two weeks**, and the
  result is pasted verbatim into `docs/eval_report.md` with its intervals. Threshold and
  feature choices are tuned on `val` + `authored_heldout` + `real_train` CV only.
- **`real_train`**: reported by 5-fold CV until it exceeds 200 calls; after that mixed into
  training with a sample weight (decided then, recorded as an ADR), keeping the synthetic
  corpus as regulariser. `build_corpus` never folds it in automatically.
- Zero overlap with `train` / `val` / `augment` is checked by normalised transcript, not
  just by id.

## 8. Retention, access, deletion

- Real batches are kept for the partner agreement's term (default: the project's
  `QORGAN_REPORT_RETENTION_DAYS`); a partner can withdraw a batch → its files are deleted
  and the manifests record the deletion. A withdrawn locked call makes the set *smaller*,
  never *replaced* (the lock hash is re-recorded with the deletion noted).
- Access to `data/real/` is limited to the intake operator's machine; it is never uploaded
  to the Hub (the `hf_upload.py` allow-patterns exclude it — do not widen them).

## 9. Legal checklist (for the legal owner)

- [ ] Consent basis for the **called party** (the citizen) and for the **caller** (whose
  speech is in the recording) — PLAN §7 item 5 asks whether the caller's speech needs a
  separate basis under the Law on Personal Data.
- [ ] Retention term and deletion obligations per partner.
- [ ] Whether scrubbed transcripts are still personal data (assume yes: treat as such).
- [ ] Cross-border: none by default (self-hosted, no cloud ASR); the Gemini pipeline is
  build-time and never sees real calls.
- [ ] Blinded re-enactment (PLAN A3) — participants' consent form.

## 10. Per-batch checklist

1. Archive hash confirmed · 2. `--dry-run` clean (counts match the partner's sheet) ·
3. `labeler_edited_lexicons: false` · 4. ingest → allocation printed · 5. raw delivery
erased · 6. `pytest tests/data/test_heldout_lock.py` green · 7. `data/README.md` provenance
row added (source, licence/basis, counts, limits, cleaning).
