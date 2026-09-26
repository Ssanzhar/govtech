"""Publish the trained bundle and the corpus to the Hugging Face Hub (after `hf auth login`).

    python scripts/hf_upload.py

Model repo  sanzh-ts/govtech     <- models/linear (heads + metadata + web/weights.json) and the
                                    lexicons the bundle hash-validates -- ALWAYS together.
Dataset repo sanzh-ts/govtech_ds <- the processed splits by explicit name + data/augment.
Never published: data/processed/{incidents,organizations,citizen_reports,audit_log}.jsonl,
real_heldout_v2.jsonl / real_train.jsonl (real partner calls, docs/DATA_INTAKE.md),
data/synthetic/, data/real/ -- the allow list below is the guard, keep it explicit.
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

from qorgan.data.publish_guard import PUBLISHED_SPLITS, find_unscrubbed

# PII gate (ADR D45): every utterance headed for the public dataset must already be a
# scrub_text fixed point -- nothing is uploaded otherwise.
_findings = find_unscrubbed(
    [Path("data/processed") / name for name in PUBLISHED_SPLITS] + sorted(Path("data/augment").glob("*.jsonl"))
)
if _findings:
    for f in _findings:
        print(f"PII gate: {f.file} {f.dialogue_id} utterance {f.utterance_index} is not scrubbed", file=sys.stderr)
    sys.exit("refusing to publish: fix the rows above (qorgan.data.build_corpus.scrub_dialogue)")

api = HfApi()

# 2026-09-21: heads trained on the device's own embeddings (ADR D33), per-tactic thresholds
# (D30), ASR-form Kazakh cues (D31), repaired corpus (D34). The bundle records
# embed_backend=device; the server loads it under the native `onnx` backend as its proxy.
MODEL_REPO = "sanzh-ts/govtech"
api.upload_folder(repo_id=MODEL_REPO, repo_type="model", folder_path="models/linear")
api.upload_folder(repo_id=MODEL_REPO, repo_type="model", folder_path="data/lexicon", path_in_repo="lexicon")
# The July embedding-only baseline (`embed_only/`) is unchanged; re-upload only if retrained.
# api.upload_folder(repo_id=MODEL_REPO, repo_type="model", folder_path="models/linear_embed_only", path_in_repo="embed_only")

DATASET_REPO = "sanzh-ts/govtech_ds"
api.upload_folder(
    repo_id=DATASET_REPO, repo_type="dataset", folder_path="data/processed",
    allow_patterns=[
        "train.jsonl", "val.jsonl", "test.jsonl", "authored_heldout.jsonl", "ood.jsonl",
        "adversarial.jsonl", "adversarial_legit.jsonl", "manifest.json",
        "shift.jsonl", "shift.manifest.json",  # the second-generator eval split (ADR D35)
    ],
    # The pre-A2 name of the hand-written set; it was never real calls, but the name misleads.
    delete_patterns=["real_heldout.jsonl"],
)
api.upload_folder(repo_id=DATASET_REPO, repo_type="dataset", folder_path="data/augment", path_in_repo="augment")
# The cards (docs/hf/): what each repo is, its limits and the honest cross-generator number.
api.upload_file(repo_id=MODEL_REPO, repo_type="model", path_or_fileobj="docs/hf/MODEL_CARD.md", path_in_repo="README.md")
api.upload_file(repo_id=DATASET_REPO, repo_type="dataset", path_or_fileobj="docs/hf/DATASET_CARD.md", path_in_repo="README.md")
print(f"uploaded models/linear + lexicons -> {MODEL_REPO}; processed splits + augment -> {DATASET_REPO}")
