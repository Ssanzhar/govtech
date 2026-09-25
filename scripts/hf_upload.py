"""Publish the trained bundle and the corpus to the Hugging Face Hub (after `hf auth login`).

    python scripts/hf_upload.py

Model repo  sanzh-ts/govtech     <- models/linear (heads + metadata + web/weights.json) and the
                                    lexicons the bundle hash-validates -- ALWAYS together.
Dataset repo sanzh-ts/govtech_ds <- the processed splits by explicit name + data/augment.
Never published: data/processed/{incidents,organizations,citizen_reports,audit_log}.jsonl,
real_heldout_v2.jsonl / real_train.jsonl (real partner calls, docs/DATA_INTAKE.md),
data/synthetic/, data/real/ -- the allow list below is the guard, keep it explicit.
"""

from huggingface_hub import HfApi

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
print(f"uploaded models/linear + lexicons -> {MODEL_REPO}; processed splits + augment -> {DATASET_REPO}")
