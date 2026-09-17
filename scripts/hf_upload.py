from huggingface_hub import HfApi

api = HfApi()
repo = 'sanzh-ts/govtech'

# Retrained hybrid bundle (2026-07-15: KK negatives + reassurance/cue lexicon update).
api.upload_folder(repo_id=repo, repo_type='model', folder_path='models/linear')
# The bundle hash-validates these lexicons -- ALWAYS upload them together with models/linear.
api.upload_folder(repo_id=repo, repo_type='model', folder_path='data/lexicon', path_in_repo='lexicon')
# Baseline unchanged this round; uncomment only if it was retrained.
# api.upload_folder(repo_id=repo, repo_type='model', folder_path='models/linear_embed_only', path_in_repo='embed_only')

repo = 'sanzh-ts/govtech_ds'

# Explicit filenames on purpose: real_heldout_v2.jsonl / real_train.jsonl (real partner calls,
# docs/DATA_INTAKE.md) live in the same dir and must NEVER be published.
api.upload_folder(
    repo_id=repo, repo_type='dataset', folder_path='data/processed',
    allow_patterns=['train.jsonl', 'val.jsonl', 'test.jsonl', 'authored_heldout.jsonl', 'ood.jsonl', 'adversarial.jsonl', 'manifest.json']
)
api.upload_folder(repo_id=repo, repo_type='dataset', folder_path='data/augment', path_in_repo='augment')