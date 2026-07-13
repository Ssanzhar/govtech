from huggingface_hub import HfApi

api = HfApi()
# repo = 'sanzh-ts/govtech'

# api.upload_folder(repo_id=repo, repo_type='model', folder_path='models/linear')
# api.upload_folder(repo_id=repo, repo_type='model', folder_path='models/linear_embed_only', path_in_repo='embed_only')

# api.upload_folder(repo_id=repo, repo_type='model', folder_path='data/lexicon', path_in_repo='lexicon')

repo = 'sanzh-ts/govtech_ds'

api.upload_folder(
    repo_id=repo, repo_type='dataset', folder_path='data/processed',
    allow_patterns=['train.jsonl', 'val.jsonl', 'test.jsonl', 'real_heldout.jsonl', 'ood.jsonl', 'manifest.json']
)
api.upload_folder(repo_id=repo, repo_type='dataset', folder_path='data/augment', path_in_repo='augment')