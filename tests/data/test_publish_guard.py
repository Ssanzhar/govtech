"""Nothing reaches the public Hub unless every utterance is already a `scrub_text` fixed
point (`qorgan.data.publish_guard`, called by `scripts/hf_upload.py` before uploading)."""

import json
from pathlib import Path

import pytest

from qorgan.data.publish_guard import PUBLISHED_SPLITS, find_unscrubbed

_REPO = Path(__file__).resolve().parents[2]


def _write(path: Path, dialogues: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in dialogues) + "\n", encoding="utf-8")
    return path


def _dialogue(did: str, *texts: str) -> dict:
    return {"id": did, "utterances": [{"speaker": "caller", "text": t} for t in texts]}


def test_clean_files_have_no_findings(tmp_path):
    path = _write(tmp_path / "a.jsonl", [_dialogue("d1", "Позвоните в [PHONE]", "Код 123456 никому")])
    assert find_unscrubbed([path]) == []


def test_pii_is_reported_with_file_id_and_kind_but_never_the_value(tmp_path):
    path = _write(
        tmp_path / "ood.jsonl",
        [_dialogue("ok", "всё чисто"), _dialogue("leak", "это ИИН 770808300300?", "карта4400123456789010")],
    )
    findings = find_unscrubbed([path])
    assert [(f.file, f.dialogue_id, f.utterance_index) for f in findings] == [
        ("ood.jsonl", "leak", 0),
        ("ood.jsonl", "leak", 1),
    ]
    assert "770808300300" not in repr(findings)  # a gate must not print what it guards


def test_missing_files_are_skipped(tmp_path):
    assert find_unscrubbed([tmp_path / "absent.jsonl"]) == []


def test_every_hub_split_is_covered():
    upload = (_REPO / "scripts" / "hf_upload.py").read_text(encoding="utf-8")
    assert "find_unscrubbed" in upload, "hf_upload.py must run the guard before uploading"
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "authored_heldout.jsonl", "ood.jsonl",
                 "adversarial.jsonl", "adversarial_legit.jsonl", "shift.jsonl"):
        assert name in PUBLISHED_SPLITS


def test_local_published_splits_are_scrubbed():
    processed = _REPO / "data" / "processed"
    paths = [processed / name for name in PUBLISHED_SPLITS] + sorted((_REPO / "data" / "augment").glob("*.jsonl"))
    if not any(p.exists() for p in paths[: len(PUBLISHED_SPLITS)]):
        pytest.skip("no local corpus (run scripts/deploy_bootstrap.py)")
    assert find_unscrubbed(paths) == []
