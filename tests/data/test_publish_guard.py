"""Nothing reaches the public Hub unless every utterance is already a `scrub_text` fixed
point (`qorgan.data.publish_guard`, called by `scripts/hf_upload.py` before uploading)."""

import json
from pathlib import Path

import pytest

from qorgan.data.publish_guard import PUBLISHED_SPLITS, find_unscrubbed, scrub_file
from qorgan.data.schema import SCAM_RISK_THRESHOLD, Dialogue, Label, Utterance, spans_from_phrases

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


# --- scrub_file: the repair that establishes the gate's invariant (ADR D45 / legal gap M11) ---


def _full(did: str, *texts: str, risk: float = 0.0, phrases: tuple[str, ...] = ()) -> dict:
    """A complete, schema-valid dialogue row as a dict (the shape the published splits hold)."""
    utterances = tuple(Utterance(speaker="caller", text=t) for t in texts)
    transcript = "\n".join(u.text for u in utterances)
    label = Label(
        risk=risk,
        trigger_spans=spans_from_phrases(list(phrases), transcript),
        is_hard_negative=risk < SCAM_RISK_THRESHOLD,
    )
    return Dialogue(id=did, language="ru", utterances=utterances, label=label).model_dump(mode="json")


def test_scrub_file_makes_the_file_a_fixed_point(tmp_path):
    path = _write(tmp_path / "ood.jsonl", [
        _full("clean", "всё чисто"),
        _full("leak", "это ИИН 770808300300?", "и карта 4400123456789010"),
    ])
    scrub_file(path)
    assert find_unscrubbed([path]) == []


def test_scrub_file_returns_what_it_fixed(tmp_path):
    path = _write(tmp_path / "ood.jsonl", [_full("clean", "всё чисто"), _full("leak", "ИИН 770808300300?")])
    fixed = scrub_file(path)
    assert [(f.file, f.dialogue_id, f.utterance_index) for f in fixed] == [("ood.jsonl", "leak", 0)]
    assert "770808300300" not in repr(fixed)  # a repair must not print what it redacted


def test_scrub_file_leaves_an_already_clean_file_byte_identical(tmp_path):
    path = _write(tmp_path / "ood.jsonl", [_full("a", "Позвоните в [PHONE]"), _full("b", "Код 123456 никому")])
    before = path.read_bytes()
    assert scrub_file(path) == []
    assert path.read_bytes() == before


def test_scrub_file_drops_a_trigger_span_that_was_itself_pii(tmp_path):
    path = _write(tmp_path / "shift.jsonl", [
        _full("scam", "Продиктуйте ИИН 770808300300 сейчас", risk=0.9, phrases=("770808300300",)),
    ])
    assert json.loads(path.read_text(encoding="utf-8"))["label"]["trigger_spans"], "fixture needs a span"
    scrub_file(path)
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["label"]["trigger_spans"] == []  # dangling span dropped, never left pointing at [IIN]
    Dialogue.model_validate(row)  # still schema-valid (verbatim-span invariant holds)


def test_scrub_file_is_idempotent(tmp_path):
    path = _write(tmp_path / "ood.jsonl", [_full("leak", "ИИН 770808300300?")])
    scrub_file(path)
    once = path.read_bytes()
    assert scrub_file(path) == []
    assert path.read_bytes() == once


def test_scrub_file_skips_a_missing_file(tmp_path):
    assert scrub_file(tmp_path / "absent.jsonl") == []


# --- never-published held-outs (ADR D65) ------------------------------------------------------


def test_shift_local_is_never_in_the_published_splits_or_the_upload_allow_list():
    from qorgan.data.publish_guard import DATASET_ALLOW_PATTERNS, NEVER_PUBLISHED, assert_publishable

    assert "shift_local.jsonl" in NEVER_PUBLISHED
    assert not set(NEVER_PUBLISHED) & set(PUBLISHED_SPLITS)
    assert_publishable(DATASET_ALLOW_PATTERNS)  # the real allow list passes


@pytest.mark.parametrize("patterns", [["shift_local.jsonl"], ["*.jsonl"], ["shift*"], ["train.jsonl", "*.manifest.json"]])
def test_an_allow_list_that_would_publish_shift_local_is_refused(patterns):
    from qorgan.data.publish_guard import PublishRefused, assert_publishable

    with pytest.raises(PublishRefused, match="shift_local"):
        assert_publishable(patterns)


def test_hf_upload_refuses_shift_local_before_uploading_anything():
    upload = (_REPO / "scripts" / "hf_upload.py").read_text(encoding="utf-8")
    guard_at = upload.index("assert_publishable(DATASET_ALLOW_PATTERNS)")
    assert guard_at < upload.index("HfApi()"), "the guard must run before any client exists"
    assert "allow_patterns=list(DATASET_ALLOW_PATTERNS)" in upload
    assert "shift_local" not in upload.split("allow_patterns=")[1].split(")")[0]


def test_shift_local_raw_and_processed_files_are_gitignored():
    import subprocess

    for path in ("data/raw/shift_local/kk.jsonl", "data/processed/shift_local.jsonl", "data/processed/shift_local.manifest.json"):
        result = subprocess.run(["git", "check-ignore", "-q", path], cwd=_REPO, check=False)
        assert result.returncode == 0, f"{path} must be gitignored"
