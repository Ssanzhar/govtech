"""The register-diversity augmentation (ADR D35 follow-up): train-only, disjoint, varied.

The point of this augmentation is that the model should learn the scam rather than one
generator's way of writing a phone call. These tests pin the invariants that make it safe:
it never touches an evaluation split, and it is genuinely a different register.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qorgan.data.build_corpus import normalize_for_dedup
from qorgan.data.schema import Dialogue

_REPO = Path(__file__).resolve().parents[2]
_AUGMENT = _REPO / "data" / "augment"
_PROCESSED = _REPO / "data" / "processed"
_FILES = ("register_diversity.jsonl", "register_diversity_negatives.jsonl")
_EVAL_SPLITS = ("val", "test", "authored_heldout", "ood", "adversarial", "adversarial_legit", "shift")


def _load(name: str) -> list[Dialogue]:
    path = _AUGMENT / name
    if not path.exists():
        pytest.skip(f"{name} not generated (scripts/augment_register_diversity.py)")
    return [Dialogue.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_scams_are_positives_and_negatives_are_hard_negatives():
    for dialogue in _load("register_diversity.jsonl"):
        assert dialogue.label.risk >= 0.5 and not dialogue.label.is_hard_negative, dialogue.id
        assert dialogue.label.tactic_tags, dialogue.id
    for dialogue in _load("register_diversity_negatives.jsonl"):
        assert dialogue.label.risk < 0.5 and dialogue.label.is_hard_negative, dialogue.id


def test_augmentation_shares_nothing_with_any_evaluation_split():
    own = {normalize_for_dedup(d.transcript()) for name in _FILES for d in _load(name)}
    for split in _EVAL_SPLITS:
        path = _PROCESSED / f"{split}.jsonl"
        if not path.exists():
            continue
        other = {normalize_for_dedup(Dialogue.model_validate_json(l).transcript())
                 for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
        assert not (own & other), f"register augmentation overlaps {split}"


def test_the_register_is_actually_varied_not_one_repeated_shape():
    """A widened register must show up as varied length and varied openings -- otherwise the
    sampling did nothing and the augmentation is just more of the same corpus."""
    dialogues = [d for name in _FILES for d in _load(name)]
    lengths = {len(d.utterances) for d in dialogues}
    assert len(lengths) >= 5, f"only {len(lengths)} distinct dialogue lengths"
    openings = {d.utterances[0].text.split()[0].lower().strip(",.!?") for d in dialogues if d.utterances[0].text.split()}
    assert len(openings) >= 10, f"only {len(openings)} distinct opening words"
    assert any(d.utterances[0].speaker == "callee" for d in dialogues), "no call opens on the callee"


def test_manifest_records_how_it_was_made():
    path = _AUGMENT / "register_diversity.manifest.json"
    if not path.exists():
        pytest.skip("manifest not generated")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["seed"] and manifest["model"]
    assert manifest["scams"] > 0 and manifest["negatives"] >= manifest["scams"], "negatives must not be outnumbered (ADR D27)"


def test_every_augment_file_is_already_scrubbed():
    """`data/README.md` promises `data/augment/` is scrubbed and therefore publishable, and
    `scripts/hf_upload.py` uploads the whole directory. A generator that forgets
    `scrub_dialogue` would publish fabricated phone numbers and IINs (found 2026-09-24)."""
    from qorgan.data.scrub import scrub_text

    offenders = []
    for path in sorted(_AUGMENT.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            dialogue = Dialogue.model_validate_json(line)
            for utterance in dialogue.utterances:
                if scrub_text(utterance.text) != utterance.text:
                    offenders.append(f"{path.name}:{dialogue.id}")
    assert not offenders, f"unscrubbed augment rows: {sorted(set(offenders))}"
