"""The demo scenes the README and STATUS describe must actually exist and behave.

The pitch rests on three scenes: a scam that alerts, a real bank call that does not, and a
KAZAKH scam that alerts — the last one is what demonstrates the bilingual claim. It existed
only in the ASR bench harness until 2026-09-24, so the shipped replay path had two.
"""

from __future__ import annotations

import pytest

from qorgan.data.demo_transcripts import LIVE_DEMO_CALLS
from qorgan.web.client_config import export_scenarios


def test_all_three_demo_scenes_are_bundled_and_labelled():
    assert set(LIVE_DEMO_CALLS) == {
        "live_scam_bank_ru",
        "live_hard_negative_bank_ru",
        "live_scam_bank_kk",
    }
    exported = {s["id"]: s for s in export_scenarios()}
    assert set(exported) == set(LIVE_DEMO_CALLS)
    for scene in exported.values():
        assert scene["label"] and scene["label"] != scene["id"], scene["id"]
        assert len(scene["lines"]) >= 4, scene["id"]


def test_the_kazakh_scene_is_actually_kazakh_and_carries_a_cue():
    from qorgan.classifier.cue_match import find_cue
    from qorgan.classifier.cue_lexicon import load_cue_lexicon

    script = LIVE_DEMO_CALLS["live_scam_bank_kk"]
    assert any(ch in script for ch in "әғқңөұүһі"), "no Kazakh-specific letters"
    lexicon = load_cue_lexicon()
    fired = {t for t, cues in lexicon.entries.items() if any(find_cue(script, c) for c in cues)}
    assert fired, "the Kazakh demo scene must trip at least one hard-signal cue"


@pytest.mark.parametrize(
    "scene_id,should_alert",
    [("live_scam_bank_ru", True), ("live_scam_bank_kk", True), ("live_hard_negative_bank_ru", False)],
)
def test_each_scene_lands_on_the_right_side_of_the_threshold(scene_id, should_alert):
    """Runs on the mock backend so it is offline and deterministic; the real-model behaviour
    is verified in the browser and recorded in docs/STATUS.md."""
    from qorgan.classifier.predict import score
    from qorgan.config import get_config

    result = score(LIVE_DEMO_CALLS[scene_id], backend="mock")
    alerted = result.risk >= get_config().risk_threshold
    assert alerted is should_alert, f"{scene_id}: risk {result.risk:.3f}"
