"""Cue matching that survives the recogniser (Tier A).

The recogniser mangles cue phrases in ways exact substring matching cannot see through —
above all by moving word boundaries («на обороте» → «наоборот де»). These tests pin the
normalisation, the bounded-edit search, and the rule that keeps the change safe: an exact
hit always wins, so clean text behaves exactly as it did before.
"""

from __future__ import annotations

import pytest

from qorgan.classifier.cue_match import (
    MATCHER_VERSION,
    budget_for,
    find_cue,
    normalize,
)


# --- normalisation ------------------------------------------------------------------------

def test_normalize_despaces_lowercases_and_folds_yo():
    text = "Продиктуйте КОД из SMS, пожалуйста!"
    normalized, index_map = normalize(text)
    assert normalized == "продиктуйтекодизsmsпожалуйста"
    assert len(index_map) == len(normalized)
    # every normalised character points back at the character it came from
    assert text[index_map[0]] == "П"
    assert text[index_map[-1]] == "а"


def test_normalize_folds_yo_and_keeps_kazakh_letters():
    normalized, _ = normalize("Счёт — қауіпсіз, ешкімге айтпаңыз")
    assert normalized == "счетқауіпсізешкімгеайтпаңыз"


def test_normalize_drops_hyphens_so_asr_forms_match():
    # the recogniser never emits hyphens; D31 had to add hyphen-free cue variants by hand
    assert normalize("push-уведомление")[0] == normalize("push уведомление")[0]


# --- the budget ---------------------------------------------------------------------------

def test_budget_is_zero_for_short_cues_and_grows_with_length():
    assert budget_for(len(normalize("AnyDesk")[0])) == 0
    assert budget_for(len(normalize("код из SMS")[0])) == 0
    assert budget_for(len(normalize("три цифры на обороте")[0])) == 1
    assert budget_for(len(normalize("Не сообщайте сотрудникам банка")[0])) == 2
    lengths = [0, 5, 11, 12, 20, 21, 30, 31, 60]
    budgets = [budget_for(n) for n in lengths]
    assert budgets == sorted(budgets), "budget must not shrink as the cue grows"


# --- matching -----------------------------------------------------------------------------

def test_exact_cue_is_found_with_verbatim_original_offsets():
    text = "Здравствуйте. Продиктуйте код из SMS, чтобы отменить списание."
    span = find_cue(text, "код из SMS")
    assert span is not None
    start, end = span
    assert text[start:end] == "код из SMS"


def test_cue_is_found_across_a_word_boundary_the_recogniser_moved():
    # real Vosk output: «три цифры на обороте» -> «три цифры наоборот де»
    hypothesis = "назовите номер карты и три цифры наоборот де чтобы мы в форме продвижение"
    span = find_cue(hypothesis, "три цифры на обороте")
    assert span is not None
    start, end = span
    assert hypothesis[start:end].startswith("три цифры")


def test_exact_hit_wins_over_a_fuzzy_one_earlier_in_the_text():
    # the fuzzy pass must never pre-empt a verbatim occurrence: clean text keeps old behaviour
    text = "три цифры наоборот де ... и назовите три цифры на обороте"
    start, end = find_cue(text, "три цифры на обороте")
    assert text[start:end] == "три цифры на обороте"


def test_unrelated_text_does_not_match():
    text = "Ваша карта готова, курьер привезёт её завтра после обеда."
    assert find_cue(text, "три цифры на обороте") is None
    assert find_cue(text, "Продиктуйте код из SMS") is None


def test_short_cue_gets_no_fuzz_so_near_words_do_not_fire():
    # "AnyDesk" -> "не доски" is a real recogniser error, but rescuing it with edit distance
    # would also match unrelated words; short cues stay exact (a lexicon variant is the fix)
    assert find_cue("установите приложения не доски", "AnyDesk") is None


def test_kazakh_agglutination_within_budget_is_matched():
    span = find_cue("сізге келген кодты айтыңызшы, тез", "келген кодты айтыңыз")
    assert span is not None


def test_empty_inputs_are_safe():
    assert find_cue("", "код из SMS") is None
    assert find_cue("продиктуйте код", "") is None


def test_matcher_version_is_pinned():
    # bumping this invalidates trained bundles: the cue features are computed with it
    assert MATCHER_VERSION == 2


@pytest.mark.parametrize("cue", ["код из SMS", "три цифры на обороте", "Ешкімге айтпаңыз"])
def test_a_cue_always_matches_itself(cue):
    assert find_cue(cue, cue) == (0, len(cue))


# --- the prefilter must never change an answer, only the time it takes ---------------------

def test_prefilter_is_sound_on_every_lexicon_cue_and_corpus_text():
    """Pigeonhole: with budget b the cue splits into b+1 blocks and one must survive intact.
    Brute force (prefilter disabled) must agree with the fast path on every pair."""
    import json
    from pathlib import Path

    import yaml

    from qorgan.classifier import cue_match

    repo = Path(__file__).resolve().parents[2]
    capture = repo / "data/asr_capture/pairs.jsonl"
    if not capture.exists():  # regenerate with scripts/spikes/asr_cue_survival/capture.py (macOS + vosk)
        pytest.skip("ASR capture corpus not present")
    cues = yaml.safe_load((repo / "data/lexicon/hard_signal_cues.yaml").read_text(encoding="utf-8"))["cues"]
    phrases = [c for cue_list in cues.values() for c in cue_list]
    texts = []
    for line in (repo / "data/processed/authored_heldout.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            texts.append("\n".join(u["text"] for u in json.loads(line)["utterances"]))
    for line in capture.read_text(encoding="utf-8").splitlines()[:60]:
        if line.strip():
            texts.append(json.loads(line)["hypothesis"])

    for text in texts:
        for phrase in phrases:
            fast = cue_match.find_cue(text, phrase)
            slow = cue_match.find_cue(text, phrase, _prefilter=False)
            assert fast == slow, f"prefilter changed the answer for {phrase!r}"


# --- the bundle must refuse a matcher it was not trained with -------------------------------

def test_a_bundle_trained_under_another_matcher_version_is_refused(tmp_path, monkeypatch):
    """The cue block is a model input, so a bundle trained with exact matching must not be
    scored with the bounded-edit one (or the features silently shift under the heads)."""
    import json

    import pytest as _pytest

    from qorgan.classifier import linear_train
    from qorgan.config import get_config

    model_dir = tmp_path / "linear"
    model_dir.mkdir()
    (model_dir / "metadata.json").write_text(
        json.dumps({
            "label_space": ["otp_request"],
            "hard_signal_enabled": True,
            "embed_backend": get_config().embed_backend,
            "cue_matcher_version": 1,
        }),
        encoding="utf-8",
    )
    with _pytest.raises(linear_train.LinearFeatureMismatchError, match="matcher"):
        linear_train.load_linear(model_dir)


# --- a cue may never be assembled across an utterance boundary -----------------------------

def test_fuzzy_matching_never_bridges_two_utterances():
    """De-spacing removes the newline that separates turns, so without an explicit boundary
    two innocuous turns could concatenate into a cue neither of them contains (found in
    review, 2026-09-23). FPR is the primary metric: the join is a wall."""
    first = "Хорошо, зафиксировано, назовите три цифры на"
    second = "обороте карты, чтобы я мог подтвердить."
    assert find_cue(first, "три цифры на обороте") is None
    assert find_cue(second, "три цифры на обороте") is None
    assert find_cue(f"{first}\n{second}", "три цифры на обороте") is None


def test_a_cue_inside_one_utterance_still_matches_when_others_surround_it():
    text = "Здравствуйте.\nНазовите три цифры наоборот де, пожалуйста.\nСпасибо."
    span = find_cue(text, "три цифры на обороте")
    assert span is not None
    start, end = span
    assert "\n" not in text[start:end]


def test_exact_match_is_still_found_anywhere_in_a_multi_turn_transcript():
    text = "Добрый день.\nВаша карта готова.\nНазовите три цифры на обороте."
    start, end = find_cue(text, "три цифры на обороте")
    assert text[start:end] == "три цифры на обороте"
