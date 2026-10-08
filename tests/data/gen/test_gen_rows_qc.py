"""Generated-row validation and the deterministic QC helpers (ADR D63)."""

from __future__ import annotations

import numpy as np

from qorgan.data.gen.grid import GridCell, Persona
from qorgan.data.gen.qc import (
    cue_free_share,
    lcs_length,
    matched_sample,
    max_rouge_l,
    nearest_cosine,
    rouge_l,
    scrub_failures,
    split_hashes,
    transcript_hash,
)
from qorgan.data.gen.rows import check_row
from qorgan.data.schema import Dialogue, Label, Utterance

V2 = frozenset({"imp_bank", "otp_request", "urgency"})
HARD = frozenset({"legit_bank_call"})
PERSONA = Persona(age_band="26-40", region="Almaty", occupation="nurse")


def _cell(kind="scam", **kw) -> GridCell:
    base = dict(cell_id="r_0000", run="r", kind=kind, language="ru", persona=PERSONA, outcome="refuses",
                channel="cold_call", stage="full" if kind == "scam" else "none", length_target=3)
    if kind == "scam":
        base.update(scheme="bank_card_phish", role="bank_security", tactics=("imp_bank", "otp_request"), phrasing="direct")
    else:
        base.update(negative_class="legit_bank_call", role="bank_security", hard_negative=True, confusable=False)
    base.update(kw)
    return GridCell(**base)


def _row(kind="scam", **kw) -> dict:
    row = {
        "id": "r_0000", "cell_id": "r_0000", "language": "ru", "kind": kind,
        "scheme": "bank_card_phish" if kind == "scam" else None,
        "negative_class": None if kind == "scam" else "legit_bank_call", "role": "bank_security",
        "utterances": [
            {"speaker": "caller", "text": "Здравствуйте, это банк. Назовите цифры, что пришли.", "stage": "hook",
             "tactics": [{"id": "imp_bank", "phrase": "это банк"}, {"id": "otp_request", "phrase": "Назовите цифры"}] if kind == "scam" else []},
            {"speaker": "callee", "text": "Нет, я перезвоню сам.", "stage": "none", "tactics": []},
        ],
    }
    row.update(kw)
    return row


def test_valid_scam_row_becomes_a_grounded_dialogue():
    check = check_row(_row(), {"r_0000": _cell()}, v2=V2, hard_negative_ids=HARD)
    assert not check.errors, check.errors
    d = check.dialogue
    assert [t.id for t in d.label.tactic_tags] == ["imp_bank", "otp_request"] and d.label.risk == 0.9
    for span in d.label.trigger_spans:
        assert d.transcript()[span.start:span.end] == span.text


def test_phrase_in_second_utterance_is_offset_correctly():
    row = _row()
    row["utterances"][1]["tactics"] = [{"id": "urgency", "phrase": "перезвоню"}]
    d = check_row(row, {"r_0000": _cell()}, v2=V2, hard_negative_ids=HARD).dialogue
    span = [s for s in d.label.trigger_spans if s.text == "перезвоню"][0]
    assert d.transcript()[span.start:span.end] == "перезвоню"


def test_non_verbatim_phrase_unknown_tactic_and_missing_tactic_are_errors():
    row = _row()
    row["utterances"][0]["tactics"] = [{"id": "imp_bank", "phrase": "это не тот банк"}, {"id": "made_up", "phrase": "банк"}]
    errors = " ".join(check_row(row, {"r_0000": _cell()}, v2=V2, hard_negative_ids=HARD).errors)
    assert "not verbatim" in errors and "unknown tactic" in errors and "required tactics" in errors


def test_negative_with_tactics_and_cell_mismatch_are_errors():
    row = _row("legit")
    row["utterances"][0]["tactics"] = [{"id": "urgency", "phrase": "это банк"}]
    assert any("negative carries" in e for e in check_row(row, {"r_0000": _cell("legit")}, v2=V2, hard_negative_ids=HARD).errors)
    assert any("language" in e for e in check_row(_row(language="kk"), {"r_0000": _cell()}, v2=V2, hard_negative_ids=HARD).errors)


def test_valid_negative_is_a_hard_negative_at_low_risk():
    d = check_row(_row("legit"), {"r_0000": _cell("legit")}, v2=V2, hard_negative_ids=HARD).dialogue
    assert d.label.is_hard_negative and not d.label.is_scam and d.label.tactic_tags == ()


def test_schema_violation_is_reported_not_raised():
    check = check_row({"id": "x", "extra": 1}, {}, v2=V2, hard_negative_ids=HARD)
    assert check.dialogue is None and check.errors


def test_length_and_script_warnings():
    check = check_row(_row(), {"r_0000": _cell(length_target=12)}, v2=V2, hard_negative_ids=HARD)
    assert any("length" in w for w in check.warnings)


def test_lcs_and_rouge_l():
    a, b = "a b c d e".split(), "a x c y e".split()
    assert lcs_length(a, b) == 3 and lcs_length(a, a) == 5 and lcs_length([], a) == 0
    assert rouge_l(a, a) == 1.0 and rouge_l(a, ["z"]) == 0.0
    assert abs(rouge_l(a, b) - 0.6) < 1e-9
    assert max_rouge_l(a, [b, a]) == 1.0


def test_lcs_matches_the_quadratic_reference():
    import random

    rng = random.Random(0)

    def ref(x, y):
        dp = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
        for i in range(len(x)):
            for j in range(len(y)):
                dp[i + 1][j + 1] = dp[i][j] + 1 if x[i] == y[j] else max(dp[i][j + 1], dp[i + 1][j])
        return dp[-1][-1]

    for _ in range(200):
        x = [rng.choice("abcd") for _ in range(rng.randint(0, 30))]
        y = [rng.choice("abcd") for _ in range(rng.randint(0, 30))]
        assert lcs_length(x, y) == ref(x, y)


def test_nearest_cosine_excludes_self():
    v = np.eye(3, dtype=np.float32)
    assert nearest_cosine(v, v, exclude_self=True).tolist() == [0.0, 0.0, 0.0]
    assert nearest_cosine(v, v, exclude_self=False).tolist() == [1.0, 1.0, 1.0]


def _d(i, text, lang="ru", risk=0.9):
    return Dialogue(id=f"d{i}", language=lang, utterances=(Utterance(speaker="caller", text=text),), label=Label(risk=risk))


def test_scrub_and_cue_free_share():
    assert scrub_failures(_d(0, "позвоните +7 701 123 45 67")) == 1
    assert scrub_failures(_d(1, "позвоните [PHONE]")) == 0
    rows = [_d(0, "код"), _d(1, "ничего")]
    assert cue_free_share(rows, lambda t: "код" in t) == (1, 2)


def test_transcript_hash_normalises_and_split_hashes_return_only_hashes(tmp_path):
    assert transcript_hash("А  б\nВ") == transcript_hash("а б в")
    (tmp_path / "test.jsonl").write_text(_d(0, "секрет").model_dump_json() + "\n", encoding="utf-8")
    hashes = split_hashes(tmp_path, ("test", "missing"))
    assert set(hashes) == {"test"} and hashes["test"] == frozenset({transcript_hash("секрет")})


def test_matched_sample_follows_the_target_strata():
    pool = [_d(i, "x", lang, risk) for i, (lang, risk) in enumerate([("ru", 0.9)] * 5 + [("kk", 0.02)] * 5)]
    target = [_d(90, "y", "ru", 0.9), _d(91, "y", "kk", 0.02), _d(92, "y", "kk", 0.02)]
    sample = matched_sample(pool, target, seed=1)
    assert sorted((d.language, d.label.is_scam) for d in sample) == [("kk", False), ("kk", False), ("ru", True)]
    assert matched_sample(pool, target, seed=1) == sample
