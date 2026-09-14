"""Tests for the hybrid feature pipeline: hard-signal cue features + embedding composition.

The golden gate is the point of the whole feature: the 3 hand-written legit RU calls that the
embedding-only model false-positives (they *mention* codes only to say they are NOT needed)
must yield an all-zero cue block, while the RU OTP scam must fire otp_request + safe_account.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qorgan.classifier import features as feat
from qorgan.classifier.cue_lexicon import load_cue_lexicon
from qorgan.config import get_config
from qorgan.data.schema import Dialogue
from qorgan.taxonomy import get_taxonomy

_LEGIT_FIXTURE_IDS = (
    "real_neg_bank_fraud_alert_ru",
    "real_neg_telecom_tariff_ru",
    "real_neg_bank_card_ready_ru",
)
_SCAM_FIXTURE_ID = "real_scam_bank_otp_ru"


def _lexicon():
    return load_cue_lexicon(get_config().cue_lexicon_path)


def _authored_heldout_by_id() -> dict[str, Dialogue]:
    path = get_config().data_dir / "processed" / "authored_heldout.jsonl"
    out: dict[str, Dialogue] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            dialogue = Dialogue.model_validate_json(line)
            out[dialogue.id] = dialogue
    return out


def test_exact_match_ru_cue_sets_column_and_verbatim_span():
    lexicon = _lexicon()
    ids = get_taxonomy().hard_signal_ids()
    text = "Пожалуйста, продиктуйте код из SMS прямо сейчас."

    matches = feat.match_cues(text, lexicon)
    otp = [m for m in matches if m.tactic_id == "otp_request"]
    assert otp, "expected an otp_request cue match"
    span = otp[0].span
    assert text[span.start : span.end] == span.text  # verbatim, grounded

    hard = feat.hard_signal_features([text], lexicon)
    assert hard[0, ids.index("otp_request")] == 1.0


def test_kazakh_cue_matches():
    lexicon = _lexicon()
    ids = get_taxonomy().hard_signal_ids()
    hard = feat.hard_signal_features(["Өтінемін, SMS-тегі кодты айтыңыз."], lexicon)
    assert hard[0, ids.index("otp_request")] == 1.0


def test_golden_legit_fixtures_yield_all_zero_block():
    lexicon = _lexicon()
    by_id = _authored_heldout_by_id()
    for fixture_id in _LEGIT_FIXTURE_IDS:
        hard = feat.hard_signal_features([by_id[fixture_id].transcript()], lexicon)
        assert hard.sum() == 0.0, f"{fixture_id} wrongly fired a hard-signal cue"


def test_golden_scam_fixture_fires_otp_and_safe_account():
    lexicon = _lexicon()
    ids = get_taxonomy().hard_signal_ids()
    scam = _authored_heldout_by_id()[_SCAM_FIXTURE_ID].transcript()
    hard = feat.hard_signal_features([scam], lexicon)
    assert hard[0, ids.index("otp_request")] == 1.0
    assert hard[0, ids.index("safe_account")] == 1.0
    assert feat.match_cues(scam, lexicon), "expected >=1 grounded cue match"


def test_hard_signal_features_are_deterministic():
    lexicon = _lexicon()
    scam = _authored_heldout_by_id()[_SCAM_FIXTURE_ID].transcript()
    assert np.array_equal(
        feat.hard_signal_features([scam], lexicon),
        feat.hard_signal_features([scam], lexicon),
    )


def test_column_order_matches_taxonomy_order():
    lexicon = _lexicon()
    ids = get_taxonomy().hard_signal_ids()
    hard = feat.hard_signal_features(["переведите на безопасный счёт немедленно"], lexicon)
    assert hard[0, ids.index("safe_account")] == 1.0
    assert hard.sum() == 1.0  # only safe_account fired


def test_asr_normalized_safe_account_variants_fire():
    lexicon = _lexicon()
    ids = get_taxonomy().hard_signal_ids()
    idx = ids.index("safe_account")
    variants = [
        "переведите деньги на безопасной счёт",  # real Vosk ASR output (genitive-ish inflection)
        "переведите деньги на безопасному счёту",  # dative inflection
        "переведите деньги на безопасный счет прямо сейчас",  # е-vs-ё spelling variant
    ]
    for text in variants:
        hard = feat.hard_signal_features([text], lexicon)
        assert hard[0, idx] == 1.0, f"expected safe_account to fire for: {text!r}"


def test_asr_normalized_otp_sms_variant_fires():
    lexicon = _lexicon()
    ids = get_taxonomy().hard_signal_ids()
    text = "Продиктуйте код из сообщения, которое вам только что пришло."

    hard = feat.hard_signal_features([text], lexicon)
    assert hard[0, ids.index("otp_request")] == 1.0

    matches = feat.match_cues(text, lexicon)
    otp = [m for m in matches if m.tactic_id == "otp_request"]
    assert otp, "expected a grounded otp_request cue match"
    span = otp[0].span
    assert text[span.start : span.end] == span.text


def test_kk_bare_imperative_safe_account_variant_fires():
    lexicon = _lexicon()
    ids = get_taxonomy().hard_signal_ids()
    text = "Ақшаны қазір қауіпсіз шотқа аудар, кейін өкінбейсіз."

    hard = feat.hard_signal_features([text], lexicon)
    assert hard[0, ids.index("safe_account")] == 1.0


def test_kk_bare_imperative_secrecy_variant_fires():
    lexicon = _lexicon()
    ids = get_taxonomy().hard_signal_ids()
    text = "Бұл туралы ешкімге айтпа, бұл құпия тексеру."

    hard = feat.hard_signal_features([text], lexicon)
    assert hard[0, ids.index("secrecy")] == 1.0


def test_matcher_is_case_insensitive_for_new_and_existing_cues():
    lexicon = _lexicon()
    ids = get_taxonomy().hard_signal_ids()
    hard = feat.hard_signal_features(
        ["ПРОДИКТУЙТЕ КОД ИЗ СООБЩЕНИЯ", "переведите на БЕЗОПАСНЫЙ СЧЕТ"], lexicon
    )
    assert hard[0, ids.index("otp_request")] == 1.0
    assert hard[1, ids.index("safe_account")] == 1.0


def test_bare_mentions_alone_still_yield_all_zero_block():
    lexicon = _lexicon()
    bare_mentions = [
        "какой у вас счёт открыт",
        "назовите код",
        "у вас есть карта",
        "пришла смс",
    ]
    for text in bare_mentions:
        hard = feat.hard_signal_features([text], lexicon)
        assert hard.sum() == 0.0, f"bare mention wrongly fired a cue: {text!r}"


def test_reassurance_style_sentences_still_yield_all_zero_block():
    lexicon = _lexicon()
    reassurance_sentences = [
        "Никакие коды называть не нужно, банк никогда их не спрашивает.",
        "Ешқандай код айтудың қажеті жоқ.",
    ]
    for text in reassurance_sentences:
        hard = feat.hard_signal_features([text], lexicon)
        assert hard.sum() == 0.0, f"reassurance sentence wrongly fired a cue: {text!r}"


def test_hybrid_matrix_dimensionality(fake_embedder):
    lexicon = _lexicon()
    k = len(get_taxonomy().hard_signal_ids())
    texts = ["продиктуйте код из SMS", "просто разговор о погоде"]

    blocks = feat.compute_feature_blocks(texts, embedder=fake_embedder, lexicon=lexicon)
    assert blocks.embedding.shape == (2, fake_embedder.dim)
    assert blocks.hard_signal.shape == (2, k)
    assert blocks.reassurance.shape == (2, 1)
    assert len(blocks.matches) == 2

    hybrid = feat.hybrid_matrix(blocks)
    # hybrid vector is embedding | hard-signal cues (K) | reassurance (1)
    assert hybrid.shape == (2, fake_embedder.dim + k + 1)
    assert np.array_equal(hybrid[:, :fake_embedder.dim], blocks.embedding)
    assert np.array_equal(hybrid[:, fake_embedder.dim : fake_embedder.dim + k], blocks.hard_signal)
    assert np.array_equal(hybrid[:, fake_embedder.dim + k :], blocks.reassurance)
