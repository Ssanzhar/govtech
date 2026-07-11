"""SMOKE tests for `qorgan.classifier.train` — tiny model + fake tokenizer, offline."""

import json

import torch

from qorgan.classifier.model import ScamClassifierModel
from qorgan.classifier.train import build_targets, train_and_export
from qorgan.data.schema import Dialogue, Label, TacticTag, Utterance, spans_from_phrases

_LABEL_SPACE = ("otp_request", "urgency", "safe_account")


def _dialogue(did, text, *, risk, tags=(), phrases=()):
    return Dialogue(
        id=did,
        language="ru",
        utterances=(Utterance(speaker="caller", text=text),),
        label=Label(
            risk=risk,
            tactic_tags=tuple(TacticTag(id=t) for t in tags),
            trigger_spans=spans_from_phrases(phrases, text),
        ),
    )


def _corpus():
    return [
        _dialogue("s1", "Продиктуйте код из SMS сейчас", risk=0.9, tags=["otp_request", "urgency"], phrases=["код из SMS"]),
        _dialogue("s2", "Переведите на безопасный счёт немедленно", risk=0.9, tags=["safe_account", "urgency"]),
        _dialogue("n1", "Подтверждаем перевод на 5000 тенге", risk=0.02),
        _dialogue("n2", "Ваша посылка прибыла в отделение", risk=0.03),
    ]


def test_build_targets_produces_risk_and_multihot():
    risk, matrix = build_targets(_corpus(), _LABEL_SPACE)
    assert risk == [1, 1, 0, 0]
    assert matrix[0] == [1.0, 1.0, 0.0]  # otp_request + urgency
    assert matrix[2] == [0.0, 0.0, 0.0]  # hard negative -> no tactics


def test_train_and_export_writes_loadable_bundle(tmp_path, tiny_encoder, fake_tokenizer):
    model = ScamClassifierModel(tiny_encoder, num_tactics=len(_LABEL_SPACE))
    out_dir = tmp_path / "xlmr"

    metadata = train_and_export(
        _corpus(),
        _corpus(),  # reuse as val for the smoke run
        model=model,
        tokenizer=fake_tokenizer,
        base_model="xlm-roberta-base",
        label_space=_LABEL_SPACE,
        out_dir=out_dir,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=2,
        max_length=32,
    )

    assert (out_dir / "model.pt").exists()
    assert (out_dir / "metadata.json").exists()
    assert metadata["label_space"] == list(_LABEL_SPACE)
    assert metadata["num_tactics"] == 3
    assert metadata["temperature"] > 0

    on_disk = json.loads((out_dir / "metadata.json").read_text())
    assert on_disk == metadata

    # weights reload into a fresh model of the same architecture
    fresh = ScamClassifierModel(tiny_encoder, num_tactics=len(_LABEL_SPACE))
    fresh.load_state_dict(torch.load(out_dir / "model.pt"))
