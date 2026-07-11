"""SMOKE tests for `qorgan.data.generate` — mocked Gemini client only, no network."""

import json
from types import SimpleNamespace

import pytest

from qorgan.data.generate import (
    CorpusConfig,
    GenerationError,
    build_generation_prompt,
    build_hard_negative_prompt,
    generate_batch,
    generate_dialogue,
    generate_hard_negative,
    load_corpus_config,
    write_dialogues_jsonl,
)
from qorgan.data.schema import Dialogue
from qorgan.taxonomy import get_taxonomy


class FakeModels:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


def _tool_response(payload: dict):
    # Gemini JSON-mode response: the payload is a JSON object in `.text`.
    return SimpleNamespace(text=json.dumps(payload), parsed=None)


VALID_DIALOGUE_PAYLOAD = {
    "utterances": [
        {"speaker": "caller", "text": "Это служба безопасности банка."},
        {"speaker": "callee", "text": "Слушаю вас."},
        {"speaker": "caller", "text": "Продиктуйте код из SMS немедленно."},
    ],
    "trigger_phrases": ["код из SMS"],
}


# --- load_corpus_config ----------------------------------------------------------------


def test_load_real_corpus_config_is_valid():
    cfg = load_corpus_config()
    assert isinstance(cfg, CorpusConfig)
    assert cfg.min_utterances <= cfg.max_utterances
    assert "ru" in cfg.languages
    assert cfg.output_path.is_absolute()


def test_load_corpus_config_missing_file_raises(tmp_path):
    with pytest.raises(GenerationError):
        load_corpus_config(tmp_path / "nope.yaml")


def test_load_corpus_config_max_below_min_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
version: 1
seed: 1
languages: ["ru"]
model_route: bulk
dialogues_per_tactic: 1
hard_negatives_per_category: 1
min_utterances: 10
max_utterances: 2
output_path: data/synthetic/x.jsonl
""",
        encoding="utf-8",
    )
    with pytest.raises(GenerationError):
        load_corpus_config(bad)


def test_load_corpus_config_negative_count_raises(tmp_path):
    bad = tmp_path / "bad2.yaml"
    bad.write_text(
        """
version: 1
seed: 1
languages: ["ru"]
model_route: bulk
dialogues_per_tactic: -1
hard_negatives_per_category: 1
min_utterances: 2
max_utterances: 5
output_path: data/synthetic/x.jsonl
""",
        encoding="utf-8",
    )
    with pytest.raises(GenerationError):
        load_corpus_config(bad)


def test_load_corpus_config_unknown_language_raises(tmp_path):
    bad = tmp_path / "bad3.yaml"
    bad.write_text(
        """
version: 1
seed: 1
languages: ["en"]
model_route: bulk
dialogues_per_tactic: 1
hard_negatives_per_category: 1
min_utterances: 2
max_utterances: 5
output_path: data/synthetic/x.jsonl
""",
        encoding="utf-8",
    )
    with pytest.raises(GenerationError):
        load_corpus_config(bad)


# --- prompt templates --------------------------------------------------------------------


def test_build_generation_prompt_mentions_tactic_and_language():
    cfg = load_corpus_config()
    tactic = get_taxonomy().get("otp_request")

    prompt = build_generation_prompt(tactic, "ru", cfg)

    assert "otp_request" in prompt
    assert "Russian" in prompt


def test_build_generation_prompt_injects_style_when_provided():
    cfg = load_corpus_config()
    tactic = get_taxonomy().get("otp_request")
    style = "STYLE: rough transcribed call with filler words"

    with_style = build_generation_prompt(tactic, "ru", cfg, style=style)
    without_style = build_generation_prompt(tactic, "ru", cfg)

    assert style in with_style
    assert style not in without_style


def test_build_hard_negative_prompt_mentions_category():
    cfg = load_corpus_config()
    category = get_taxonomy().negatives[0]

    prompt = build_hard_negative_prompt(category, "kk", cfg)

    assert category.id in prompt
    assert "Kazakh" in prompt


# --- generate_dialogue / generate_hard_negative -------------------------------------------


def test_generate_dialogue_returns_valid_schema_dialogue():
    cfg = load_corpus_config()
    client = FakeClient([_tool_response(VALID_DIALOGUE_PAYLOAD)])

    dialogue = generate_dialogue("otp_request", "ru", client=client, cfg=cfg)

    assert isinstance(dialogue, Dialogue)
    assert dialogue.label.tactic_tags[0].id == "otp_request"
    assert dialogue.label.trigger_spans[0].text == "код из SMS"
    assert dialogue.label.is_hard_negative is False
    assert len(client.models.calls) == 1


def test_generate_dialogue_uses_bulk_or_quality_model_route(monkeypatch):
    cfg = load_corpus_config()
    client = FakeClient([_tool_response(VALID_DIALOGUE_PAYLOAD)])

    generate_dialogue("otp_request", "ru", client=client, cfg=cfg)

    used_model = client.models.calls[0]["model"]
    from qorgan.config import get_config

    app_cfg = get_config()
    expected = app_cfg.llm_model_quality if cfg.model_route == "quality" else app_cfg.llm_model_bulk
    assert used_model == expected


def test_generate_hard_negative_forces_empty_spans_and_low_risk():
    cfg = load_corpus_config()
    payload = {
        "utterances": [
            {"speaker": "caller", "text": "Здравствуйте, подтверждаю перевод на 5000 тенге."},
            {"speaker": "callee", "text": "Хорошо, спасибо."},
        ],
        # Model might still (wrongly) include a phrase; hard negatives must force empty.
        "trigger_phrases": ["перевод"],
    }
    client = FakeClient([_tool_response(payload)])
    category_id = get_taxonomy().negatives[0].id

    dialogue = generate_hard_negative(category_id, "ru", client=client, cfg=cfg)

    assert dialogue.label.trigger_spans == ()
    assert dialogue.label.tactic_tags == ()
    assert dialogue.label.is_hard_negative is True
    assert dialogue.label.risk < 0.1


def test_generate_dialogue_no_utterances_raises_generation_error():
    cfg = load_corpus_config()
    client = FakeClient([_tool_response({"utterances": [], "trigger_phrases": []})])

    with pytest.raises(GenerationError):
        generate_dialogue("otp_request", "ru", client=client, cfg=cfg)


def test_generate_dialogue_drops_hallucinated_trigger_phrase():
    cfg = load_corpus_config()
    payload = {
        "utterances": VALID_DIALOGUE_PAYLOAD["utterances"],
        "trigger_phrases": ["код из SMS", "фраза которой здесь нет"],
    }
    client = FakeClient([_tool_response(payload)])

    dialogue = generate_dialogue("otp_request", "ru", client=client, cfg=cfg)

    assert len(dialogue.label.trigger_spans) == 1
    assert dialogue.label.trigger_spans[0].text == "код из SMS"


def test_generate_dialogue_unknown_tactic_raises_keyerror():
    cfg = load_corpus_config()
    client = FakeClient([_tool_response(VALID_DIALOGUE_PAYLOAD)])

    with pytest.raises(KeyError):
        generate_dialogue("not_a_real_tactic", "ru", client=client, cfg=cfg)


# --- generate_batch ------------------------------------------------------------------------


def test_generate_batch_produces_expected_count(tmp_path):
    bad = tmp_path / "small.yaml"
    bad.write_text(
        """
version: 1
seed: 1
languages: ["ru"]
model_route: bulk
dialogues_per_tactic: 1
hard_negatives_per_category: 1
min_utterances: 2
max_utterances: 5
output_path: data/synthetic/small.jsonl
""",
        encoding="utf-8",
    )
    cfg = load_corpus_config(bad)
    taxonomy = get_taxonomy()
    n_tactics = len(taxonomy.tactics)
    n_negatives = len(taxonomy.negatives)

    responses = [_tool_response(VALID_DIALOGUE_PAYLOAD) for _ in range(n_tactics)]
    responses += [
        _tool_response({"utterances": VALID_DIALOGUE_PAYLOAD["utterances"], "trigger_phrases": []})
        for _ in range(n_negatives)
    ]
    client = FakeClient(responses)

    dialogues = generate_batch(cfg, client=client)

    assert len(dialogues) == n_tactics + n_negatives
    assert all(isinstance(d, Dialogue) for d in dialogues)
    assert len(client.models.calls) == n_tactics + n_negatives


# --- write_dialogues_jsonl -----------------------------------------------------------------


def test_write_dialogues_jsonl_round_trips(tmp_path):
    cfg = load_corpus_config()
    client = FakeClient([_tool_response(VALID_DIALOGUE_PAYLOAD)])
    dialogue = generate_dialogue("otp_request", "ru", client=client, cfg=cfg)
    out_path = tmp_path / "out" / "dialogues.jsonl"

    write_dialogues_jsonl([dialogue], out_path)

    assert out_path.exists()
    lines = [line for line in out_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    round_tripped = Dialogue.model_validate_json(lines[0])
    assert round_tripped == dialogue
