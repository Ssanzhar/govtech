"""Synthetic dialogue generation via Gemini JSON mode (D1-10 scaffolding).

Generates schema-valid `Dialogue`s: one call per (tactic, language) or
(hard-negative category, language) combination, using a documented prompt template and
the same JSON-mode pattern as `classifier/llm_classifier.py`. Day 1 scope is scaffolding +
a first small batch; D2-1 extends this to the full corpus (all tactics x languages x
hard negatives, ~500 incidents worth of dialogues) plus dedup/splits (`build_corpus.py`).

Design note: the `Label` attached here reflects the *generation target* (the tactic or
hard-negative category the model was asked to write toward), i.e. cheap "self-instruct"
labeling at generation time. Day 2's `label.py` independently re-labels and validates
each dialogue (may add tags beyond the seed tactic, catches generation drift) -- this
mirrors the TeleAntiFraud-28k methodology referenced in `data/README.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qorgan.config import get_config
from qorgan.data.schema import Dialogue, Label, TacticTag, Utterance, spans_from_phrases
from qorgan.llm_tools import LLMResponseError, generate_json, thinking_budget_for
from qorgan.taxonomy import NegativeCategory, TacticDefinition, get_taxonomy

# Generous ceiling: a multi-turn dialogue in JSON is long, and Gemini 2.5 thinking tokens
# count against this budget (see `llm_tools.thinking_budget_for`). Too low truncates the
# JSON mid-object.
_MAX_TOKENS = 4096

# Provisional risk assigned to generation-time (pre-label.py) dialogues. Day 2's
# label.py replaces these with independently-assessed risk scores.
_SEED_POSITIVE_RISK = 0.9
_SEED_HARD_NEGATIVE_RISK = 0.02

_LANGUAGE_INSTRUCTIONS = {
    "ru": "Write the entire dialogue in Russian.",
    "kk": "Write the entire dialogue in Kazakh.",
    "mixed": (
        "Write the dialogue in natural Kazakh-Russian code-switched speech, as commonly "
        "spoken in everyday Kazakhstani phone calls."
    ),
}

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "utterances": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string", "description": "'caller' or 'callee'"},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "text"],
            },
        },
        "trigger_phrases": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Verbatim phrases copied from the utterances that signal the tactic "
                "(empty for hard negatives)."
            ),
        },
    },
    "required": ["utterances", "trigger_phrases"],
}


class GenerationError(RuntimeError):
    """Raised when corpus config is invalid or a generated dialogue fails validation."""


class CorpusConfig(BaseModel):
    """Validated `configs/corpus.yaml` contents."""

    model_config = ConfigDict(frozen=True)

    version: int
    seed: int
    languages: tuple[Literal["ru", "kk", "mixed"], ...]
    model_route: Literal["quality", "bulk"]
    dialogues_per_tactic: int = Field(ge=0)
    hard_negatives_per_category: int = Field(ge=0)
    min_utterances: int = Field(ge=1)
    max_utterances: int
    output_path: Path

    @field_validator("languages")
    @classmethod
    def _languages_not_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("languages must not be empty")
        return value

    @model_validator(mode="after")
    def _max_not_below_min(self) -> "CorpusConfig":
        if self.max_utterances < self.min_utterances:
            raise ValueError(
                f"max_utterances ({self.max_utterances}) must be >= "
                f"min_utterances ({self.min_utterances})"
            )
        return self


def load_corpus_config(path: Path | None = None) -> CorpusConfig:
    """Load and validate `configs/corpus.yaml` (or `path`). Raises `GenerationError`."""
    cfg = get_config()
    resolved_path = path or cfg.corpus_config_path
    if not resolved_path.exists():
        raise GenerationError(f"Corpus config not found: {resolved_path}")

    try:
        raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise GenerationError(f"Invalid YAML in {resolved_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise GenerationError(f"{resolved_path} must contain a YAML mapping at the top level")

    output_path = raw.get("output_path")
    if isinstance(output_path, str) and not Path(output_path).is_absolute():
        raw = {**raw, "output_path": cfg.repo_root / output_path}

    try:
        return CorpusConfig.model_validate(raw)
    except ValueError as exc:
        raise GenerationError(f"Corpus config validation failed for {resolved_path}: {exc}") from exc


def build_generation_prompt(tactic: TacticDefinition, language: str, cfg: CorpusConfig) -> str:
    """Documented prompt template for one synthetic scam-dialogue generation call."""
    lang_instruction = _LANGUAGE_INSTRUCTIONS[language]
    return (
        "Write a realistic phone-call transcript between a SCAM CALLER and a CALLEE in "
        f"Kazakhstan, {cfg.min_utterances}-{cfg.max_utterances} utterances long, "
        "alternating turns.\n\n"
        f'The caller must clearly employ this scam tactic: "{tactic.description}" '
        f"(tactic id: {tactic.id}).\n\n"
        f"{lang_instruction}\n\n"
        "Respond with a single JSON object with keys `utterances` (array of "
        "{speaker, text}) and `trigger_phrases` (array of strings). Every trigger_phrases "
        "entry must be an exact, verbatim substring of one of the utterances you write -- "
        "never paraphrase or invent a phrase."
    )


def build_hard_negative_prompt(category: NegativeCategory, language: str, cfg: CorpusConfig) -> str:
    """Documented prompt template for one synthetic hard-negative dialogue."""
    lang_instruction = _LANGUAGE_INSTRUCTIONS[language]
    return (
        "Write a realistic phone-call transcript between two people in Kazakhstan, "
        f"{cfg.min_utterances}-{cfg.max_utterances} utterances long, alternating turns. "
        "This call must be legitimate -- NOT a scam.\n\n"
        f'Scenario: "{category.note}" (category id: {category.id}). The caller must NOT '
        "ask for OTP codes, card details/CVV, or request moving money to a 'safe' "
        "account -- this is a hard-negative example used to measure false positives.\n\n"
        f"{lang_instruction}\n\n"
        "Respond with a single JSON object with keys `utterances` (array of "
        "{speaker, text}) and `trigger_phrases` (leave it empty)."
    )


def generate_dialogue(
    tactic_id: str,
    language: str,
    *,
    client: Any,
    cfg: CorpusConfig | None = None,
    dialogue_id: str | None = None,
) -> Dialogue:
    """Generate one schema-valid `Dialogue` exhibiting `tactic_id`, in `language`."""
    active_cfg = cfg or load_corpus_config()
    tactic = get_taxonomy().get(tactic_id)
    prompt = build_generation_prompt(tactic, language, active_cfg)
    payload = _call_tool(client, prompt, active_cfg)
    return _build_dialogue(
        payload,
        dialogue_id=dialogue_id or f"{tactic_id}_{language}",
        language=language,
        tags=(TacticTag(id=tactic_id, weight=1.0),),
        risk=_SEED_POSITIVE_RISK,
        is_hard_negative=False,
    )


def generate_hard_negative(
    category_id: str,
    language: str,
    *,
    client: Any,
    cfg: CorpusConfig | None = None,
    dialogue_id: str | None = None,
) -> Dialogue:
    """Generate one schema-valid hard-negative `Dialogue` for `category_id`, in `language`."""
    active_cfg = cfg or load_corpus_config()
    category = _get_negative_category(category_id)
    prompt = build_hard_negative_prompt(category, language, active_cfg)
    payload = _call_tool(client, prompt, active_cfg)
    return _build_dialogue(
        payload,
        dialogue_id=dialogue_id or f"neg_{category_id}_{language}",
        language=language,
        tags=(),
        risk=_SEED_HARD_NEGATIVE_RISK,
        is_hard_negative=True,
        force_empty_spans=True,
    )


def generate_batch(cfg: CorpusConfig, *, client: Any) -> list[Dialogue]:
    """Generate the full (tactic x language) + (hard-negative x language) batch per `cfg`.

    Deterministic iteration order (sorted tactic/category ids x configured languages) so
    output is stable given the same config and a deterministic client/model.
    """
    taxonomy = get_taxonomy()
    dialogues: list[Dialogue] = []

    for tactic_id in sorted(taxonomy.tactic_ids()):
        for language in cfg.languages:
            for i in range(cfg.dialogues_per_tactic):
                dialogues.append(
                    generate_dialogue(
                        tactic_id,
                        language,
                        client=client,
                        cfg=cfg,
                        dialogue_id=f"{tactic_id}_{language}_{i}",
                    )
                )

    for category_id in sorted(taxonomy.negative_ids()):
        for language in cfg.languages:
            for i in range(cfg.hard_negatives_per_category):
                dialogues.append(
                    generate_hard_negative(
                        category_id,
                        language,
                        client=client,
                        cfg=cfg,
                        dialogue_id=f"neg_{category_id}_{language}_{i}",
                    )
                )

    return dialogues


def write_dialogues_jsonl(dialogues: list[Dialogue], path: Path) -> None:
    """Write `dialogues` as JSONL (one `Dialogue` per line), creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (dialogue.model_dump_json() for dialogue in dialogues)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _get_negative_category(category_id: str) -> NegativeCategory:
    taxonomy = get_taxonomy()
    for category in taxonomy.negatives:
        if category.id == category_id:
            return category
    raise KeyError(f"Unknown negative category id: {category_id!r}")


def _resolve_model(cfg: CorpusConfig) -> str:
    """Map `cfg.model_route` ("quality" | "bulk") to an actual model id via config.py --
    the only place model ids are allowed to be hardcoded (no hardcoded values elsewhere)."""
    app_cfg = get_config()
    return app_cfg.llm_model_quality if cfg.model_route == "quality" else app_cfg.llm_model_bulk


def _call_tool(client: Any, prompt: str, cfg: CorpusConfig) -> dict[str, Any]:
    model = _resolve_model(cfg)
    try:
        return generate_json(
            client,
            model=model,
            prompt=prompt,
            response_schema=_RESPONSE_SCHEMA,
            max_output_tokens=_MAX_TOKENS,
            thinking_budget=thinking_budget_for(model),
        )
    except LLMResponseError as exc:
        raise GenerationError(str(exc)) from exc


def _build_dialogue(
    payload: dict[str, Any],
    *,
    dialogue_id: str,
    language: str,
    tags: tuple[TacticTag, ...],
    risk: float,
    is_hard_negative: bool,
    force_empty_spans: bool = False,
) -> Dialogue:
    raw_utterances = payload.get("utterances") or []
    utterances = [
        Utterance(speaker=item["speaker"], text=item["text"])
        for item in raw_utterances
        if isinstance(item, dict) and item.get("speaker") and item.get("text")
    ]
    if not utterances:
        raise GenerationError(f"Generated dialogue {dialogue_id!r} has no valid utterances: {payload!r}")

    transcript = "\n".join(u.text for u in utterances)
    spans = () if force_empty_spans else spans_from_phrases(payload.get("trigger_phrases") or [], transcript)

    try:
        return Dialogue(
            id=dialogue_id,
            language=language,
            utterances=utterances,
            label=Label(risk=risk, tactic_tags=tags, trigger_spans=spans, is_hard_negative=is_hard_negative),
        )
    except ValueError as exc:
        raise GenerationError(f"Generated dialogue {dialogue_id!r} failed schema validation: {exc}") from exc


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI (live network)
    """CLI: `python -m qorgan.data.generate [--config configs/corpus.yaml]`.

    Generates the full batch per the corpus config and writes it to `output_path`.
    """
    from qorgan.llm_tools import build_client

    parser = argparse.ArgumentParser(description="Generate the Qorgan synthetic corpus.")
    parser.add_argument("--config", type=Path, default=None, help="Path to configs/corpus.yaml")
    args = parser.parse_args(argv)

    cfg = load_corpus_config(args.config)
    client = build_client(get_config().gemini_api_key)
    dialogues = generate_batch(cfg, client=client)
    write_dialogues_jsonl(dialogues, cfg.output_path)
    print(f"wrote {len(dialogues)} dialogues -> {cfg.output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
