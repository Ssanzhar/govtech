"""Independent LLM re-labeling of dialogues (D2-2).

`generate.py` attaches a cheap "self-instruct" label at generation time (the tactic the
model was *asked* to write toward). This module re-labels each dialogue independently:
Gemini reads the finished transcript and assigns `{risk, tactic_tags, trigger_phrases}`
with no knowledge of the generation seed, which catches generation drift and adds tactics
beyond the seed (the TeleAntiFraud-28k methodology cited in `data/README.md`).

Two grounding guarantees, enforced here rather than trusted from the model:
- tactic ids not present in the taxonomy are dropped (never invent a label);
- trigger phrases are re-located as verbatim substrings via `spans_from_phrases`, so any
  hallucinated phrase is silently discarded (CLAUDE.md SS6).

The Gemini client is injected by callers, so importing this module never needs an API key
and tests can pass a fake client with no network.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qorgan.config import get_config
from qorgan.data.schema import Dialogue, Label, TacticTag, spans_from_phrases
from qorgan.llm_tools import LLMResponseError, generate_json, thinking_budget_for
from qorgan.taxonomy import Taxonomy, get_taxonomy

_MAX_TOKENS = 2048

_SYSTEM_PROMPT_HEADER = (
    "You are labeling phone-call transcripts for Qorgan, a Kazakhstani scam-detection "
    "corpus. Read the transcript (Kazakh, Russian, or code-switched) and assess how likely "
    "the CALLER is running a social-engineering scam. Respond with a single JSON object: "
    "risk (number 0-1), tactic_tags (array of {id, weight}), trigger_phrases (array of "
    "strings). Only quote a trigger_phrases entry if it appears verbatim in the transcript "
    "-- never paraphrase or invent. For tactic_tags.id you MUST use ONLY the exact ids "
    "listed below (any other id is discarded); assign each a weight in [0, 1] by how "
    "strongly the tactic is present.\n\nAvailable tactic ids:"
)


def _build_system_prompt(taxonomy: Taxonomy) -> str:
    """System prompt enumerating the exact taxonomy ids the labeler may emit.

    Without the explicit id list the model invents plausible-but-wrong ids that then get
    dropped by `_valid_tags`, yielding empty tag sets -- so the label space is injected
    into the prompt directly from the taxonomy.
    """
    lines = [
        f"- {tactic.id}: {tactic.description}" for tactic in taxonomy.tactics
    ]
    return _SYSTEM_PROMPT_HEADER + "\n" + "\n".join(lines)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "risk": {"type": "number", "description": "Scam probability in [0, 1]."},
        "tactic_tags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "weight": {"type": "number"},
                },
                "required": ["id"],
            },
        },
        "trigger_phrases": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["risk", "tactic_tags", "trigger_phrases"],
}


class LabelingError(RuntimeError):
    """Raised when a re-labeling response cannot be turned into a valid `Label`."""


def label_dialogue(dialogue: Dialogue, *, client: Any, model: str | None = None) -> Dialogue:
    """Return a NEW `Dialogue` with an independently LLM-assigned `Label`.

    Identity (`id`, `language`, `utterances`) and the `is_hard_negative` provenance flag
    are preserved; only risk/tags/spans are replaced by the model's assessment.
    """
    cfg = get_config()
    # Labeling is a bulk pass over the whole corpus -> the cheaper flash model
    # (CLAUDE.md SS4: "gemini-2.5-flash for bulk gen/labeling").
    active_model = model or cfg.llm_model_bulk
    transcript = dialogue.transcript()

    try:
        payload = generate_json(
            client,
            model=active_model,
            prompt=transcript,
            response_schema=_RESPONSE_SCHEMA,
            system_instruction=_build_system_prompt(get_taxonomy()),
            max_output_tokens=_MAX_TOKENS,
            thinking_budget=thinking_budget_for(active_model),
        )
    except LLMResponseError as exc:
        raise LabelingError(str(exc)) from exc

    new_label = _build_label(payload, transcript, is_hard_negative=dialogue.label.is_hard_negative)
    return Dialogue(
        id=dialogue.id,
        language=dialogue.language,
        utterances=dialogue.utterances,
        label=new_label,
    )


def label_corpus(dialogues: Any, *, client: Any, model: str | None = None) -> list[Dialogue]:
    """Re-label every dialogue in `dialogues`, preserving order."""
    return [label_dialogue(d, client=client, model=model) for d in dialogues]


def _build_label(payload: dict[str, Any], transcript: str, *, is_hard_negative: bool) -> Label:
    try:
        risk = float(payload["risk"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LabelingError(f"Labeling response missing/invalid 'risk': {payload!r}") from exc
    risk = max(0.0, min(1.0, risk))

    tags = _valid_tags(payload.get("tactic_tags") or [])
    spans = spans_from_phrases(payload.get("trigger_phrases") or [], transcript)
    return Label(
        risk=risk, tactic_tags=tags, trigger_spans=spans, is_hard_negative=is_hard_negative
    )


def _valid_tags(raw_tags: list[Any]) -> tuple[TacticTag, ...]:
    """Keep only tags whose id exists in the taxonomy; clamp weights into [0, 1]."""
    valid_ids = set(get_taxonomy().tactic_ids())
    tags: list[TacticTag] = []
    for item in raw_tags:
        if not isinstance(item, dict):
            continue
        tag_id = item.get("id")
        if not tag_id or str(tag_id) not in valid_ids:
            continue
        weight = item.get("weight", 1.0)
        try:
            weight = max(0.0, min(1.0, float(weight)))
        except (TypeError, ValueError):
            weight = 1.0
        tags.append(TacticTag(id=str(tag_id), weight=weight))
    return tuple(tags)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI (live network)
    """CLI: `python -m qorgan.data.label --in dialogues.jsonl --out labeled.jsonl`.

    Independently re-labels an existing JSONL corpus and writes the re-labeled copy.
    """
    from qorgan.data.generate import write_dialogues_jsonl
    from qorgan.llm_tools import build_client

    parser = argparse.ArgumentParser(description="Independently re-label a dialogue corpus.")
    parser.add_argument("--in", dest="in_path", type=Path, required=True, help="Input JSONL")
    parser.add_argument("--out", dest="out_path", type=Path, required=True, help="Output JSONL")
    args = parser.parse_args(argv)

    lines = [ln for ln in args.in_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    dialogues = [Dialogue.model_validate_json(ln) for ln in lines]
    client = build_client(get_config().gemini_api_key)
    relabeled = label_corpus(dialogues, client=client)
    write_dialogues_jsonl(relabeled, args.out_path)
    print(f"re-labeled {len(relabeled)} dialogues -> {args.out_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
