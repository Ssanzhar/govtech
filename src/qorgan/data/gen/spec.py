"""The chunk spec a generating agent receives (ADR D63).

Self-contained: instructions, the output JSON schema, glossaries of tactics / schemes / roles /
negative classes built from the reviewed YAML (descriptions and names only -- never the
`examples_*` seed phrases, never any file under data/lexicon), and the cells to write.
The agent reads this one file and writes one JSONL file; it sees no corpus, no lexicon and
no evaluation data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from qorgan.data.gen.grid import GridCell

SPEC_VERSION = 2  # 2: look-alike negatives (`lookalike_of`), phase 2 of ADR D63

INSTRUCTIONS = (
    "You write realistic phone-call transcripts for a scam-detection training corpus for Kazakhstan. "
    "Write exactly one dialogue per cell, in cell order, as one JSON object per line (JSONL), and nothing else.",
    "Register: kk = Kazakh only; ru = Russian only; mixed = natural Kazakh-Russian code-switching inside "
    "and across turns, as spoken in Kazakhstan; en = English spoken by a resident of Kazakhstan (Kaspi, Halyk, "
    "eGov, tenge, IIN stay local). Write as people actually talk on the phone: fillers, interruptions, "
    "hesitation, short turns, imperfect grammar. No narration, no stage directions, no speaker names inside text.",
    "Speakers are exactly 'caller' and 'callee'. The caller is the person who placed the call.",
    "Length: aim for `length_target` utterances (within +-2).",
    "Scam cells (kind=scam): the caller runs the given `scheme`, claims the given `role` (null = a private "
    "person or an unnamed company), and the dialogue MUST show every tactic in `tactics`; it may show other "
    "tactics from the glossary if natural, and then they must be annotated too. `stage` says which phases "
    "the call covers (hook_only, hook_escalation, full = hook -> escalation -> extraction, extraction_handoff = "
    "the call starts mid-scheme). `phrasing=indirect`: the key request is implied, euphemised or spread over "
    "several turns, never in stock wording -- a careful listener understands it, a keyword filter would not. "
    "`phrasing=direct`: the request is explicit. `outcome` is how the callee ends the call.",
    "Negative cells (kind=legit): a genuinely legitimate call of `negative_class`, caller role as given. It "
    "contains NO scam tactic and the `tactics` lists stay empty. If `confusable` is true, the legitimate "
    "caller says at least one thing that would sound like a scam request out of context but is clearly "
    "legitimate in context (e.g. a courier at the door asks for the code shown in the recipient's own app; "
    "a bank warns never to tell anyone a code; HR asks to keep a reorganisation quiet until Monday).",
    "Look-alike negatives (kind=legit with `lookalike_of` set): the legitimate call is about the SAME subject "
    "as that scheme's pretext (see the scheme glossary) -- the same kind of organisation, the same topic, the "
    "same ordinary vocabulary (a parcel, a clinic appointment, an EDS certificate, a loan application, a "
    "contract, a refund, a job) -- but it makes none of the scheme's asks: no code read out over the phone, "
    "no transfer, no app install, no secrecy, no threat. A reader should need the asks, not the topic, to "
    "tell the two apart. Vary how it ends; the callee may be suspicious at first and then reassured.",
    "Annotation: every utterance has `stage` (hook, escalation, extraction, handoff; `none` for negatives and "
    "small talk) and `tactics`: a list of {id, phrase} where `phrase` is copied CHARACTER-FOR-CHARACTER from "
    "that utterance's text -- the shortest span that carries the tactic (a few words, not the whole turn).",
    "Persona: the callee matches `persona` (age band, region, occupation) in how they speak and what they know.",
    "Privacy: never write a real-looking phone number, card number, IIN or e-mail. If one is needed, write "
    "the placeholder [PHONE], [CARD], [IIN] or [EMAIL]. Amounts in tenge, short codes (4-6 digits) are fine.",
    "Variety: no two dialogues may share an opening line or a sentence. Vary names, banks, cities, amounts, "
    "products and the order of moves. Do not reuse phrasing across cells.",
    "Copy `id`, `cell_id`, `language`, `kind`, `scheme`, `negative_class` and `role` "
    "from the cell exactly.",
)

# Added only to specs whose cells carry a `scenario` (v2.1), so earlier specs stay byte-identical.
SCENARIO_INSTRUCTION = (
    "Scenario (negative cells with `scenario` set): the brief says what the legitimate call is concretely "
    "about. Write that situation, in your own words and details (names, amounts, places, who says what); "
    "never copy the brief's wording into the dialogue."
)

# Added only to specs with a scam cell that carries a `scenario` (v2.1 `scam_scenarios`).
SCAM_SCENARIO_INSTRUCTION = (
    "Scenario (scam cells with `scenario` set): the brief describes a move the scammer makes in this call, on "
    "top of the scheme, tactics, stage and phrasing. Write it in your own words and details; never copy the "
    "brief's wording into the dialogue. The move named in the brief is not itself a tactic: annotate only "
    "the glossary tactics the call shows."
)

# Added only to specs with a cell that carries a `constraint` (a regenerated cell, v2.1).
CONSTRAINT_INSTRUCTION = (
    "Constraint (cells with `constraint` set): the constraint is a hard rule for that one dialogue. It "
    "overrides every other instruction, example and brief, including the confusable example above."
)

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "cell_id", "language", "kind", "scheme", "negative_class", "role", "utterances"],
    "properties": {
        "id": {"type": "string"},
        "cell_id": {"type": "string"},
        "language": {"enum": ["kk", "ru", "mixed", "en"]},
        "kind": {"enum": ["scam", "legit"]},
        "scheme": {"type": ["string", "null"]},
        "negative_class": {"type": ["string", "null"]},
        "role": {"type": ["string", "null"]},
        "utterances": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["speaker", "text", "stage", "tactics"],
                "properties": {
                    "speaker": {"enum": ["caller", "callee"]},
                    "text": {"type": "string"},
                    "stage": {"enum": ["hook", "escalation", "extraction", "handoff", "none"]},
                    "tactics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "phrase"],
                            "properties": {"id": {"type": "string"}, "phrase": {"type": "string"}},
                        },
                    },
                },
            },
        },
    },
}


def glossaries(taxonomy, schemes, roles) -> dict:
    """Names + descriptions only. No `examples_*`, no cue lists (the agent must not see a lexicon)."""
    v2 = taxonomy.label_space("v2")
    return {
        "tactics": {t: {"name": taxonomy.get(t).en, "description": taxonomy.get(t).description} for t in v2},
        "schemes": {s.id: {"name": s.names["en"], "description": s.description} for s in schemes.schemes},
        "roles": {r.id: r.names["en"] for r in roles.roles},
        "negative_classes": {n.id: n.note for n in taxonomy.negatives},
    }


def _cell_json(cell: GridCell) -> dict:
    exclude = ({"run", "hard_negative"} | ({"scenario"} if cell.scenario is None else set())
               | ({"constraint"} if cell.constraint is None else set()))
    return json.loads(cell.model_dump_json(exclude=exclude))


def build_chunk_spec(cells: Sequence[GridCell], *, run: str, chunk: str, taxonomy, schemes, roles) -> dict:
    with_scenarios = any(c.scenario and c.kind == "legit" for c in cells)
    with_scam_scenarios = any(c.scenario and c.kind == "scam" for c in cells)
    with_constraints = any(c.constraint for c in cells)
    return {
        "spec_version": SPEC_VERSION,
        "run": run,
        "chunk": chunk,
        "instructions": list(INSTRUCTIONS) + ([SCENARIO_INSTRUCTION] if with_scenarios else [])
        + ([SCAM_SCENARIO_INSTRUCTION] if with_scam_scenarios else []) + ([CONSTRAINT_INSTRUCTION] if with_constraints else []),
        "output_schema": OUTPUT_SCHEMA,
        "glossary": glossaries(taxonomy, schemes, roles),
        "cells": [_cell_json(c) for c in cells],
    }


def chunk_cells(cells: Sequence[GridCell], size: int) -> list[tuple[str, tuple[GridCell, ...]]]:
    """Split into chunks of `size`, interleaving (cell i -> chunk i % n) so every chunk mixes
    classes and registers instead of one agent writing all the negatives."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    n = max(1, -(-len(cells) // size))
    buckets: list[list[GridCell]] = [[] for _ in range(n)]
    for index, cell in enumerate(cells):
        buckets[index % n].append(cell)
    return [(f"chunk_{i:02d}", tuple(bucket)) for i, bucket in enumerate(buckets)]


def write_spec(spec: dict, path: Path) -> str:
    """Write the spec; returns its sha256 (chunks are idempotent: same cells -> same hash)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(spec, ensure_ascii=False, indent=1, sort_keys=True)
    path.write_text(blob + "\n", encoding="utf-8")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
