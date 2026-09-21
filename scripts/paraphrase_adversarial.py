"""Build the `adversarial` eval split: every scam in test + ood paraphrased by Gemini so
that NO hard-signal cue survives (PLAN_2026-09 A9, ADR D15). Build-time only; needs
GEMINI_API_KEY. Writes the committed source `data/adversarial/adversarial.jsonl` (+ a
manifest with the lexicon hash, model, counts and failures) and copies it into
`data/processed/` for the harness. Verification is local (`features`-equivalent cue
matching); the model is never trusted to have complied.

Run: `python scripts/paraphrase_adversarial.py [--style cue_free|legit_sounding] [--max-attempts 3] [--limit N]`
Then: `QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.adversarial`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from qorgan.classifier.cue_lexicon import lexicon_hash, load_cue_lexicon
from qorgan.config import get_config
from qorgan.data.adversarial import (
    ADVERSARIAL_ID_PREFIX,
    DEFAULT_MAX_ATTEMPTS,
    PARAPHRASE_STYLES,
    ParaphraseFailure,
    build_paraphrase_prompt,
    paraphrase_dialogue,
    source_positives,
    split_name_for,
)
from qorgan.data.schema import Dialogue
from qorgan.eval.adversarial import SOURCE_SPLITS
from qorgan.eval.run import load_split
from qorgan.llm_tools import LLMResponseError, build_client, generate_json, thinking_budget_for

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "data" / "adversarial"
_MAX_TOKENS = 4096
_TRANSIENT_RETRIES = 3
_BACKOFF_SECONDS = 2.0
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "utterances": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"speaker": {"type": "string"}, "text": {"type": "string"}},
                "required": ["speaker", "text"],
            },
        }
    },
    "required": ["utterances"],
}


def _gemini_paraphraser(client, model: str):
    def paraphrase(prompt: str) -> dict:
        for attempt in range(1, _TRANSIENT_RETRIES + 1):
            try:
                return generate_json(
                    client, model=model, prompt=prompt, response_schema=_RESPONSE_SCHEMA,
                    max_output_tokens=_MAX_TOKENS, thinking_budget=thinking_budget_for(model),
                )
            except LLMResponseError:
                return {}  # a malformed answer is one failed attempt, not a crash
            except Exception as exc:  # noqa: BLE001 - network/quota: back off, then give up this attempt
                if attempt == _TRANSIENT_RETRIES:
                    print(f"  transient error, giving up this attempt: {exc}", file=sys.stderr)
                    return {}
                time.sleep(_BACKOFF_SECONDS * attempt)
        return {}

    return paraphrase


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paraphrase test+ood scams to avoid every lexicon cue.")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--limit", type=int, default=None, help="only the first N sources (smoke)")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--style", choices=PARAPHRASE_STYLES, default="cue_free", help="cue_free (A9) or legit_sounding (A9b)")
    args = parser.parse_args(argv)
    split = split_name_for(args.style)

    cfg = get_config()
    if not cfg.gemini_api_key:
        print("GEMINI_API_KEY is not set", file=sys.stderr)
        return 2
    processed = args.processed_dir or cfg.data_dir / "processed"
    lexicon = load_cue_lexicon()
    split_of = {d.id: name for name in SOURCE_SPLITS for d in source_positives(load_split(processed, name))}
    sources = [d for name in SOURCE_SPLITS for d in source_positives(load_split(processed, name))]
    if args.limit:
        sources = sources[: args.limit]
    model = cfg.llm_model_bulk
    paraphrase = _gemini_paraphraser(build_client(cfg.gemini_api_key), model)
    print(f"{len(sources)} source scams from {SOURCE_SPLITS}; model {model}; lexicon {lexicon_hash(lexicon)[:12]}")

    produced: list[Dialogue] = []
    failures: list[str] = []
    started = time.perf_counter()
    for index, source in enumerate(sources, 1):
        try:
            produced.append(paraphrase_dialogue(source, lexicon, paraphrase=paraphrase, max_attempts=args.max_attempts, style=args.style))
            status = "ok"
        except ParaphraseFailure as exc:
            failures.append(source.id)
            status = f"FAILED ({exc})"
        print(f"[{index}/{len(sources)}] {source.id} ({source.language}) {status}", flush=True)

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    payload = "".join(d.model_dump_json() + "\n" for d in produced)
    (SOURCE_DIR / f"{split}.jsonl").write_text(payload, encoding="utf-8")
    (processed / f"{split}.jsonl").write_text(payload, encoding="utf-8")
    prompt_hash = hashlib.sha256(build_paraphrase_prompt(sources[0], lexicon, style=args.style).encode("utf-8")).hexdigest() if sources else None
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "style": args.style,
        "split": split,
        "model": model,
        "lexicon_hash": lexicon_hash(lexicon),
        "prompt_sha256_first_source": prompt_hash,
        "max_attempts": args.max_attempts,
        "sources": {name: sum(1 for d in produced if split_of.get(d.id[len(ADVERSARIAL_ID_PREFIX):]) == name) for name in SOURCE_SPLITS},
        "produced": len(produced),
        "failed": failures,
        "by_language": {lang: sum(1 for d in produced if d.language == lang) for lang in ("ru", "kk", "mixed")},
        "seconds": round(time.perf_counter() - started, 1),
    }
    manifest_name = "manifest.json" if args.style == "cue_free" else f"manifest_{split}.json"
    (SOURCE_DIR / manifest_name).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"produced {len(produced)}, failed {len(failures)} -> {SOURCE_DIR / (split + '.jsonl')} (+ processed copy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
