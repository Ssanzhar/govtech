"""Training augmentation against the legit-sounding adversary (PLAN_2026-09 A9b → A9c).

`adversarial_legit` (paraphrases of test + ood scams in a calm institutional register, with
the reassurances a real bank gives) cut recall from 0.899 to 0.651 (2026-09-18). The fix
that follows the sprint lesson ("pair a feature with data, never tune the feature alone")
is training data of the same style, built ONLY from TRAIN scams so the evaluation stays
clean: `data/augment/legit_style_scams.jsonl`, folded into train by `build_corpus`.

Run: `python scripts/augment_legit_style_scams.py [--limit N] [--max-attempts 3]`
Then: build_corpus → linear_train → gates (eval.run, eval.stream, eval.adversarial both splits,
npm test) — rollback `models/linear_prev` if any FPR gate moves.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from qorgan.classifier.cue_lexicon import lexicon_hash, load_cue_lexicon
from qorgan.config import get_config
from qorgan.data.adversarial import DEFAULT_MAX_ATTEMPTS, ParaphraseFailure, paraphrase_dialogue, source_positives
from qorgan.data.schema import Dialogue
from qorgan.eval.run import load_split
from qorgan.llm_tools import build_client

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "data" / "augment" / "legit_style_scams.jsonl"
MANIFEST = REPO_ROOT / "data" / "augment" / "legit_style_scams.manifest.json"
STYLE = "legit_sounding"
ID_PREFIX = "aug-legit-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Legit-sounding paraphrases of TRAIN scams as training augmentation.")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = get_config()
    if not cfg.gemini_api_key:
        print("GEMINI_API_KEY is not set", file=sys.stderr)
        return 2
    from paraphrase_adversarial import _gemini_paraphraser  # same transient-error handling

    processed = args.processed_dir or cfg.data_dir / "processed"
    lexicon = load_cue_lexicon()
    # Only genuine train scams: augment rows already in train are skipped so we never paraphrase a paraphrase.
    sources = [d for d in source_positives(load_split(processed, "train")) if not d.id.startswith("aug")]
    if args.limit:
        sources = sources[: args.limit]
    paraphrase = _gemini_paraphraser(build_client(cfg.gemini_api_key), cfg.llm_model_bulk)
    print(f"{len(sources)} train scams; style {STYLE}; model {cfg.llm_model_bulk}; lexicon {lexicon_hash(lexicon)[:12]}")

    produced: list[Dialogue] = []
    failures: list[str] = []
    started = time.perf_counter()
    for index, source in enumerate(sources, 1):
        try:
            dialogue = paraphrase_dialogue(source, lexicon, paraphrase=paraphrase, max_attempts=args.max_attempts, style=STYLE)
            produced.append(dialogue.model_copy(update={"id": f"{ID_PREFIX}{source.id}"}))
            status = "ok"
        except ParaphraseFailure as exc:
            failures.append(source.id)
            status = f"FAILED ({exc})"
        print(f"[{index}/{len(sources)}] {source.id} ({source.language}) {status}", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(d.model_dump_json() + "\n" for d in produced), encoding="utf-8")
    MANIFEST.write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(), "style": STYLE, "source_split": "train",
        "model": cfg.llm_model_bulk, "lexicon_hash": lexicon_hash(lexicon), "produced": len(produced), "failed": failures,
        "by_language": {lang: sum(1 for d in produced if d.language == lang) for lang in ("ru", "kk", "mixed")},
        "seconds": round(time.perf_counter() - started, 1),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"produced {len(produced)}, failed {len(failures)} -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
