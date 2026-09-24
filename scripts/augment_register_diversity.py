"""Widen the training corpus's REGISTER, so the model learns the scam and not the house style.

ADR D35 measured the problem: on 66 calls written by a second generator the shipped model
recalls 8/33, against 0.95 on `test`. Every existing split shares its generator and its
prompts with `train`, so the model had no reason to learn anything but one model's way of
writing a phone call.

This generates fresh scam dialogues and fresh hard negatives through the SAME Gemini pipeline
(`data/generate.py`) but with a **sampled register persona** injected into the existing
`style` hook: who is calling, who is answering, how the call opens, how people actually
speak. The register is widened; the generator is not changed.

    python scripts/augment_register_diversity.py --per-tactic 2 --negatives 45

**Why not write it myself.** The `shift` split is authored by Claude. Training on Claude-
written text would make `shift` a test of the model's own author, destroying the only
cross-generator signal the project has. Gemini under wider prompting keeps `shift` honest.

**Why negatives too.** ADR D27: scam-side augmentation alone brings the old false positives
back (authored FPR 0.083); paired with negatives in the same register it holds every gate.
The counts default to that pairing.

Output: `data/augment/register_diversity{,_negatives}.jsonl` (+ a manifest), folded into
TRAIN only by `build_corpus`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qorgan.config import get_config  # noqa: E402
from qorgan.data.generate import (  # noqa: E402
    generate_dialogue,
    generate_hard_negative,
    load_corpus_config,
    write_dialogues_jsonl,
)
from qorgan.llm_tools import build_client  # noqa: E402
from qorgan.taxonomy import get_taxonomy  # noqa: E402

OUTPUT = REPO_ROOT / "data" / "augment" / "register_diversity.jsonl"
OUTPUT_NEGATIVES = REPO_ROOT / "data" / "augment" / "register_diversity_negatives.jsonl"
MANIFEST = REPO_ROOT / "data" / "augment" / "register_diversity.manifest.json"

# Each axis is sampled independently, so the prompt differs every call. The point is not
# realism in any single axis but VARIANCE: the corpus's own register is one point in here.
CALLER_STYLE = [
    "a polished corporate voice that never stumbles",
    "a rushed, slightly sloppy caller who repeats himself and mangles a word or two",
    "an over-familiar caller who calls the callee by diminutive name and jokes",
    "a bored bureaucrat reading from a script, flat and impatient",
    "an aggressive caller who talks over the callee and raises his voice",
    "a soft, patronising voice speaking slowly as if to a child",
    "a nervous junior employee who apologises and checks notes mid-sentence",
]
CALLEE_STYLE = [
    "an elderly person who mishears and asks the caller to repeat things",
    "a young, impatient person answering between other tasks",
    "a suspicious person who pushes back and asks who exactly is calling",
    "a distracted parent with children audible in the background",
    "a polite, compliant person who agrees with everything",
    "someone driving, with a bad connection, asking 'what? I can't hear you'",
]
OPENING = [
    "the call opens mid-sentence, as if the callee picked up late",
    "the call opens with a long institutional greeting and a case number",
    "the caller has to confirm he reached the right person before anything else",
    "the callee answers expecting someone else entirely",
    "the line is bad at first and the first exchange is about hearing each other",
]
TEXTURE = [
    "include natural filler words, self-corrections and one interruption",
    "keep sentences short and clipped, the way people really speak on the phone",
    "let the callee interrupt the caller at least twice",
    "include one moment where the caller repeats a number because the callee did not catch it",
    "include a short irrelevant aside before the call returns to its point",
]
LENGTH = [
    "Make it SHORT: 4-5 utterances, the call is cut off before it finishes.",
    "Make it LONG: 12-16 utterances, with the callee resisting and being talked round.",
    "",  # the configured default length
]


# Widening the scams' register swamps the reassurance signal the July fix installed: the first
# run of this script put `real_neg_bank_fraud_alert_ru` and `real_neg_telecom_tariff_ru` back
# over the threshold -- the very anchors ADR D27 was written about, at the very same FPR
# (0.083). So a third of the negatives carry the counter-signal IN THE NEW REGISTERS too.
# Reused verbatim from `augment_reassurance_negatives.py` so both paths teach the same thing.
REASSURANCE_STYLE = (
    "CRITICAL REALISM -- fraud-safety reassurance: the legitimate caller must PROACTIVELY "
    "REASSURE the customer at least once, exactly as a security-conscious real institution "
    "does. Naturally and in varied wording, have the caller make clear that the "
    "bank/operator/service will NEVER ask for an SMS/OTP code, card number, CVV, PIN, or "
    "password over the phone; that the customer should never share such data with anyone, "
    "even someone claiming to be staff; that no code or card data is needed to resolve this "
    "matter; and that anything sensitive is handled in person at a branch/office with an ID "
    "document. The caller still NEVER asks for any of those things and never redirects money."
)
REASSURANCE_CATEGORIES = ("legit_bank_call", "legit_gov_service", "delivery_notification")


def style_for(rng: random.Random, *, scam: bool, reassuring: bool = False) -> str:
    """One sampled register instruction for the generator's `style` hook."""
    who = "caller" if scam else "caller (a legitimate one)"
    parts = [
        "REGISTER (follow it closely; do NOT write a tidy, evenly-paced transcript):",
        f"- The {who} is {rng.choice(CALLER_STYLE)}.",
        f"- The callee is {rng.choice(CALLEE_STYLE)}.",
        f"- Opening: {rng.choice(OPENING)}.",
        f"- Texture: {rng.choice(TEXTURE)}.",
        rng.choice(LENGTH),
        "Vary the institution, the city and the names from the obvious defaults.",
    ]
    if reassuring:
        parts.append("")
        parts.append(REASSURANCE_STYLE)
    return "\n".join(p for p in parts if p)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-tactic", type=int, default=2, help="scam dialogues per tactic (spread over languages)")
    parser.add_argument("--negatives", type=int, default=60, help="hard negatives in the same registers")
    parser.add_argument("--reassurance-share", type=float, default=0.4,
                        help="fraction of negatives that also carry the fraud-safety reassurance (ADR D27)")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--dry-run", action="store_true", help="print one sampled prompt and exit")
    args = parser.parse_args(argv)

    cfg = get_config()
    corpus_cfg = load_corpus_config()
    taxonomy = get_taxonomy()
    rng = random.Random(args.seed)
    languages = list(corpus_cfg.languages)

    if args.dry_run:
        print(style_for(rng, scam=True))
        print("\n--- negative ---\n")
        print(style_for(rng, scam=False))
        return 0

    if not cfg.gemini_api_key:
        print("GEMINI_API_KEY is not set; nothing to do.", file=sys.stderr)
        return 1
    client = build_client(cfg.gemini_api_key)

    scams, negatives, failures = [], [], []
    tactics = list(taxonomy.tactics)
    for tactic in tactics:
        for slot in range(args.per_tactic):
            language = languages[(tactics.index(tactic) + slot) % len(languages)]
            style = style_for(rng, scam=True)
            try:
                dialogue = generate_dialogue(
                    tactic.id, language, client=client, cfg=corpus_cfg, style=style,
                    dialogue_id=f"reg_{tactic.id}_{language}_{slot}",
                )
            except Exception as exc:  # noqa: BLE001 - one bad generation must not stop the batch
                failures.append({"kind": "scam", "tactic": tactic.id, "language": language, "error": str(exc)[:200]})
                continue
            scams.append(dialogue)
            print(f"  scam  {tactic.id:32s} {language:5s} {len(dialogue.utterances)} turns", flush=True)

    categories = list(taxonomy.negatives)
    reassuring_categories = [c for c in categories if c.id in REASSURANCE_CATEGORIES] or categories
    for index in range(args.negatives):
        reassuring = (index % 10) < round(args.reassurance_share * 10)
        pool = reassuring_categories if reassuring else categories
        category = pool[index % len(pool)]
        language = languages[index % len(languages)]
        style = style_for(rng, scam=False, reassuring=reassuring)
        try:
            dialogue = generate_hard_negative(
                category.id, language, client=client, cfg=corpus_cfg, style=style,
                dialogue_id=f"reg_neg{'_reassure' if reassuring else ''}_{category.id}_{language}_{index}",
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"kind": "negative", "category": category.id, "language": language, "error": str(exc)[:200]})
            continue
        negatives.append(dialogue)
        print(f"  neg   {category.id:32s} {language:5s} {len(dialogue.utterances)} turns", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_dialogues_jsonl(scams, OUTPUT)
    write_dialogues_jsonl(negatives, OUTPUT_NEGATIVES)
    MANIFEST.write_text(json.dumps({
        "purpose": "register diversity for TRAIN (ADR D35 follow-up): same generator, widened prompting",
        "seed": args.seed,
        "model": cfg.llm_model_bulk,
        "scams": len(scams),
        "negatives": len(negatives),
        "failures": failures,
        "reassurance_share": args.reassurance_share,
        "axes": {"caller": len(CALLER_STYLE), "callee": len(CALLEE_STYLE), "opening": len(OPENING),
                 "texture": len(TEXTURE), "length": len(LENGTH)},
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{len(scams)} scams -> {OUTPUT}\n{len(negatives)} negatives -> {OUTPUT_NEGATIVES}\n{len(failures)} failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
