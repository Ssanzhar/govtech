"""Generate the `shift_local` held-out calls with a LOCAL open-weights model (ADR D65).

    llama-server -m models/llm/<repo>/<file>.gguf --port 8091 -ngl 99 -c 4096 --jinja
    PYTHONPATH=src python scripts/gen_shift_local.py --generator qwen35_kk
    PYTHONPATH=src python scripts/gen_shift_local.py --generator qwen35_kk --split shift_local_2

Runs every planned slot whose language is written by `--generator` (see
`qorgan.data.shift_local_plan.LANGUAGE_GENERATOR`). Each slot gets up to `MAX_ATTEMPTS`
deterministic seeds; an attempt is accepted or rejected by `shift_local_checks.accept`
ONLY (format / turns / language / echo / loops / duplicates) -- never by a classifier.

`--split` picks a registered batch (`shift_local_plan.LOCAL_SPLITS`); a later batch's duplicate
check also counts the earlier batches' accepted calls of the same language.

Writes under `data/raw/<split>/` (gitignored, never published):
  <language>.jsonl          accepted calls, raw shape read by `shift_set.parse_raw_row`
  meta/attempts.jsonl       one line per attempt: slot, seed, reason, sha256 of the raw output
Resumable: slots already accepted, or already out of attempts, are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from qorgan.data.shift_local_checks import accept
from qorgan.data.shift_local_plan import (
    LOCAL_SPLITS, MAX_ATTEMPTS, MAX_TOKENS, TEMPERATURE, system_prompt, TOP_P, Slot, build_plan, load_briefs, raw_row,
    user_prompt,
)
from qorgan.data.shift_local_set import prior_accepted_texts

_REPO = Path(__file__).resolve().parents[1]
_BRIEFS = _REPO / "configs" / "shift_local_briefs.yaml"
_RAW = _REPO / "data" / "raw"
_REQUEST_TIMEOUT_S = 900


def _complete(server: str, slot: Slot, attempt: int) -> str:
    body = {
        "messages": [{"role": "system", "content": system_prompt(slot.language)}, {"role": "user", "content": user_prompt(slot)}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": slot.attempt_seed(attempt),
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_prompt": False,
    }
    request = urllib.request.Request(
        f"{server}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_S) as response:
        payload = json.loads(response.read())
    return payload["choices"][0]["message"]["content"] or ""


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(
    server: str, generator: str, out_dir: Path, limit: int | None, plan_seed: int,
    languages: list[str] | None = None, split: str = "shift_local",
) -> None:
    spec = LOCAL_SPLITS[split]
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = out_dir / "meta" / "attempts.jsonl"  # a subdir: the split builder globs *.jsonl
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    past = _read_jsonl(attempts_path)
    planned = build_plan(load_briefs(_BRIEFS), seed=plan_seed, targets=spec.language_targets, prefix=spec.slot_prefix)
    slots = [s for s in planned if s.generator == generator and (not languages or s.language in languages)]
    if limit:  # a smoke test samples both labels: the plan lists each language's scams first
        slots = slots[::max(1, len(slots) // limit)][:limit]
    for slot in slots:
        lang_path = out_dir / f"{slot.language}.jsonl"
        done = {r["id"] for r in _read_jsonl(lang_path)}
        tried = [a for a in past if a["slot"] == slot.slot_id]
        if slot.slot_id in done or len(tried) >= MAX_ATTEMPTS:
            continue
        for attempt in range(len(tried), MAX_ATTEMPTS):
            accepted_texts = [
                *prior_accepted_texts(raw_root=_RAW, split=split, language=slot.language),
                *("\n".join(u["text"] for u in r["utterances"]) for r in _read_jsonl(lang_path)),
            ]
            started = time.monotonic()
            try:
                raw = _complete(server, slot, attempt)
            except (urllib.error.URLError, TimeoutError, KeyError) as exc:
                sys.exit(f"{slot.slot_id}: generation failed ({exc}); fix the server and re-run (resumable)")
            prompt = system_prompt(slot.language) + "\n" + user_prompt(slot)
            verdict = accept(raw, language=slot.language, prompt=prompt, accepted=accepted_texts)
            record = {
                "slot": slot.slot_id, "attempt": attempt, "seed": slot.attempt_seed(attempt),
                "generator": generator, "language": slot.language, "label": slot.brief.label,
                "reason": verdict.reason or "accepted", "turns": len(verdict.utterances),
                "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "seconds": round(time.monotonic() - started, 1),
            }
            _append(attempts_path, record)
            past.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if verdict.accepted:
                _append(lang_path, raw_row(slot, verdict.utterances))
                break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generator", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8091")
    parser.add_argument("--split", default="shift_local", choices=sorted(LOCAL_SPLITS))
    parser.add_argument("--out-dir", type=Path, default=None, help="default: data/raw/<split>")
    parser.add_argument("--limit", type=int, default=None, help="only N evenly spaced slots (smoke test)")
    parser.add_argument("--languages", nargs="+", default=None, help="only these languages (parallel runs)")
    parser.add_argument(
        "--plan-seed", type=int, default=None,
        help="a different seed gives different slots and texts: use one for smoke tests a person may read",
    )
    args = parser.parse_args()
    split_seed = LOCAL_SPLITS[args.split].plan_seed
    plan_seed = split_seed if args.plan_seed is None else args.plan_seed
    out_dir = args.out_dir or _RAW / args.split
    if plan_seed != split_seed and any(out_dir.resolve() == (_RAW / name).resolve() for name in LOCAL_SPLITS):
        sys.exit("a non-default --plan-seed must write to another --out-dir (it is a pilot, not the split)")
    run(args.server, args.generator, out_dir, args.limit, plan_seed, args.languages, args.split)


if __name__ == "__main__":
    main()
