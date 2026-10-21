"""GigaAM-Multilingual 220M decodes of the span-vote DEV set (ADR D70, endpoint-pass tag inheritance).

The TEST set already has them (`cs_decodes.jsonl → gigaam220`); DEV did not, so nothing about how
GigaAM's words inherit the live tags could be chosen without looking at TEST. Same decoder as
`decode_cs.py` (sherpa-onnx, one thread, greedy CTC).

    python scripts/spikes/asr_wordlevel/spanvote/decode_dev_gigaam.py   # -> data/asr_capture/spanvote_dev_gigaam.jsonl

Resumable: rows already in the output are skipped.
"""

from __future__ import annotations

import json
import sys

from common import MANIFESTS, REPO, key_of

sys.path.insert(0, str(REPO / "scripts/spikes/asr_wordlevel"))
from decode_cs import GIGAAM_MODELS, decode_gigaam, gigaam_recognizer, read_samples  # noqa: E402

OUT = REPO / "data/asr_capture/spanvote_dev_gigaam.jsonl"


def main() -> None:
    rows = [json.loads(l) for l in MANIFESTS["dev"].read_text(encoding="utf-8").splitlines() if l.strip()]
    done = set()
    if OUT.exists():
        done = {key_of(json.loads(l)) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in rows if key_of(r) not in done]
    print(f"{len(todo)} rows to decode, {len(done)} done", flush=True)
    recognizer = gigaam_recognizer(GIGAAM_MODELS["gigaam220"])
    with OUT.open("a", encoding="utf-8") as handle:
        for i, row in enumerate(todo):
            decoded = decode_gigaam(recognizer, read_samples(REPO / row["wav"]))
            out = {"id": row["id"], "condition": row["condition"], "channel": row["channel"], "gigaam220": decoded}
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
