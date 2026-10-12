"""CPU cost of SetPartialWords: single-thread decode time with partial word confidences on vs off.

Both arms feed 0.25 s chunks and call PartialResult() after every chunk (the browser's Vosklet
emits a partial per chunk either way); only SetPartialWords differs. Cost is process CPU time
(time.process_time), not wall time, so other load on the machine does not inflate it. Arms are
interleaved per row and repeated; each row/model/arm keeps its cheapest run.

    python .../cpu_cost.py [--rows 60 --repeats 3]   # -> data/asr_capture/spanvote_cpu_cost.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ["OMP_NUM_THREADS"] = "1"

from common import LANGS, MANIFESTS, REPO  # noqa: E402
from decode_partials import VOSK_MODELS, decode_one, read_raw  # noqa: E402

OUT = REPO / "data/asr_capture/spanvote_cpu_cost.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    from vosk import Model, SetLogLevel

    SetLogLevel(-1)
    models = {l: Model(model_name=VOSK_MODELS[l]) for l in LANGS}
    rows = [json.loads(l) for l in MANIFESTS["test"].read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[:: max(1, len(rows) // args.rows)][: args.rows]
    audio = sum(r["duration"] for r in rows)
    totals = {l: {"on": 0.0, "off": 0.0} for l in LANGS}
    for row in rows:
        raw = read_raw(REPO / row["wav"])
        for l in LANGS:
            best = {"on": float("inf"), "off": float("inf")}
            for _ in range(args.repeats):
                for arm in ("off", "on"):
                    t0 = time.process_time()
                    decode_one(models[l], raw, partial_words=arm == "on")
                    best[arm] = min(best[arm], time.process_time() - t0)
            for arm in best:
                totals[l][arm] += best[arm]
    result = {"rows": len(rows), "audio_s": round(audio, 2), "repeats": args.repeats,
              "cpu_rtf_1thread": {l: {arm: round(v / audio, 4) for arm, v in t.items()} for l, t in totals.items()},
              "overhead_pct": {l: round(100 * (t["on"] / t["off"] - 1), 1) for l, t in totals.items()}}
    all_on, all_off = sum(t["on"] for t in totals.values()), sum(t["off"] for t in totals.values())
    result["three_models"] = {"rtf_on": round(all_on / audio, 4), "rtf_off": round(all_off / audio, 4),
                              "overhead_pct": round(100 * (all_on / all_off - 1), 1)}
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
