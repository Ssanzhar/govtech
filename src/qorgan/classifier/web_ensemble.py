"""A two-model risk ensemble as one browser weights file (ADR D75).

The exported risk head is the mean over calibrated CV members (`web_bundle`), so the equal
mix of two heads that have the same member count and read the same features is simply both
member lists in one head. `WebScorer` and `site/core/head.js` score the result unchanged,
so the browser needs no new code and parity holds by construction.

The base (the shipped v1) keeps the alert line calm on family and everyday calls; the
partner (a retrain) brings the recall, and its tactic head and label space are kept whole.

    python -m qorgan.classifier.web_ensemble --base site/models/weights.json \\
        --partner models/linear_v31_device/web/weights.json --out models/linear_v31e/web/weights.json \\
        --enter 0.66 --exit 0.56
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Fields that define what the risk head reads; both heads must agree on every one.
_INPUT_FIELDS = (
    "format_version", "embed_model_name", "embed_backend", "embedding_dim",
    "feature_order", "feature_version", "cue_matcher_version",
)
ENSEMBLE_KEY = "risk_ensemble"


class EnsembleMismatchError(ValueError):
    """The two weights files cannot be mixed member-for-member."""


def fuse_risk_heads(
    base: Mapping[str, Any], partner: Mapping[str, Any], *, sources: Mapping[str, str],
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """The partner's weights with the risk head replaced by the 50/50 mix of both heads.

    Returns a new dict; neither input is changed. `sources` names each input (e.g. its
    sha256) and is recorded in the output. `thresholds` (the ensemble's own operating point,
    chosen on validation) replaces the partner's; by default the partner's are kept."""
    _check_compatible(base, partner)
    base_members = base["risk_head"]["members"]
    partner_members = partner["risk_head"]["members"]
    fused = copy.deepcopy(dict(partner))
    fused["risk_head"] = {**fused["risk_head"], "members": copy.deepcopy(base_members) + copy.deepcopy(partner_members)}
    fused[ENSEMBLE_KEY] = {
        "weights": {"base": 0.5, "partner": 0.5},
        "members": {"base": len(base_members), "partner": len(partner_members)},
        "sources": {"base": sources["base"], "partner": sources["partner"]},
    }
    if thresholds is not None:
        fused["thresholds"] = {k: float(v) for k, v in thresholds.items()}
    return fused


def _check_compatible(base: Mapping[str, Any], partner: Mapping[str, Any]) -> None:
    for name, data in (("base", base), ("partner", partner)):
        if ENSEMBLE_KEY in data:
            raise EnsembleMismatchError(f"the {name} is already an ensemble; fuse the original heads")
    for field in _INPUT_FIELDS:
        if base.get(field) != partner.get(field):
            raise EnsembleMismatchError(f"{field} differs: {base.get(field)!r} vs {partner.get(field)!r}")
    if base.get("lexicon") != partner.get("lexicon"):
        raise EnsembleMismatchError("the embedded lexicon differs, so the cue features would differ")
    if base["risk_head"]["type"] != partner["risk_head"]["type"]:
        raise EnsembleMismatchError("risk head types differ")
    counts = len(base["risk_head"]["members"]), len(partner["risk_head"]["members"])
    if counts[0] != counts[1] or counts[0] == 0:
        raise EnsembleMismatchError(f"member counts {counts} differ, so the mix would not be 50/50")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description="Fuse two browser risk heads 50/50 (ADR D75).")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--partner", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--enter", type=float, default=None, help="the ensemble's alert point (exit = --exit)")
    parser.add_argument("--exit", type=float, default=None)
    args = parser.parse_args(argv)
    if (args.enter is None) != (args.exit is None):
        parser.error("--enter and --exit go together")
    thresholds = None if args.enter is None else {"risk": args.enter, "enter": args.enter, "exit": args.exit}
    base = json.loads(args.base.read_text(encoding="utf-8"))
    partner = json.loads(args.partner.read_text(encoding="utf-8"))
    fused = fuse_risk_heads(base, partner, sources={"base": _sha256(args.base), "partner": _sha256(args.partner)},
                            thresholds=thresholds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Same serialisation as `web_bundle.write_web_bundle`.
    args.out.write_text(json.dumps(fused, sort_keys=True, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"fused -> {args.out} sha256 {_sha256(args.out)}")


if __name__ == "__main__":  # pragma: no cover
    main()
