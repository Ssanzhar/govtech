"""Is this deployment what the repo says, and ready to demo? Read-only: GET and HEAD only.

    python scripts/preflight.py                          # the public demo (DEFAULT_BASE)
    python scripts/preflight.py --base http://127.0.0.1:8000

Prints one line per check and exits non-zero if any fails. A warning does not fail: stale demo
data looks bad on a chart but breaks nothing. What it looks at: the three pages, the health route,
the headers the voice mode needs, every model and runtime file (HEAD, nothing large is downloaded),
that the served weights, client config and service worker are byte-for-byte this checkout's, and
how fresh the analyst console's demo data is. Standard library only, so it runs before any install.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "https://govtech-production.up.railway.app"
TIMEOUT_SECONDS = 30
USER_AGENT = "qorgan-preflight"
# The console's chart covers 30 days and its trend compares the last 7 with the 7 before: older
# seeds show a falling line and "0 this week". Redeploying reseeds (ADR D60).
FRESH_DATA_DAYS = 7

PAGES = ("/", "/live.html", "/admin.html")
# These must carry the headers SharedArrayBuffer and the worker scripts need (api_limits.py).
ISOLATED_PATHS = ("/live.html", "/core/device.js", "/core/embed-worker.js")
# (path, why): served by HEAD so nothing large moves.
ASSETS = (
    ("/fonts/inter-latin-wght-normal.woff2", "self-hosted fonts"),
    ("/vendor/vosklet/Vosklet.js", "quick speech recogniser runtime"),
    ("/models/vosk/vosk-model-small-kz-0.42.tar.gz", "Kazakh quick speech model"),
    ("/models/vosk/vosk-model-small-ru-0.22.tar.gz", "Russian quick speech model"),
    ("/vendor/sherpa-onnx/1.13.8/sherpa-onnx-asr.js", "GigaAM runtime"),
    ("/vendor/sherpa-onnx/1.13.8/sherpa-onnx-wasm-web.wasm", "GigaAM runtime"),
    ("/models/gigaam/gigaam-multilingual-220m-ctc-int8/tokens.txt", "GigaAM tokens"),
    ("/models/gigaam/gigaam-multilingual-220m-ctc-int8/model.int8.onnx", "GigaAM model"),
    ("/models/Xenova/multilingual-e5-base/onnx/model_quantized.onnx", "embedding model"),
)
# Must be byte-for-byte this checkout: a difference means the deployed build is another commit.
SAME_AS_CHECKOUT = ("/models/weights.json", "/core/qorgan-config.json", "/sw.js", "/i18n.js")


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict[str, str]
    body: bytes = b""


@dataclass(frozen=True)
class Check:
    name: str
    level: str  # ok | warn | fail
    detail: str = ""


Fetch = Callable[[str, str], Response]


def http_fetch(url: str, method: str = "GET") -> Response:
    request = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as answer:  # noqa: S310 -- the operator's own URL
            return Response(answer.status, {k.lower(): v for k, v in answer.headers.items()}, answer.read() if method == "GET" else b"")
    except urllib.error.HTTPError as exc:
        return Response(exc.code, {k.lower(): v for k, v in exc.headers.items()})
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return Response(0, {"x-error": str(exc)})


def _status(path: str, response: Response, want: int = 200) -> Check:
    if response.status == want:
        return Check(path, "ok")
    return Check(path, "fail", response.headers.get("x-error") or f"HTTP {response.status}, wanted {want}")


def check_pages(base: str, fetch: Fetch) -> list[Check]:
    return [_status(path, fetch(base + path, "GET")) for path in PAGES]


def check_health(base: str, fetch: Fetch) -> list[Check]:
    response = fetch(f"{base}/api/health", "GET")
    if response.status != 200:
        return [_status("/api/health", response)]
    try:
        body = json.loads(response.body)
    except ValueError:
        return [Check("/api/health", "fail", "not JSON")]
    checks = [Check("health: status", "ok" if body.get("status") == "ok" else "fail", str(body.get("status")))]
    source = body.get("linear_model_source")
    checks.append(Check(
        "health: serving the linear model",
        "ok" if body.get("configured_backend") == "linear" and source in ("web", "bundle") else "fail",
        f"backend {body.get('configured_backend')}, source {source}, threshold {body.get('threshold')}",
    ))
    if body.get("commit"):
        checks.append(Check("health: deployed commit", "ok", str(body["commit"])))
    return checks


def check_isolation(base: str, fetch: Fetch) -> list[Check]:
    checks = []
    for path in ISOLATED_PATHS:
        headers = fetch(base + path, "HEAD").headers
        good = headers.get("cross-origin-opener-policy") == "same-origin" and headers.get("cross-origin-embedder-policy") == "require-corp"
        checks.append(Check(f"isolation headers on {path}", "ok" if good else "fail", "" if good else "the full voice mode cannot start"))
    return checks


def check_assets(base: str, fetch: Fetch) -> list[Check]:
    checks = []
    for path, why in ASSETS:
        response = fetch(base + path, "HEAD")
        size = response.headers.get("content-length")
        detail = f"{why}, {int(size) / 1e6:.1f} MB" if response.status == 200 and size else why
        checks.append(Check(path, "ok" if response.status == 200 else "fail", detail if response.status == 200 else f"{why}: HTTP {response.status}"))
    return checks


def check_same_as_checkout(base: str, fetch: Fetch, repo: Path = REPO) -> list[Check]:
    checks = []
    for path in SAME_AS_CHECKOUT:
        local = (repo / "site" / path.lstrip("/")).read_bytes()
        response = fetch(base + path, "GET")
        if response.status != 200:
            checks.append(_status(path, response))
            continue
        same = hashlib.sha256(response.body).hexdigest() == hashlib.sha256(local).hexdigest()
        checks.append(Check(f"same as this checkout: {path}", "ok" if same else "fail", "" if same else "the deployed build is another commit"))
    return checks


def check_demo_data(base: str, fetch: Fetch, *, today: date | None = None) -> list[Check]:
    """Only where the console is open to visitors; a keyed deployment answers 401 and is skipped."""
    today = today or datetime.now(UTC).date()
    session = fetch(f"{base}/api/admin/session", "GET")
    if session.status != 200:
        return [Check("analyst console", "ok", f"closed to visitors (HTTP {session.status}): demo data not checked")]
    response = fetch(f"{base}/api/admin/overview?locale=ru", "GET")
    if response.status != 200:
        return [_status("/api/admin/overview", response)]
    overview = json.loads(response.body)
    if not overview.get("available"):
        return [Check("demo data", "fail", "no analysis yet: a server that just started seeds for a few minutes")]
    organizations = overview.get("organizations", [])
    checks = [Check("demo data", "ok" if organizations else "fail", f"{overview['kpis']['incidents']} incidents, {len(organizations)} organizations")]
    dates = [o["last_activity"] for o in organizations if o.get("last_activity")]
    if dates:
        age = today - date.fromisoformat(max(dates))
        stale = age > timedelta(days=FRESH_DATA_DAYS)
        checks.append(Check(
            "demo data is fresh",
            "warn" if stale else "ok",
            f"newest incident {age.days} days ago" + (": the console's trend will read 0 this week; redeploy to reseed" if stale else ""),
        ))
    return checks


CHECKS = (check_pages, check_health, check_isolation, check_assets, check_same_as_checkout, check_demo_data)


def run(base: str, fetch: Fetch = http_fetch) -> list[Check]:
    base = base.rstrip("/")
    return [check for run_checks in CHECKS for check in run_checks(base, fetch)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=DEFAULT_BASE, help=f"deployment to check (default {DEFAULT_BASE})")
    args = parser.parse_args(argv)
    checks = run(args.base)
    marks = {"ok": "  ok ", "warn": "WARN ", "fail": "FAIL "}
    for check in checks:
        print(f"{marks[check.level]} {check.name}{f'  ({check.detail})' if check.detail else ''}")
    failed = sum(check.level == "fail" for check in checks)
    warned = sum(check.level == "warn" for check in checks)
    print(f"\n{args.base}: {len(checks) - failed - warned} ok, {warned} warnings, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
