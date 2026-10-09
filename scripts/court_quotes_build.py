"""Build the local `court_quotes` split from saved verdict pages (ADR D68).

    python scripts/court_quotes_build.py [--collected-on 2026-10-09]

Inputs (all under data/raw/court_quotes/, gitignored):
  html/<id>.html   sudact.ru /regular/doc/<id>/ pages, fetched politely (robots.txt-allowed,
                   >= 12 s apart, identified user agent; discovery via a web search engine,
                   because sudact's own search endpoint is robots-disallowed)
  kz_sources.json  short court press-release passages from sud.gov.kz (re-use with a hyperlink)
Outputs:
  data/raw/court_quotes/quotes.jsonl          the split (text; local only)
  data/manifests/court_quotes.manifest.json   counts + citations, no text (committed)

`data/manifests/court_quotes.exclusions.json` lists rows a human review removed because they
are not caller speech (sha256 prefix of the text + reason; no text). Exclusions never add or
edit text, so they cannot leak held-out content into anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from qorgan.config import get_config
from qorgan.data.court_quotes import (
    MANIFEST_RELPATH,
    QUOTES_FILENAME,
    RAW_SUBDIR,
    QuoteRow,
    VerdictMeta,
    build_manifest,
    build_rows,
    parse_sudact,
    write_rows,
)

EXCLUSIONS_RELPATH = Path("data") / "manifests" / "court_quotes.exclusions.json"


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _sudact_rows(html_dir: Path) -> tuple[list[QuoteRow], int, int]:
    rows: list[QuoteRow] = []
    dropped = read = 0
    for page in sorted(html_dir.glob("*.html")):
        read += 1
        meta, text = parse_sudact(page.read_text(encoding="utf-8", errors="ignore"), page.stem)
        verdict_rows, verdict_dropped = build_rows(meta, text, language="ru")
        rows.extend(verdict_rows)
        dropped += verdict_dropped
    return rows, dropped, read


def _kz_rows(path: Path) -> tuple[list[QuoteRow], int, int]:
    if not path.exists():
        return [], 0, 0
    rows: list[QuoteRow] = []
    dropped = 0
    sources = json.loads(path.read_text(encoding="utf-8"))
    for source in sources:
        meta = VerdictMeta(
            verdict_id=source["verdict_id"], source=source["source"], url=source["url"],
            court=source["court"], date=source["date"], title=source["title"],
        )
        verdict_rows, verdict_dropped = build_rows(meta, source["text"], language=source.get("language", "ru"))
        rows.extend(verdict_rows)
        dropped += verdict_dropped
    return rows, dropped, len(sources)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the local court_quotes split (ADR D68).")
    parser.add_argument("--collected-on", default=date.today().isoformat())
    args = parser.parse_args(argv)

    cfg = get_config()
    raw = cfg.data_dir / RAW_SUBDIR
    exclusions_path = cfg.repo_root / EXCLUSIONS_RELPATH
    excluded = {e["key"] for e in json.loads(exclusions_path.read_text(encoding="utf-8"))} if exclusions_path.exists() else set()

    ru_rows, ru_dropped, ru_read = _sudact_rows(raw / "html")
    kz_rows, kz_dropped, kz_read = _kz_rows(raw / "kz_sources.json")
    candidates = ru_rows + kz_rows
    seen: set[str] = set()
    rows: list[QuoteRow] = []
    for row in candidates:
        key = text_key(row.text)
        if key in excluded or key in seen:
            continue
        seen.add(key)
        rows.append(row)

    write_rows(rows, raw / QUOTES_FILENAME)
    manifest = build_manifest(
        rows, collected_on=args.collected_on, dropped_pii=ru_dropped + kz_dropped, verdicts_read=ru_read + kz_read
    )
    manifest["candidates_before_review"] = len(candidates)
    manifest["excluded_by_review"] = sum(1 for r in candidates if text_key(r.text) in excluded)
    manifest_path = cfg.repo_root / MANIFEST_RELPATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(
        f"court_quotes: {len(rows)} rows from {manifest['verdicts_with_quotes']} of {manifest['verdicts_read']} "
        f"sources ({len(candidates)} candidates, {manifest['excluded_by_review']} excluded by review, "
        f"{manifest['dropped_residual_pii']} dropped for residual PII) -> {raw / QUOTES_FILENAME}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
