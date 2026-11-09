"""The Android wrappers stay buildable and point at pages the site really serves (ADR D87).

`android/twa/twa-manifest.json` feeds Bubblewrap, and `site/.well-known/assetlinks.json` is what
Chrome reads to trust the app. Neither is built in CI (no Android SDK there), so these checks pin the
facts a TWA build depends on.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
TWA_MANIFEST = ROOT / "android" / "twa" / "twa-manifest.json"
ASSET_LINKS = ROOT / "site" / ".well-known" / "assetlinks.json"
SITE = ROOT / "site"

# com.google.androidbrowserhelper 2.x (what Bubblewrap generates) declares minSdk 24; a lower
# value fails the manifest merger, so the app cannot be built at all.
BROWSER_HELPER_MIN_SDK = 24


def _twa() -> dict:
    return json.loads(TWA_MANIFEST.read_text(encoding="utf-8"))


def _site_path(url_or_path: str) -> Path:
    path = urlparse(url_or_path).path or url_or_path
    return SITE / path.lstrip("/")


def test_twa_min_sdk_is_buildable_with_android_browser_helper():
    assert _twa()["minSdkVersion"] >= BROWSER_HELPER_MIN_SDK


def test_asset_links_name_the_twa_package():
    links = json.loads(ASSET_LINKS.read_text(encoding="utf-8"))
    packages = {entry["target"]["package_name"] for entry in links}
    assert packages == {_twa()["packageId"]}


def test_twa_urls_point_at_files_the_site_serves():
    twa = _twa()
    for key in ("startUrl", "iconUrl", "webManifestUrl"):
        assert _site_path(twa[key]).is_file(), f"{key}={twa[key]} is not a file under site/"


def test_twa_urls_share_one_host():
    twa = _twa()
    hosts = {urlparse(twa[key]).hostname for key in ("iconUrl", "webManifestUrl", "fullScopeUrl")}
    assert hosts == {twa["host"]}


def test_signing_key_is_not_in_the_repository():
    keystore = TWA_MANIFEST.parent / Path(_twa()["signingKey"]["path"]).name
    assert not keystore.exists()
