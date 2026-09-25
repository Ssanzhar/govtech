"""Tests for `qorgan.asr.web_models` (PLAN B9): USTAR tarballs Vosklet accepts, built
from the cached model dir or a downloaded zip, macOS metadata skipped."""

from __future__ import annotations

import io
import tarfile
import zipfile

import pytest

from qorgan.asr.web_models import VOSKLET_FILES, ensure_vosk_model_tarball, ensure_vosklet_runtime, package_vosk_model


def _fake_model(root, name="vosk-model-small-xx-0.1"):
    model = root / name
    (model / "conf").mkdir(parents=True)
    (model / "am").mkdir()
    (model / "graph").mkdir()
    (model / "conf" / "model.conf").write_text("--sample-frequency=16000\n", encoding="utf-8")
    (model / "am" / "final.mdl").write_bytes(b"\x00" * 64)
    (model / "graph" / "HCLr.fst").write_bytes(b"\x01" * 32)
    (model / "README").write_text("fake\n", encoding="utf-8")
    (model / "._README").write_bytes(b"apple double")  # macOS metadata
    (model / ".DS_Store").write_bytes(b"ds")
    return model


def test_package_writes_a_ustar_archive_rooted_at_the_model_name(tmp_path):
    model = _fake_model(tmp_path / "cache")
    target = package_vosk_model(model, tmp_path / "site" / "vosk" / "m.tar.gz")

    with tarfile.open(target) as archive:
        members = archive.getmembers()
        # USTAR: only plain files and directories, no PAX extended headers (Vosklet rejects them)
        assert {m.type for m in members} <= {tarfile.REGTYPE, tarfile.DIRTYPE}
        names = [m.name for m in members]
    assert names[0] == "vosk-model-small-xx-0.1" and members[0].isdir()  # root entry first (Vosklet needs it)
    assert names[1] == "vosk-model-small-xx-0.1/README"
    assert "vosk-model-small-xx-0.1/conf/model.conf" in names and "vosk-model-small-xx-0.1/am/final.mdl" in names
    assert not any("._" in n or ".DS_Store" in n for n in names)
    assert all(n.startswith("vosk-model-small-xx-0.1") for n in names)


def test_package_refuses_a_dir_that_is_not_a_model(tmp_path):
    (tmp_path / "nope").mkdir()
    with pytest.raises(ValueError):
        package_vosk_model(tmp_path / "nope", tmp_path / "out.tar.gz")


def test_ensure_uses_the_cache_and_is_idempotent(tmp_path):
    cache = tmp_path / "cache"
    _fake_model(cache)
    calls = []
    target = ensure_vosk_model_tarball("vosk-model-small-xx-0.1", site_models_dir=tmp_path / "site", cache_dir=cache, downloader=lambda url: calls.append(url) or b"")
    assert target == tmp_path / "site" / "vosk" / "vosk-model-small-xx-0.1.tar.gz" and target.exists()
    assert calls == []
    stamp = target.stat().st_mtime_ns
    assert ensure_vosk_model_tarball("vosk-model-small-xx-0.1", site_models_dir=tmp_path / "site", cache_dir=cache, downloader=lambda url: b"") == target
    assert target.stat().st_mtime_ns == stamp


def test_ensure_downloads_and_extracts_the_official_zip_when_not_cached(tmp_path):
    staging = tmp_path / "staging"
    _fake_model(staging)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipped:
        for path in sorted((staging / "vosk-model-small-xx-0.1").rglob("*")):
            if path.is_file():
                zipped.write(path, arcname=str(path.relative_to(staging)))
    urls = []

    def downloader(url):
        urls.append(url)
        return buffer.getvalue()

    target = ensure_vosk_model_tarball("vosk-model-small-xx-0.1", site_models_dir=tmp_path / "site", cache_dir=tmp_path / "cache", downloader=downloader)

    assert urls == ["https://alphacephei.com/vosk/models/vosk-model-small-xx-0.1.zip"]
    assert (tmp_path / "cache" / "vosk-model-small-xx-0.1" / "am" / "final.mdl").exists()
    with tarfile.open(target) as archive:
        assert "vosk-model-small-xx-0.1/conf/model.conf" in archive.getnames()


def test_ensure_rejects_zip_slip(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipped:
        zipped.writestr("../evil.txt", "x")
    with pytest.raises(RuntimeError):
        ensure_vosk_model_tarball("vosk-model-small-xx-0.1", site_models_dir=tmp_path / "site", cache_dir=tmp_path / "cache", downloader=lambda url: buffer.getvalue())


def test_vosklet_runtime_is_pinned_by_hash(tmp_path, monkeypatch):
    import hashlib

    good = {name: b"payload-" + name.encode() for name in VOSKLET_FILES}
    monkeypatch.setattr("qorgan.asr.web_models.VOSKLET_FILES", {n: hashlib.sha256(b).hexdigest() for n, b in good.items()})
    urls = []
    target = ensure_vosklet_runtime(tmp_path / "site", downloader=lambda url: urls.append(url) or good[url.rsplit("/", 1)[1]])
    assert (target / "Vosklet.js").read_bytes() == good["Vosklet.js"] and (target / "Vosklet.wasm").exists()
    assert all(u.startswith("https://cdn.jsdelivr.net/gh/msqr1/Vosklet@") for u in urls)
    # already present with the right hash: no download
    assert ensure_vosklet_runtime(tmp_path / "site", downloader=lambda url: (_ for _ in ()).throw(AssertionError("fetched"))) == target
    # a tampered file is refused and nothing is written
    with pytest.raises(RuntimeError):
        ensure_vosklet_runtime(tmp_path / "other", downloader=lambda url: b"tampered")
    assert not (tmp_path / "other" / "vendor").exists()
