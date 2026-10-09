"""The GigaAM endpoint pass's self-hosted assets (ADR D64): pinned, hash-checked, never trusted."""

from __future__ import annotations

import hashlib

import pytest

from qorgan.asr import gigaam_web as gw


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def runtime_pins(monkeypatch):
    files = {name: f"// {name}\n".encode() for name in gw.SHERPA_WASM_FILES}
    monkeypatch.setattr(gw, "SHERPA_WASM_FILES", {name: _sha(data) for name, data in files.items()})
    return files


def test_runtime_installs_from_a_local_build_when_every_hash_matches(tmp_path, runtime_pins):
    build = tmp_path / "out"
    build.mkdir()
    for name, data in runtime_pins.items():
        (build / name).write_bytes(data)
    target = gw.ensure_sherpa_runtime(tmp_path / "site", build_dir=build)
    assert target == tmp_path / "site" / gw.SHERPA_VENDOR_SUBDIR
    assert {p.name for p in target.iterdir()} == set(runtime_pins)


def test_runtime_refuses_a_tampered_file(tmp_path, runtime_pins):
    build = tmp_path / "out"
    build.mkdir()
    for name, data in runtime_pins.items():
        (build / name).write_bytes(data + b"evil")
    with pytest.raises(RuntimeError, match="refusing"):
        gw.ensure_sherpa_runtime(tmp_path / "site", build_dir=build)
    assert not (tmp_path / "site" / gw.SHERPA_VENDOR_SUBDIR).exists() or not any((tmp_path / "site" / gw.SHERPA_VENDOR_SUBDIR).iterdir())


def test_runtime_downloads_from_a_mirror_and_checks_hashes(tmp_path, runtime_pins):
    seen = []

    def fake(url: str) -> bytes:
        seen.append(url)
        return runtime_pins[url.rsplit("/", 1)[1]]

    target = gw.ensure_sherpa_runtime(tmp_path / "site", base_url="https://mirror.example/sherpa/", downloader=fake)
    assert sorted(seen) == sorted(f"https://mirror.example/sherpa/{n}" for n in runtime_pins)
    assert all((target / n).read_bytes() == d for n, d in runtime_pins.items())


def test_runtime_without_a_source_says_how_to_build_it(tmp_path, runtime_pins):
    with pytest.raises(RuntimeError, match="scripts/sherpa_wasm/build.sh"):
        gw.ensure_sherpa_runtime(tmp_path / "site")


def test_an_installed_runtime_with_the_right_hashes_is_kept(tmp_path, runtime_pins):
    target = tmp_path / "site" / gw.SHERPA_VENDOR_SUBDIR
    target.mkdir(parents=True)
    for name, data in runtime_pins.items():
        (target / name).write_bytes(data)
    assert gw.ensure_sherpa_runtime(tmp_path / "site", downloader=lambda url: pytest.fail("no download")) == target


@pytest.fixture()
def model_pins(monkeypatch):
    files = {"model.int8.onnx": b"onnx-bytes", "tokens.txt": b"a 0\n"}
    monkeypatch.setattr(gw, "GIGAAM_FILES", {name: _sha(data) for name, data in files.items()})
    return files


def test_model_is_copied_from_the_local_cache_when_hashes_match(tmp_path, model_pins):
    cache = tmp_path / "cache"
    cache.mkdir()
    for name, data in model_pins.items():
        (cache / name).write_bytes(data)
    target = gw.ensure_gigaam_model(tmp_path / "site_models", cache_dir=cache, downloader=lambda url: pytest.fail("no download"))
    assert target == tmp_path / "site_models" / "gigaam" / gw.GIGAAM_MODEL_ID
    assert (target / "model.int8.onnx").read_bytes() == model_pins["model.int8.onnx"]


def test_model_downloads_at_the_pinned_revision_and_refuses_a_wrong_hash(tmp_path, model_pins):
    urls = []

    def fake(url: str) -> bytes:
        urls.append(url)
        return model_pins[url.rsplit("/", 1)[1]]

    target = gw.ensure_gigaam_model(tmp_path / "m", cache_dir=tmp_path / "none", downloader=fake)
    assert all(f"/resolve/{gw.GIGAAM_REVISION}/" in u for u in urls)
    assert (target / "tokens.txt").read_bytes() == model_pins["tokens.txt"]
    with pytest.raises(RuntimeError, match="refusing"):
        gw.ensure_gigaam_model(tmp_path / "m2", cache_dir=tmp_path / "none", downloader=lambda url: b"tampered")


def test_pins_are_the_measured_artifacts():
    assert gw.GIGAAM_FILES["model.int8.onnx"] == "2d94f93ffd4ef58e7899c9de885c25bbbc8c9f1073618868d118a674450ba5f7"
    assert gw.GIGAAM_REVISION == "9f5a77e8975211abe8511693accd3a63ee1e9f43"
    assert set(gw.SHERPA_WASM_FILES) == {"sherpa-onnx-wasm-web.js", "sherpa-onnx-wasm-web.wasm", "sherpa-onnx-asr.js"}
