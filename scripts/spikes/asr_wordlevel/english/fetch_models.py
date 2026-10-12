"""Download the English ASR candidates into models/asr/<name>/ (gitignored).

The network here is per-connection throttled, so each archive is fetched as parallel HTTP range
chunks with curl (resumable: finished chunks are kept), then joined and extracted into its own new
directory.

    python scripts/spikes/asr_wordlevel/english/fetch_models.py [names...]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
MODELS = REPO / "models/asr"
DL = MODELS / "_dl"
SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{}.tar.bz2"
SOURCES = {
    "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8": SHERPA,
    "sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27": SHERPA,
    "sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8": SHERPA,
    "vosk-model-en-us-0.22-lgraph": "https://alphacephei.com/vosk/models/{}.zip",
    "sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-480ms-int8": SHERPA,
    "sherpa-onnx-moonshine-base-en-quantized-2026-02-27": SHERPA,
    "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms": SHERPA,  # fp32, to test int8 loss
}
CHUNK = 1024 * 1024
WORKERS = 6
RETRIES = 30


def content_length(url: str) -> int:
    out = subprocess.run(["curl", "-sSLI", url], capture_output=True, text=True, check=True).stdout
    lengths = [int(l.split(":")[1]) for l in out.splitlines() if l.lower().startswith("content-length:")]
    if not lengths:
        raise RuntimeError(f"no content-length for {url}")
    return lengths[-1]


def fetch_chunk(url: str, part: Path, start: int, end: int) -> None:
    """Fetch one byte range, appending to a partial chunk on retry instead of starting over."""
    want = end - start + 1
    for _ in range(RETRIES):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return
        if have > want:
            part.unlink()
            continue
        tmp = part.with_suffix(".cur")
        subprocess.run(["curl", "-sSL", "--max-time", "240", "-r", f"{start + have}-{end}", "-o", str(tmp), url], check=False)
        if tmp.exists():
            with part.open("ab") as sink:
                sink.write(tmp.read_bytes())
            tmp.unlink()
    if not part.exists() or part.stat().st_size != want:
        raise RuntimeError(f"chunk {part.name} failed")


def download(name: str) -> Path:
    url = SOURCES[name].format(name)
    target = DL / Path(url).name
    if target.exists():
        return target
    size = content_length(url)
    parts_dir = DL / f"{name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = [(i, s, min(s + CHUNK, size) - 1) for i, s in enumerate(range(0, size, CHUNK))]
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda r: fetch_chunk(url, parts_dir / f"{r[0]:05d}", r[1], r[2]), ranges))
    tmp = target.with_suffix(".tmp")
    with tmp.open("wb") as sink:
        for i, _, _ in ranges:
            sink.write((parts_dir / f"{i:05d}").read_bytes())
    tmp.rename(target)
    for p in parts_dir.iterdir():
        p.unlink()
    parts_dir.rmdir()
    print(f"downloaded {target.name} ({size / 1e6:.1f} MB)", flush=True)
    return target


def extract(archive: Path, name: str) -> None:
    dest = MODELS / name
    if dest.exists():
        return
    staging = MODELS / f"_x_{name}"
    staging.mkdir(parents=True, exist_ok=False)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(staging)
    else:
        with tarfile.open(archive) as t:
            t.extractall(staging, filter="data")
    inner = staging / name
    (inner if inner.is_dir() else staging).rename(dest)
    if staging.exists():
        staging.rmdir()
    print(f"extracted {dest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=list(SOURCES))
    args = parser.parse_args()
    for name in args.names:
        try:
            extract(download(name), name)
        except Exception as exc:  # report and continue with the next candidate
            print(f"FAILED {name}: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
