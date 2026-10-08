"""Fetch FLEURS test utterances (real read speech) for the English recogniser test.

FLEURS ships each split as one large `audio/test.tar.gz` (en_us test = 628 MB). Only the first
few tens of MB are needed, so the archive prefix is fetched in parallel byte-range chunks, then
stream-decompressed until the truncation point. Members are taken in archive order; the reference
is the normalised `transcription` column of `test.tsv`.

- `fleurs_en`        -- 200 en_us test utterances (the main English set);
- `probe_fleurs_kk`  -- 75 kk_kz test utterances (false-win probe);
- `probe_fleurs_ru`  -- 75 ru_ru test utterances (false-win probe).

Writes `data/asr_capture/english_fleurs.jsonl`; audio to gitignored `data/asr_capture/_wav/english/fleurs/`.

    python scripts/spikes/asr_wordlevel/english/fetch_fleurs.py
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from fetch_models import CHUNK, fetch_chunk

REPO = Path(__file__).resolve().parents[4]
BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{}/{}"
CACHE = REPO / "models/asr/_dl/fleurs"
WAV_DIR = REPO / "data/asr_capture/_wav/english/fleurs"
OUT = REPO / "data/asr_capture/english_fleurs.jsonl"
PLAN = (("en_us", "fleurs_en", "en", 200, 90), ("kk_kz", "probe_fleurs_kk", "kk", 75, 64), ("ru_ru", "probe_fleurs_ru", "ru", 75, 40))
WORKERS = 6
SAMPLE_RATE = 16000


def references(locale: str) -> dict[str, str]:
    path = CACHE / f"{locale}_test.tsv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sSL", "--retry", "10", "-o", str(path), BASE.format(locale, "test.tsv")], check=True)
    rows = csv.reader(path.read_text(encoding="utf-8").splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE)
    return {r[1]: r[3] for r in rows if len(r) > 3}


def prefix(locale: str, megabytes: int) -> Path:
    url = BASE.format(locale, "audio/test.tar.gz")
    parts = CACHE / f"{locale}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    size = megabytes * 1024 * 1024
    ranges = [(i, s, min(s + CHUNK, size) - 1) for i, s in enumerate(range(0, size, CHUNK))]
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda r: fetch_chunk(url, parts / f"{r[0]:05d}", r[1], r[2]), ranges))
    out = CACHE / f"{locale}_prefix.tar.gz"
    with out.open("wb") as sink:
        for i, _, _ in ranges:
            sink.write((parts / f"{i:05d}").read_bytes())
    return out


def to_int16(raw: bytes) -> np.ndarray:
    """FLEURS WAVs are 32-bit float (format 3), which `wave` cannot read; scipy can."""
    from scipy.io import wavfile

    rate, data = wavfile.read(io.BytesIO(raw))
    if rate != SAMPLE_RATE or data.ndim != 1:
        raise ValueError(f"unexpected wav: {rate} Hz, shape {data.shape}")
    if data.dtype.kind == "f":
        return (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    return data.astype(np.int16)


def members(archive: Path):
    """Yield (name, bytes) for complete members; stop quietly at the truncation point."""
    try:
        with tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                if member.isfile():
                    data = tar.extractfile(member).read()
                    if len(data) == member.size:
                        yield Path(member.name).name, data
    except (EOFError, tarfile.ReadError, OSError, Exception) as exc:  # truncated gzip ends the stream
        print(f"  {archive.name}: stopped at truncation ({type(exc).__name__})", flush=True)


def main() -> None:
    from build_cs_set import write_wav  # noqa: PLC0415  (sys.path set below)

    rows = []
    for locale, set_name, lang, wanted, megabytes in PLAN:
        refs = references(locale)
        archive = prefix(locale, megabytes)
        taken = 0
        for name, data in members(archive):
            if taken >= wanted:
                break
            if name not in refs:
                continue
            samples = to_int16(data)
            wav = WAV_DIR / locale / name
            write_wav(wav, samples)
            rows.append({"id": f"{locale}_{Path(name).stem}", "set": set_name, "lang": lang, "voice": "fleurs",
                         "channel": "clean", "wav": str(wav.relative_to(REPO)),
                         "duration": round(len(samples) / SAMPLE_RATE, 3), "reference": refs[name]})
            taken += 1
        print(f"{locale}: {taken} utterances", flush=True)
    with OUT.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
