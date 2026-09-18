"""Synthesize the B7 bench clips (macOS `say`: Milena ru_RU, Aru kk_KZ) as 16 kHz mono WAV."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
KK_SCAM = (
    "Сәлеметсіз бе. Бұл банктің қауіпсіздік қызметі. Сіздің картаңыздан күдікті операция тіркелді. "
    "Ешкімге айтпаңыз. SMS-тен келген кодты айтыңыз, біз операцияны тоқтатамыз."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scenarios = {s["id"]: s for s in json.loads((REPO_ROOT / "site/core/scenarios.json").read_text(encoding="utf-8"))["scenarios"]}
    clips = {
        "ru_scam": ("Milena", " ".join(scenarios["live_scam_bank_ru"]["lines"])),
        "ru_legit": ("Milena", " ".join(scenarios["live_hard_negative_bank_ru"]["lines"])),
        "kk_scam": ("Aru", KK_SCAM),
    }
    for name, (voice, text) in clips.items():
        aiff = args.out / f"{name}.aiff"
        subprocess.run(["say", "-v", voice, "-r", "175", "-o", str(aiff), text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(args.out / f"{name}.wav")], check=True)
        aiff.unlink()
    (args.out / "meta.json").write_text(json.dumps({k: {"voice": v[0], "text": v[1]} for k, v in clips.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(clips)} clips -> {args.out}")


if __name__ == "__main__":
    main()
