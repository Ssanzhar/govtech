"""Tests for `qorgan.asr.stream` — pseudo-streaming utterance source (design spec §06).

`replay_transcript` (deterministic, used by the live demo and tests) gets real coverage;
`stream_transcribe` is exercised with an injected fake whisper model, mirroring how
`asr/transcribe.py` is tested — the real model never loads in tests.
"""

import math
from dataclasses import dataclass

import pytest

from qorgan.asr.stream import CommittedUtterance, replay_transcript, stream_transcribe

# --- replay_transcript ---------------------------------------------------------------------


def test_replay_yields_one_utterance_per_line_in_order():
    utterances = list(replay_transcript("Алло\nЭто банк\nНазовите код"))

    assert [u.text for u in utterances] == ["Алло", "Это банк", "Назовите код"]


def test_replay_default_confidence_is_full():
    utterances = list(replay_transcript("Алло\nЭто банк"))

    assert all(u.confidence == 1.0 for u in utterances)


def test_replay_custom_confidence_is_applied():
    utterances = list(replay_transcript("Алло", default_confidence=0.7))

    assert utterances[0].confidence == 0.7


def test_replay_skips_blank_lines():
    utterances = list(replay_transcript("Алло\n\n  \nЭто банк"))

    assert [u.text for u in utterances] == ["Алло", "Это банк"]


def test_replay_rejects_blank_transcript():
    with pytest.raises(ValueError):
        list(replay_transcript("   \n  "))


def test_replay_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        list(replay_transcript("Алло", default_confidence=1.5))


def test_committed_utterance_is_frozen_and_validated():
    utterance = CommittedUtterance(text="Алло", confidence=0.9)
    with pytest.raises(Exception):
        utterance.text = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError):
        CommittedUtterance(text="   ", confidence=0.9)


# --- stream_transcribe (fake whisper model) ---------------------------------------------------


@dataclass
class _FakeSegment:
    text: str
    avg_logprob: float
    start: float
    end: float


class _FakeWhisperModel:
    def transcribe(self, path):
        segments = [
            _FakeSegment(text=" Алло ", avg_logprob=-0.1, start=0.0, end=1.2),
            _FakeSegment(text=" Это банк ", avg_logprob=-0.7, start=1.4, end=3.0),
        ]
        return iter(segments), {"language": "ru"}


def test_stream_transcribe_yields_segments_with_confidence(tmp_path):
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"\x00")

    utterances = list(stream_transcribe(audio, model=_FakeWhisperModel()))

    assert [u.text for u in utterances] == ["Алло", "Это банк"]
    assert utterances[0].confidence == pytest.approx(math.exp(-0.1))
    assert utterances[1].confidence == pytest.approx(math.exp(-0.7))
    assert utterances[0].start_s == 0.0
    assert utterances[1].end_s == 3.0


def test_stream_transcribe_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(stream_transcribe(tmp_path / "missing.wav", model=_FakeWhisperModel()))
