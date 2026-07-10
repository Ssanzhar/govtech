"""SMOKE tests for `qorgan.asr.transcribe` — no model download, no real audio decoding."""

from types import SimpleNamespace

import pytest

from qorgan.asr.transcribe import transcribe


def test_transcribe_manual_transcript_passthrough_returns_text_unchanged():
    assert transcribe(manual_transcript="Привет, это тест.") == "Привет, это тест."


def test_transcribe_manual_transcript_blank_raises():
    with pytest.raises(ValueError):
        transcribe(manual_transcript="   ")


def test_transcribe_missing_audio_and_no_manual_raises_value_error():
    with pytest.raises(ValueError):
        transcribe()


def test_transcribe_audio_path_not_found_raises_file_not_found(tmp_path):
    missing = tmp_path / "does_not_exist.wav"
    with pytest.raises(FileNotFoundError):
        transcribe(missing)


class FakeWhisperModel:
    def __init__(self, segments):
        self._segments = segments
        self.calls = []

    def transcribe(self, path):
        self.calls.append(path)
        return self._segments, SimpleNamespace(language="ru")


def test_transcribe_calls_whisper_model_and_joins_segments(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake-audio-bytes")
    segments = [SimpleNamespace(text=" Привет "), SimpleNamespace(text="как дела? ")]
    fake_model = FakeWhisperModel(segments)

    result = transcribe(audio, model=fake_model)

    assert result == "Привет как дела?"
    assert fake_model.calls == [str(audio)]


def test_transcribe_prefers_manual_transcript_over_audio_path(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake-audio-bytes")
    fake_model = FakeWhisperModel([SimpleNamespace(text="should not be used")])

    result = transcribe(audio, manual_transcript="Ручной транскрипт.", model=fake_model)

    assert result == "Ручной транскрипт."
    assert fake_model.calls == []


def test_transcribe_empty_segments_returns_empty_string(tmp_path):
    audio = tmp_path / "silence.wav"
    audio.write_bytes(b"fake-audio-bytes")
    fake_model = FakeWhisperModel([])

    assert transcribe(audio, model=fake_model) == ""
