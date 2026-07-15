"""TDD tests for `qorgan.asr.vosk_stream` — dual-language streaming recognition with
per-utterance confidence voting (design spec §06).

Uses scripted fake recognizers (the vosk `KaldiRecognizer` contract: `AcceptWaveform` /
`Result` / `PartialResult` / `FinalResult` returning JSON strings) — the real models never
load in tests, mirroring `_FakeWhisperModel` in `test_stream.py`.
"""

import json

import pytest

from qorgan.asr.stream import CommittedUtterance
from qorgan.asr.vosk_stream import Partial, recognize_stream


class FakeRecognizer:
    """Scripted vosk.KaldiRecognizer stand-in.

    `accepts[i]` is what the i-th `AcceptWaveform` call returns; `Result`/`FinalResult`/
    `PartialResult` pop from their scripts (falling back to empty JSON).
    """

    def __init__(self, accepts=(), results=(), partials=(), finals=()):
        self.accepts = list(accepts)
        self.results = list(results)
        self.partials = list(partials)
        self.finals = list(finals)
        self.fed: list[bytes] = []
        self.final_calls = 0

    def AcceptWaveform(self, chunk):  # noqa: N802 - vosk API name
        self.fed.append(chunk)
        return self.accepts.pop(0) if self.accepts else False

    def Result(self):  # noqa: N802
        return self.results.pop(0) if self.results else "{}"

    def PartialResult(self):  # noqa: N802
        return self.partials.pop(0) if self.partials else json.dumps({"partial": ""})

    def FinalResult(self):  # noqa: N802
        self.final_calls += 1
        return self.finals.pop(0) if self.finals else "{}"


def _result(text, *confs):
    return json.dumps(
        {"text": text, "result": [{"word": w, "conf": c} for w, c in zip(text.split(), confs)]}
    )


def _partial(text):
    return json.dumps({"partial": text})


CHUNK = b"\x00\x01"


# --- endpoint voting --------------------------------------------------------------------------


def test_kk_wins_endpoint_when_more_confident():
    kk = FakeRecognizer(accepts=[True], results=[_result("салем банк", 0.95, 0.9)])
    ru = FakeRecognizer(accepts=[False], finals=[_result("салам бак", 0.4, 0.5)])

    events = list(recognize_stream([CHUNK], recognizers={"kk": kk, "ru": ru}))
    finals = [e for e in events if isinstance(e, CommittedUtterance)]

    assert len(finals) == 1
    assert finals[0].text == "салем банк"
    assert finals[0].language == "kk"
    assert finals[0].confidence == pytest.approx((0.95 + 0.9) / 2)
    # The loser was flushed at the endpoint boundary (a second, empty flush also happens
    # at end-of-stream — harmless on real vosk recognizers).
    assert ru.final_calls >= 1


def test_ru_wins_endpoint_when_more_confident():
    kk = FakeRecognizer(accepts=[True], results=[_result("бул банк", 0.3, 0.4)])
    ru = FakeRecognizer(accepts=[False], finals=[_result("это банк", 0.9, 0.95)])

    events = list(recognize_stream([CHUNK], recognizers={"kk": kk, "ru": ru}))
    finals = [e for e in events if isinstance(e, CommittedUtterance)]

    assert finals[0].text == "это банк"
    assert finals[0].language == "ru"


def test_every_chunk_feeds_both_recognizers():
    kk = FakeRecognizer(accepts=[False, False])
    ru = FakeRecognizer(accepts=[False, False])

    list(recognize_stream([b"a", b"b"], recognizers={"kk": kk, "ru": ru}))

    assert kk.fed == [b"a", b"b"]
    assert ru.fed == [b"a", b"b"]


def test_silence_endpoint_emits_nothing():
    kk = FakeRecognizer(accepts=[True], results=[json.dumps({"text": ""})])
    ru = FakeRecognizer(accepts=[False], finals=[json.dumps({"text": ""})])

    events = list(recognize_stream([CHUNK], recognizers={"kk": kk, "ru": ru}))

    assert [e for e in events if isinstance(e, CommittedUtterance)] == []


def test_missing_word_confidences_default_to_full():
    kk = FakeRecognizer(accepts=[True], results=[json.dumps({"text": "алло"})])
    ru = FakeRecognizer(accepts=[False], finals=[json.dumps({"text": ""})])

    events = list(recognize_stream([CHUNK], recognizers={"kk": kk, "ru": ru}))
    finals = [e for e in events if isinstance(e, CommittedUtterance)]

    assert finals[0].confidence == 1.0


def test_confidence_is_clamped_to_unit_interval():
    kk = FakeRecognizer(accepts=[True], results=[_result("алло", 1.7)])
    ru = FakeRecognizer(accepts=[False], finals=[json.dumps({"text": ""})])

    events = list(recognize_stream([CHUNK], recognizers={"kk": kk, "ru": ru}))
    finals = [e for e in events if isinstance(e, CommittedUtterance)]

    assert finals[0].confidence == 1.0


# --- partials ---------------------------------------------------------------------------------


def test_partials_stream_between_endpoints():
    kk = FakeRecognizer(accepts=[False, False], partials=[_partial("это"), _partial("это служба")])
    ru = FakeRecognizer(accepts=[False, False], partials=[_partial(""), _partial("")])

    events = list(recognize_stream([CHUNK, CHUNK], recognizers={"kk": kk, "ru": ru}))
    partials = [e for e in events if isinstance(e, Partial)]

    assert [p.text for p in partials] == ["это", "это служба"]


def test_identical_consecutive_partials_are_deduplicated():
    kk = FakeRecognizer(accepts=[False, False], partials=[_partial("алло"), _partial("алло")])
    ru = FakeRecognizer(accepts=[False, False])

    events = list(recognize_stream([CHUNK, CHUNK], recognizers={"kk": kk, "ru": ru}))
    partials = [e for e in events if isinstance(e, Partial)]

    assert len(partials) == 1


def test_partial_falls_back_to_other_language_when_preferred_is_silent():
    kk = FakeRecognizer(accepts=[False], partials=[_partial("")])
    ru = FakeRecognizer(accepts=[False], partials=[_partial("это банк")])

    events = list(recognize_stream([CHUNK], recognizers={"kk": kk, "ru": ru}))
    partials = [e for e in events if isinstance(e, Partial)]

    assert partials[0].text == "это банк"
    assert partials[0].language == "ru"


# --- end-of-stream flush ----------------------------------------------------------------------


def test_stream_end_flushes_pending_hypothesis():
    kk = FakeRecognizer(accepts=[False], finals=[_result("соңғы сөз", 0.8, 0.8)])
    ru = FakeRecognizer(accepts=[False], finals=[json.dumps({"text": ""})])

    events = list(recognize_stream([CHUNK], recognizers={"kk": kk, "ru": ru}))
    finals = [e for e in events if isinstance(e, CommittedUtterance)]

    assert len(finals) == 1
    assert finals[0].text == "соңғы сөз"


def test_empty_stream_yields_nothing():
    kk = FakeRecognizer()
    ru = FakeRecognizer()

    assert list(recognize_stream([], recognizers={"kk": kk, "ru": ru})) == []
