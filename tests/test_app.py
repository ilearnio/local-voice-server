import logging
import struct
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from local_voice_server.app import MAX_INPUT_CHARS, create_app

from .conftest import make_voice


class FakeEngine:
    model_id = "test-tts"
    sample_rate = 24_000

    def __init__(self, error: Exception | None = None, stream_chunks=None):
        self.error = error
        self.stream_chunks = stream_chunks if stream_chunks is not None else [[0.5, -0.5], [1.0]]
        self.calls: list[tuple[str, str]] = []

    async def synthesize(self, text, voice):
        self.calls.append((text, voice.name))
        if self.error:
            raise self.error
        return np.zeros(2400, dtype=np.float32), 24_000

    async def stream(self, text, voice):
        self.calls.append((text, voice.name))
        if self.error:
            raise self.error
        for chunk in self.stream_chunks:
            yield np.array(chunk, dtype=np.float32)


def speech_body(**overrides):
    body = {"model": "test-tts", "input": "Hi", "voice": "my-voice", "response_format": "wav"}
    body.update(overrides)
    return body


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def client(engine, voices_dir: Path):
    return TestClient(create_app(engine, voices_dir))


def test_models_lists_tts_model(client):
    assert client.get("/v1/models").json()["data"] == [
        {"id": "test-tts", "object": "model", "task": "text-to-speech", "owned_by": "local"}
    ]


def test_voices_sorted_and_rescanned(client, voices_dir):
    assert client.get("/v1/audio/voices").json() == {"voices": ["my-voice"]}
    make_voice(voices_dir, "alpha")
    assert client.get("/v1/audio/voices").json() == {"voices": ["alpha", "my-voice"]}


def test_new_voice_usable_without_restart(client, voices_dir, engine):
    make_voice(voices_dir, "alpha")
    assert client.post("/v1/audio/speech", json=speech_body(voice="alpha")).status_code == 200
    assert engine.calls == [("Hi", "alpha")]


def test_speech_returns_wav(client):
    response = client.post("/v1/audio/speech", json=speech_body())
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    body = response.content
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"
    fmt, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[20:36])
    assert (fmt, channels, rate, bits) == (1, 1, 24_000, 16)
    assert body[36:40] == b"data"
    assert struct.unpack("<I", body[40:44])[0] == 4800


def test_speech_empty_model_uses_default(client):
    assert client.post("/v1/audio/speech", json=speech_body(model="")).status_code == 200


def test_speech_unknown_model_400(client):
    assert client.post("/v1/audio/speech", json=speech_body(model="tts-1")).status_code == 400


def test_speech_unknown_voice_404(client):
    response = client.post("/v1/audio/speech", json=speech_body(voice="nobody"))
    assert response.status_code == 404
    assert "nobody" in response.json()["detail"]


def test_speech_blank_input_400(client):
    assert client.post("/v1/audio/speech", json=speech_body(input="   ")).status_code == 400


def test_speech_non_wav_format_400(client):
    assert client.post("/v1/audio/speech", json=speech_body(response_format="mp3")).status_code == 400


def test_speech_input_at_cap_ok(client):
    assert client.post("/v1/audio/speech", json=speech_body(input="a" * MAX_INPUT_CHARS)).status_code == 200


def test_speech_input_over_cap_400(client):
    response = client.post("/v1/audio/speech", json=speech_body(input="a" * (MAX_INPUT_CHARS + 1)))
    assert response.status_code == 400
    assert str(MAX_INPUT_CHARS) in response.json()["detail"]


def test_speech_voice_path_traversal_404(client, engine):
    assert client.post("/v1/audio/speech", json=speech_body(voice="../my-voice")).status_code == 404
    assert engine.calls == []


def test_speech_passes_events_and_unicode_unchanged(client, engine):
    text = '(laugh) "Yes" — ✅'
    assert client.post("/v1/audio/speech", json=speech_body(input=text)).status_code == 200
    assert engine.calls == [(text, "my-voice")]


def test_speech_engine_error_500(voices_dir, caplog):
    client = TestClient(create_app(FakeEngine(error=RuntimeError("boom")), voices_dir))
    with caplog.at_level(logging.ERROR):
        response = client.post("/v1/audio/speech", json=speech_body())
    assert response.status_code == 500
    assert "boom" in response.json()["detail"]
    assert "Traceback" in caplog.text or any(r.exc_info for r in caplog.records)


def test_pcm_streams_raw_audio(client, engine):
    response = client.post("/v1/audio/speech", json=speech_body(response_format="pcm"))
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"
    assert response.headers["x-sample-rate"] == "24000"
    assert list(struct.unpack("<3h", response.content)) == [16384, -16384, 32767]
    assert engine.calls == [("Hi", "my-voice")]


def test_pcm_start_failure_500(voices_dir):
    client = TestClient(create_app(FakeEngine(error=RuntimeError("no gpu")), voices_dir))
    response = client.post("/v1/audio/speech", json=speech_body(response_format="pcm"))
    assert response.status_code == 500
    assert "no gpu" in response.json()["detail"]


def test_pcm_validates_like_wav(client, engine):
    assert client.post("/v1/audio/speech", json=speech_body(response_format="pcm", voice="nobody")).status_code == 404
    assert client.post("/v1/audio/speech", json=speech_body(response_format="pcm", input=" ")).status_code == 400
    assert engine.calls == []


def test_pcm_reports_model_sample_rate(voices_dir):
    engine = FakeEngine()
    engine.sample_rate = 44_100
    client = TestClient(create_app(engine, voices_dir))
    response = client.post("/v1/audio/speech", json=speech_body(response_format="pcm"))
    assert response.status_code == 200
    assert response.headers["x-sample-rate"] == "44100"
