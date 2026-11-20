import asyncio
import os
import time
from pathlib import Path

import numpy as np
import pytest

from local_voice_server.engine import Engine
from local_voice_server.models import create_model
from local_voice_server.models.breeze import BreezeModel
from local_voice_server.voices import Voice, scan_voices

VOICE = Voice("my-voice", Path("/voices/my-voice/reference.wav"), "Hello there.")


class FakeModel:
    """A `TtsModel` that yields scripted chunks; an Exception in the script is raised."""

    id = "fake-tts"
    sample_rate = 24_000

    def __init__(self, chunks=None, delay=0.0, load_error=None):
        self.chunks = chunks if chunks is not None else [[0.1, 0.2], [0.3]]
        self.delay = delay
        self.load_error = load_error
        self.calls: list[tuple[str, str, bool]] = []
        self.spans: list[tuple[float, float]] = []
        self.yielded = 0

    def load(self):
        if self.load_error:
            raise self.load_error

    def generate(self, text, voice, *, stream):
        start = time.monotonic()
        self.calls.append((text, voice.name, stream))
        time.sleep(self.delay)
        self.spans.append((start, time.monotonic()))
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            self.yielded += 1
            yield np.array(chunk, dtype=np.float32)


def loaded(model) -> Engine:
    engine = Engine(model)
    asyncio.run(engine.load())
    return engine


async def collect(engine, limit=None):
    out = []
    chunks = engine.stream("Hi", VOICE)
    async for chunk in chunks:
        out.append(chunk.tolist())
        if limit is not None and len(out) >= limit:
            break
    await chunks.aclose()
    return out


def test_synthesize_concatenates_chunks():
    model = FakeModel()
    audio, rate = asyncio.run(loaded(model).synthesize("Hi", VOICE))
    assert rate == 24_000 and audio.dtype == np.float32
    np.testing.assert_allclose(audio, [0.1, 0.2, 0.3], rtol=1e-6)
    assert model.calls == [("Hi", "my-voice", False)]


def test_engine_exposes_model_identity():
    engine = Engine(FakeModel())
    assert (engine.model_id, engine.sample_rate) == ("fake-tts", 24_000)


def test_calls_before_load_raise():
    engine = Engine(FakeModel())
    with pytest.raises(RuntimeError, match="model not loaded"):
        asyncio.run(engine.synthesize("Hi", VOICE))
    with pytest.raises(RuntimeError, match="model not loaded"):
        asyncio.run(collect(engine))


def test_empty_output_raises():
    with pytest.raises(RuntimeError, match="no audio"):
        asyncio.run(loaded(FakeModel(chunks=[])).synthesize("Hi", VOICE))
    with pytest.raises(RuntimeError, match="no audio"):
        asyncio.run(collect(loaded(FakeModel(chunks=[]))))


def test_calls_are_serialised():
    model = FakeModel(delay=0.2)
    engine = Engine(model)

    async def go():
        await engine.load()
        await asyncio.gather(engine.synthesize("a", VOICE), engine.synthesize("b", VOICE))

    asyncio.run(go())
    (_, first_end), (second_start, _) = sorted(model.spans)
    assert first_end <= second_start


def test_load_failure_propagates():
    with pytest.raises(FileNotFoundError, match="weights missing"):
        loaded(FakeModel(load_error=FileNotFoundError("weights missing")))


def test_stream_yields_chunks_in_order():
    model = FakeModel(chunks=[[0.5], [0.25, 0.125]])
    assert asyncio.run(collect(loaded(model))) == [[0.5], [0.25, 0.125]]
    assert model.calls == [("Hi", "my-voice", True)]


def test_stream_stops_when_consumer_stops():
    model = FakeModel(chunks=[[0.1]] * 50)
    engine = Engine(model)

    async def go():
        await engine.load()
        await collect(engine, limit=1)
        streamed = model.yielded
        await engine.synthesize("again", VOICE)  # the engine is free again
        return streamed

    assert asyncio.run(go()) < 50


def test_stream_error_propagates():
    with pytest.raises(RuntimeError, match="codec broke"):
        asyncio.run(collect(loaded(FakeModel(chunks=[[0.1], RuntimeError("codec broke")]))))


def test_create_model_rejects_unknown_adapter():
    with pytest.raises(ValueError, match="unknown VOICE_MODEL 'nope'"):
        create_model("nope")


def test_breeze_missing_weights_folder_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        BreezeModel(str(tmp_path / "no-such-model")).load()


def test_breeze_weights_env_override(monkeypatch):
    monkeypatch.setenv("BREEZE_MODEL", "mlx-community/Breeze-TTS-2-mlx-8bit")
    assert BreezeModel().weights == "mlx-community/Breeze-TTS-2-mlx-8bit"


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("VOICE_SERVER_SLOW") != "1", reason="set VOICE_SERVER_SLOW=1 to load the real model")
def test_real_breeze_short_clip_and_stream():
    voices = scan_voices(Path(__file__).resolve().parents[1] / "voices")
    assert "my-voice" in voices, "voices/my-voice is required for the slow test"
    engine = loaded(BreezeModel())
    audio, rate = asyncio.run(engine.synthesize("Testing one two three.", voices["my-voice"]))
    assert 1.0 < len(audio) / rate < 6.0

    async def stream():
        return [chunk async for chunk in engine.stream("Testing one two three.", voices["my-voice"])]

    chunks = asyncio.run(stream())
    assert len(chunks) > 1 and 1.0 < sum(len(c) for c in chunks) / rate < 6.0
