"""OpenAI-compatible routes in the shape OpenBot's voice server client expects.

`/v1/audio/speech` answers `response_format: "wav"` with one complete WAV file, and
`response_format: "pcm"` with raw mono PCM16 streamed while it is generated (the format OpenAI's
speech API streams). OpenAI's PCM is 24 kHz; the `X-Sample-Rate` header states the actual rate.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from .voices import Voice, scan_voices
from .wav import encode_pcm16, encode_wav_pcm16

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 3000
RESPONSE_FORMATS = ("wav", "pcm")


class EngineLike(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def sample_rate(self) -> int: ...

    async def synthesize(self, text: str, voice: Voice) -> tuple[np.ndarray, int]: ...

    def stream(self, text: str, voice: Voice) -> AsyncIterator[np.ndarray]: ...


class SpeechRequest(BaseModel):
    model: str = ""
    input: str
    voice: str
    response_format: str = "wav"


def create_app(engine: EngineLike, voices_dir: Path) -> FastAPI:
    app = FastAPI(title="Local voice server", docs_url=None, redoc_url=None)

    @app.get("/v1/models")
    def models() -> dict:
        return {
            "object": "list",
            "data": [{"id": engine.model_id, "object": "model", "task": "text-to-speech", "owned_by": "local"}],
        }

    @app.get("/v1/audio/voices")
    def voices() -> dict:
        return {"voices": sorted(scan_voices(voices_dir))}

    @app.post("/v1/audio/speech")
    async def speech(request: SpeechRequest) -> Response:
        if request.model and request.model != engine.model_id:
            raise HTTPException(400, f"unknown model {request.model!r}; this server serves {engine.model_id!r}")
        if request.response_format not in RESPONSE_FORMATS:
            raise HTTPException(
                400, f"response_format {request.response_format!r} not supported; use 'wav' or 'pcm'"
            )
        text = request.input.strip()
        if not text:
            raise HTTPException(400, "input is empty")
        if len(text) > MAX_INPUT_CHARS:
            raise HTTPException(400, f"input is {len(text)} characters; the limit is {MAX_INPUT_CHARS}")
        voice = scan_voices(voices_dir).get(request.voice)
        if voice is None:
            raise HTTPException(404, f"unknown voice {request.voice!r}")

        if request.response_format == "pcm":
            return await _stream_pcm(engine, text, voice)

        try:
            audio, sample_rate = await engine.synthesize(text, voice)
        except Exception as exc:
            logger.exception("speech generation failed (voice=%s, chars=%d)", voice.name, len(text))
            raise HTTPException(500, f"speech generation failed: {exc}") from exc

        wav = await asyncio.to_thread(encode_wav_pcm16, audio, sample_rate)
        logger.info(
            "spoke %d chars as %s: %.1fs of audio", len(text), voice.name, len(audio) / sample_rate
        )
        return Response(content=wav, media_type="audio/wav")

    return app


async def _stream_pcm(engine: EngineLike, text: str, voice: Voice) -> StreamingResponse:
    rate = engine.sample_rate
    chunks = engine.stream(text, voice)
    # Wait for the first chunk before answering, so a failure to start is still a clean 500.
    try:
        first = await anext(chunks)
    except Exception as exc:
        await chunks.aclose()
        logger.exception("speech streaming failed to start (voice=%s, chars=%d)", voice.name, len(text))
        raise HTTPException(500, f"speech generation failed: {exc}") from exc

    async def body() -> AsyncIterator[bytes]:
        samples = first.size
        try:
            yield encode_pcm16(first)
            async for chunk in chunks:
                samples += chunk.size
                yield encode_pcm16(chunk)
        except Exception:
            # Headers are already sent; ending the stream early is the only signal left.
            logger.exception("speech streaming failed after %.1fs of audio", samples / rate)
            return
        finally:
            await chunks.aclose()
        logger.info("streamed %d chars as %s: %.1fs of audio", len(text), voice.name, samples / rate)

    return StreamingResponse(
        body(), media_type="audio/pcm", headers={"X-Sample-Rate": str(rate)}
    )
