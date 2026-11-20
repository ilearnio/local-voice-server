"""Runs one TTS model adapter on a single worker thread, one generation at a time."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .models import TtsModel
from .voices import Voice

_DONE = object()


class Engine:
    """Owns a `TtsModel`.

    Model calls always run on the same worker thread (MLX streams, for one, are thread-local).
    The lock makes concurrent requests queue instead of interleaving on that thread.
    """

    def __init__(self, model: TtsModel):
        self._model = model
        self._loaded = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-model")
        self._lock = asyncio.Lock()

    @property
    def model_id(self) -> str:
        return self._model.id

    @property
    def sample_rate(self) -> int:
        return self._model.sample_rate

    async def load(self) -> None:
        async with self._lock:
            await asyncio.get_running_loop().run_in_executor(self._executor, self._model.load)
            self._loaded = True

    async def synthesize(self, text: str, voice: Voice) -> tuple[np.ndarray, int]:
        """The whole utterance as float32 mono, plus its sample rate."""
        self._require_loaded()
        async with self._lock:
            audio = await asyncio.get_running_loop().run_in_executor(
                self._executor, self._synthesize_sync, text, voice
            )
        return audio, self.sample_rate

    async def stream(self, text: str, voice: Voice) -> AsyncIterator[np.ndarray]:
        """Yield float32 mono chunks as the model produces them.

        Generation runs on the worker thread and hands chunks over through a queue. If the
        consumer stops early (client disconnected), the worker stops at the next chunk.
        """
        self._require_loaded()
        async with self._lock:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[object] = asyncio.Queue()
            stop = threading.Event()

            def produce() -> None:
                produced = 0
                try:
                    for chunk in self._model.generate(text, voice, stream=True):
                        if stop.is_set():
                            return
                        if chunk.size:
                            produced += chunk.size
                            loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    if produced == 0:
                        raise RuntimeError("model produced no audio")
                except Exception as exc:  # handed to the consumer, which re-raises it
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, _DONE)

            worker = loop.run_in_executor(self._executor, produce)
            try:
                while (item := await queue.get()) is not _DONE:
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                stop.set()
                # Hold the lock until the worker is idle. Shielded so a cancelled request still
                # lets the worker finish its current chunk instead of abandoning it mid-call.
                await asyncio.shield(worker)

    def _synthesize_sync(self, text: str, voice: Voice) -> np.ndarray:
        chunks = [chunk for chunk in self._model.generate(text, voice, stream=False) if chunk.size]
        if not chunks:
            raise RuntimeError("model produced no audio")
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("model not loaded")
