"""Model adapters. Each adapter wraps one TTS model behind the `TtsModel` interface.

To add a model: create `models/<name>.py` with a class implementing `TtsModel`, and register a
factory in `ADAPTERS`. The server picks one with env `VOICE_MODEL` (default: `breeze`).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import Protocol

import numpy as np

from ..voices import Voice


class TtsModel(Protocol):
    """A voice-cloning TTS model. All methods are called on one worker thread, never concurrently."""

    @property
    def id(self) -> str:
        """Public model id listed on /v1/models, e.g. "breeze-tts-2"."""
        ...

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz; valid after `load()`."""
        ...

    def load(self) -> None:
        """Load weights. Raise if they are missing; never download silently."""
        ...

    def generate(self, text: str, voice: Voice, *, stream: bool) -> Iterator[np.ndarray]:
        """Yield float32 mono chunks for `text` spoken in `voice`.

        With `stream=True` chunks should be small and arrive as soon as possible; with
        `stream=False` the adapter may yield the whole utterance at once.
        """
        ...


def _breeze() -> TtsModel:
    from .breeze import BreezeModel

    return BreezeModel()


ADAPTERS: dict[str, Callable[[], TtsModel]] = {"breeze": _breeze}
DEFAULT_ADAPTER = "breeze"


def create_model(name: str | None = None) -> TtsModel:
    name = name or os.environ.get("VOICE_MODEL") or DEFAULT_ADAPTER
    factory = ADAPTERS.get(name)
    if factory is None:
        raise ValueError(f"unknown VOICE_MODEL {name!r}; available: {', '.join(sorted(ADAPTERS))}")
    return factory()
