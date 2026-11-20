"""Breeze TTS 2 (BreezeBlue) on Apple Silicon through mlx-audio.

Weights: the full-precision (bf16) MLX build `mlx-community/Breeze-TTS-2-mlx`. Streaming hides the
generation speed, so quantized builds only trade quality for speed that is not needed at 1x playback.
Override with env `BREEZE_MODEL` (a local folder or a Hugging Face repo id).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..voices import Voice

logger = logging.getLogger(__name__)

MODEL_ID = "breeze-tts-2"
DEFAULT_WEIGHTS = "mlx-community/Breeze-TTS-2-mlx"
# Seconds of audio per streamed chunk; mlx-audio decodes this much codec output at a time.
STREAM_INTERVAL_SECONDS = 0.16
FALLBACK_SAMPLE_RATE = 24_000


def configured_weights() -> str:
    return os.environ.get("BREEZE_MODEL") or DEFAULT_WEIGHTS


class BreezeModel:
    def __init__(self, weights: str | None = None):
        self._weights = weights or configured_weights()
        self._model: Any = None
        self._sample_rate = FALLBACK_SAMPLE_RATE

    @property
    def id(self) -> str:
        return MODEL_ID

    @property
    def weights(self) -> str:
        return self._weights

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load(self) -> None:
        weights = self._weights
        if weights.startswith(("/", "~", ".")) and not Path(weights).expanduser().is_dir():
            raise FileNotFoundError(f"model folder {weights} does not exist")
        from mlx_audio.tts import load  # lazy: heavy import, keeps unit tests MLX-free

        model = load(weights)
        install_reference_cache(model)
        rate = getattr(model, "sample_rate", None)
        if rate is None:
            logger.warning("model has no sample_rate attribute; assuming %d Hz", FALLBACK_SAMPLE_RATE)
            rate = FALLBACK_SAMPLE_RATE
        self._model = model
        self._sample_rate = int(rate)

    def generate(self, text: str, voice: Voice, *, stream: bool) -> Iterator[np.ndarray]:
        if self._model is None:
            raise RuntimeError("model not loaded")
        options: dict[str, Any] = {"stream": True, "streaming_interval": STREAM_INTERVAL_SECONDS} if stream else {}
        results = self._model.generate(
            text=text, ref_audio=str(voice.reference), ref_text=voice.transcript, verbose=False, **options
        )
        try:
            for result in results:
                yield np.asarray(result.audio, dtype=np.float32).reshape(-1)
        finally:
            close = getattr(results, "close", None)
            if close is not None:
                close()


def install_reference_cache(model: Any) -> Any:
    """Cache each voice's reference encoding (lossless).

    mlx-audio re-encodes the reference clip through the codec on every request (~0.3 s for a 30 s
    clip on an M5 Max). The encoding depends only on the file, so it is computed once per
    (path, size, mtime) and reused; a changed file is re-encoded automatically.
    """
    import mlx.core as mx  # lazy: MLX only exists on Apple Silicon

    original = model._encode_reference
    per_path: dict[str, tuple[tuple[int, int], Any]] = {}

    def encode_reference(ref_audio):
        if not isinstance(ref_audio, (str, Path)):
            return original(ref_audio)
        path = os.path.abspath(str(ref_audio))
        stat = os.stat(path)
        key = (stat.st_size, stat.st_mtime_ns)
        hit = per_path.get(path)
        if hit is not None and hit[0] == key:
            return hit[1]
        codes = original(ref_audio)
        mx.eval(codes)
        per_path[path] = (key, codes)
        return codes

    model._encode_reference = encode_reference
    return model
