"""PCM16 encoding: raw little-endian samples, and a minimal mono WAV wrapper around them."""

from __future__ import annotations

import struct

import numpy as np


def encode_pcm16(audio: np.ndarray) -> bytes:
    samples = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return np.round(samples * 32767.0).astype("<i2").tobytes()


def encode_wav_pcm16(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = encode_pcm16(audio)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # PCM
        1,  # mono
        sample_rate,
        sample_rate * 2,  # byte rate
        2,  # block align
        16,  # bits per sample
        b"data",
        len(pcm),
    )
    return header + pcm
