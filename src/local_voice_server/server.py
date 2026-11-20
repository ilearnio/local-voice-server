"""Entrypoint: load the model, then serve on 127.0.0.1."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import uvicorn

from .app import create_app
from .engine import Engine
from .models import create_model
from .voices import scan_voices

logger = logging.getLogger("local_voice_server")

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST = "127.0.0.1"
DEFAULT_PORT = 8880


async def _run() -> None:
    port = int(os.environ.get("VOICE_SERVER_PORT", DEFAULT_PORT))
    voices_dir = Path(os.environ.get("VOICE_SERVER_VOICES_DIR", REPO_ROOT / "voices")).expanduser()

    try:
        model = create_model()
        engine = Engine(model)
        logger.info("loading model %s ...", engine.model_id)
        await engine.load()
    except Exception as exc:
        logger.exception("failed to load model: %s", exc)
        sys.exit(1)
    logger.info("model %s loaded (%d Hz)", engine.model_id, engine.sample_rate)

    voices = scan_voices(voices_dir)
    logger.info("voices in %s: %s", voices_dir, ", ".join(sorted(voices)) or "(none)")
    await _warm_up(engine, voices)

    config = uvicorn.Config(create_app(engine, voices_dir), host=HOST, port=port, log_level="info")
    logger.info("ready on http://%s:%d/v1", HOST, port)
    await uvicorn.Server(config).serve()


WARM_UP_TEXT = "Ready."


async def _warm_up(engine: Engine, voices: dict) -> None:
    """Run one throwaway generation per voice so GPU kernels and reference encodings are ready.

    Without this the first real request after a restart takes ~2x longer.
    A failed warm-up is logged but does not stop the server: real requests surface the same
    error to the caller with full detail.
    """
    for name, voice in sorted(voices.items()):
        started = time.perf_counter()
        try:
            await engine.synthesize(WARM_UP_TEXT, voice)
        except Exception:
            logger.exception("warm-up for voice %s failed", name)
            continue
        logger.info("warmed up voice %s in %.2fs", name, time.perf_counter() - started)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    asyncio.run(_run())


if __name__ == "__main__":
    main()
