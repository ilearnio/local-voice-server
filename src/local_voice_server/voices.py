"""Folder-based voice registry: voices/<name>/reference.wav + transcript.txt."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

REFERENCE_FILE = "reference.wav"
TRANSCRIPT_FILE = "transcript.txt"


@dataclass(frozen=True)
class Voice:
    name: str
    reference: Path
    transcript: str


def scan_voices(root: Path) -> dict[str, Voice]:
    """Return every valid voice folder under ``root``, keyed by folder name.

    Invalid folders (missing reference, missing or empty transcript) are skipped
    with a warning so one broken voice never takes the others down.
    """
    if not root.is_dir():
        logger.warning("voices folder %s does not exist; no voices available", root)
        return {}

    voices: dict[str, Voice] = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        reference = folder / REFERENCE_FILE
        transcript_path = folder / TRANSCRIPT_FILE
        if not reference.is_file():
            logger.warning("skipping voice %r: %s is missing", folder.name, reference)
            continue
        if not transcript_path.is_file():
            logger.warning("skipping voice %r: %s is missing", folder.name, transcript_path)
            continue
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        if not transcript:
            logger.warning("skipping voice %r: %s is empty", folder.name, transcript_path)
            continue
        voices[folder.name] = Voice(folder.name, reference, transcript)
    return voices
