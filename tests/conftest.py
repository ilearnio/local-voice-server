from pathlib import Path

import pytest


def make_voice(root: Path, name: str, transcript: str = "Hello there.") -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "reference.wav").write_bytes(b"RIFF....WAVE")
    (folder / "transcript.txt").write_text(transcript, encoding="utf-8")
    return folder


@pytest.fixture
def voices_dir(tmp_path: Path) -> Path:
    root = tmp_path / "voices"
    root.mkdir()
    make_voice(root, "my-voice")
    return root
