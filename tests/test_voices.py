import logging
from pathlib import Path

from local_voice_server.voices import Voice, scan_voices

from .conftest import make_voice


def test_scan_finds_valid_voice(tmp_path: Path):
    folder = make_voice(tmp_path, "my-voice", "  Hello there.\n")
    assert scan_voices(tmp_path) == {"my-voice": Voice("my-voice", folder / "reference.wav", "Hello there.")}


def test_scan_skips_missing_reference(tmp_path: Path, caplog):
    folder = make_voice(tmp_path, "broken")
    (folder / "reference.wav").unlink()
    with caplog.at_level(logging.WARNING):
        assert scan_voices(tmp_path) == {}
    assert "broken" in caplog.text


def test_scan_skips_missing_transcript(tmp_path: Path, caplog):
    folder = make_voice(tmp_path, "broken")
    (folder / "transcript.txt").unlink()
    with caplog.at_level(logging.WARNING):
        assert scan_voices(tmp_path) == {}
    assert "broken" in caplog.text


def test_scan_skips_empty_transcript(tmp_path: Path, caplog):
    make_voice(tmp_path, "blank", "  \n\t")
    with caplog.at_level(logging.WARNING):
        assert scan_voices(tmp_path) == {}
    assert "blank" in caplog.text


def test_scan_ignores_files_and_hidden_dirs(tmp_path: Path):
    (tmp_path / "foo.txt").write_text("x")
    make_voice(tmp_path, ".hidden")
    assert scan_voices(tmp_path) == {}


def test_scan_missing_root_returns_empty_with_warning(tmp_path: Path, caplog):
    with caplog.at_level(logging.WARNING):
        assert scan_voices(tmp_path / "nope") == {}
    assert "does not exist" in caplog.text
