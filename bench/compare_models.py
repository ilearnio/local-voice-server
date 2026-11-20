"""Compare Breeze checkpoints (e.g. 8-bit vs 4-bit): speed (RTF) and audio files to listen to.

Usage: HF_HUB_OFFLINE=1 uv run python bench/compare_models.py MODEL [MODEL ...] [--runs 3]
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from local_voice_server.models.breeze import install_reference_cache
from local_voice_server.voices import scan_voices

REPO = Path(__file__).resolve().parents[1]
TEXTS = [
    "Got it. I will handle that right now.",
    "Discipline is doing what needs to be done, even when you don't feel like it. "
    "Most people wait for motivation. They wait, and they wait, and nothing ever happens.",
    "Let me tell you something about time. Every single day you waste is gone forever. "
    "You don't get it back. So the question is simple. What are you going to do with today?",
]


def generate(model, voice, text, seed):
    mx.random.seed(seed)
    mx.synchronize()
    started = time.perf_counter()
    chunks = model.generate(text=text, ref_audio=str(voice.reference), ref_text=voice.transcript, verbose=False)
    audio = np.concatenate([np.asarray(c.audio).reshape(-1) for c in chunks])
    return time.perf_counter() - started, audio


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--out", type=Path, default=REPO / "bench" / "out")
    p.add_argument("--voice", help="voice folder name under voices/ (default: the first one)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from mlx_audio.tts import load

    voices = scan_voices(REPO / "voices")
    if not voices:
        raise SystemExit("no voices found in voices/")
    name = args.voice or sorted(voices)[0]
    if name not in voices:
        raise SystemExit(f"voice {name!r} not found; have: {', '.join(sorted(voices))}")
    voice = voices[name]
    table: dict[str, list[tuple[float, float, float]]] = {}
    peaks: dict[str, float] = {}
    for repo in args.models:
        mx.reset_peak_memory()
        model = install_reference_cache(load(repo))
        tag = Path(repo).name
        generate(model, voice, "Warming up.", seed=0)
        rows = []
        for i, text in enumerate(TEXTS):
            runs = [generate(model, voice, text, seed=42 + run) for run in range(args.runs)]
            sf.write(args.out / f"text{i + 1}_{tag}.wav", runs[0][1], model.sample_rate)
            durations = [len(audio) / model.sample_rate for _, audio in runs]
            rows.append(
                (
                    statistics.median(durations),
                    statistics.median(elapsed for elapsed, _ in runs),
                    statistics.median(e / d for (e, _), d in zip(runs, durations)),
                )
            )
        table[tag] = rows
        peaks[tag] = mx.get_peak_memory() / 1e9
        del model
        mx.clear_cache()

    tags = list(table)
    print("sentence | audio s | " + " | ".join(f"{t} time s / RTF" for t in tags))
    for i in range(len(TEXTS)):
        cells = [f"{table[t][i][1]:.2f} / {table[t][i][2]:.3f}" for t in tags]
        print(f"text{i + 1} | {table[tags[0]][i][0]:.1f} | " + " | ".join(cells))
    overall = {t: statistics.median(r[2] for r in table[t]) for t in tags}
    base = overall[tags[0]]
    print("median RTF | | " + " | ".join(f"{overall[t]:.3f} ({base / overall[t]:.2f}x vs {tags[0]})" for t in tags))
    print("peak GB | | " + " | ".join(f"{peaks[t]:.1f}" for t in tags))

if __name__ == "__main__":
    main()
