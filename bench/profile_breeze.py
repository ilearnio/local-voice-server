"""Profile where Breeze generation time goes (reference encode, prefill, per-frame loop, codec decode).

Usage: HF_HUB_OFFLINE=1 uv run python bench/profile_breeze.py [--text "..."] [--runs 3]
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

from local_voice_server.models.breeze import DEFAULT_WEIGHTS as MODEL_REPO
from local_voice_server.voices import scan_voices

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TEXT = (
    "Discipline is doing what needs to be done, even when you don't feel like it. "
    "Most people wait for motivation. They wait, and they wait, and nothing ever happens."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--voice", default="my-voice")
    parser.add_argument("--model", default=MODEL_REPO)
    args = parser.parse_args()

    from mlx_audio.tts import load

    model = load(args.model)
    voice = scan_voices(REPO / "voices")[args.voice]
    timings: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    def timed(name, fn):
        def wrapper(*a, **k):
            mx.synchronize()
            t = time.perf_counter()
            out = fn(*a, **k)
            if isinstance(out, mx.array):
                mx.eval(out)
            mx.synchronize()
            timings[name] += time.perf_counter() - t
            counts[name] += 1
            return out

        return wrapper

    model._encode_reference = timed("ref_encode (codec)", model._encode_reference)
    model.text_encoder.__call__  # noqa: B018 - ensure attribute exists
    model._depth_tokens = timed("depth_decoder (15 codebooks/frame)", model._depth_tokens)
    model._sample = timed("backbone sample (+lm_head)", model._sample)
    model._decode_codes = timed("codec decode", model._decode_codes)
    orig_backbone = model.backbone_model.__call__
    backbone_cls = type(model.backbone_model)
    backbone_cls.__call__ = timed("backbone forward (prefill+steps)", orig_backbone.__func__)

    for run in range(args.runs + 1):  # first run is warm-up
        timings.clear()
        counts.clear()
        mx.synchronize()
        t0 = time.perf_counter()
        chunks = list(
            model.generate(text=args.text, ref_audio=str(voice.reference), ref_text=voice.transcript, verbose=False)
        )
        mx.synchronize()
        total = time.perf_counter() - t0
        samples = sum(c.audio.shape[0] for c in chunks)
        dur = samples / model.sample_rate
        label = "warm-up" if run == 0 else f"run {run}"
        print(f"\n== {label}: {total:.2f}s for {dur:.2f}s audio  RTF {total / dur:.3f}")
        accounted = 0.0
        for name, secs in sorted(timings.items(), key=lambda kv: -kv[1]):
            accounted += secs
            print(f"   {name:<40} {secs:6.2f}s  {100 * secs / total:5.1f}%  calls={counts[name]}")
        print(f"   {'other (python, text encoder, glue)':<40} {total - accounted:6.2f}s  {100 * (total - accounted) / total:5.1f}%")


if __name__ == "__main__":
    main()
