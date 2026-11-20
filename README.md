# Local voice server

A local text-to-speech server for [OpenBot](https://github.com/nightly-labs/openbot) and other clients
of the OpenAI speech API. It speaks in cloned voices on your Mac.

- **Streaming:** `response_format: "pcm"` streams audio while it is generated. The first audio
  arrives in about 0.14 s on an M5 Max, also with the full-precision model.
- **Pluggable models:** the first adapter runs [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
  on the Apple silicon GPU through MLX (`mlx-audio` 0.6.7), at full precision (bf16).
- **Voice folders:** each voice is a folder with a reference clip and its transcript.

| Route | Returns |
|---|---|
| `GET  /v1/models` | the loaded model (task `text-to-speech`), e.g. `breeze-tts-2` |
| `GET  /v1/audio/voices` | `{"voices": [...]}`, one per folder in `voices/` |
| `POST /v1/audio/speech` | `{model, input, voice, response_format}`. `"wav"` returns one mono WAV. `"pcm"` streams raw mono PCM16 while it is generated, with the rate in `X-Sample-Rate` (24 kHz for Breeze) |

It listens on `http://127.0.0.1:8880/v1` only, so it is not reachable from other machines.

## Quick start (Apple silicon)

Requirements: macOS on Apple silicon with 32 GB of memory or more, [uv](https://docs.astral.sh/uv/),
and about 8 GB of free disk space.

```bash
git clone https://github.com/ilearnio/local-voice-server.git
cd local-voice-server
uv sync

# 1. Download the Breeze model (7.6 GB, once).
uv run hf download mlx-community/Breeze-TTS-2-mlx

# 2. Add a voice: a clean 15–30 s clip of one speaker, plus its exact transcript.
mkdir -p voices/my-voice
cp /path/to/clip.wav voices/my-voice/reference.wav
echo "The exact words spoken in the clip." > voices/my-voice/transcript.txt

# 3. Run it in the foreground...
HF_HUB_OFFLINE=1 uv run local-voice-server
# ...or install it as a login service that restarts if it crashes.
scripts/install-launchagent.sh
```

Why full precision: the audio streams, so a reply starts as soon as the first audio is ready. The
model only has to stay ahead of playback, and the full-precision model does that with room to spare at
normal speed. It sounds clearly better than the quantized builds.

On a slower Mac, or to play replies faster than about 1.5x, use a quantized build instead, for example
`BREEZE_MODEL=mlx-community/Breeze-TTS-2-mlx-8bit` (4.6 GB). Download it first with `uv run hf download`.

Try it:

```bash
curl -s -X POST http://127.0.0.1:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"voice":"my-voice","input":"(laugh) Hello from the local voice server."}' \
  -o out.wav && afplay out.wav
```

## Use it with OpenBot

In OpenBot, open Settings → Voice → Voice server:

- Address: `http://127.0.0.1:8880/v1`
- API key: leave empty
- Spoken replies: on. OpenBot picks the model and the voice.

OpenBot builds that stream spoken replies (`voice:speak-stream`) start to speak after about 0.15 s,
for a short answer or a long text. Older builds request WAV, which this server also supports.

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `VOICE_MODEL` | `breeze` | which adapter to load |
| `VOICE_SERVER_PORT` | `8880` | port on 127.0.0.1 |
| `VOICE_SERVER_VOICES_DIR` | `<repo>/voices` | voices folder |
| `BREEZE_MODEL` | `mlx-community/Breeze-TTS-2-mlx` | Breeze weights: a local folder or HF repo id |

## Voices

Each folder in `voices/` is one voice, named after the folder:

```
voices/my-voice/reference.wav     clean speech, ideally 15–30 s, one speaker, no music
voices/my-voice/transcript.txt    the exact words spoken in reference.wav
```

New folders show up at once, with no restart. A folder missing a file, or with an empty transcript,
is skipped with a warning in the log.

Breeze speaks English and Chinese. Inline events like `(laugh)`, `(sigh)` and `(clears throat)` work.

## Service

`scripts/install-launchagent.sh` installs a LaunchAgent (`io.github.ilearnio.local-voice-server`;
set `VOICE_SERVER_LABEL` to change it). It starts at login, restarts after a crash, and warms up
each voice at start. It starts with `HF_HUB_OFFLINE=1`, so it never downloads at boot.

```bash
launchctl kickstart -k gui/$(id -u)/io.github.ilearnio.local-voice-server   # restart
launchctl bootout gui/$(id -u)/io.github.ilearnio.local-voice-server        # stop and unload
tail -f ~/Library/Logs/local-voice-server/server.log                        # logs
```

## Adding a model

1. Create `src/local_voice_server/models/<name>.py` with a class that implements `TtsModel`
   (`id`, `sample_rate`, `load()`, `generate(text, voice, *, stream)` yielding float32 mono chunks).
2. Register it in `ADAPTERS` in `models/__init__.py`.
3. Set `VOICE_MODEL=<name>`.

Routes, voices, streaming, warm-up and the request queue need no changes.

## Performance (M5 Max, `bench/compare_models.py`)

Median real-time factor: the time to make the audio divided by its length (lower is faster). Streamed
playback runs out of audio when this is more than 1 divided by the playback speed.

| Model | Real-time factor | Long reply (885 characters) | Peak memory |
|---|---|---|---|
| bf16 (default) | 0.480 | 0.49–0.64 | 15.4 GB |
| 8-bit | 0.342 | | |
| 5-bit (local conversion) | 0.288 | 0.33 | 7.9 GB |
| 4-bit (local conversion) | 0.275 | | |

A local conversion is `uv run python -m mlx_audio.convert --hf-path <bf16 snapshot folder>
--mlx-path models/<name> -q --q-bits 5 --q-group-size 64 --q-mode affine`.

The server caches each voice's reference encoding. This changes no output and saves about 0.3 s
per request.

## Tests

```bash
uv run pytest -m "not slow"                       # unit tests, no model
VOICE_SERVER_SLOW=1 uv run pytest -k real         # loads the real Breeze model
```

## Licence and responsible use

The code is MIT licensed (see `LICENSE`). The Breeze TTS 2 weights are not part of this repository.
BreezeBlue licenses them for **research and non-commercial use only**.

Cloned voices are convincing. Clone only voices that you have the right to use. Label generated
audio as AI-generated, and never present it as something a real person said.
