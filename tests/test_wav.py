import struct

import numpy as np

from local_voice_server.wav import encode_wav_pcm16


def test_encode_clips_and_header():
    wav = encode_wav_pcm16(np.array([2.0, -2.0, 0.0], dtype=np.float32), 24_000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert struct.unpack("<I", wav[4:8])[0] == 36 + 6
    assert struct.unpack("<I", wav[40:44])[0] == 6
    assert list(struct.unpack("<3h", wav[44:])) == [32767, -32767, 0]
