import numpy as np

from tdm.asr import filter_segments
from tdm.audio import load_audio, rms_dbfs
from tdm.config import Config


def test_audio_resampled_to_16k_mono(three_scene_video):
    samples = load_audio(three_scene_video, 16_000)
    assert samples.dtype == np.float32 and samples.ndim == 1
    assert abs(len(samples) / 16_000 - 6.0) < 0.15
    # Sinusoide di ampiezza 0.5 su entrambi i canali. Il downmix di FFmpeg fa
    # mono = (L + R) / sqrt(2): con canali identici l'ampiezza diventa 0.5 * sqrt(2),
    # quindi RMS = 0.5 e il livello è circa -6 dBFS (non -9 come sul singolo canale).
    assert abs(rms_dbfs(samples) - 20 * np.log10(0.5)) < 1.0


def test_no_audio_track(long_gop_video):
    assert load_audio(long_gop_video) is None
    assert rms_dbfs(np.zeros(16_000, dtype=np.float32)) < -100


def test_whisper_segment_filter():
    cfg = Config()
    segments = [
        {"start": 0, "end": 2, "text": " a man explains the recipe", "no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.2},
        {"start": 2, "end": 4, "text": " Thank you.", "no_speech_prob": 0.9, "avg_logprob": -1.5, "compression_ratio": 1.0},
        {"start": 4, "end": 6, "text": " la la la la la la la la", "no_speech_prob": 0.2, "avg_logprob": -0.5, "compression_ratio": 3.1},
    ]
    kept = filter_segments(segments, cfg)
    assert [s["text"] for s in kept] == ["a man explains the recipe"]
