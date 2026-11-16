"""Strumenti per i test: video sintetici ed encoder fittizi.

Gli encoder fittizi hanno gli stessi metodi di quelli reali ma non scaricano modelli:
l'"embedding" di un'immagine è il suo istogramma HSV, quello di un testo è l'istogramma
di un'immagine del colore nominato nel testo. Bastano a verificare che la pipeline
colleghi correttamente fotogrammi, segmenti, punteggi e metriche.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from tdm.keyframes import hsv_histogram

COLORS = {
    "red": (220, 30, 30),
    "green": (30, 200, 40),
    "blue": (30, 40, 220),
    "yellow": (230, 220, 30),
}
BINS = (16, 4, 4)


def make_video(path: Path, scenes: list[tuple[str, float]], fps: int = 25, gop: int = 25,
               audio_tone: float | None = 440.0, size: tuple[int, int] = (160, 120)) -> Path:
    """Crea un mp4 H.264 con una scena a tinta unita (e un quadrato in movimento) per ogni
    elemento di `scenes`, un fotogramma I ogni `gop` fotogrammi e, se richiesto, un tono
    sinusoidale stereo a 44.1 kHz."""
    w, h = size
    with av.open(str(path), mode="w") as container:
        v = container.add_stream("libx264", rate=fps)
        v.width, v.height, v.pix_fmt = w, h, "yuv420p"
        v.codec_context.gop_size = gop
        # Nessun fotogramma I aggiuntivo ai cambi di scena: i fotogrammi I sono solo quelli del GOP.
        v.options = {"sc_threshold": "0", "bf": "0"}
        a = None
        if audio_tone is not None:
            a = container.add_stream("aac", rate=44100, layout="stereo")
        i = 0
        for color, seconds in scenes:
            for _ in range(int(seconds * fps)):
                img = np.zeros((h, w, 3), dtype=np.uint8)
                img[:] = COLORS[color]
                x = (i * 3) % (w - 20)
                img[50:70, x : x + 20] = 255  # quadrato bianco in movimento
                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                frame.pts = i
                frame.time_base = Fraction(1, fps)
                for packet in v.encode(frame):
                    container.mux(packet)
                i += 1
        for packet in v.encode():
            container.mux(packet)

        if a is not None:
            total = sum(s for _, s in scenes)
            sr, n = 44100, 1024
            t = np.arange(int(total * sr)) / sr
            tone = (0.5 * np.sin(2 * np.pi * audio_tone * t)).astype(np.float32)
            for start in range(0, len(tone), n):
                chunk = tone[start : start + n]
                stereo = np.stack([chunk, chunk])  # formato planare (canali, campioni)
                frame = av.AudioFrame.from_ndarray(stereo, format="fltp", layout="stereo")
                frame.sample_rate = sr
                frame.pts = start
                frame.time_base = Fraction(1, sr)
                for packet in a.encode(frame):
                    container.mux(packet)
            for packet in a.encode():
                container.mux(packet)
    return path


def color_image(color: str) -> np.ndarray:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:] = COLORS[color]
    return img


def _unit(x: np.ndarray) -> np.ndarray:
    return (x / max(np.linalg.norm(x), 1e-12)).astype(np.float32)


def _hash_vector(text: str, dim: int) -> np.ndarray:
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(size=dim)


class FakeClipEncoder:
    dim = int(np.prod(BINS))

    def encode_images(self, images):
        return np.stack([_unit(hsv_histogram(im, BINS)) for im in images]) if images else np.zeros((0, self.dim), np.float32)

    def encode_texts(self, texts):
        out = []
        for t in texts:
            colors = [c for c in COLORS if c in t.lower()]
            vec = sum(hsv_histogram(color_image(c), BINS) for c in colors) if colors else _hash_vector(t, self.dim)
            out.append(_unit(np.asarray(vec, dtype=np.float64)))
        return np.stack(out)


class FakeTextEncoder:
    """Bag-of-words con hashing delle parole in 64 dimensioni."""

    dim = 64

    def encode(self, texts):
        out = []
        for t in texts:
            vec = np.zeros(self.dim)
            for word in t.lower().split():
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
            out.append(_unit(vec) if vec.any() else _unit(np.ones(self.dim)))
        return np.stack(out)
