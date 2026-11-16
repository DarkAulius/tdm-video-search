"""Estrazione della traccia audio con PyAV (binding Python delle librerie di FFmpeg).

L'audio viene ricampionato a 16 kHz mono in float32, il formato atteso da Whisper:
- 16 kHz: per il teorema di Nyquist conserva le frequenze fino a 8 kHz, che contengono
  quasi tutta l'informazione utile del parlato, con un terzo dei campioni rispetto a 48 kHz;
- mono: i canali vengono combinati (FFmpeg usa mono = (L + R) / sqrt(2)); la posizione
  spaziale non serve alla trascrizione;
- float32 in [-1, 1]: i campioni interi (es. 16 bit) vengono riscalati dal resampler.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np


def load_audio(path: Path, sample_rate: int = 16_000) -> np.ndarray | None:
    """Restituisce i campioni audio come array float32 mono, o None se la clip non ha audio."""
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # svuota il buffer interno del resampler
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return None
    return np.concatenate(chunks).astype(np.float32)


def rms_dbfs(samples: np.ndarray) -> float:
    """Livello RMS in dBFS (0 dBFS = fondo scala). Il silenzio digitale dà valori molto negativi."""
    if samples is None or samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-6))
