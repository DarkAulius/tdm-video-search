"""Selezione dei fotogrammi e segmentazione in scene.

Tre strategie a confronto:

- "uniform": K fotogrammi equidistanti nel tempo. Richiede di decodificare tutta la clip,
  perché i fotogrammi P e B dipendono dai precedenti.
- "iframes": solo i fotogrammi I (intra) del GOP. Si decodificano senza riferimenti ad altri
  fotogrammi, quindi il decoder può saltare tutti gli altri (skip_frame = NONKEY). Inoltre
  l'encoder x264 inserisce un fotogramma I ai cambi di scena, quindi cadono spesso nei punti
  in cui il contenuto cambia.
- "iframes_hsv": i fotogrammi I filtrati con un rilevatore di cambi di scena basato sugli
  istogrammi HSV. Per ogni scena si tiene un solo fotogramma rappresentativo, così si
  codificano con CLIP meno immagini.

Per tutte le strategie i fotogrammi vengono raggruppati in scene con la stessa regola HSV,
e le scene diventano i segmenti temporali restituiti dalla ricerca.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np

from .config import Config


@dataclass
class Frame:
    time: float          # istante del fotogramma in secondi
    image: np.ndarray    # immagine RGB uint8 (H, W, 3)
    is_iframe: bool      # True se è un fotogramma I (chiave) del flusso compresso


@dataclass
class Selection:
    """Risultato della selezione dei fotogrammi per una clip."""

    duration: float
    frames: list[Frame]                    # fotogrammi da codificare con CLIP
    frame_segment: list[int]               # indice del segmento di ciascun fotogramma
    segments: list[tuple[float, float]]    # (inizio, fine) di ciascun segmento in secondi
    boundary_distances: list[float]        # distanza HSV che ha aperto ciascun segmento
    decoded: int                           # fotogrammi decodificati (misura del costo)
    used_fallback: bool = False            # True se i fotogrammi I erano troppo pochi
    candidate_times: list[float] = field(default_factory=list)  # fotogrammi considerati prima del filtro


# --------------------------------------------------------------------------- decodifica

def _duration(container: av.container.InputContainer, stream: av.video.stream.VideoStream) -> float:
    """Durata della clip in secondi, letta dai metadati del contenitore."""
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base  # container.duration è in microsecondi
    return 0.0


def _frame_time(frame: av.VideoFrame, index: int, fps: float) -> float:
    return float(frame.time) if frame.time is not None else index / fps


def sample_uniform(path: Path, k: int) -> tuple[list[Frame], float, int]:
    """Campiona k fotogrammi equidistanti: il primo fotogramma a partire da ciascun istante
    t_i = (i + 0.5) * durata / k. Restituisce (fotogrammi, durata, fotogrammi decodificati)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate or 25)
        duration = _duration(container, stream)
        targets = [(i + 0.5) * duration / k for i in range(k)]
        frames: list[Frame] = []
        decoded = 0
        last = None
        for decoded, frame in enumerate(container.decode(stream), start=1):
            t = _frame_time(frame, decoded - 1, fps)
            last = (t, frame)
            # La conversione in RGB si fa solo per i fotogrammi scelti: è la parte costosa.
            while len(frames) < k and t >= targets[len(frames)]:
                frames.append(Frame(t, frame.to_ndarray(format="rgb24"), bool(frame.key_frame)))
            if len(frames) == k:
                break
        # Se la durata nei metadati supera l'ultimo fotogramma, l'ultimo istante può non
        # essere mai raggiunto: si usa l'ultimo fotogramma decodificato.
        if last is not None and len(frames) < k and (not frames or frames[-1].time < last[0]):
            t, frame = last
            frames.append(Frame(t, frame.to_ndarray(format="rgb24"), bool(frame.key_frame)))
    return frames, duration, decoded


def extract_iframes(path: Path, max_frames: int) -> tuple[list[Frame], float, int]:
    """Decodifica solo i fotogrammi I. Restituisce (fotogrammi, durata, fotogrammi decodificati)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # Il decoder scarta i fotogrammi non chiave senza ricostruirli.
        stream.codec_context.skip_frame = "NONKEY"
        fps = float(stream.average_rate or 25)
        duration = _duration(container, stream)
        frames = [
            Frame(_frame_time(f, i, fps), f.to_ndarray(format="rgb24"), True)
            for i, f in enumerate(container.decode(stream))
        ]
    decoded = len(frames)
    if len(frames) > max_frames:
        keep = np.unique(np.linspace(0, len(frames) - 1, max_frames).round().astype(int))
        frames = [frames[i] for i in keep]
    return frames, duration, decoded


# --------------------------------------------------------------------------- scene (HSV)

def rgb_to_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """Conversione RGB -> HSV pixel per pixel.

    V = max(R, G, B): luminosità; S = (max - min) / max: saturazione;
    H = angolo sul cerchio dei colori (0-360 gradi), calcolato rispetto al canale dominante.
    Restituisce un array (..., 3) con H in [0, 360) e S, V in [0, 1].
    """
    rgb = image_rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    v = rgb.max(axis=-1)
    delta = v - rgb.min(axis=-1)
    s = np.where(v > 0, delta / np.maximum(v, 1e-12), 0.0)
    h = np.zeros_like(v)
    chroma = delta > 0                       # i grigi (delta = 0) non hanno tinta: H = 0
    safe = np.maximum(delta, 1e-12)
    r_max = chroma & (v == r)
    g_max = chroma & (v == g) & ~r_max
    b_max = chroma & ~r_max & ~g_max
    h[r_max] = (60.0 * (g - b)[r_max] / safe[r_max]) % 360.0
    h[g_max] = 60.0 * (b - r)[g_max] / safe[g_max] + 120.0
    h[b_max] = 60.0 * (r - g)[b_max] / safe[b_max] + 240.0
    return np.stack([h, s, v], axis=-1)


def hsv_histogram(image_rgb: np.ndarray, bins: tuple[int, int, int]) -> np.ndarray:
    """Istogramma 3D nello spazio colore HSV, normalizzato a somma 1.

    Si usa un pixel ogni due per riga e per colonna: la distribuzione dei colori
    resta praticamente la stessa e il calcolo è quattro volte più veloce.
    """
    hsv = rgb_to_hsv(image_rgb[::2, ::2]).reshape(-1, 3)
    hist, _ = np.histogramdd(hsv, bins=bins, range=[(0, 360), (0, 1), (0, 1)])
    hist = hist.ravel()
    return hist / max(hist.sum(), 1e-12)


def bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    """Distanza di Bhattacharyya (forma di Hellinger) tra due istogrammi normalizzati:
    0 se identici, 1 se non hanno nessun intervallo in comune."""
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(p * q)))))


def group_by_scene(frames: list[Frame], threshold: float, bins: tuple[int, int, int]) -> tuple[list[list[int]], list[float]]:
    """Raggruppa fotogrammi consecutivi in scene.

    Un fotogramma apre una nuova scena se il suo istogramma HSV dista più di `threshold`
    dal primo fotogramma della scena corrente. Il confronto con l'inizio della scena,
    invece che con il fotogramma precedente, rileva anche i cambiamenti graduali
    (panoramiche, dissolvenze) che tra fotogrammi vicini sarebbero piccoli.
    Restituisce (gruppi di indici, distanza che ha aperto ciascun gruppo).
    """
    if not frames:
        return [], []
    hists = [hsv_histogram(f.image, bins) for f in frames]
    groups = [[0]]
    distances = [0.0]
    for i in range(1, len(frames)):
        d = bhattacharyya(hists[groups[-1][0]], hists[i])
        if d > threshold:
            groups.append([i])
            distances.append(d)
        else:
            groups[-1].append(i)
    return groups, distances


def _segments_from_groups(frames: list[Frame], groups: list[list[int]], duration: float) -> list[tuple[float, float]]:
    """Ogni scena va dal suo primo fotogramma al primo fotogramma della scena successiva."""
    starts = [0.0] + [frames[g[0]].time for g in groups[1:]]
    ends = starts[1:] + [max(duration, frames[-1].time)]
    return [(round(s, 3), round(e, 3)) for s, e in zip(starts, ends)]


# --------------------------------------------------------------------------- strategie

def select_frames(path: Path, strategy: str, cfg: Config) -> Selection:
    """Applica una delle tre strategie e restituisce fotogrammi e segmenti."""
    used_fallback = False
    if strategy == "uniform":
        frames, duration, decoded = sample_uniform(path, cfg.uniform_k)
    elif strategy in ("iframes", "iframes_hsv"):
        frames, duration, decoded = extract_iframes(path, cfg.max_frames)
        if len(frames) < cfg.min_frames:
            # GOP lunghi o clip statiche: troppo pochi fotogrammi I per descrivere la clip.
            # Si integrano con campioni uniformi (qui serve la decodifica completa).
            extra, _, extra_decoded = sample_uniform(path, cfg.min_frames)
            known = [f.time for f in frames]
            frames += [f for f in extra if all(abs(f.time - t) > 0.05 for t in known)]
            frames.sort(key=lambda f: f.time)
            decoded += extra_decoded
            used_fallback = True
    else:
        raise ValueError(f"Strategia sconosciuta: {strategy}")

    if not frames:
        raise RuntimeError(f"Nessun fotogramma decodificato da {path}")
    if duration <= 0:
        duration = frames[-1].time + 1.0

    groups, distances = group_by_scene(frames, cfg.scene_threshold, cfg.hsv_bins)
    segments = _segments_from_groups(frames, groups, duration)
    candidate_times = [round(f.time, 3) for f in frames]

    if strategy == "iframes_hsv":
        # Un solo fotogramma per scena: quello centrale, meno esposto alle transizioni.
        keep = [g[len(g) // 2] for g in groups]
        selected = [frames[i] for i in keep]
        frame_segment = list(range(len(groups)))
    else:
        selected = frames
        frame_segment = [0] * len(frames)
        for s, g in enumerate(groups):
            for i in g:
                frame_segment[i] = s

    return Selection(
        duration=round(duration, 3),
        frames=selected,
        frame_segment=frame_segment,
        segments=segments,
        boundary_distances=[round(d, 4) for d in distances],
        decoded=decoded,
        used_fallback=used_fallback,
        candidate_times=candidate_times,
    )
