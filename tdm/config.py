"""Parametri del sistema, raccolti in un unico punto.

Ogni valore è commentato con il motivo della scelta, così è facile
cambiarlo e capire l'effetto (per esempio la soglia dei cambi di scena).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Seme unico per tutte le scelte casuali (campione di validazione, didascalia scelta, torch).
SEED = 42

# Strategie di selezione dei fotogrammi confrontate nell'elaborato.
STRATEGIES = ("uniform", "iframes", "iframes_hsv")

# Modi di aggregare gli embedding dei fotogrammi in un punteggio per clip.
AGGREGATIONS = ("mean_frames", "mean_segments", "max_segments")


@dataclass
class Config:
    # --- Percorsi -----------------------------------------------------------
    data_dir: Path = Path("data")        # video MSR-VTT e metadati scaricati
    cache_dir: Path = Path("cache")      # risultati intermedi (embedding, trascrizioni)
    results_dir: Path = Path("results")  # metriche, grafo RDF, tabelle

    # --- Dati ---------------------------------------------------------------
    hf_repo: str = "friedrichor/MSR-VTT"  # metadati di MSR-VTT (didascalie, split, categorie)
    # Video: i file originali di MSR-VTT, con traccia audio, dall'archivio distribuito dagli
    # autori di Frozen in Time (la fonte indicata da CLIP4Clip). Si possono elencare più
    # sorgenti (URL di un .zip o repository Hugging Face), provate in ordine per i video mancanti.
    # I video di "friedrichor/MSR-VTT" non si usano: sono ricodificati a 3 fps e senza audio.
    video_sources: tuple[str, ...] = (
        "https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip",
    )
    video_tag: str = "msrvtt_originali"   # nome della cartella dei video e della cache
    n_test: int = 1000                     # split di test standard "1k-A"
    n_val: int = 300                       # clip del training usate per tarare i parametri
    # La validazione si prende tra le clip del test originale di MSR-VTT (id >= 7010) non
    # incluse in 1k-A: sono clip dello stesso tipo e periodo di quelle di test.
    val_min_id: int = 7010

    # --- Selezione dei fotogrammi -------------------------------------------
    uniform_k: int = 8          # fotogrammi per clip nel campionamento uniforme
    min_frames: int = 3         # sotto questa soglia di fotogrammi I si integra con campioni uniformi
    max_frames: int = 32        # tetto per contenere il costo su clip con GOP molto corti
    # Istogramma HSV: più livelli sulla tinta (H), pochi su saturazione e luminosità (S, V),
    # così il rilevatore reagisce ai cambi di contenuto e poco a flash o variazioni di esposizione.
    hsv_bins: tuple[int, int, int] = (16, 4, 4)
    # Distanza di Bhattacharyya (0 = istogrammi identici, 1 = disgiunti) oltre la quale
    # un fotogramma apre una nuova scena.
    scene_threshold: float = 0.35

    # --- Audio e parlato ----------------------------------------------------
    sample_rate: int = 16_000    # Whisper è addestrato a 16 kHz: banda utile fino a 8 kHz (Nyquist)
    silence_dbfs: float = -50.0  # sotto questo livello RMS la clip è considerata muta
    whisper_model: str = "base"  # "small" è più preciso ma circa 3 volte più lento
    # Regole di Whisper per scartare segmenti senza parlato o allucinati.
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    min_words: int = 3           # trascrizioni più corte sono quasi sempre rumore ("Thank you.")

    # --- Modelli pre-addestrati ---------------------------------------------
    clip_model: str = "openai/clip-vit-base-patch32"
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32

    # --- Fusione ------------------------------------------------------------
    # Pesi provati per il parlato: punteggio = z_visivo + w * z_parlato.
    weight_grid: tuple[float, ...] = field(default_factory=lambda: tuple(i / 20 for i in range(21)))

    @property
    def source_tag(self) -> str:
        """Nome della sorgente video usato nei percorsi: video e cache di sorgenti diverse
        restano separati e non si mescolano."""
        return self.video_tag

    def split_cache(self, split: str) -> Path:
        path = self.cache_dir / self.source_tag / split
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def video_dir(self) -> Path:
        path = self.data_dir / f"videos_{self.source_tag}"
        path.mkdir(parents=True, exist_ok=True)
        return path


def set_seed(seed: int = SEED) -> None:
    """Fissa i semi casuali per rendere l'esecuzione riproducibile."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # i test girano anche senza torch
        pass
