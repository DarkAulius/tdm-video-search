"""Modelli pre-addestrati usati come estrattori di embedding.

- CLIP proietta immagini e testi nello stesso spazio vettoriale: il coseno tra una query
  e un fotogramma misura quanto il testo descrive l'immagine.
- Un modello Sentence-Transformers confronta la query con le trascrizioni del parlato:
  CLIP è addestrato su didascalie brevi (max 77 token) e non è adatto a testi parlati lunghi.

Tutti gli embedding vengono normalizzati a norma 1, quindi il prodotto scalare è il coseno.
I test usano encoder fittizi con gli stessi metodi (vedi tests/fakes.py).
"""

from __future__ import annotations

import numpy as np

from .config import Config


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # GPU dei Mac con Apple Silicon
        return "mps"
    return "cpu"


class ClipEncoder:
    def __init__(self, cfg: Config):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = pick_device()
        self.batch_size = cfg.batch_size
        self.model = CLIPModel.from_pretrained(cfg.clip_model).to(self.device).eval()
        # Processore "lento" (basato su Pillow), lo stesso con cui il modello è stato pubblicato.
        self.processor = CLIPProcessor.from_pretrained(cfg.clip_model, use_fast=False)
        self.dim = self.model.config.projection_dim

    def _features(self, out) -> np.ndarray:
        # Con transformers 4.x get_*_features restituisce direttamente il tensore proiettato.
        if not isinstance(out, self.torch.Tensor):
            for name in ("image_embeds", "text_embeds", "pooler_output"):
                if getattr(out, name, None) is not None:
                    out = getattr(out, name)
                    break
        feats = out.float().cpu().numpy()
        # Immagini e testi devono stare nello stesso spazio (dimensione della proiezione).
        assert feats.shape[-1] == self.dim, f"embedding di dimensione {feats.shape[-1]}, attesa {self.dim}"
        return _normalize(feats)

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        """Embedding CLIP di una lista di immagini RGB uint8. Il processor le ridimensiona a
        224x224 e normalizza i canali con media e deviazione standard del training di CLIP."""
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        feats = []
        with self.torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                inputs = self.processor(images=images[i : i + self.batch_size], return_tensors="pt").to(self.device)
                feats.append(self._features(self.model.get_image_features(**inputs)))
        return np.concatenate(feats).astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        feats = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                inputs = self.processor(
                    text=texts[i : i + self.batch_size], padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                feats.append(self._features(self.model.get_text_features(**inputs)))
        return np.concatenate(feats).astype(np.float32)


class SentenceEncoder:
    def __init__(self, cfg: Config):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(cfg.text_model, device=pick_device())

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)
