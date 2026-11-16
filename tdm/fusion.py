"""Fusione tardiva dei punteggi visivo e del parlato, e taratura del peso.

I due punteggi vengono da modelli diversi (CLIP e Sentence-Transformers) e hanno scale
diverse: non si possono sommare direttamente. Per ogni query si standardizzano i punteggi
sulle clip candidate (z-score: media 0, deviazione standard 1) e poi si combinano:

    punteggio = z_visivo + w * z_parlato

Le clip senza parlato ricevono z_parlato = 0, cioè il valore medio: il parlato non le
premia né le penalizza. Il peso w si sceglie sulla validazione, mai sul test.
"""

from __future__ import annotations

import numpy as np

from .metrics import ranks, recall_sum


def zscore_rows(scores: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Standardizza ogni riga (query). Se `mask` è dato, media e deviazione standard si
    calcolano solo sulle colonne valide e le altre colonne valgono 0."""
    if mask is None:
        mask = np.ones(scores.shape[1], dtype=bool)
    out = np.zeros_like(scores, dtype=np.float64)
    if mask.sum() < 2:
        return out
    valid = scores[:, mask]
    mu = valid.mean(axis=1, keepdims=True)
    sd = valid.std(axis=1, keepdims=True) + 1e-8
    out[:, mask] = (valid - mu) / sd
    return out


def fuse(vis: np.ndarray, speech: np.ndarray, has_speech: np.ndarray, w: float) -> np.ndarray:
    return zscore_rows(vis) + w * zscore_rows(speech, has_speech)


def tune_weight(vis: np.ndarray, speech: np.ndarray, has_speech: np.ndarray, grid) -> tuple[float, list[dict]]:
    """Prova ogni peso della griglia e restituisce quello con R@1+R@5+R@10 più alto.
    A parità si preferisce il peso più piccolo (modello più semplice)."""
    curve = []
    for w in grid:
        curve.append({"w": float(w), "recall_sum": recall_sum(ranks(fuse(vis, speech, has_speech, w)))})
    best = max(curve, key=lambda c: (c["recall_sum"], -c["w"]))
    return best["w"], curve
