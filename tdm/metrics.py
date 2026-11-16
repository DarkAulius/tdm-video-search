"""Metriche di recupero testo -> video.

Per ogni query i (didascalia della clip i) si guarda in che posizione il sistema mette la
clip corretta tra tutte le clip della collezione:
- Recall@k: percentuale di query con la clip corretta tra i primi k risultati;
- Median Rank (MdR) e Mean Rank (MnR): posizione mediana e media della clip corretta
  (1 = prima posizione; più basso è meglio).
"""

from __future__ import annotations

import numpy as np


def ranks(scores: np.ndarray, targets: np.ndarray | None = None) -> np.ndarray:
    """Posizione (da 1) della clip corretta per ogni query.

    scores[i, j] è il punteggio della clip j per la query i; targets[i] è la clip corretta
    (di default la clip i). A parità di punteggio si conta la posizione peggiore, per non
    sovrastimare i risultati.
    """
    n = scores.shape[0]
    targets = np.arange(n) if targets is None else targets
    correct = scores[np.arange(n), targets][:, None]
    return (scores >= correct).sum(axis=1)


def summarize(r: np.ndarray) -> dict:
    return {
        "R@1": round(100 * float(np.mean(r <= 1)), 1),
        "R@5": round(100 * float(np.mean(r <= 5)), 1),
        "R@10": round(100 * float(np.mean(r <= 10)), 1),
        "MdR": float(np.median(r)),
        "MnR": round(float(np.mean(r)), 1),
    }


def recall_sum(r: np.ndarray) -> float:
    """R@1 + R@5 + R@10: criterio unico usato per scegliere i parametri sulla validazione."""
    s = summarize(r)
    return s["R@1"] + s["R@5"] + s["R@10"]
