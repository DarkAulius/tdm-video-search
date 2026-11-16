"""Ricerca ibrida: filtri strutturati via SPARQL sul knowledge graph + ranking semantico.

1. SPARQL restituisce le clip che rispettano i filtri (categoria, durata, parlato);
2. la query viene codificata con CLIP (per i fotogrammi) e con Sentence-Transformers
   (per le trascrizioni);
3. i punteggi visivo e del parlato vengono fusi con il peso scelto sulla validazione;
4. per ogni clip restituita si indica il segmento più pertinente come URI Media Fragments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdflib import Graph

from .fusion import fuse
from .index import SpeechIndex, VisualIndex
from .kg import candidate_ids, fragment_uri, speech_fragments


@dataclass
class Hit:
    rank: int
    video_id: str
    score: float
    fragment: str        # URI Media Fragments del segmento più pertinente
    start: float
    end: float
    speech: str          # parlato nel segmento (se presente)


class SearchEngine:
    def __init__(self, visual: VisualIndex, speech: SpeechIndex, graph: Graph,
                 clip_encoder, text_encoder, aggregation: str, w: float):
        self.visual, self.speech, self.graph = visual, speech, graph
        self.clip_encoder, self.text_encoder = clip_encoder, text_encoder
        self.aggregation, self.w = aggregation, w
        self._speech_pos = {vid: i for i, vid in enumerate(speech.video_ids)}

    def search(self, query: str, top_k: int = 5, category: str | None = None,
               min_duration: float | None = None, max_duration: float | None = None,
               has_speech: bool | None = None) -> list[Hit]:
        ids = set(candidate_ids(self.graph, category, min_duration, max_duration, has_speech))
        cols = np.array([i for i, vid in enumerate(self.visual.video_ids) if vid in ids and vid in self._speech_pos])
        if cols.size == 0:
            return []
        speech_cols = np.array([self._speech_pos[self.visual.video_ids[i]] for i in cols])

        q_clip = self.clip_encoder.encode_texts([query])
        q_text = self.text_encoder.encode([query])
        s_vis = self.visual.scores(q_clip, self.aggregation, cols)
        s_sp = self.speech.scores(q_text, speech_cols)
        scores = fuse(s_vis, s_sp, self.speech.has_speech[speech_cols], self.w)[0]

        hits = []
        for rank, j in enumerate(np.argsort(-scores)[:top_k], start=1):
            clip_idx = int(cols[j])
            vid = self.visual.video_ids[clip_idx]
            seg = self.visual.best_segment(q_clip[0], clip_idx)
            start, end = (float(x) for x in self.visual.seg_bounds[seg])
            spoken = [f["text"] for f in speech_fragments(self.graph, vid) if f["end"] > start and f["start"] < end]
            hits.append(Hit(rank, vid, round(float(scores[j]), 3), str(fragment_uri(vid, start, end)),
                            start, end, " ".join(spoken)))
        return hits
