"""Costruzione degli indici visivo e del parlato, e calcolo dei punteggi query-clip.

Indice visivo: per ogni clip gli embedding CLIP dei fotogrammi selezionati, ciascuno
associato al proprio segmento (scena). Da qui si ottengono tre modi di aggregare:
- "mean_frames":   media di tutti i fotogrammi della clip;
- "mean_segments": prima la media per segmento, poi la media dei segmenti; ogni scena pesa
                   uguale, anche se dura poco, e una scena lunga e statica non domina;
- "max_segments":  il punteggio della clip è quello del suo segmento migliore; premia le
                   query che descrivono una sola scena della clip.

Indice del parlato: un embedding testuale della trascrizione per ogni clip con parlato.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import load_audio
from .config import Config
from .dataset import Clip, video_path
from .keyframes import select_frames


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


# =========================================================================== indice visivo

@dataclass
class VisualIndex:
    video_ids: list[str]
    frame_emb: np.ndarray    # (F, d) embedding CLIP dei fotogrammi
    frame_seg: np.ndarray    # (F,)   indice globale del segmento di ciascun fotogramma
    frame_time: np.ndarray   # (F,)   istante del fotogramma (s)
    seg_clip: np.ndarray     # (S,)   indice della clip di ciascun segmento (segmenti contigui per clip)
    seg_bounds: np.ndarray   # (S, 2) inizio e fine di ciascun segmento (s)
    clip_meta: list[dict]    # durata, fotogrammi decodificati, distanze HSV, ...
    stats: dict              # tempi e conteggi complessivi della strategia

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path.with_suffix(".npz"),
            frame_emb=self.frame_emb, frame_seg=self.frame_seg, frame_time=self.frame_time,
            seg_clip=self.seg_clip, seg_bounds=self.seg_bounds,
        )
        meta = {"video_ids": self.video_ids, "clip_meta": self.clip_meta, "stats": self.stats}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, path: Path) -> "VisualIndex":
        arrays = np.load(path.with_suffix(".npz"))
        meta = json.loads(path.with_suffix(".json").read_text())
        return cls(meta["video_ids"], arrays["frame_emb"], arrays["frame_seg"], arrays["frame_time"],
                   arrays["seg_clip"], arrays["seg_bounds"], meta["clip_meta"], meta["stats"])

    # ------------------------------------------------------------------ aggregazione

    def segment_embeddings(self) -> np.ndarray:
        """Media normalizzata degli embedding dei fotogrammi di ciascun segmento."""
        n_seg = len(self.seg_clip)
        sums = np.zeros((n_seg, self.frame_emb.shape[1]), dtype=np.float64)
        np.add.at(sums, self.frame_seg, self.frame_emb)
        return _normalize(sums).astype(np.float32)

    def clip_embeddings(self, mode: str) -> np.ndarray:
        n = len(self.video_ids)
        if mode == "mean_frames":
            frame_clip = self.seg_clip[self.frame_seg]
            sums = np.zeros((n, self.frame_emb.shape[1]))
            np.add.at(sums, frame_clip, self.frame_emb)
        elif mode == "mean_segments":
            sums = np.zeros((n, self.frame_emb.shape[1]))
            np.add.at(sums, self.seg_clip, self.segment_embeddings())
        else:
            raise ValueError(f"clip_embeddings non definito per {mode}")
        return _normalize(sums).astype(np.float32)

    def scores(self, query_emb: np.ndarray, mode: str, clip_subset: np.ndarray | None = None) -> np.ndarray:
        """Matrice (query, clip) di similarità coseno visiva secondo il modo di aggregazione."""
        if mode == "max_segments":
            seg_scores = query_emb @ self.segment_embeddings().T            # (Q, S)
            starts = np.flatnonzero(np.r_[True, np.diff(self.seg_clip) != 0])  # primo segmento di ogni clip
            out = np.maximum.reduceat(seg_scores, starts, axis=1)          # massimo per clip
        else:
            out = query_emb @ self.clip_embeddings(mode).T
        return out if clip_subset is None else out[:, clip_subset]

    def best_segment(self, query_emb: np.ndarray, clip_idx: int) -> int:
        """Indice globale del segmento della clip più simile alla query (per la localizzazione)."""
        seg_ids = np.flatnonzero(self.seg_clip == clip_idx)
        seg_scores = self.segment_embeddings()[seg_ids] @ query_emb.reshape(-1)
        return int(seg_ids[np.argmax(seg_scores)])


def build_visual_index(cfg: Config, clips: list[Clip], strategy: str, encoder) -> VisualIndex:
    """Seleziona i fotogrammi con la strategia indicata e li codifica con CLIP."""
    embs, frame_seg, frame_time, seg_clip, seg_bounds, clip_meta = [], [], [], [], [], []
    t_select = t_encode = 0.0
    n_decoded = n_fallback = 0
    video_ids = []

    for n_done, clip in enumerate(clips, start=1):
        if n_done % 100 == 0:
            print(f"  {n_done}/{len(clips)} clip ({t_select + t_encode:.0f} s)")
        t0 = time.perf_counter()
        try:
            sel = select_frames(video_path(cfg, clip), strategy, cfg)
        except Exception as exc:  # video corrotto: la clip viene esclusa e segnalata
            print(f"  salto {clip.video_id}: {exc}")
            continue
        t1 = time.perf_counter()
        emb = encoder.encode_images([f.image for f in sel.frames])
        t2 = time.perf_counter()
        t_select += t1 - t0
        t_encode += t2 - t1

        clip_idx = len(video_ids)
        video_ids.append(clip.video_id)
        seg_offset = len(seg_clip)
        embs.append(emb)
        frame_seg += [seg_offset + s for s in sel.frame_segment]
        frame_time += [f.time for f in sel.frames]
        seg_clip += [clip_idx] * len(sel.segments)
        seg_bounds += sel.segments
        n_decoded += sel.decoded
        n_fallback += int(sel.used_fallback)
        clip_meta.append({
            "duration": sel.duration,
            "decoded": sel.decoded,
            "encoded": len(sel.frames),
            "used_fallback": sel.used_fallback,
            "candidate_times": sel.candidate_times,
            "frame_is_iframe": [f.is_iframe for f in sel.frames],
            "boundary_distances": sel.boundary_distances,
        })

    n = max(len(video_ids), 1)
    stats = {
        "strategy": strategy,
        "clips": len(video_ids),
        "frames_encoded_per_clip": round(sum(m["encoded"] for m in clip_meta) / n, 2),
        "frames_decoded_per_clip": round(n_decoded / n, 1),
        "segments_per_clip": round(len(seg_clip) / n, 2),
        "fallback_clips": n_fallback,
        "selection_seconds": round(t_select, 1),
        "encoding_seconds": round(t_encode, 1),
        "total_seconds": round(t_select + t_encode, 1),
    }
    return VisualIndex(
        video_ids=video_ids,
        frame_emb=np.concatenate(embs).astype(np.float32),
        frame_seg=np.asarray(frame_seg, dtype=np.int64),
        frame_time=np.asarray(frame_time, dtype=np.float32),
        seg_clip=np.asarray(seg_clip, dtype=np.int64),
        seg_bounds=np.asarray(seg_bounds, dtype=np.float32).reshape(-1, 2),
        clip_meta=clip_meta,
        stats=stats,
    )


# =========================================================================== indice del parlato

@dataclass
class SpeechIndex:
    video_ids: list[str]
    transcripts: list[dict]   # output di Transcriber.transcribe per ogni clip
    emb: np.ndarray           # (N, d) embedding della trascrizione, zeri se non c'è parlato
    has_speech: np.ndarray    # (N,) bool

    def save(self, path: Path) -> None:
        np.savez_compressed(path.with_suffix(".npz"), emb=self.emb, has_speech=self.has_speech)
        path.with_suffix(".json").write_text(json.dumps({"video_ids": self.video_ids}, indent=1))

    @classmethod
    def load(cls, path: Path, transcripts_path: Path) -> "SpeechIndex":
        arrays = np.load(path.with_suffix(".npz"))
        video_ids = json.loads(path.with_suffix(".json").read_text())["video_ids"]
        all_tr = json.loads(transcripts_path.read_text())
        return cls(video_ids, [all_tr[v] for v in video_ids], arrays["emb"], arrays["has_speech"])

    def scores(self, query_emb: np.ndarray, clip_subset: np.ndarray | None = None) -> np.ndarray:
        out = query_emb @ self.emb.T
        return out if clip_subset is None else out[:, clip_subset]


def transcribe_clips(cfg: Config, clips: list[Clip], transcriber, transcripts_path: Path) -> dict:
    """Trascrive le clip salvando i risultati ogni 20 clip, così un'interruzione non fa
    ripartire da zero."""
    done = json.loads(transcripts_path.read_text()) if transcripts_path.exists() else {}
    todo = [c for c in clips if c.video_id not in done]
    t0 = time.perf_counter()
    for i, clip in enumerate(todo, start=1):
        samples = load_audio(video_path(cfg, clip), cfg.sample_rate)
        done[clip.video_id] = transcriber.transcribe(samples)
        if i % 20 == 0 or i == len(todo):
            transcripts_path.write_text(json.dumps(done, indent=1))
            print(f"  trascritte {i}/{len(todo)} ({time.perf_counter() - t0:.0f} s)")
    return done


def build_speech_index(clips: list[Clip], transcripts: dict, text_encoder) -> SpeechIndex:
    video_ids = [c.video_id for c in clips if c.video_id in transcripts]
    trs = [transcripts[v] for v in video_ids]
    has_speech = np.array([t["has_speech"] for t in trs], dtype=bool)
    texts = [t["text"] for t, s in zip(trs, has_speech) if s]
    dim_probe = text_encoder.encode(["probe"]).shape[1]
    emb = np.zeros((len(video_ids), dim_probe), dtype=np.float32)
    if texts:
        emb[has_speech] = text_encoder.encode(texts)
    return SpeechIndex(video_ids, trs, emb, has_speech)
