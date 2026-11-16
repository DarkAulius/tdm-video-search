"""Trascrizione del parlato con Whisper e filtro dei segmenti senza parlato.

Molte clip di MSR-VTT contengono solo musica o rumore: su questi tratti Whisper tende a
"inventare" frasi (es. "Thank you for watching"). Il filtro combina:
1. un controllo di energia prima della trascrizione (clip mute: Whisper non viene chiamato);
2. le stesse regole che Whisper usa internamente, applicate a ogni segmento:
   - no_speech_prob alta e avg_logprob bassa  -> tratto senza parlato;
   - compression_ratio alto                    -> testo ripetitivo, tipico delle allucinazioni;
3. una lunghezza minima della trascrizione complessiva.

Si usa task="translate": Whisper traduce in inglese il parlato in altre lingue, così le
trascrizioni sono nella stessa lingua delle query di MSR-VTT.
"""

from __future__ import annotations

import numpy as np

from .audio import rms_dbfs
from .config import Config


def filter_segments(segments: list[dict], cfg: Config) -> list[dict]:
    """Tiene solo i segmenti di Whisper che contengono parlato plausibile."""
    kept = []
    for s in segments:
        silent = s["no_speech_prob"] > cfg.no_speech_threshold and s["avg_logprob"] < cfg.logprob_threshold
        repetitive = s["compression_ratio"] > cfg.compression_ratio_threshold
        text = s["text"].strip()
        if silent or repetitive or not text:
            continue
        kept.append({"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2), "text": text})
    return kept


def empty_transcript(reason: str, level_dbfs: float | None = None) -> dict:
    return {"has_speech": False, "reason": reason, "level_dbfs": level_dbfs, "language": None, "segments": [], "text": ""}


class Transcriber:
    """Incapsula il modello Whisper (pacchetto openai-whisper)."""

    def __init__(self, cfg: Config):
        import whisper  # importato qui per non richiederlo nei test

        self.cfg = cfg
        self.model = whisper.load_model(cfg.whisper_model, device="cpu")

    def transcribe(self, samples: np.ndarray | None) -> dict:
        if samples is None:
            return empty_transcript("nessuna traccia audio")
        level = rms_dbfs(samples)
        if level < self.cfg.silence_dbfs:
            return empty_transcript("audio sotto la soglia di silenzio", level)

        result = self.model.transcribe(
            samples,
            task="translate",
            temperature=0.0,                    # decodifica greedy: risultato deterministico
            condition_on_previous_text=False,   # evita che un errore si propaghi ai segmenti successivi
            fp16=False,                         # su CPU si lavora in float32
            verbose=None,
        )
        segments = filter_segments(result["segments"], self.cfg)
        text = " ".join(s["text"] for s in segments)
        if len(text.split()) < self.cfg.min_words:
            out = empty_transcript("parlato assente o troppo breve", level)
            out["language"] = result.get("language")
            return out
        return {
            "has_speech": True,
            "reason": "",
            "level_dbfs": round(level, 1),
            "language": result.get("language"),
            "segments": segments,
            "text": text,
        }
