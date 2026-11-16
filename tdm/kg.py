"""Knowledge graph RDF della collezione video.

Vocabolari:
- W3C Ontology for Media Resources (ma:): MediaResource, MediaFragment, isFragmentOf,
  duration, hasGenre, locator, hasLanguage;
- un piccolo vocabolario locale (tdm:) per ciò che ma: non prevede: istanti dei
  fotogrammi chiave, tipo di fotogramma (I), distanza HSV che ha aperto la scena,
  trascrizione, presenza di parlato.

Ogni segmento è identificato da un URI W3C Media Fragments, cioè l'URI del video seguito
da "#t=inizio,fine" in secondi: il risultato di una ricerca punta direttamente al tratto
di video pertinente e un lettore compatibile può riprodurlo da lì.

Le interrogazioni SPARQL selezionano le clip candidate con filtri strutturati (categoria,
durata, presenza di parlato); il ranking semantico lavora poi solo su quelle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rdflib import RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef

from .dataset import CATEGORIES, Clip
from .index import SpeechIndex, VisualIndex

MA = Namespace("http://www.w3.org/ns/ma-ont#")
TDM = Namespace("http://example.org/tdm/ontology#")   # vocabolario locale dell'elaborato
VIDEO = Namespace("http://example.org/tdm/video/")     # risorse video della collezione
GENRE = Namespace("http://example.org/tdm/genre/")


def video_uri(video_id: str) -> URIRef:
    return VIDEO[video_id]


def fragment_uri(video_id: str, start: float, end: float) -> URIRef:
    """URI Media Fragments del segmento temporale [start, end) in secondi."""
    return URIRef(f"{VIDEO[video_id]}#t={start:.2f},{end:.2f}")


def _decimal(x: float) -> Literal:
    return Literal(round(float(x), 3), datatype=XSD.decimal)


def build_graph(clips: list[Clip], visual: VisualIndex, speech: SpeechIndex | None = None) -> Graph:
    g = Graph()
    g.bind("ma", MA)
    g.bind("tdm", TDM)
    g.bind("video", VIDEO)
    g.bind("genre", GENRE)

    for i, name in enumerate(CATEGORIES):
        g.add((GENRE[str(i)], RDF.type, TDM.Genre))
        g.add((GENRE[str(i)], RDFS.label, Literal(name)))

    by_id = {c.video_id: c for c in clips}
    speech_by_id = {}
    if speech is not None:
        speech_by_id = dict(zip(speech.video_ids, speech.transcripts))

    for clip_idx, vid in enumerate(visual.video_ids):
        clip = by_id[vid]
        meta = visual.clip_meta[clip_idx]
        v = video_uri(vid)
        g.add((v, RDF.type, MA.MediaResource))
        g.add((v, MA.title, Literal(vid)))
        g.add((v, MA.duration, _decimal(meta["duration"])))
        g.add((v, MA.hasGenre, GENRE[str(clip.category)]))
        g.add((v, MA.locator, Literal(f"{clip.url}&t={int(clip.start)}", datatype=XSD.anyURI)))
        g.add((v, TDM.sourceStart, _decimal(clip.start)))
        g.add((v, TDM.sourceEnd, _decimal(clip.end)))
        g.add((v, TDM.selectionStrategy, Literal(visual.stats["strategy"])))

        # Segmenti visivi (scene) con i loro fotogrammi chiave.
        seg_ids = np.flatnonzero(visual.seg_clip == clip_idx)
        for k, s in enumerate(seg_ids):
            start, end = (float(x) for x in visual.seg_bounds[s])
            f = fragment_uri(vid, start, end)
            g.add((f, RDF.type, MA.MediaFragment))
            g.add((f, TDM.fragmentKind, Literal("scene")))
            g.add((f, MA.isFragmentOf, v))
            g.add((v, MA.hasFragment, f))
            g.add((f, TDM.startTime, _decimal(start)))
            g.add((f, TDM.endTime, _decimal(end)))
            g.add((f, TDM.sceneIndex, Literal(k)))
            if k < len(meta["boundary_distances"]):
                g.add((f, TDM.hsvDistance, _decimal(meta["boundary_distances"][k])))
            for t in visual.frame_time[visual.frame_seg == s]:
                g.add((f, TDM.keyframeAt, _decimal(t)))

        # Parlato: trascrizione complessiva e segmenti di Whisper come frammenti temporali.
        tr = speech_by_id.get(vid)
        if tr is not None:
            g.add((v, TDM.hasSpeech, Literal(bool(tr["has_speech"]))))
            if tr["has_speech"]:
                g.add((v, TDM.transcript, Literal(tr["text"], lang="en")))
                if tr.get("language"):
                    g.add((v, MA.hasLanguage, Literal(tr["language"])))
                for seg in tr["segments"]:
                    f = fragment_uri(vid, seg["start"], seg["end"])
                    g.add((f, RDF.type, MA.MediaFragment))
                    g.add((f, TDM.fragmentKind, Literal("speech")))
                    g.add((f, MA.isFragmentOf, v))
                    g.add((v, MA.hasFragment, f))
                    g.add((f, TDM.startTime, _decimal(seg["start"])))
                    g.add((f, TDM.endTime, _decimal(seg["end"])))
                    g.add((f, TDM.text, Literal(seg["text"], lang="en")))
    return g


def save_graph(g: Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(path), format="turtle")


def load_graph(path: Path) -> Graph:
    g = Graph()
    g.parse(str(path), format="turtle")
    return g


def candidate_ids(g: Graph, category: str | None = None, min_duration: float | None = None,
                  max_duration: float | None = None, has_speech: bool | None = None) -> list[str]:
    """Clip che rispettano i filtri strutturati. Le costanti vengono inserite nella query
    come letterali RDF tipizzati (metodo n3()), non come testo libero."""
    filters = []
    if category is not None:
        filters.append(f"CONTAINS(LCASE(STR(?genreLabel)), LCASE({Literal(category).n3()}))")
    if min_duration is not None:
        filters.append(f"?duration >= {_decimal(min_duration).n3()}")
    if max_duration is not None:
        filters.append(f"?duration <= {_decimal(max_duration).n3()}")
    speech_pattern = ""
    if has_speech is not None:
        speech_pattern = f"?video tdm:hasSpeech {Literal(has_speech).n3()} ."
    filter_clause = f"FILTER ({' && '.join(filters)})" if filters else ""

    query = f"""
        PREFIX ma: <{MA}>
        PREFIX tdm: <{TDM}>
        PREFIX rdfs: <{RDFS}>
        SELECT ?id WHERE {{
            ?video a ma:MediaResource ;
                   ma:title ?id ;
                   ma:duration ?duration ;
                   ma:hasGenre ?genre .
            ?genre rdfs:label ?genreLabel .
            {speech_pattern}
            {filter_clause}
        }}
        ORDER BY ?id
    """
    return [str(row.id) for row in g.query(query)]


def speech_fragments(g: Graph, video_id: str) -> list[dict]:
    """Segmenti di parlato di una clip, in ordine di tempo."""
    query = f"""
        PREFIX ma: <{MA}>
        PREFIX tdm: <{TDM}>
        SELECT ?f ?start ?end ?text WHERE {{
            ?f ma:isFragmentOf {video_uri(video_id).n3()} ;
               tdm:fragmentKind "speech" ;
               tdm:startTime ?start ; tdm:endTime ?end ; tdm:text ?text .
        }}
        ORDER BY ?start
    """
    return [{"uri": str(r.f), "start": float(r.start), "end": float(r.end), "text": str(r.text)} for r in g.query(query)]
