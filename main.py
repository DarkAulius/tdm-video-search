"""Ricerca testuale in una collezione video con fusione di contenuto visivo e parlato.

Elaborato di Trattamento Dati Multimediali - Vincenzo Chianese.

Uso (dettagli nel README):
    python main.py all                         # pipeline completa: dati, indici, valutazione, grafo
    python main.py all --n-test 50 --n-val 30  # prova veloce su poche clip
    python main.py search "a man is cooking on stage" --category cooking
    python main.py sparql queries/clip_per_categoria.rq
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tdm.config import SEED, STRATEGIES, Config, set_seed
from tdm.dataset import CATEGORIES, available, ensure_videos, load_split, media_summary
from tdm.index import SpeechIndex, VisualIndex, build_speech_index, build_visual_index, transcribe_clips

SPLITS = ("val", "test")


# --------------------------------------------------------------------------- percorsi della cache

def n_of(cfg: Config, split: str) -> int:
    return cfg.n_test if split == "test" else cfg.n_val


def visual_path(cfg: Config, split: str, strategy: str) -> Path:
    return cfg.split_cache(split) / f"visual_{strategy}_{n_of(cfg, split)}"


def speech_path(cfg: Config, split: str) -> Path:
    return cfg.split_cache(split) / f"speech_{n_of(cfg, split)}"


def transcripts_path(cfg: Config, split: str) -> Path:
    return cfg.split_cache(split) / "transcripts.json"


def evaluation_path(cfg: Config) -> Path:
    return cfg.results_dir / "evaluation.json"


def kg_path(cfg: Config) -> Path:
    return cfg.results_dir / "kg_test.ttl"


# --------------------------------------------------------------------------- comandi

def cmd_prepare(cfg: Config, args) -> None:
    for split in args.splits:
        clips = load_split(cfg, split)
        print(f"[{split}] {len(clips)} clip: scarico i video mancanti...")
        ensure_videos(cfg, clips)
        info = media_summary(cfg, clips)
        print(f"[{split}] video disponibili: {len(available(cfg, clips))}/{len(clips)}; "
              f"con traccia audio {info['with_audio']}/{info['checked']} controllati; fps mediano {info['median_fps']:.0f}")
        # Senza audio la parte del parlato non ha senso: meglio fermarsi subito.
        if info["checked"] and info["with_audio"] == 0 and not args.allow_no_audio:
            raise SystemExit(f"I video scaricati non hanno traccia audio "
                             "(usa --allow-no-audio per proseguire con il solo visivo).")


def cmd_visual(cfg: Config, args) -> None:
    from tdm.encoders import ClipEncoder

    encoder = ClipEncoder(cfg)
    # Riscaldamento: la prima chiamata paga l'inizializzazione della GPU e falserebbe
    # il confronto dei tempi a sfavore della prima strategia misurata.
    encoder.encode_images([np.zeros((224, 224, 3), dtype=np.uint8)] * 4)
    for split in args.splits:
        clips = available(cfg, load_split(cfg, split))
        for strategy in args.strategies:
            path = visual_path(cfg, split, strategy)
            if path.with_suffix(".npz").exists() and not args.force:
                # Si riusa l'indice solo se copre le clip disponibili (tolleranza 1% per
                # eventuali video illeggibili, che vengono comunque esclusi).
                cached = set(VisualIndex.load(path).video_ids)
                missing = [c for c in clips if c.video_id not in cached]
                if len(missing) <= 0.01 * len(clips):
                    print(f"[{split}/{strategy}] indice già presente, salto (usa --force per rifarlo)")
                    continue
                print(f"[{split}/{strategy}] indice incompleto ({len(cached)} clip su {len(clips)}): lo ricalcolo")
            print(f"[{split}/{strategy}] selezione fotogrammi e codifica CLIP di {len(clips)} clip...")
            index = build_visual_index(cfg, clips, strategy, encoder)
            index.save(path)
            print(f"[{split}/{strategy}] {json.dumps(index.stats)}")


def cmd_speech(cfg: Config, args) -> None:
    from tdm.asr import Transcriber
    from tdm.encoders import SentenceEncoder

    transcriber = Transcriber(cfg)
    text_encoder = SentenceEncoder(cfg)
    for split in args.splits:
        clips = available(cfg, load_split(cfg, split))
        print(f"[{split}] trascrizione con Whisper '{cfg.whisper_model}' di {len(clips)} clip...")
        transcripts = transcribe_clips(cfg, clips, transcriber, transcripts_path(cfg, split))
        index = build_speech_index(clips, transcripts, text_encoder)
        index.save(speech_path(cfg, split))
        print(f"[{split}] clip con parlato: {int(index.has_speech.sum())}/{len(index.video_ids)}")


def _load_split_indices(cfg: Config, split: str):
    visuals = {s: VisualIndex.load(visual_path(cfg, split, s)) for s in STRATEGIES}
    speech = SpeechIndex.load(speech_path(cfg, split), transcripts_path(cfg, split))
    return visuals, speech


def cmd_evaluate(cfg: Config, args) -> None:
    from tdm.encoders import ClipEncoder, SentenceEncoder
    from tdm.experiments import align, evaluate, markdown_summary, tune

    clip_encoder, text_encoder = ClipEncoder(cfg), SentenceEncoder(cfg)
    results = {"seed": SEED}
    for split in SPLITS:
        clips = load_split(cfg, split)
        captions = {c.video_id: c.caption for c in clips}
        visuals, speech = _load_split_indices(cfg, split)
        common, cols, speech_cols = align(visuals, speech, [c.video_id for c in clips])
        queries = [captions[v] for v in common]
        q_clip, q_text = clip_encoder.encode_texts(queries), text_encoder.encode(queries)
        if split == "val":
            results["tuning"] = tune(visuals, speech, q_clip, q_text, cols, speech_cols, cfg)
            print(f"[val] configurazione scelta: {results['tuning']['best']}")
        else:
            results["test"] = evaluate(visuals, speech, q_clip, q_text, cols, speech_cols, results["tuning"])

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path(cfg).write_text(json.dumps(results, indent=1))
    summary = markdown_summary(results)
    (cfg.results_dir / "summary.md").write_text(summary)
    print(summary)


def _best_config(cfg: Config) -> dict:
    if evaluation_path(cfg).exists():
        return json.loads(evaluation_path(cfg).read_text())["tuning"]["best"]
    return {"strategy": "iframes_hsv", "aggregation": "mean_segments", "w": 0.5}  # prima della valutazione


def cmd_kg(cfg: Config, args) -> None:
    from tdm.kg import build_graph, save_graph

    best = _best_config(cfg)
    clips = load_split(cfg, "test")
    visual = VisualIndex.load(visual_path(cfg, "test", best["strategy"]))
    speech = SpeechIndex.load(speech_path(cfg, "test"), transcripts_path(cfg, "test"))
    graph = build_graph(clips, visual, speech)
    save_graph(graph, kg_path(cfg))
    print(f"Grafo salvato in {kg_path(cfg)}: {len(graph)} triple (strategia {best['strategy']})")


def cmd_search(cfg: Config, args) -> None:
    from tdm.encoders import ClipEncoder, SentenceEncoder
    from tdm.kg import load_graph
    from tdm.search import SearchEngine

    best = _best_config(cfg)
    engine = SearchEngine(
        visual=VisualIndex.load(visual_path(cfg, "test", best["strategy"])),
        speech=SpeechIndex.load(speech_path(cfg, "test"), transcripts_path(cfg, "test")),
        graph=load_graph(kg_path(cfg)),
        clip_encoder=ClipEncoder(cfg),
        text_encoder=SentenceEncoder(cfg),
        aggregation=best["aggregation"],
        w=best["w"],
    )
    has_speech = {"yes": True, "no": False}.get(args.speech)
    hits = engine.search(args.query, args.top_k, args.category, args.min_duration, args.max_duration, has_speech)
    if not hits:
        print("Nessuna clip rispetta i filtri.")
    for h in hits:
        print(f"{h.rank}. {h.video_id}  punteggio {h.score:+.3f}  segmento {h.start:.1f}-{h.end:.1f} s")
        print(f"   {h.fragment}")
        if h.speech:
            print(f"   parlato: \"{h.speech[:120]}\"")


def cmd_sparql(cfg: Config, args) -> None:
    from tdm.kg import load_graph

    text = Path(args.query).read_text() if Path(args.query).exists() else args.query
    for row in load_graph(kg_path(cfg)).query(text):
        print(" | ".join(str(x) for x in row))


def cmd_all(cfg: Config, args) -> None:
    args.splits = list(SPLITS)
    args.strategies = list(STRATEGIES)
    cmd_prepare(cfg, args)
    cmd_visual(cfg, args)
    cmd_speech(cfg, args)
    cmd_evaluate(cfg, args)
    cmd_kg(cfg, args)


# --------------------------------------------------------------------------- argomenti

def build_parser() -> argparse.ArgumentParser:
    # Opzioni comuni, accettate dopo qualunque comando (es. "main.py all --n-test 50").
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=Path("data"))
    common.add_argument("--cache-dir", type=Path, default=Path("cache"))
    common.add_argument("--results-dir", type=Path, default=Path("results"))
    common.add_argument("--n-test", type=int, default=1000, help="clip di test (max 1000)")
    common.add_argument("--n-val", type=int, default=300, help="clip di validazione")
    common.add_argument("--whisper-model", default="base", help="tiny, base, small, ...")
    common.add_argument("--video-source", help="sorgente unica dei video (URL di un .zip o repository HF); di default si provano quelle in tdm/config.py")
    common.add_argument("--allow-no-audio", action="store_true", help="prosegue anche con video senza audio")
    common.add_argument("--force", action="store_true", help="ricalcola gli indici già presenti")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    for name, help_text in [("prepare", "scarica metadati e video"), ("visual", "indice visivo"),
                            ("speech", "trascrizioni e indice del parlato")]:
        sp = sub.add_parser(name, help=help_text, parents=[common])
        sp.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
        if name == "visual":
            sp.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))

    sub.add_parser("evaluate", help="taratura sulla validazione e valutazione sul test", parents=[common])
    sub.add_parser("kg", help="costruisce il knowledge graph RDF del test", parents=[common])
    sub.add_parser("all", help="esegue tutta la pipeline", parents=[common])

    sp = sub.add_parser("search", help="ricerca testuale con filtri", parents=[common])
    sp.add_argument("query")
    sp.add_argument("--top-k", type=int, default=5)
    sp.add_argument("--category", help=f"filtro sulla categoria, es. {', '.join(CATEGORIES[:4])}")
    sp.add_argument("--min-duration", type=float)
    sp.add_argument("--max-duration", type=float)
    sp.add_argument("--speech", choices=("yes", "no"), help="solo clip con (yes) o senza (no) parlato")

    sp = sub.add_parser("sparql", help="esegue una query SPARQL sul grafo (testo o file .rq)", parents=[common])
    sp.add_argument("query")
    return p


def use_system_certificates() -> None:
    """Il Python di python.org su macOS non usa i certificati di sistema: lo scaricamento dei
    pesi di Whisper fallirebbe con CERTIFICATE_VERIFY_FAILED. truststore fa verificare i
    certificati con il portachiavi del sistema operativo (la verifica resta attiva)."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass


def main() -> None:
    args = build_parser().parse_args()
    use_system_certificates()
    set_seed(SEED)
    cfg = Config(data_dir=args.data_dir, cache_dir=args.cache_dir, results_dir=args.results_dir,
                 n_test=args.n_test, n_val=args.n_val, whisper_model=args.whisper_model,
                 **({"video_sources": (args.video_source,)} if args.video_source else {}))
    commands = {"prepare": cmd_prepare, "visual": cmd_visual, "speech": cmd_speech, "evaluate": cmd_evaluate,
                "kg": cmd_kg, "search": cmd_search, "sparql": cmd_sparql, "all": cmd_all}
    commands[args.command](cfg, args)


if __name__ == "__main__":
    main()
