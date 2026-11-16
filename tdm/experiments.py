"""Taratura sulla validazione e valutazione sul test.

1. Validazione: per ogni combinazione (strategia di selezione, aggregazione) si sceglie il
   peso w del parlato; la configurazione finale è quella con R@1+R@5+R@10 più alto.
2. Test: si misurano tutte le combinazioni con i pesi scelti sulla validazione, in tre varianti
   (solo visivo, solo parlato, fusione), più il costo computazionale di ciascuna strategia.
   Il test non viene mai usato per scegliere parametri.
"""

from __future__ import annotations

import numpy as np

from .config import AGGREGATIONS, Config
from .fusion import fuse, tune_weight, zscore_rows
from .index import SpeechIndex, VisualIndex
from .metrics import ranks, recall_sum, summarize


def align(visuals: dict[str, VisualIndex], speech: SpeechIndex, order: list[str]) -> tuple[list[str], dict, np.ndarray]:
    """Clip presenti in tutti gli indici, nell'ordine dello split.
    Restituisce (video_id comuni, colonne per ogni indice visivo, colonne nell'indice del parlato)."""
    sets = [set(v.video_ids) for v in visuals.values()] + [set(speech.video_ids)]
    common = [vid for vid in order if all(vid in s for s in sets)]
    cols = {}
    for name, v in visuals.items():
        pos = {vid: i for i, vid in enumerate(v.video_ids)}
        cols[name] = np.array([pos[vid] for vid in common])
    pos = {vid: i for i, vid in enumerate(speech.video_ids)}
    speech_cols = np.array([pos[vid] for vid in common])
    return common, cols, speech_cols


def tune(visuals, speech, q_clip, q_text, cols, speech_cols, cfg: Config) -> dict:
    s_speech = speech.scores(q_text, speech_cols)
    has_speech = speech.has_speech[speech_cols]
    configs = []
    for strategy, visual in visuals.items():
        for agg in AGGREGATIONS:
            s_vis = visual.scores(q_clip, agg, cols[strategy])
            w, curve = tune_weight(s_vis, s_speech, has_speech, cfg.weight_grid)
            configs.append({
                "strategy": strategy,
                "aggregation": agg,
                "w": w,
                "val_visual_recall_sum": recall_sum(ranks(s_vis)),
                "val_fused_recall_sum": recall_sum(ranks(fuse(s_vis, s_speech, has_speech, w))),
                "curve": curve,
            })
    # A parità di risultato si preferisce la strategia che codifica meno fotogrammi (più economica).
    cost = {s: v.stats["frames_encoded_per_clip"] for s, v in visuals.items()}
    best = max(configs, key=lambda c: (c["val_fused_recall_sum"], -cost[c["strategy"]]))
    return {
        "best": {k: best[k] for k in ("strategy", "aggregation", "w")},
        "val_clips": len(speech_cols),
        "val_speech_clips": int(has_speech.sum()),
        "configs": configs,
    }


def evaluate(visuals, speech, q_clip, q_text, cols, speech_cols, tuning: dict) -> dict:
    s_speech = speech.scores(q_text, speech_cols)
    has_speech = speech.has_speech[speech_cols]
    speech_only = ranks(zscore_rows(s_speech, has_speech))
    weights = {(c["strategy"], c["aggregation"]): c["w"] for c in tuning["configs"]}

    rows = []
    for strategy, visual in visuals.items():
        for agg in AGGREGATIONS:
            w = weights[(strategy, agg)]
            s_vis = visual.scores(q_clip, agg, cols[strategy])
            r_vis = ranks(s_vis)
            r_fus = ranks(fuse(s_vis, s_speech, has_speech, w))
            rows.append({
                "strategy": strategy,
                "aggregation": agg,
                "w": w,
                "visual": summarize(r_vis),
                "fusion": summarize(r_fus),
                # Sottoinsieme delle query la cui clip contiene parlato: qui la fusione può incidere.
                "visual_on_speech_clips": summarize(r_vis[has_speech]) if has_speech.any() else None,
                "fusion_on_speech_clips": summarize(r_fus[has_speech]) if has_speech.any() else None,
            })

    return {
        "clips": len(speech_cols),
        "speech_clips": int(has_speech.sum()),
        "best": tuning["best"],
        "speech_only": summarize(speech_only),
        "speech_only_on_speech_clips": summarize(speech_only[has_speech]) if has_speech.any() else None,
        "cost": {name: v.stats for name, v in visuals.items()},
        "table": rows,
    }


def markdown_summary(results: dict) -> str:
    """Tabelle in Markdown per il README e per il report."""
    t = results["test"]
    best = t["best"]
    lines = [
        "# Risultati",
        "",
        f"Test: {t['clips']} clip di MSR-VTT 1k-A, di cui {t['speech_clips']} con parlato rilevato. "
        f"Configurazione scelta sulla validazione ({results['tuning']['val_clips']} clip): "
        f"strategia `{best['strategy']}`, aggregazione `{best['aggregation']}`, peso del parlato w = {best['w']:.2f}.",
        "",
        "## Recupero testo -> video (test)",
        "",
        "| Strategia | Aggregazione | w | Visivo R@1 | R@5 | R@10 | MdR | Fusione R@1 | R@5 | R@10 | MdR |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in t["table"]:
        v, f = r["visual"], r["fusion"]
        mark = " **(scelta)**" if (r["strategy"], r["aggregation"]) == (best["strategy"], best["aggregation"]) else ""
        lines.append(
            f"| {r['strategy']}{mark} | {r['aggregation']} | {r['w']:.2f} | {v['R@1']} | {v['R@5']} | {v['R@10']} | {v['MdR']:.0f} "
            f"| {f['R@1']} | {f['R@5']} | {f['R@10']} | {f['MdR']:.0f} |"
        )
    so = t["speech_only"]
    lines += [
        "",
        f"Solo parlato (tutte le query): R@1 {so['R@1']}, R@5 {so['R@5']}, R@10 {so['R@10']}, MdR {so['MdR']:.0f}.",
    ]
    if t["speech_only_on_speech_clips"]:
        s2 = t["speech_only_on_speech_clips"]
        lines.append(f"Solo parlato (query con clip parlata): R@1 {s2['R@1']}, R@5 {s2['R@5']}, R@10 {s2['R@10']}.")
    lines += [
        "",
        "## Costo delle strategie di selezione dei fotogrammi (test)",
        "",
        "| Strategia | Fotogrammi decodificati/clip | Fotogrammi codificati/clip | Segmenti/clip | Clip con integrazione uniforme | Tempo selezione (s) | Tempo CLIP (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in t["cost"].items():
        lines.append(
            f"| {name} | {s['frames_decoded_per_clip']} | {s['frames_encoded_per_clip']} | {s['segments_per_clip']} "
            f"| {s['fallback_clips']} | {s['selection_seconds']} | {s['encoding_seconds']} |"
        )
    return "\n".join(lines) + "\n"
