"""Test end-to-end con video sintetici ed encoder fittizi: indici, taratura, valutazione,
knowledge graph e ricerca ibrida."""

import numpy as np
import pytest

from fakes import FakeClipEncoder, FakeTextEncoder, make_video
from tdm.config import AGGREGATIONS, STRATEGIES, Config
from tdm.dataset import Clip, video_path
from tdm.experiments import align, evaluate, markdown_summary, tune
from tdm.index import VisualIndex, build_speech_index, build_visual_index
from tdm.kg import build_graph, candidate_ids, fragment_uri, load_graph, save_graph, speech_fragments
from tdm.search import SearchEngine

SPEC = [
    # video_id, scene, didascalia (query), categoria, parlato
    ("video0", [("red", 2), ("green", 2)], "red and green", 17, "the chef cooks pasta with tomato"),
    ("video1", [("blue", 3)], "a blue sky", 13, None),
    ("video2", [("yellow", 2), ("blue", 2)], "yellow then blue", 4, "the news anchor talks about elections"),
    ("video3", [("green", 3)], "green grass", 3, "the crowd cheers for the goal"),
]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    cfg = Config(data_dir=root / "data", cache_dir=root / "cache", results_dir=root / "results", uniform_k=4)
    clips, transcripts = [], {}
    for vid, scenes, caption, cat, spoken in SPEC:
        clip = Clip(vid, caption, cat, f"https://www.youtube.com/watch?v={vid}", 10.0, 10.0 + sum(s for _, s in scenes))
        make_video(video_path(cfg, clip), scenes, gop=25)
        clips.append(clip)
        if spoken:
            transcripts[vid] = {"has_speech": True, "reason": "", "level_dbfs": -20.0, "language": "en",
                                "segments": [{"start": 0.0, "end": 1.5, "text": spoken}], "text": spoken}
        else:
            transcripts[vid] = {"has_speech": False, "reason": "test", "level_dbfs": -80.0, "language": None,
                                "segments": [], "text": ""}
    enc, txt = FakeClipEncoder(), FakeTextEncoder()
    visuals = {s: build_visual_index(cfg, clips, s, enc) for s in STRATEGIES}
    speech = build_speech_index(clips, transcripts, txt)
    return cfg, clips, visuals, speech, enc, txt


def test_visual_index_roundtrip(setup, tmp_path):
    _, _, visuals, _, _, _ = setup
    v = visuals["iframes_hsv"]
    assert v.stats["frames_encoded_per_clip"] < visuals["iframes"].stats["frames_encoded_per_clip"]
    assert v.stats["frames_decoded_per_clip"] < visuals["uniform"].stats["frames_decoded_per_clip"]
    v.save(tmp_path / "idx")
    w = VisualIndex.load(tmp_path / "idx")
    assert np.allclose(v.frame_emb, w.frame_emb) and w.video_ids == v.video_ids
    # video0 e video2 hanno due scene, video1 e video3 una sola
    assert np.bincount(v.seg_clip).tolist() == [2, 1, 2, 1]


def test_scores_shapes_for_all_aggregations(setup):
    _, _, visuals, _, enc, _ = setup
    q = enc.encode_texts(["red", "blue"])
    for agg in AGGREGATIONS:
        s = visuals["uniform"].scores(q, agg)
        assert s.shape == (2, 4)
    # La query "red" con max_segments preferisce video0 (unica clip con una scena rossa).
    assert int(np.argmax(visuals["uniform"].scores(q, "max_segments")[0])) == 0


def test_tune_and_evaluate(setup):
    cfg, clips, visuals, speech, enc, txt = setup
    order = [c.video_id for c in clips]
    common, cols, speech_cols = align(visuals, speech, order)
    assert common == order
    queries = [c.caption for c in clips]
    q_clip, q_text = enc.encode_texts(queries), txt.encode(queries)
    tuning = tune(visuals, speech, q_clip, q_text, cols, speech_cols, cfg)
    assert len(tuning["configs"]) == len(STRATEGIES) * len(AGGREGATIONS)
    test = evaluate(visuals, speech, q_clip, q_text, cols, speech_cols, tuning)
    assert test["clips"] == 4 and test["speech_clips"] == 3
    assert len(test["table"]) == 9
    md = markdown_summary({"tuning": tuning, "test": test})
    assert "Recupero testo -> video" in md and "iframes_hsv" in md


def test_knowledge_graph_and_search(setup):
    cfg, clips, visuals, speech, enc, txt = setup
    g = build_graph(clips, visuals["iframes_hsv"], speech)
    path = cfg.results_dir / "kg.ttl"
    save_graph(g, path)
    g = load_graph(path)

    assert candidate_ids(g) == ["video0", "video1", "video2", "video3"]
    assert candidate_ids(g, category="news") == ["video2"]
    assert candidate_ids(g, has_speech=False) == ["video1"]
    assert candidate_ids(g, max_duration=3.5) == ["video1", "video3"]
    frags = speech_fragments(g, "video0")
    assert frags[0]["uri"].endswith("video0#t=0.00,1.50") and "chef" in frags[0]["text"]
    assert str(fragment_uri("video9", 1, 2.5)).endswith("video9#t=1.00,2.50")

    engine = SearchEngine(visuals["iframes_hsv"], speech, g, enc, txt, "max_segments", w=0.5)
    hits = engine.search("red", top_k=2)
    assert hits[0].video_id == "video0"
    assert hits[0].start == 0.0 and hits[0].end == pytest.approx(2.0, abs=0.05)  # la scena rossa
    assert "chef" in hits[0].speech
    assert engine.search("red", category="animals")[0].video_id == "video1"
    assert engine.search("anything", category="nonexistent") == []
