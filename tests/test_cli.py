"""Verifica il collegamento dei comandi di main.py (all, search, sparql) senza scaricare
dati né modelli: i metadati vengono scritti a mano e i modelli sostituiti con quelli finti."""

import json
import sys
from dataclasses import asdict

import pytest

import main
from fakes import FakeClipEncoder, FakeTextEncoder, make_video
from tdm.config import Config
from tdm.dataset import Clip, video_path


class FakeTranscriber:
    def __init__(self, cfg):
        pass

    def transcribe(self, samples):
        # Le clip con audio "parlano" di cucina, le altre sono mute.
        if samples is None:
            return {"has_speech": False, "reason": "nessuna traccia audio", "level_dbfs": None,
                    "language": None, "segments": [], "text": ""}
        text = "the chef explains how to cook pasta"
        return {"has_speech": True, "reason": "", "level_dbfs": -10.0, "language": "en",
                "segments": [{"start": 0.0, "end": 2.0, "text": text}], "text": text}


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tdm.encoders.ClipEncoder", lambda cfg: FakeClipEncoder())
    monkeypatch.setattr("tdm.encoders.SentenceEncoder", lambda cfg: FakeTextEncoder())
    monkeypatch.setattr("tdm.asr.Transcriber", FakeTranscriber)
    cfg = Config()
    specs = {
        "test": [("video100", "red", True), ("video101", "blue", False), ("video102", "green", True)],
        "val": [("video200", "yellow", True), ("video201", "red", False), ("video202", "blue", True)],
    }
    for split, items in specs.items():
        clips = []
        for vid, color, audio in items:
            clip = Clip(vid, f"a {color} scene", 17 if audio else 13, "https://www.youtube.com/watch?v=x", 0.0, 3.0)
            make_video(video_path(cfg, clip), [(color, 3)], audio_tone=440.0 if audio else None)
            clips.append(clip)
        name = "test_pool.json" if split == "test" else f"val_pool_min{cfg.val_min_id}.json"
        (cfg.data_dir / name).write_text(json.dumps([asdict(c) for c in clips]))
    return tmp_path


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    main.main()


def test_full_cli(project, monkeypatch, capsys):
    run(monkeypatch, "all", "--n-test", "3", "--n-val", "3")
    results = json.loads((project / "results" / "evaluation.json").read_text())
    assert results["test"]["clips"] == 3 and results["test"]["speech_clips"] == 2
    assert (project / "results" / "summary.md").exists()
    assert (project / "results" / "kg_test.ttl").exists()

    run(monkeypatch, "search", "red", "--n-test", "3", "--top-k", "1")
    out = capsys.readouterr().out
    assert "1. video100" in out and "#t=" in out

    run(monkeypatch, "search", "red", "--n-test", "3", "--speech", "no")
    assert "1. video101" in capsys.readouterr().out

    query = "PREFIX ma: <http://www.w3.org/ns/ma-ont#> SELECT (COUNT(?f) AS ?n) WHERE { ?f a ma:MediaFragment }"
    run(monkeypatch, "sparql", query, "--n-test", "3")
    assert int(capsys.readouterr().out.strip()) >= 3
