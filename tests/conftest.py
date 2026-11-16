import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fakes import make_video  # noqa: E402


@pytest.fixture(scope="session")
def three_scene_video(tmp_path_factory):
    """6 s a 25 fps: rosso, verde, blu (2 s ciascuno), un fotogramma I al secondo, tono a 440 Hz."""
    path = tmp_path_factory.mktemp("videos") / "three_scenes.mp4"
    return make_video(path, [("red", 2), ("green", 2), ("blue", 2)], fps=25, gop=25)


@pytest.fixture(scope="session")
def long_gop_video(tmp_path_factory):
    """Un solo fotogramma I (GOP più lungo della clip) e nessuna traccia audio."""
    path = tmp_path_factory.mktemp("videos") / "long_gop.mp4"
    return make_video(path, [("yellow", 3), ("blue", 3)], fps=25, gop=1000, audio_tone=None)
