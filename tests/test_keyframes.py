import numpy as np

from fakes import color_image
from tdm.config import Config
from tdm.keyframes import bhattacharyya, extract_iframes, hsv_histogram, sample_uniform, select_frames


def test_histogram_distance_bounds():
    red = hsv_histogram(color_image("red"), (16, 4, 4))
    blue = hsv_histogram(color_image("blue"), (16, 4, 4))
    assert abs(red.sum() - 1) < 1e-9
    assert bhattacharyya(red, red) < 1e-6
    assert bhattacharyya(red, blue) > 0.99


def test_iframes_follow_gop(three_scene_video):
    frames, duration, decoded = extract_iframes(three_scene_video, max_frames=32)
    assert abs(duration - 6.0) < 0.1
    # Un fotogramma I al secondo: 0, 1, 2, 3, 4, 5 s, e il decoder non ricostruisce gli altri.
    assert [round(f.time) for f in frames] == [0, 1, 2, 3, 4, 5]
    assert decoded == 6
    assert all(f.is_iframe for f in frames)


def test_uniform_sampling_decodes_more(three_scene_video):
    frames, _, decoded = sample_uniform(three_scene_video, k=6)
    times = [f.time for f in frames]
    assert len(frames) == 6
    assert times == sorted(times)
    assert np.allclose(times, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], atol=0.05)
    assert decoded > 100  # serve decodificare (quasi) tutta la clip


def test_iframes_hsv_finds_three_scenes(three_scene_video):
    sel = select_frames(three_scene_video, "iframes_hsv", Config())
    assert len(sel.segments) == 3
    assert [s for s, _ in sel.segments] == [0.0, 2.0, 4.0]
    assert sel.segments[-1][1] >= 5.9
    assert len(sel.frames) == 3          # un fotogramma rappresentativo per scena
    assert sel.frame_segment == [0, 1, 2]
    assert sel.boundary_distances[0] == 0.0 and min(sel.boundary_distances[1:]) > 0.35


def test_uniform_and_iframes_share_segments(three_scene_video):
    cfg = Config(uniform_k=6)
    uni = select_frames(three_scene_video, "uniform", cfg)
    ifr = select_frames(three_scene_video, "iframes", cfg)
    assert len(uni.frames) == 6 and len(ifr.frames) == 6
    assert len(uni.segments) == len(ifr.segments) == 3
    assert uni.frame_segment == [0, 0, 1, 1, 2, 2]


def test_fallback_when_few_iframes(long_gop_video):
    sel = select_frames(long_gop_video, "iframes", Config(min_frames=3))
    assert sel.used_fallback
    assert len(sel.frames) >= 3
    assert len(sel.segments) == 2  # giallo e blu
