"""Pure-logic unit tests (no network / LLM / TTS / FFmpeg required)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings, get_settings  # noqa: E402
from src.modules.downloader import is_static, sec_to_ts  # noqa: E402
from src.modules.transcript_llm import StoryPlanError, _SYSTEM_PROMPT, _parse_hook, parse_story_plan  # noqa: E402
from src.modules.tts_generator import _compensate_rate  # noqa: E402
from src.modules.video_editor import (  # noqa: E402
    _clamp_crop_x,
    allocate_caption_timings,
)
from src.modules.youtube_search import (  # noqa: E402
    has_english_marker,
    is_english_candidate,
    parse_iso_duration,
)


def test_settings_defaults():
    s = get_settings()
    assert s.target_script_words == 360
    assert s.max_scenes == 30
    assert s.min_scene_seconds == 5.0
    assert s.max_scene_seconds == 5.0
    assert s.target_duration_seconds == 150
    assert s.movie_genres == "thriller,horror,romantic,survival"


def test_long_video_and_hook_defaults():
    s = get_settings()
    assert s.hook_seconds == 3.0
    assert s.make_long_video is True
    assert s.long_duration_seconds == 300
    assert s.long_output_width == 1920
    assert s.long_output_height == 1080


def test_parse_hook_clamps_to_hook_seconds():
    s = Settings(hook_seconds=3.0)
    hook = _parse_hook({"start_time": 100, "end_time": 120}, s, None)
    assert hook is not None
    assert hook.start_time == 100
    assert hook.duration == 3.0


def test_parse_hook_bounds_to_video_duration():
    s = Settings(hook_seconds=3.0)
    hook = _parse_hook({"start_time": 10, "end_time": 20}, s, video_duration=11)
    assert hook is not None
    assert hook.end_time == 11


def test_parse_hook_rejects_bad():
    s = Settings(hook_seconds=3.0)
    assert _parse_hook(None, s, None) is None
    assert _parse_hook({"start_time": 5, "end_time": 5}, s, None) is None
    assert _parse_hook({"start_time": -1, "end_time": 10}, s, None) is None
    assert _parse_hook({"start_time": "bad", "end_time": 10}, s, None) is None


def test_parse_story_plan_carries_hook():
    s = Settings()
    data = {
        "hindi_script": "पहला वाक्य। दूसरा वाक्य!",
        "hook": {"start_time": 200, "end_time": 210},
        "timestamps": [{"start_time": 10, "end_time": 18}],
    }
    plan = parse_story_plan(data, "abc123", s)
    assert plan.hook is not None
    assert plan.hook.start_time == 200


def test_system_prompt_asks_for_hook():
    assert "hook" in _SYSTEM_PROMPT
    assert "most-viewed" in _SYSTEM_PROMPT


def test_parse_iso_duration():
    assert parse_iso_duration("PT1H32M10S") == 5530
    assert parse_iso_duration("PT2H") == 7200
    assert parse_iso_duration("PT45M") == 2700
    assert parse_iso_duration("PT30S") == 30
    assert parse_iso_duration("") == 0
    assert parse_iso_duration("garbage") == 0


def test_is_english_candidate():
    assert is_english_candidate("The Silence of the Lambs | Full Movie")
    assert is_english_candidate("A Quiet Place (2018) Full English Movie")
    assert not is_english_candidate("Jai Vikraanta Hindi Full Movie")
    assert not is_english_candidate("रहना है तेरे दिल में हिंदी मूवी")
    assert not is_english_candidate("Avengers Endgame Hindi Dubbed Movie")
    assert not is_english_candidate("Theri Tamil Full Movie")


def test_has_english_marker():
    assert has_english_marker("Horror Movie Lost at Sea | Full Movies in English HD")
    assert has_english_marker("Awesome Action Movie in English")
    assert not has_english_marker("Entertainment | Full Movie | Akshay Kumar")
    assert not has_english_marker("A Quiet Place (2018) Full Movie")


def test_is_static():
    assert is_static(0.0, 2.5)
    assert is_static(1.2, 2.5)
    assert not is_static(4.0, 2.5)
    assert not is_static(2.5, 2.5)


def test_settings_static_motion_default():
    s = get_settings()
    assert s.min_clip_motion == 2.5


def test_system_prompt_covers_climax():
    assert "CLIMAX" in _SYSTEM_PROMPT
    assert "climax" in _SYSTEM_PROMPT
    assert "static title" in _SYSTEM_PROMPT


def test_sec_to_ts():
    assert sec_to_ts(0) == "00:00:00"
    assert sec_to_ts(59.9) == "00:01:00"
    assert sec_to_ts(61) == "00:01:01"
    assert sec_to_ts(3661) == "01:01:01"


def test_allocate_caption_timings_proportional():
    sentences = ["short", "a much longer sentence than the first one"]
    timings = allocate_caption_timings(sentences, total_duration=100.0)
    assert len(timings) == 2
    (s1, e1), (s2, e2) = timings
    assert s1 == 0.0
    assert e1 <= s2
    assert abs(e2 - 100.0) < 1e-6
    assert (e1 - s1) < (e2 - s2)  # shorter sentence -> shorter window


def test_allocate_caption_timings_empty():
    assert allocate_caption_timings([], 100.0) == []


def test_compensate_rate():
    assert _compensate_rate("+0%", ratio=2.0) == "-20%"  # clamped
    assert _compensate_rate("+0%", ratio=1.0) == "+0%"
    assert _compensate_rate("+0%", ratio=0.5) == "+20%"  # clamped
    assert _compensate_rate("+0%", ratio=0.8) == "+20%"


def test_parse_story_plan_valid():
    settings = Settings()
    data = {
        "hindi_script": "पहला वाक्य। दूसरा वाक्य!",
        "timestamps": [
            {"start_time": 10, "end_time": 18},
            {"start_time": 40, "end_time": 50},
        ],
    }
    plan = parse_story_plan(data, "abc123", settings)
    assert plan.source_video_id == "abc123"
    assert len(plan.timestamps) == 2
    assert plan.word_count == 4


def test_parse_story_plan_clamps_and_merges():
    settings = Settings()
    data = {
        "hindi_script": "एक कहानी।",
        "timestamps": [
            {"start_time": 5, "end_time": 3},   # end <= start -> dropped
            {"start_time": 0, "end_time": 2},   # too short -> clamped up
            {"start_time": 1, "end_time": 100},  # overlaps previous -> shifted; too long -> clamped
        ],
    }
    plan = parse_story_plan(data, "x", settings)
    assert len(plan.timestamps) == 2
    first, second = plan.timestamps
    assert first.duration == settings.min_scene_seconds
    assert second.start_time >= first.end_time
    assert second.duration <= settings.max_scene_seconds


def test_parse_story_plan_bounds_to_video_duration():
    settings = Settings()
    data = {
        "hindi_script": "एक कहानी।",
        "timestamps": [{"start_time": 5000, "end_time": 5020}],
    }
    plan = parse_story_plan(data, "x", settings, video_duration=5005)
    assert plan.timestamps[0].end_time == 5005


def test_parse_story_plan_rejects_bad():
    settings = Settings()
    with pytest.raises(StoryPlanError):
        parse_story_plan({"hindi_script": "", "timestamps": []}, "x", settings)
    with pytest.raises(StoryPlanError):
        parse_story_plan({"hindi_script": "ok", "timestamps": []}, "x", settings)
    with pytest.raises(StoryPlanError):
        parse_story_plan(
            {"hindi_script": "ok", "timestamps": [{"start_time": "bad", "end_time": 10}]},
            "x",
            settings,
        )


def test_clamp_crop_x_center():
    assert _clamp_crop_x(50.0, 60.0, 100.0) == pytest.approx(20.0)
    assert _clamp_crop_x(100.0, 60.0, 100.0) == pytest.approx(40.0)
    assert _clamp_crop_x(0.0, 60.0, 100.0) == pytest.approx(0.0)
    assert _clamp_crop_x(200.0, 60.0, 100.0) == pytest.approx(40.0)  # beyond right edge
    assert _clamp_crop_x(50.0, 150.0, 100.0) == 0.0  # window wider than frame


def test_used_movies_roundtrip(tmp_path):
    from main import load_used_movies, mark_movie_used, reset_used_movies

    path = tmp_path / "used.json"
    assert load_used_movies(path) == set()
    mark_movie_used(path, "aaa")
    mark_movie_used(path, "bbb")
    mark_movie_used(path, "aaa")
    assert load_used_movies(path) == {"aaa", "bbb"}
    reset_used_movies(path)
    assert load_used_movies(path) == set()


def test_has_existing_clips(tmp_path):
    from main import _has_existing_clips

    s = Settings(downloads_dir=tmp_path)
    assert not _has_existing_clips(s, "abcdefghijk")
    (tmp_path / "abcdefgh_00.mp4").write_bytes(b"x")
    assert _has_existing_clips(s, "abcdefghijk")


def test_backup_outputs_copies_and_prunes(tmp_path):
    from main import backup_outputs, prune_backups

    settings = Settings(backup_dir=tmp_path / "backups", backup_retention_hours=48)
    src = tmp_path / "final_short.mp4"
    src.write_bytes(b"fake-video")
    backup_outputs(settings, src)

    folders = list((tmp_path / "backups").iterdir())
    assert len(folders) == 1
    assert (folders[0] / "final_short.mp4").read_bytes() == b"fake-video"
    assert settings.backup_dir.exists()

    # Nothing older than the window is pruned; an old folder is.
    old = settings.backup_dir / "old"
    old.mkdir()
    old_time = time.time() - 49 * 3600.0
    import os

    os.utime(old, (old_time, old_time))
    prune_backups(settings.backup_dir, settings.backup_retention_hours)
    assert not old.exists()
    assert len(list(settings.backup_dir.iterdir())) == 1