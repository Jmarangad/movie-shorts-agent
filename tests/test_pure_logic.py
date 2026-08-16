"""Pure-logic unit tests (no network / LLM / TTS / FFmpeg required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings, get_settings  # noqa: E402
from src.modules.downloader import sec_to_ts  # noqa: E402
from src.modules.transcript_llm import StoryPlanError, parse_story_plan  # noqa: E402
from src.modules.tts_generator import _compensate_rate  # noqa: E402
from src.modules.video_editor import (  # noqa: E402
    allocate_caption_timings,
    sequence_to_fit_duration,
)
from src.modules.youtube_search import parse_iso_duration  # noqa: E402


def test_settings_defaults():
    s = get_settings()
    assert s.target_script_words == 280
    assert s.max_scenes == 6
    assert s.min_scene_seconds == 5.0
    assert s.max_scene_seconds == 12.0


def test_parse_iso_duration():
    assert parse_iso_duration("PT1H32M10S") == 5530
    assert parse_iso_duration("PT2H") == 7200
    assert parse_iso_duration("PT45M") == 2700
    assert parse_iso_duration("PT30S") == 30
    assert parse_iso_duration("") == 0
    assert parse_iso_duration("garbage") == 0


def test_sec_to_ts():
    assert sec_to_ts(0) == "00:00:00"
    assert sec_to_ts(59.9) == "00:01:00"
    assert sec_to_ts(61) == "00:01:01"
    assert sec_to_ts(3661) == "01:01:01"


def test_sequence_to_fit_duration():
    assert sequence_to_fit_duration(30.0, 120.0) == (4, 120.0)
    assert sequence_to_fit_duration(60.0, 120.0) == (2, 120.0)
    assert sequence_to_fit_duration(100.0, 120.0) == (2, 200.0)
    assert sequence_to_fit_duration(130.0, 120.0) == (1, 130.0)
    assert sequence_to_fit_duration(0.0, 120.0) == (1, 0.0)


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
    assert _compensate_rate("+0%", ratio=2.0) == "-100%".replace("-100%", "-60%")  # clamped
    assert _compensate_rate("+0%", ratio=1.0) == "+0%"
    assert _compensate_rate("+0%", ratio=0.5) == "+50%"


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