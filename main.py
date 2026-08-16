"""Orchestrator: run the full movie-shorts pipeline end to end.

CLI usage::

    python main.py                          # search thriller/romantic/horror, new movie each run
    python main.py --video-id dQw4w9WgXcQ   # skip search, use a known video
    python main.py --dry-run                # no TTS/download/edit, plan only
    python main.py --reset-used             # forget all previously-used movies
    python main.py --schedule               # run now, then repeat every SCHEDULE_INTERVAL_HOURS
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

from config import Settings, get_settings
from src.modules.downloader import DownloadError, download_scenes
from src.modules.transcript_llm import (
    StoryPlanError,
    TranscriptError,
    fetch_transcript,
    generate_story_plan,
)
from src.modules.tts_generator import fit_rate_to_target, media_duration
from src.modules.video_editor import build_short
from src.modules.youtube_search import MovieCandidate, YouTubeSearchError, search_movies

logger = logging.getLogger("movie_shorts")

_PIPELINE_ERRORS = (YouTubeSearchError, TranscriptError, StoryPlanError, DownloadError)


def _setup_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _output_dirs(settings: Settings) -> tuple[Path, Path]:
    out = settings.output_dir
    down = settings.downloads_dir
    out.mkdir(parents=True, exist_ok=True)
    down.mkdir(parents=True, exist_ok=True)
    return out, down


# --- used-movie tracking -----------------------------------------------------
def load_used_movies(path: Path) -> set[str]:
    """Read the set of video_ids already turned into Shorts."""
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    return set(data.get("video_ids", []) if isinstance(data, dict) else data)


def mark_movie_used(path: Path, video_id: str) -> None:
    """Append ``video_id`` to the used-movies registry."""
    used = load_used_movies(path)
    used.add(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"video_ids": sorted(used)}, indent=2),
        encoding="utf-8",
    )


def reset_used_movies(path: Path) -> None:
    """Clear the used-movies registry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"video_ids": []}, indent=2), encoding="utf-8")


def _genres(settings: Settings) -> list[str]:
    return [g.strip() for g in settings.movie_genres.split(",") if g.strip()]


# --- output backup -----------------------------------------------------------
def prune_backups(backup_dir: Path, retention_hours: float) -> int:
    """Delete backup folders older than ``retention_hours``. Returns count."""
    if retention_hours <= 0 or not backup_dir.exists():
        return 0
    cutoff = time.time() - retention_hours * 3600.0
    pruned = 0
    for entry in backup_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            too_old = entry.stat().st_mtime < cutoff
        except OSError:
            too_old = False
        if too_old:
            shutil.rmtree(entry, ignore_errors=True)
            logger.info("pruned backup folder older than %dh: %s", retention_hours, entry.name)
            pruned += 1
    return pruned


def backup_outputs(
    settings: Settings,
    *paths: Path,
) -> Path:
    """Copy the run's artifacts into a fresh timestamped backup folder.

    The canonical ``output/`` files (final_short.mp4, narration.mp3,
    story_plan.json) stay in place so consumers read the latest; each run's
    copy is kept under ``settings.backup_dir`` until the retention window
    expires. Never raises: a failed backup only logs a warning.
    """
    dest = settings.backup_dir / time.strftime("%Y%m%d_%H%M%S")
    copied = 0
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for path in paths:
            if path.exists():
                shutil.copy2(path, dest / path.name)
                copied += 1
        logger.info("backed up %d artifact(s) to %s", copied, dest)
    except OSError as exc:
        logger.warning("backup to %s failed: %s", dest, exc)
    prune_backups(settings.backup_dir, settings.backup_retention_hours)
    return dest


def _select_candidate(
    settings: Settings,
    used: set[str],
    query: str | None,
    video_id: str | None,
) -> tuple[MovieCandidate, str]:
    """Pick the most-viewed, language-matching, not-yet-used movie with a transcript.

    Returns ``(candidate, transcript)``.
    """
    if video_id:
        candidate = MovieCandidate(
            video_id=video_id,
            title=video_id,
            channel="unknown",
            published_at="",
            duration_seconds=0,
            view_count=0,
            url=f"https://www.youtube.com/watch?v={video_id}",
        )
        logger.info("using explicit video %s", video_id)
        transcript = fetch_transcript(video_id, max_chars=settings.max_transcript_chars)
        return candidate, transcript

    candidates = search_movies(
        settings.youtube_api_key,
        query=query,
        genres=_genres(settings),
        max_results=5,
        language=settings.search_language,
    )
    # search_movies already ranks English-marked titles above view count;
    # iterate in that order, skipping movies we already turned into a Short.
    for cand in candidates:
        if cand.video_id in used:
            logger.info("skipping already-used movie %s (%s)", cand.video_id, cand.title)
            continue
        logger.info(
            "trying transcript for %s (%s, %.0f views) %s",
            cand.title, cand.duration_display, cand.view_count, cand.url,
        )
        try:
            transcript = fetch_transcript(cand.video_id, max_chars=settings.max_transcript_chars)
            return cand, transcript
        except TranscriptError as exc:
            logger.warning("no transcript for %s: %s", cand.video_id, exc)
    raise TranscriptError(
        "no unused candidate had a usable transcript (run --reset-used to allow repeats)"
    )


def run_pipeline(settings: Settings, query: str | None = None, video_id: str | None = None, dry_run: bool = False) -> Path:
    """Execute the five-module pipeline and return the final Short path."""
    start = time.time()
    out_dir, down_dir = _output_dirs(settings)
    prune_backups(settings.backup_dir, settings.backup_retention_hours)

    # 1-2. Search + transcript ----------------------------------------------------
    used = load_used_movies(settings.used_movies_path)
    candidate, transcript = _select_candidate(settings, used, query, video_id)
    logger.info("chosen: %s (%s, %.0f views)", candidate.title, candidate.duration_display, candidate.view_count)
    logger.info("transcript: %d chars", len(transcript))

    plan = generate_story_plan(
        candidate.video_id,
        transcript,
        settings,
        video_duration=candidate.duration_seconds or None,
    )
    logger.info("plan: %d scenes, %d words", len(plan.timestamps), plan.word_count)

    plan_path = out_dir / "story_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "source_video_id": candidate.video_id,
                "hindi_script": plan.hindi_script,
                "timestamps": [{"start_time": s.start_time, "end_time": s.end_time} for s in plan.timestamps],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    # Committed as soon as the movie is selected, so the next run picks a new one.
    mark_movie_used(settings.used_movies_path, candidate.video_id)
    if dry_run:
        logger.info("dry-run: wrote story_plan.json, stopping before TTS/download/edit")
        return plan_path

    # 3. Targeted clip download --------------------------------------------------
    clips = download_scenes(
        candidate.url,
        plan.timestamps,
        down_dir,
        settings,
        prefix=candidate.video_id[:8],
    )
    logger.info("downloaded %d clips", len(clips))

    # 4. TTS fitted to the distinct clip timeline --------------------------------
    # Every scene plays exactly once in the final Short, so the narration is
    # targeted at the total length of the downloaded clips, never longer.
    available = sum(media_duration(c, settings.ffprobe_binary) for c in clips)
    tts_target = min(float(settings.target_duration_seconds), max(available - 1.5, 10.0))
    logger.info("distinct clip timeline: %.1fs; narration target: %.1fs", available, tts_target)
    audio_path = out_dir / "narration.mp3"
    if audio_path.exists():
        audio_path.unlink()
    audio_path, audio_duration = fit_rate_to_target(
        plan.hindi_script, audio_path, settings, target_seconds=tts_target,
    )
    logger.info("narration: %.1fs -> %s", audio_duration, audio_path.name)

    # 5. Edit --------------------------------------------------------------------
    final_path = out_dir / "final_short.mp4"
    final_path = build_short(clips, audio_path, final_path, plan, settings)

    # Keep this run's artifacts in a timestamped backup folder (48 h by default).
    backup_outputs(settings, final_path, audio_path, plan_path)

    if not settings.keep_clips:
        for clip in clips:
            clip.unlink(missing_ok=True)
            logger.debug("removed clip %s", clip.name)

    logger.info("done in %.1fs -> %s", time.time() - start, final_path)
    return final_path


def run_scheduled(settings: Settings, query: str | None = None, video_id: str | None = None, dry_run: bool = False) -> int:
    """Run the pipeline forever, sleeping between runs."""
    interval = max(settings.schedule_interval_hours, 0.1) * 3600.0
    logger.info("scheduler started: every %.1f h", settings.schedule_interval_hours)
    while True:
        try:
            run_pipeline(settings, query=query, video_id=video_id, dry_run=dry_run)
        except _PIPELINE_ERRORS as exc:
            logger.error("scheduled run failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            logger.error("scheduled run crashed: %s", exc)
        logger.info("next run in %.1f h", interval / 3600.0)
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="movie-shorts-agent")
    parser.add_argument("--query", help="YouTube search query (overrides the genre-based search)")
    parser.add_argument("--video-id", help="skip search and process this video id directly")
    parser.add_argument("--dry-run", action="store_true", help="stop after the story plan is written")
    parser.add_argument("--schedule", action="store_true", help="run now, then repeat every SCHEDULE_INTERVAL_HOURS")
    parser.add_argument("--reset-used", action="store_true", help="forget all previously-used movies first")
    args = parser.parse_args(argv)

    settings = get_settings()
    _setup_logging(settings)

    if args.reset_used:
        reset_used_movies(settings.used_movies_path)
        logger.info("cleared used-movies registry at %s", settings.used_movies_path)

    if args.schedule:
        run_scheduled(settings, query=args.query, video_id=args.video_id, dry_run=args.dry_run)
        return 0  # unreachable; kept for typing

    try:
        run_pipeline(settings, query=args.query, video_id=args.video_id, dry_run=args.dry_run)
    except _PIPELINE_ERRORS as exc:
        logger.error("pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())