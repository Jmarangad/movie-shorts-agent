"""Orchestrator: run the full movie-shorts pipeline end to end.

CLI usage::

    python main.py --query "hindi action movie full movie"
    python main.py --video-id dQw4w9WgXcQ   # skip search, use a known video
    python main.py --dry-run                # no TTS/download/edit, plan only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from config import Settings, get_settings
from src.modules.downloader import download_scenes
from src.modules.transcript_llm import (
    StoryPlanError,
    TranscriptError,
    fetch_transcript,
    generate_story_plan,
)
from src.modules.tts_generator import fit_rate_to_target
from src.modules.video_editor import build_short
from src.modules.youtube_search import MovieCandidate, YouTubeSearchError, search_movies

logger = logging.getLogger("movie_shorts")


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


def run_pipeline(settings: Settings, query: str | None = None, video_id: str | None = None, dry_run: bool = False) -> Path:
    """Execute the five-module pipeline and return the final Short path."""
    start = time.time()
    out_dir, down_dir = _output_dirs(settings)

    # 1. Search ---------------------------------------------------------------
    if video_id:
        candidates = [
            MovieCandidate(
                video_id=video_id,
                title=video_id,
                channel="unknown",
                published_at="",
                duration_seconds=0,
                view_count=0,
                url=f"https://www.youtube.com/watch?v={video_id}",
            )
        ]
        logger.info("using explicit video %s", video_id)
    else:
        if not query:
            query = "full movie"
        candidates = search_movies(settings.youtube_api_key, query=query, max_results=5)
        candidates.sort(key=lambda c: c.duration_seconds, reverse=True)
        if not candidates:
            raise YouTubeSearchError(f"no movie found for query {query!r}")

    # 2. Transcript + story plan ------------------------------------------------
    candidate: MovieCandidate | None = None
    transcript: str | None = None
    for cand in candidates:
        logger.info("trying transcript for %s (%s) %s", cand.title, cand.duration_display, cand.url)
        try:
            transcript = fetch_transcript(cand.video_id, max_chars=settings.max_transcript_chars)
            candidate = cand
            break
        except TranscriptError as exc:
            logger.warning("no transcript for %s: %s", cand.video_id, exc)
    if candidate is None or transcript is None:
        raise TranscriptError("no candidate had a usable transcript")
    logger.info("chosen: %s (%s)", candidate.title, candidate.duration_display)
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
    if dry_run:
        logger.info("dry-run: wrote story_plan.json, stopping before TTS/download/edit")
        return plan_path

    # 3. TTS --------------------------------------------------------------------
    audio_path = out_dir / "narration.mp3"
    if audio_path.exists():
        audio_path.unlink()
    audio_path, audio_duration = fit_rate_to_target(plan.hindi_script, audio_path, settings)
    logger.info("narration: %.1fs -> %s", audio_duration, audio_path.name)

    # 4. Targeted clip download --------------------------------------------------
    clips = download_scenes(
        candidate.url,
        plan.timestamps,
        down_dir,
        settings,
        prefix=candidate.video_id[:8],
    )
    logger.info("downloaded %d clips", len(clips))

    # 5. Edit --------------------------------------------------------------------
    final_path = out_dir / "final_short.mp4"
    final_path = build_short(clips, audio_path, final_path, plan, settings)

    if not settings.keep_clips:
        for clip in clips:
            clip.unlink(missing_ok=True)
            logger.debug("removed clip %s", clip.name)

    logger.info("done in %.1fs -> %s", time.time() - start, final_path)
    return final_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="movie-shorts-agent")
    parser.add_argument("--query", help="YouTube search query (e.g. 'hindi action movie full movie')")
    parser.add_argument("--video-id", help="skip search and process this video id directly")
    parser.add_argument("--dry-run", action="store_true", help="stop after the story plan is written")
    args = parser.parse_args(argv)

    settings = get_settings()
    _setup_logging(settings)

    try:
        run_pipeline(settings, query=args.query, video_id=args.video_id, dry_run=args.dry_run)
    except (YouTubeSearchError, TranscriptError, StoryPlanError) as exc:
        logger.error("pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())