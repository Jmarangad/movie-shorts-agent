"""Module 1: Find full-length movies on YouTube.

Uses the YouTube Data API v3 ``search.list`` with ``videoDuration=long`` and
``videoCategoryId=1`` (Film & Animation), then enriches results with per-video
duration and view counts via ``videos.list``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_FILM_CATEGORY_ID = "1"
_ISO_DURATION_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


class YouTubeSearchError(RuntimeError):
    """Raised when the YouTube Data API cannot be queried."""


@dataclass(frozen=True)
class MovieCandidate:
    """A full-length movie candidate returned by the search."""

    video_id: str
    title: str
    channel: str
    published_at: str
    duration_seconds: int
    view_count: int
    url: str
    language: str = ""

    @property
    def duration_display(self) -> str:
        hours, rem = divmod(self.duration_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"


def parse_iso_duration(value: str) -> int:
    """Convert an ISO-8601 duration (``PT1H32M10S``) to seconds."""
    if not value:
        return 0
    match = _ISO_DURATION_RE.fullmatch(value.strip())
    if not match:
        return 0
    hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _build_client(api_key: str) -> Any:
    try:
        from googleapiclient.discovery import build
    except ImportError as exc:  # pragma: no cover - deps installed in image
        raise YouTubeSearchError("google-api-python-client is not installed") from exc
    if not api_key:
        raise YouTubeSearchError("YOUTUBE_API_KEY is not set (see .env.example)")
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def search_movies(
    api_key: str,
    query: str | None = None,
    genres: list[str] | None = None,
    max_results: int = 5,
    category_id: str = _FILM_CATEGORY_ID,
    language: str = "en",
) -> list[MovieCandidate]:
    """Search for long, film-category movies and rank them by view count.

    One query per genre is issued (or a single free-text query when ``genres``
    is empty). Candidates are deduplicated by ``video_id``, enriched with
    duration/views/audio language, filtered to the preferred language, and
    sorted by view count descending.

    Args:
        api_key: YouTube Data API v3 key.
        query: free-text search term, used only when ``genres`` is empty.
        genres: genre terms to search, e.g. ``["thriller", "horror"]``.
        max_results: number of candidates to fetch per genre query.
        category_id: YouTube video category id; default ``1`` = Film & Animation.
        language: preferred audio language (e.g. ``en``); candidates with an
            unknown language are kept, otherwise only matching ones.

    Returns:
        A list of :class:`MovieCandidate` sorted by view count descending.

    Raises:
        YouTubeSearchError: on API/network failures or a missing API key.
    """
    client = _build_client(api_key)
    queries = [f"{g} movie full movie" for g in genres] if genres else [query or "full movie"]

    seen: dict[str, dict[str, Any]] = {}
    try:
        for q in queries:
            search_resp = (
                client.search()
                .list(
                    part="snippet",
                    q=q,
                    type="video",
                    videoDuration="long",
                    videoCategoryId=category_id,
                    maxResults=max_results,
                    order="viewCount",
                )
                .execute()
            )
            for item in search_resp.get("items", []):
                vid = item.get("id", {}).get("videoId")
                if vid and vid not in seen:
                    seen[vid] = item
    except Exception as exc:
        raise YouTubeSearchError(f"search.list failed: {exc}") from exc

    if not seen:
        logger.info("no movie candidates found for queries %r", queries)
        return []

    video_ids = list(seen)
    snippets = {vid: item["snippet"] for vid, item in seen.items()}
    details: dict[str, dict[str, Any]] = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i : i + 50]
        try:
            videos_resp = (
                client.videos()
                .list(part="snippet,contentDetails,statistics", id=",".join(chunk))
                .execute()
            )
        except Exception as exc:
            raise YouTubeSearchError(f"videos.list failed: {exc}") from exc
        for video in videos_resp.get("items", []):
            details[video["id"]] = video

    candidates: list[MovieCandidate] = []
    for video_id in video_ids:
        snippet = snippets.get(video_id, {})
        video = details.get(video_id, {})
        content = video.get("contentDetails", {})
        stats = video.get("statistics", {})
        v_snippet = video.get("snippet", {})
        lang = v_snippet.get("defaultAudioLanguage") or v_snippet.get("defaultLanguage") or ""
        try:
            views = int(stats.get("viewCount", 0) or 0)
        except (TypeError, ValueError):
            views = 0
        candidates.append(
            MovieCandidate(
                video_id=video_id,
                title=snippet.get("title", video_id),
                channel=snippet.get("channelTitle", "unknown"),
                published_at=snippet.get("publishedAt", ""),
                duration_seconds=parse_iso_duration(content.get("duration", "")),
                view_count=views,
                url=f"https://www.youtube.com/watch?v={video_id}",
                language=lang,
            )
        )

    # Keep only candidates whose audio language matches the preference,
    # falling back to everything when no candidate declares a language.
    if language:
        matching = [c for c in candidates if c.language.startswith(language)]
        if matching:
            candidates = matching

    candidates.sort(key=lambda c: c.view_count, reverse=True)
    for c in candidates:
        logger.info(
            "candidate: %s (%s) views=%d lang=%r url=%s",
            c.title, c.duration_display, c.view_count, c.language, c.url,
        )
    return candidates