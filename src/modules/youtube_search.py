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
    query: str = "full movie",
    max_results: int = 5,
    category_id: str = _FILM_CATEGORY_ID,
) -> list[MovieCandidate]:
    """Search for long, film-category videos and rank them by view count.

    Args:
        api_key: YouTube Data API v3 key.
        query: free-text search term (e.g. a genre or movie title).
        max_results: number of candidates to fetch (each is an HTTP request).
        category_id: YouTube video category id; default ``1`` = Film & Animation.

    Returns:
        A list of :class:`MovieCandidate` sorted by view count descending.

    Raises:
        YouTubeSearchError: on API/network failures or a missing API key.
    """
    client = _build_client(api_key)
    try:
        search_resp = (
            client.search()
            .list(
                part="snippet",
                q=query,
                type="video",
                videoDuration="long",
                videoCategoryId=category_id,
                maxResults=max_results,
                order="viewCount",
            )
            .execute()
        )
    except Exception as exc:
        raise YouTubeSearchError(f"search.list failed: {exc}") from exc

    items = search_resp.get("items", [])
    if not items:
        logger.info("no movie candidates found for query %r", query)
        return []

    video_ids = [item["id"]["videoId"] for item in items if item.get("id", {}).get("videoId")]
    # Reuse the snippet fields for channel / title and fill in duration + views.
    snippets = {
        item["id"]["videoId"]: item["snippet"]
        for item in items
        if item.get("id", {}).get("videoId")
    }
    details: dict[str, dict[str, Any]] = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i : i + 50]
        try:
            videos_resp = (
                client.videos()
                .list(part="contentDetails,statistics", id=",".join(chunk))
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
            )
        )

    candidates.sort(key=lambda c: c.view_count, reverse=True)
    for c in candidates:
        logger.info(
            "candidate: %s (%s) views=%d url=%s",
            c.title, c.duration_display, c.view_count, c.url,
        )
    return candidates