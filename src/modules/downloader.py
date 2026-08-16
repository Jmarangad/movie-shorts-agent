"""Module 4: Download only the selected movie segments with yt-dlp.

Uses ``yt-dlp`` with ``--download-sections`` (handled by the external ffmpeg
downloader) so only the 5-12 second clips chosen by the LLM are fetched --
never the full movie.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from config import Settings
from src.modules.transcript_llm import Scene

logger = logging.getLogger(__name__)

_FORMAT = "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b"


class DownloadError(RuntimeError):
    """Raised when a segment clip cannot be downloaded."""


def sec_to_ts(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS`` for yt-dlp's download-sections."""
    total = max(int(round(seconds)), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _ytdlp_command(
    video_url: str,
    start: float,
    end: float,
    out_path: Path,
    ffmpeg_binary: str,
) -> list[str]:
    return [
        "yt-dlp",
        "--download-sections", f"*{sec_to_ts(start)}-{sec_to_ts(end)}",
        "--force-keyframes-at-cuts",
        "--external-downloader", "ffmpeg",
        "--external-downloader-args", f"ffmpeg_i:-nostdin",
        "-f", _FORMAT,
        "--merge-output-format", "mp4",
        "--no-part",
        "--no-mtime",
        "--retries", "2",
        "--fragment-retries", "2",
        "--socket-timeout", "30",
        "--no-overwrites",
        "--quiet",
        "--no-warnings",
        "-o", str(out_path),
        video_url,
    ]


def download_clip(
    video_url: str,
    scene: Scene,
    out_path: Path,
    ffmpeg_binary: str = "ffmpeg",
    retries: int = 3,
) -> Path:
    """Download a single segment and verify the clip exists and is non-empty."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _ytdlp_command(video_url, scene.start_time, scene.end_time, out_path, ffmpeg_binary)
    last_error: str | None = None
    for attempt in range(1, retries + 1):
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            logger.info(
                "clip %s [%.1f-%.1f] -> %s (%.1f KB)",
                Path(video_url).name if video_url.endswith(".mp4") else video_url.split("=")[-1][:8],
                scene.start_time, scene.end_time, out_path.name,
                out_path.stat().st_size / 1024,
            )
            return out_path
        last_error = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
        logger.warning(
            "download attempt %d/%d failed for %s: %s",
            attempt, retries, out_path.name, last_error[-300:],
        )
        if attempt < retries:
            time.sleep(2 * attempt)
    raise DownloadError(
        f"failed to download segment {scene.start_time:.1f}-{scene.end_time:.1f}s: {last_error}"
    )


def download_scenes(
    video_url: str,
    scenes: list[Scene],
    downloads_dir: Path,
    settings: Settings,
    prefix: str = "clip",
) -> list[Path]:
    """Download all scenes sequentially and return the produced clip paths."""
    clips: list[Path] = []
    for index, scene in enumerate(scenes):
        out_path = downloads_dir / f"{prefix}_{index:02d}.mp4"
        clip = download_clip(
            video_url,
            scene,
            out_path,
            ffmpeg_binary=settings.ffmpeg_binary,
            retries=settings.max_download_retries,
        )
        clips.append(clip)
    return clips