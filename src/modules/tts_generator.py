"""Module 3: Hindi voiceover generation with ``edge-tts``.

Synthesises the narration asynchronously (``hi-IN-SwaraNeural`` by default),
measures the produced duration with ffprobe, and automatically re-synthesises
at a compensated speaking rate until the track lands close to the target
Short length.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

from config import Settings

logger = logging.getLogger(__name__)

_TARGET_TOLERANCE_S = 3.0
# Narrow band keeps the voice natural; edge-tts beyond ~±20% sounds robotic.
_RATE_CLAMP_PERCENT = 20


class TTSError(RuntimeError):
    """Raised when edge-tts fails to produce audio."""


def media_duration(path: Path, ffprobe_binary: str = "ffprobe") -> float:
    """Return the media duration in seconds using ffprobe."""
    proc = subprocess.run(
        [
            ffprobe_binary,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(proc.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def _compensate_rate(current_rate: str, ratio: float) -> str:
    """Scale a ``+X%`` / ``-X%`` edge-tts rate so speech length matches ``ratio``.

    ``ratio`` = desired_duration / measured_duration (>1 slows down, <1 speeds up).
    """
    percent = int(current_rate.replace("%", "")) if current_rate.strip() else 0
    new_percent = round(percent + (1.0 - ratio) * 100.0)
    new_percent = max(-_RATE_CLAMP_PERCENT, min(_RATE_CLAMP_PERCENT, new_percent))
    return f"{new_percent:+d}%"


async def synthesize_async(
    text: str,
    out_path: Path,
    voice: str,
    rate: str = "+0%",
) -> Path:
    """Synthesise ``text`` to an MP3 at ``out_path`` (async)."""
    import edge_tts

    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    try:
        await communicate.save(str(out_path))
    except Exception as exc:
        raise TTSError(f"edge-tts synthesis failed: {exc}") from exc
    return out_path


def synthesize(
    text: str,
    out_path: Path,
    settings: Settings,
    rate: str | None = None,
) -> float:
    """Synthesise and return the resulting audio duration in seconds."""
    asyncio.run(
        synthesize_async(
            text,
            out_path,
            voice=settings.tts_voice,
            rate=rate if rate is not None else settings.tts_rate,
        )
    )
    return media_duration(out_path, settings.ffprobe_binary)


def fit_rate_to_target(
    text: str,
    out_path: Path,
    settings: Settings,
    target_seconds: float | None = None,
) -> tuple[Path, float]:
    """Synthesise the narration and adjust the rate to hit the target length.

    Returns ``(audio_path, duration)``. The first pass uses the configured
    ``tts_rate``; subsequent passes correct the rate from the measured ratio,
    up to a few attempts. ``target_seconds`` overrides
    ``settings.target_duration_seconds`` (e.g. when the narration must fit the
    length of the distinct clips, with no repetition).
    """
    target = float(target_seconds or settings.target_duration_seconds)
    rate = settings.tts_rate
    best = (out_path, 0.0)
    for attempt in range(3):
        duration = synthesize(text, out_path, settings, rate=rate)
        best = (out_path, duration)
        logger.info("tts attempt %d: rate=%s duration=%.1fs", attempt, rate, duration)
        if abs(duration - target) <= _TARGET_TOLERANCE_S:
            return best
        ratio = target / duration if duration > 0 else 1.0
        if ratio <= 0.01 or ratio >= 100.0:
            break
        new_rate = _compensate_rate(rate, ratio)
        if new_rate == rate:
            break
        rate = new_rate
    return best