"""Module 5: Assemble the final 9:16 vertical Short.

Pipeline:
1. Load each downloaded clip and center-crop it from 16:9 to 9:16.
2. Concatenate the clips, repeating the montage until it covers the Hindi
   audio duration, then trim to exact length.
3. Use the generated Hindi narration MP3 as the primary audio track.
4. Burn animated (fade in/out) Hindi captions synchronised to the narration.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable

from config import Settings
from src.modules.transcript_llm import StoryPlan
from src.modules.tts_generator import media_duration

logger = logging.getLogger(__name__)


class VideoEditError(RuntimeError):
    """Raised when the final Short cannot be rendered."""


def _clamp_crop_x(x_center: float, new_w: float, frame_w: float) -> float:
    """Left edge of a crop window centered on ``x_center``, clamped to the frame.

    Pure/testable. ``x_center`` is a pixel coordinate within ``[0, frame_w]``.
    """
    if new_w >= frame_w:
        return 0.0
    return max(0.0, min(x_center - new_w / 2.0, frame_w - new_w))


def focus_center_x(clip_path: Path, sample_interval: float = 0.5, max_frames: int = 40) -> float:
    """Find the normalized x-centre (0..1) of the in-focus subject in a clip.

    Samples frames and accumulates a per-column sharpness map (Sobel gradient
    magnitude) across the clip; a moving/cut subject keeps the accumulated
    centre weighted towards where the most sharpness lives. Falls back to the
    frame centre (0.5) when OpenCV is unavailable or the clip cannot be read.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return 0.5

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return 0.5
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(fps * sample_interval), 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if width <= 0:
        cap.release()
        return 0.5

    col_sharp = np.zeros(width, dtype=np.float64)
    count = 0
    idx = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            col_sharp += mag.mean(axis=0)
            count += 1
        idx += 1
    cap.release()

    total = float(col_sharp.sum())
    if total <= 0:
        return 0.5
    cols = np.arange(width)
    return float((col_sharp * cols).sum() / total) / width


def _vertical_crop(clip, out_w: int, out_h: int, focus_cx: float = 0.5):
    """Center-crop ``clip`` to the ``out_w:out_h`` (9:16) aspect, then resize.

    For landscape sources the horizontal crop window is centered on the
    in-focus subject (``focus_cx``, normalized 0..1) instead of the frame
    centre, so the object of focus stays in the middle of the Short.
    """
    w, h = clip.w, clip.h
    target_aspect = out_w / out_h
    src_aspect = w / h
    if src_aspect > target_aspect:  # too wide -> narrow horizontal strip
        new_w = h * target_aspect
        x1 = _clamp_crop_x(focus_cx * w, new_w, w)
        cropped = clip.cropped(x1=x1, x2=x1 + new_w, y1=0, y2=h)
    else:  # too tall -> vertical slice
        new_h = w / target_aspect
        y1 = (h - new_h) / 2
        cropped = clip.cropped(x1=0, x2=w, y1=y1, y2=y1 + new_h)
    return cropped.resized((int(out_w), int(out_h)))


def sequence_to_fit_duration(seq_duration: float, target: float) -> tuple[int, float]:
    """How many repetitions of a montage are needed to cover ``target``.

    Pure/testable: returns ``(repetitions, fitted_duration)``.
    """
    if seq_duration <= 0:
        return 1, 0.0
    reps = math.ceil(target / seq_duration)
    return max(reps, 1), reps * seq_duration


def allocate_caption_timings(
    sentences: Iterable[str],
    total_duration: float,
    min_duration: float = 1.4,
) -> list[tuple[float, float]]:
    """Distribute caption on-screen windows across the narration length.

    Pure/testable: durations are proportional to sentence length.
    Returns ``[(start, end), ...]``.
    """
    items = list(sentences)
    if not items:
        return []
    weights = [max(len(s), 1) for s in items]
    total_weight = sum(weights)
    starts: list[tuple[float, float]] = []
    cursor = 0.0
    for weight in weights:
        dur = max(total_duration * weight / total_weight, min_duration)
        starts.append((cursor, cursor + dur))
        cursor += dur
    return starts


def _split_sentences(text: str) -> list[str]:
    import re

    parts = [p.strip() for p in re.split(r"[।\.!\?…]+", text)]
    return [p for p in parts if p]


def _pil_text_clip(text: str, font_path: str, font_size: int, out_w: int):
    """Render a caption frame with Pillow (fallback if ImageMagick is unavailable)."""
    import numpy as np
    from moviepy import ImageClip

    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(font_path, size=font_size)
    max_line_chars = max(14, int(out_w / (font_size * 0.62)))
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if len(trial) <= max_line_chars or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    tmp = Image.new("RGBA", (out_w, 800), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)
    line_height = font_size + 14
    y = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=3)
        tw = bbox[2] - bbox[0]
        x = (out_w - tw) // 2
        draw.text(
            (x, y), line, font=font, fill=(255, 255, 255, 255),
            stroke_width=3, stroke_fill=(0, 0, 0, 255),
        )
        y += line_height
    # Trim to content height.
    bbox = tmp.getbbox()
    if bbox:
        tmp = tmp.crop(bbox)
    frame = np.array(tmp)
    return ImageClip(frame)


def _moviepy_text_clip(
    text: str,
    font: str,
    font_size: int,
    stroke_width: int,
    out_w: int,
):
    """Render a caption with MoviePy's ImageMagick-backed TextClip."""
    from moviepy import TextClip

    return TextClip(
        text=text,
        font=font,
        font_size=font_size,
        color="white",
        stroke_color="black",
        stroke_width=stroke_width,
        method="caption",
        size=(int(out_w * 0.92), None),
    )


def _caption_clip(
    text: str,
    start: float,
    duration: float,
    out_w: int,
    out_h: int,
    settings: Settings,
):
    """Build one animated caption clip (fade in/out, lower third)."""
    from moviepy.video.fx import FadeIn, FadeOut

    try:
        base = _moviepy_text_clip(
            text, settings.hindi_font, settings.caption_font_size,
            settings.caption_stroke, out_w,
        )
    except Exception as exc:
        logger.debug("TextClip failed (%s); using PIL caption", exc)
        base = _pil_text_clip(
            text, settings.hindi_font, settings.caption_font_size, out_w,
        )

    duration = max(duration, 0.4)
    fade = min(0.25, duration / 4)
    return (
        base.with_duration(duration)
        .with_start(start)
        .with_position(("center", int(out_h * 0.72)))
        .with_effects([FadeIn(fade), FadeOut(fade)])
    )


def build_short(
    clip_paths: list[Path],
    audio_path: Path,
    output_path: Path,
    story: StoryPlan,
    settings: Settings,
) -> Path:
    """Render the final 9:16 Short with narration audio + animated captions."""
    try:
        from moviepy import (
            AudioFileClip,
            CompositeVideoClip,
            VideoFileClip,
            concatenate_videoclips,
        )
    except ImportError as exc:  # pragma: no cover
        raise VideoEditError("moviepy is not installed") from exc

    if not clip_paths:
        raise VideoEditError("no clips to edit")
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = AudioFileClip(str(audio_path))
    audio_duration = audio.duration if audio.duration else media_duration(audio_path, settings.ffprobe_binary)
    logger.info("audio duration: %.1fs", audio_duration)

    sources = []
    try:
        verticals = []
        for path in clip_paths:
            focus_cx = focus_center_x(path)
            source = VideoFileClip(str(path), audio=False)
            sources.append(source)
            verticals.append(
                _vertical_crop(source, settings.output_width, settings.output_height, focus_cx=focus_cx)
            )

        montage = concatenate_videoclips(verticals, method="compose")
        reps, fitted = sequence_to_fit_duration(montage.duration, audio_duration)
        if reps > 1:
            logger.info("repeating montage %d× (%.1fs -> %.1fs)", reps, montage.duration, fitted)
            montage = concatenate_videoclips([montage] * reps, method="compose")
        if montage.duration > audio_duration:
            montage = montage.subclipped(0, audio_duration)

        base = montage.with_audio(audio)

        sentences = story.sentences
        caption_clips: list = []
        if sentences:
            timings = allocate_caption_timings(sentences, audio_duration)
            for sentence, (start, end) in zip(sentences, timings):
                caption_clips.append(
                    _caption_clip(
                        sentence, start, end - start,
                        settings.output_width, settings.output_height, settings,
                    )
                )

        final = CompositeVideoClip(
            [base] + caption_clips, size=(settings.output_width, settings.output_height)
        )

        logger.info(
            "rendering %s (%d×%d, %.1fs)",
            output_path.name, settings.output_width, settings.output_height, final.duration,
        )
        final.write_videofile(
            str(output_path),
            fps=settings.output_fps,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            bitrate="4000k",
            preset="veryfast",
            threads=2,
            logger=None,
        )
    finally:
        for source in sources:
            source.close()
        audio.close()
    return output_path