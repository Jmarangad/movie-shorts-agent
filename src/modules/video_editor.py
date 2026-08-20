"""Module 5: Assemble the final vertical Short and 16:9 long video.

Pipeline:
1. Load each downloaded clip and center-crop it from 16:9 to 9:16 (or fit the
   full frame for the 16:9 long format).
2. Prepend the "most-viewed" hook clip (which keeps its ORIGINAL dialog audio),
   then concatenate the clips, repeating the montage until it covers the audio
   duration (long format only), then trim to exact length.
3. Use the generated Hindi narration MP3 as the primary audio track, played
   AFTER the hook's original dialog ends.
4. Burn animated (fade in/out) Hindi captions synchronised to the narration.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
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


def _probe_dims(path: Path, ffprobe_binary: str) -> tuple[int, int] | None:
    """Return (width, height) of the video stream, or None if it cannot be read."""
    proc = subprocess.run(
        [
            ffprobe_binary, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    parts = proc.stdout.strip().split(",")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _portrait_transcode(
    clip_path: Path,
    out_path: Path,
    ffmpeg_binary: str,
    ffprobe_binary: str,
    out_w: int,
    out_h: int,
    focus_cx: float = 0.5,
    fps: int = 24,
) -> Path | None:
    """Transcode a clip to a portrait 9:16 CFR MP4 via ffmpeg (low memory).

    Unlike building a MoviePy montage from many open clips at once, this
    renders each clip to a small 1080x1920 file on disk (streaming through
    ffmpeg) so the final step only opens a single concatenated video.
    Returns ``None`` when the source cannot be decoded at all.
    """
    return _transcode(
        clip_path, out_path, ffmpeg_binary, ffprobe_binary,
        out_w, out_h, focus_cx=focus_cx, fps=fps, landscape=False,
    )


def _landscape_transcode(
    clip_path: Path,
    out_path: Path,
    ffmpeg_binary: str,
    ffprobe_binary: str,
    out_w: int,
    out_h: int,
    focus_cx: float = 0.5,
    fps: int = 24,
) -> Path | None:
    """Transcode a clip to a 16:9 landscape CFR MP4 (for the long video)."""
    return _transcode(
        clip_path, out_path, ffmpeg_binary, ffprobe_binary,
        out_w, out_h, focus_cx=focus_cx, fps=fps, landscape=True,
    )


def _transcode(
    clip_path: Path,
    out_path: Path,
    ffmpeg_binary: str,
    ffprobe_binary: str,
    out_w: int,
    out_h: int,
    focus_cx: float = 0.5,
    fps: int = 24,
    landscape: bool = False,
) -> Path | None:
    """Transcode a clip to a fixed-size CFR MP4 via ffmpeg (low memory).

    Portrait output (``landscape=False``) center-crops to the target aspect
    ratio; landscape output fits the whole frame into the target size without
    cropping (letterboxed if the source aspect differs). Returns ``None`` when
    the source cannot be decoded at all.
    """
    if out_path.exists():
        return out_path
    dims = _probe_dims(clip_path, ffprobe_binary)
    if dims is None:
        return None
    w, h = dims
    target_aspect = out_w / out_h
    src_aspect = w / h
    if landscape:
        # Fit the full frame inside the target box, letterbox top/bottom or
        # left/right when the source aspect differs from the target.
        vf = (f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
              f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:color=black")
    elif src_aspect > target_aspect:  # too wide -> narrow horizontal strip
        new_w = h * target_aspect
        x1 = _clamp_crop_x(focus_cx * w, new_w, w)
        vf = f"crop={new_w}:{h}:{x1}:0,scale={out_w}:{out_h}"
    else:  # too tall -> vertical slice
        new_h = w / target_aspect
        y1 = (h - new_h) / 2
        vf = f"crop={w}:{new_h}:0:{y1},scale={out_w}:{out_h}"

    proc = subprocess.run(
        [
            ffmpeg_binary, "-y", "-i", str(clip_path),
            "-vf", vf, "-r", str(fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-an", "-movflags", "+faststart",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not out_path.exists():
        logger.warning(
            "clip normalization failed (%s); skipping clip %s",
            proc.stderr.strip().splitlines()[-1][:160] if proc.stderr else "unknown",
            clip_path.name,
        )
        return None
    return out_path


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


def _montage_path(
    transcoded_paths: list[Path],
    tmp_dir: Path,
    ffmpeg_binary: str,
) -> Path:
    """Concatenate already-normalized clips into a single montage MP4."""
    list_file = tmp_dir / "concat.txt"
    list_file.write_text("".join(f"file '{p.name}'\n" for p in transcoded_paths))
    montage_path = tmp_dir / "montage.mp4"
    proc = subprocess.run(
        [
            ffmpeg_binary, "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy", str(montage_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not montage_path.exists():
        raise VideoEditError("could not build ffmpeg montage")
    return montage_path


def _build(
    clip_paths: list[Path],
    hook_path: Path | None,
    audio_path: Path,
    output_path: Path,
    story: StoryPlan,
    settings: Settings,
    landscape: bool,
    loop_to_audio: bool,
) -> Path:
    """Render a vertical Short or landscape long video with narration + captions.

    Both formats prepend the ``hook_path`` clip (the movie's most-viewed
    moment) before the main montage. The hook keeps its ORIGINAL dialog audio
    at the start; the narration (and its captions) start after the hook ends.
    When ``loop_to_audio`` is set (long format), the montage is repeated until
    it covers the full audio so the video length matches the ~5-minute target;
    the Short instead never repeats clips and warns if the narration runs
    longer than the footage.
    """
    try:
        from moviepy import (
            AudioFileClip,
            CompositeVideoClip,
            VideoFileClip,
            concatenate_audioclips,
            concatenate_videoclips,
        )
    except ImportError as exc:  # pragma: no cover
        raise VideoEditError("moviepy is not installed") from exc

    if not clip_paths:
        raise VideoEditError("no clips to edit")
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    out_w = settings.long_output_width if landscape else settings.output_width
    out_h = settings.long_output_height if landscape else settings.output_height

    narration = AudioFileClip(str(audio_path))
    narration_duration = narration.duration if narration.duration else media_duration(audio_path, settings.ffprobe_binary)
    logger.info("narration duration: %.1fs", narration_duration)

    # The hook keeps its original dialog, played before the narration begins.
    hook_audio = None
    hook_duration = 0.0
    if hook_path is not None:
        try:
            hook_audio = AudioFileClip(str(hook_path))
            hook_duration = hook_audio.duration or media_duration(hook_path, settings.ffprobe_binary)
            logger.info("hook audio: %.1fs of original dialog", hook_duration)
        except Exception as exc:  # noqa: BLE001 - fall back to a silent hook
            logger.warning("could not read hook audio (%s); hook will be silent", exc)
            hook_audio = None
            hook_duration = 0.0

    # Total timeline: hook (original dialog) + narration.
    full_duration = hook_duration + narration_duration
    logger.info("total timeline: %.1fs", full_duration)

    sources = []
    try:
        with tempfile.TemporaryDirectory(prefix="motion_") as tmp:
            tmp_dir = Path(tmp)
            transcoded_paths = []
            # The hook clip is prepended to every output (short and long).
            order = ([hook_path] if hook_path else []) + clip_paths
            for i, path in enumerate(order):
                focus_cx = focus_center_x(path)
                out = tmp_dir / f"{i:03d}_{path.stem}_v.mp4"
                transcode = _landscape_transcode if landscape else _portrait_transcode
                pt = transcode(
                    path, out, settings.ffmpeg_binary, settings.ffprobe_binary,
                    out_w, out_h, focus_cx, settings.output_fps,
                )
                if pt is None:
                    logger.warning("skipping undecodable clip %s", path.name)
                    continue
                transcoded_paths.append(pt)

            if not transcoded_paths:
                raise VideoEditError("no clips survived normalization")

            montage_path = _montage_path(transcoded_paths, tmp_dir, settings.ffmpeg_binary)
            montage = VideoFileClip(str(montage_path), audio=False)
            sources.append(montage)
            if montage.duration > full_duration:
                montage = montage.subclipped(0, full_duration)
            elif montage.duration < full_duration:
                if loop_to_audio:
                    logger.info(
                        "looping montage (%.1fs) to cover audio (%.1fs)",
                        montage.duration, full_duration,
                    )
                    loops = [montage] * (int(full_duration // montage.duration) + 1)
                    montage = concatenate_videoclips(loops).subclipped(0, full_duration)
                else:
                    logger.warning(
                        "distinct clip timeline (%.1fs) shorter than narration (%.1fs); "
                        "clips are never repeated, so the video ends early",
                        montage.duration, narration_duration,
                    )

            # Audio = original hook dialog, then the Hindi narration.
            if hook_audio is not None:
                full_audio = concatenate_audioclips([hook_audio, narration])
            else:
                full_audio = narration
            base = montage.with_audio(full_audio)

            sentences = story.sentences
            caption_clips: list = []
            if sentences:
                timings = allocate_caption_timings(sentences, narration_duration)
                for sentence, (start, end) in zip(sentences, timings):
                    caption_clips.append(
                        _caption_clip(
                            sentence, start + hook_duration, end - start,
                            out_w, out_h, settings,
                        )
                    )

            final = CompositeVideoClip(
                [base] + caption_clips, size=(out_w, out_h)
            )

            logger.info(
                "rendering %s (%d×%d, %.1fs)",
                output_path.name, out_w, out_h, final.duration,
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
        if hook_audio is not None:
            hook_audio.close()
        narration.close()
    return output_path


def build_short(
    clip_paths: list[Path],
    audio_path: Path,
    output_path: Path,
    story: StoryPlan,
    settings: Settings,
    hook_path: Path | None = None,
) -> Path:
    """Render the final 9:16 Short with narration audio + animated captions.

    ``hook_path`` (the most-viewed 3s opening moment) is prepended when given.
    """
    return _build(
        clip_paths, hook_path, audio_path, output_path, story, settings,
        landscape=False, loop_to_audio=False,
    )


def build_long(
    clip_paths: list[Path],
    audio_path: Path,
    output_path: Path,
    story: StoryPlan,
    settings: Settings,
    hook_path: Path | None = None,
) -> Path:
    """Render a 16:9 ~5-minute long video from the same clips + hook.

    The montage reuses the Short's clips and loops them to cover the longer
    narration, so the long video tells the full story without new downloads.
    """
    return _build(
        clip_paths, hook_path, audio_path, output_path, story, settings,
        landscape=True, loop_to_audio=True,
    )