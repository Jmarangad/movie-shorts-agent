"""Module 2: Fetch the movie transcript and turn it into a Hindi story plan.

Two responsibilities:

1. ``fetch_transcript`` pulls timestamped (auto-generated or manual) subtitles
   via ``youtube-transcript-api``.
2. ``generate_story_plan`` asks an LLM (Google Gemini via ``google-genai``,
   or OpenAI via the ``openai`` package) for a ~220-word Hindi narration
   plus 4-6 key visual scenes with start/end times in seconds.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from config import Settings

logger = logging.getLogger(__name__)

# Devanagari danda / sentence punctuation used to split the Hindi narration.
_SENTENCE_SPLIT_RE = re.compile(r"[।\.!\?…]+")

_SYSTEM_PROMPT = (
    "You are a skilled Hindi storyteller writing the voiceover for a "
    "{target_seconds}-second YouTube Short about a full-length movie. Given the "
    "timestamped transcript, "
    "you write a natural, human narration and pick the most visually striking scenes.\n"
    "Rules:\n"
    "- hindi_script: around {target_words} words of natural, conversational Hindi "
    "in Devanagari script. Write the way a real storyteller talks to a friend: "
    "short punchy sentences, casual yet vivid phrasing, occasional rhetorical "
    "questions, and a rhythm that breathes. Avoid stiff, literal or list-like "
    "narration.\n"
    "- Tell the COMPLETE story: the setup, the conflicts, and the CLIMAX and "
    "ending - reveal how the film concludes (spoilers are expected and desired). "
    "Narrate the climax DRAMATICALLY: build tension, use vivid language and "
    "beat-pauses (e.g. '\u0914\u0930 \u092b\u093f\u0930...', '\u0909\u0938\u0940 \u092a\u0932...'), "
    "and make the emotional payoff land. End the script with a hook line.\n"
    "- timestamps: exactly 4 to {max_scenes} scenes, each {min_scene:.0f} to "
    "{max_scene:.0f} seconds long, in ascending order, all inside the video "
    "duration. Pick the highest-impact visual moments, and each scene must "
    "directly show the on-screen moment the narration is talking about at "
    "that point (the most-viewed, most dramatic beats of the film). Include "
    "at least one scene from the climax / final act, and avoid static title "
    "cards or photo montages.\n"
    "Return ONLY valid JSON matching this schema:\n"
    '{{"hindi_script": str, '
    '"timestamps": [{{"start_time": float, "end_time": float}}]}}'
)


class TranscriptError(RuntimeError):
    """Raised when no usable transcript exists for a video."""


class StoryPlanError(RuntimeError):
    """Raised when the LLM output cannot be parsed into a valid story plan."""


@dataclass
class Scene:
    """A single visual clip to cut out of the movie."""

    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class StoryPlan:
    """The LLM-generated plan: Hindi narration + selected scenes."""

    hindi_script: str
    timestamps: list[Scene]
    source_video_id: str

    @property
    def sentences(self) -> list[str]:
        parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(self.hindi_script)]
        return [p for p in parts if p]

    @property
    def word_count(self) -> int:
        return len(self.hindi_script.split())


# --------------------------------------------------------------------------- #
# Transcript fetching
# --------------------------------------------------------------------------- #
def fetch_transcript(
    video_id: str,
    max_chars: int = 250_000,
    preferred_languages: tuple[str, ...] = ("en", "en-US", "en-GB"),
) -> str:
    """Return a timestamped transcript string ``[seconds] text`` for a video.

    Prefers manually created English subtitles, then auto-generated ones.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:  # pragma: no cover
        raise TranscriptError("youtube-transcript-api is not installed") from exc

    api = YouTubeTranscriptApi()
    try:
        transcript_list = api.list(video_id)
    except Exception as exc:
        raise TranscriptError(f"unable to list transcripts for {video_id}: {exc}") from exc

    transcript = None
    try:
        transcript = transcript_list.find_transcript(list(preferred_languages))
    except Exception:
        try:
            transcript = transcript_list.find_generated_transcript()
        except Exception:
            try:
                transcript = transcript_list.find_manually_created_transcript()
            except Exception:
                transcript = None

    if transcript is None:
        raise TranscriptError(
            f"no usable transcript found for {video_id} "
            "(manual or auto-generated, incl. English)"
        )

    try:
        entries = transcript.fetch()
    except Exception as exc:
        raise TranscriptError(f"failed to fetch transcript for {video_id}: {exc}") from exc

    lines: list[str] = []
    chars = 0
    for entry in entries:
        if isinstance(entry, dict):
            start = float(entry.get("start", 0))
            text = str(entry.get("text", "")).strip()
        else:
            start = float(getattr(entry, "start", 0))
            text = str(getattr(entry, "text", "")).strip()
        if not text:
            continue
        line = f"[{start:07.1f}] {text}"
        chars += len(line)
        if chars > max_chars:
            break
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# LLM story-plan generation
# --------------------------------------------------------------------------- #
def _build_prompt(transcript: str, settings: Settings) -> str:
    system = _SYSTEM_PROMPT.format(
        target_seconds=settings.target_duration_seconds,
        target_words=settings.target_script_words,
        max_scenes=settings.max_scenes,
        min_scene=settings.min_scene_seconds,
        max_scene=settings.max_scene_seconds,
    )
    return (
        f"{system}\n\n"
        f"Movie transcript (timestamped, seconds into the video):\n"
        f"<transcript>\n{transcript}\n</transcript>\n"
        f"\nTarget narration word count: {settings.target_script_words} words."
    )


def _call_gemini(prompt: str, settings: Settings) -> dict[str, Any]:
    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "hindi_script": {"type": "string"},
            "timestamps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_time": {"type": "number"},
                        "end_time": {"type": "number"},
                    },
                    "required": ["start_time", "end_time"],
                },
            },
        },
        "required": ["hindi_script", "timestamps"],
    }
    resp = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.6,
        ),
    )
    return json.loads(resp.text)


def _call_openai(prompt: str, settings: Settings) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": "You produce valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.6,
    )
    return _parse_json_content(resp.choices[0].message.content or "{}")


def _call_deepseek(prompt: str, settings: Settings) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    resp = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[
            {"role": "system", "content": "You produce valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.6,
    )
    return _parse_json_content(resp.choices[0].message.content or "{}")


def _call_opencode(prompt: str, settings: Settings) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.opencode_api_key or "opencode",
        base_url=settings.opencode_base_url,
    )
    resp = client.chat.completions.create(
        model=settings.opencode_model,
        messages=[
            {"role": "system", "content": "You produce valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.6,
        max_tokens=4000,
    )
    return _parse_json_content(resp.choices[0].message.content or "{}")


def _call_ollama(prompt: str, settings: Settings) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.ollama_api_key or "ollama",
        base_url=settings.ollama_base_url,
    )
    resp = client.chat.completions.create(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": "You produce valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        max_tokens=4000,
    )
    return _parse_json_content(resp.choices[0].message.content or "{}")


def _parse_json_content(content: str) -> dict[str, Any]:
    """Parse ``content`` as JSON, tolerating stray text around the object."""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    start = content.find("{")
    if start == -1:
        raise StoryPlanError("LLM response contained no JSON object")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(content)):
        ch = content[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(content[start : i + 1])
    raise StoryPlanError("LLM response contained an unbalanced JSON object")


def _call_llm(prompt: str, settings: Settings) -> dict[str, Any]:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise StoryPlanError("OPENAI_API_KEY is not set")
        return _call_openai(prompt, settings)
    if settings.llm_provider == "deepseek":
        if not settings.deepseek_api_key:
            raise StoryPlanError("DEEPSEEK_API_KEY is not set")
        return _call_deepseek(prompt, settings)
    if settings.llm_provider == "opencode":
        return _call_opencode(prompt, settings)
    if settings.llm_provider == "ollama":
        return _call_ollama(prompt, settings)
    if not settings.gemini_api_key:
        raise StoryPlanError("GEMINI_API_KEY is not set")
    return _call_gemini(prompt, settings)


def parse_story_plan(
    data: dict[str, Any],
    video_id: str,
    settings: Settings,
    video_duration: float | None = None,
) -> StoryPlan:
    """Validate and normalise raw LLM JSON into a :class:`StoryPlan`.

    Pure and testable: scene timings are clamped to the configured ranges,
    sorted, deduplicated and bounded by ``video_duration``.
    """
    script = str(data.get("hindi_script") or "").strip()
    if not script:
        raise StoryPlanError("LLM returned an empty hindi_script")

    raw_scenes = data.get("timestamps") or []
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise StoryPlanError("LLM returned no timestamps")

    scenes: list[Scene] = []
    for item in raw_scenes[: settings.max_scenes]:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start_time"))
            end = float(item.get("end_time"))
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start:
            continue
        scenes.append(Scene(start_time=start, end_time=end))

    if not scenes:
        raise StoryPlanError("LLM returned no usable scenes")

    scenes.sort(key=lambda s: s.start_time)
    merged: list[Scene] = []
    for scene in scenes:
        if merged and scene.start_time < merged[-1].end_time:
            scene = Scene(start_time=merged[-1].end_time, end_time=scene.end_time)
        if scene.end_time <= scene.start_time:
            continue
        duration = scene.end_time - scene.start_time
        if duration < settings.min_scene_seconds:
            scene = Scene(
                start_time=scene.start_time,
                end_time=scene.start_time + settings.min_scene_seconds,
            )
        elif duration > settings.max_scene_seconds:
            scene = Scene(
                start_time=scene.start_time,
                end_time=scene.start_time + settings.max_scene_seconds,
            )
        if video_duration is not None and scene.end_time > video_duration:
            scene = Scene(start_time=scene.start_time, end_time=video_duration)
        if scene.end_time > scene.start_time:
            merged.append(scene)
        if len(merged) >= settings.max_scenes:
            break

    return StoryPlan(hindi_script=script, timestamps=merged, source_video_id=video_id)


def generate_story_plan(
    video_id: str,
    transcript: str,
    settings: Settings,
    video_duration: float | None = None,
) -> StoryPlan:
    """Run the LLM over the transcript and return a validated story plan."""
    prompt = _build_prompt(transcript, settings)
    data = _call_llm(prompt, settings)
    plan = parse_story_plan(data, video_id, settings, video_duration=video_duration)
    logger.info(
        "story plan: %d words, %d scenes for %s",
        plan.word_count, len(plan.timestamps), video_id,
    )
    return plan