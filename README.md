# movie-shorts-agent

An autonomous, end-to-end Python agent that creates 2-minute Hindi YouTube
Shorts which summarise full-length movies found on YouTube.

The agent searches for a full movie, fetches its transcript, has an LLM write a
fast-paced Hindi narration plus the most striking 5–12s scenes, synthesises the
voiceover, downloads *only* those segments (never the full film), and edits the
clips into a 1080×1920 (9:16) Short with animated Hindi captions.

## Architecture

| # | Module | File | Responsibility |
|---|--------|------|----------------|
| 1 | Movie search | `src/modules/youtube_search.py` | YouTube Data API v3, `videoDuration=long`, Film & Animation category, ranked by views |
| 2 | Transcript + LLM | `src/modules/transcript_llm.py` | timestamped transcript via `youtube-transcript-api`; Gemini (or OpenAI) → Hindi narration + scene timestamps |
| 3 | TTS | `src/modules/tts_generator.py` | async `edge-tts` (hi-IN-SwaraNeural), auto rate-fit to ~120 s via ffprobe |
| 4 | Targeted download | `src/modules/downloader.py` | `yt-dlp --download-sections` (external ffmpeg) fetches only the chosen clips |
| 5 | Video editing | `src/modules/video_editor.py` | MoviePy: center-crop 16:9→9:16, montage loop to fill audio, captions burned with ImageMagick/PIL |
| 6 | Orchestration | `main.py` | CLI, error handling, `story_plan.json` + `final_short.mp4` output |
| 7 | Deployment | `Dockerfile` | slim multi-stage image: ffmpeg, ImageMagick (relaxed policy), Deva fonts, non-root |
| 8 | Compose | `docker-compose.yml` | env-file, output/downloads volumes, host network |

## Quick start

```bash
cp .env.example .env     # then fill in YOUTUBE_API_KEY and GEMINI_API_KEY
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# Plan only (no TTS / download / edit):
.venv/bin/python main.py --query "hindi action movie full movie" --dry-run

# Full pipeline:
.venv/bin/python main.py --query "hindi action movie full movie"

# Or target a specific video directly:
.venv/bin/python main.py --video-id dQw4w9WgXcQ
```

Output: `output/story_plan.json`, `output/narration.mp3`, `output/final_short.mp4`.
Downloaded clips live in `downloads/` and are removed unless `KEEP_CLIPS=true`.

## Docker

```bash
docker compose build
docker compose run --rm agent --query "sci-fi movie full movie"
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

Pure-logic helpers are unit-tested; network/LLM/TTS/FFmpeg paths are not
exercised by the test suite.

## Configuration

See `.env.example` for every setting. Key ones:

| Variable | Default | Meaning |
|----------|---------|---------|
| `YOUTUBE_API_KEY` | — | YouTube Data API v3 key (module 1) |
| `LLM_PROVIDER` | `gemini` | `gemini`, `openai`, `deepseek`, `opencode` or `ollama` (Ollama Cloud: `OLLAMA_API_KEY`, `OLLAMA_BASE_URL=https://ollama.com/v1`, `OLLAMA_MODEL`) |
| `GEMINI_API_KEY` | — | Gemini API key (module 2) |
| `TTS_VOICE` | `hi-IN-SwaraNeural` | edge-tts Hindi voice |
| `TARGET_DURATION_SECONDS` | `120` | target Short length; TTS auto-fits |
| `MAX_SCENES` | `6` | max LLM-chosen clips |
| `MIN/MAX_SCENE_SECONDS` | `5`/`12` | allowed clip lengths |
| `HINDI_FONT` | Noto Sans Devanagari | caption font path |
| `KEEP_CLIPS` | `false` | keep downloaded clip files |

## Notes / limitations

- Requires a *transcript* (manual or auto-generated, English preferred) — speech
  is matched to the movie timeline.
- Scene timings are clamped to `[MIN_SCENE_SECONDS, MAX_SCENE_SECONDS]` and to
  the actual video duration.
- Captions use ImageMagick `TextClip`; if the policy/font is unavailable the
  editor transparently falls back to Pillow-rendered frames.