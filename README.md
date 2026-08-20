# movie-shorts-agent

An autonomous, end-to-end Python agent that creates 2-minute Hindi YouTube
Shorts which summarise full-length movies found on YouTube.

The agent searches for a full movie, fetches its transcript, has an LLM write a
fast-paced Hindi narration plus the most striking 5–12s scenes, synthesises the
voiceover, downloads *only* those segments (never the full film), and edits the
clips into a 1080×1920 (9:16) Short with animated Hindi captions. Every output
opens with a 3-second "hook" — the movie's single most-viewed, most eye-catching
moment — and the pipeline also renders a 1920×1080 (16:9) ~5-minute long video
(`final_long.mp4`) that reuses the same clips, looped, with a longer narration.

## Architecture

| # | Module | File | Responsibility |
|---|--------|------|----------------|
| 1 | Movie search | `src/modules/youtube_search.py` | YouTube Data API v3, English thriller/horror/romantic/survival genre queries, ranked by views, language-filtered |
| 2 | Transcript + LLM | `src/modules/transcript_llm.py` | timestamped transcript via `youtube-transcript-api`; Gemini/OpenAI/DeepSeek/OpenCode/Ollama → Hindi narration + scene timestamps |
| 3 | TTS | `src/modules/tts_generator.py` | async `edge-tts` (hi-IN-SwaraNeural), auto rate-fit to ~120 s via ffprobe |
| 4 | Targeted download | `src/modules/downloader.py` | `yt-dlp --download-sections` (external ffmpeg) fetches only the narration-relevant clips |
| 5 | Video editing | `src/modules/video_editor.py` | MoviePy: focus-aware 16:9→9:16 crop (keeps the in-focus subject centred) for the Short, full-frame fit for the 16:9 long video, each scene plays exactly once (long video loops to fill), 3s most-viewed hook prepended, captions burned with ImageMagick/PIL |
| 6 | Orchestration | `main.py` | CLI, used-movie tracking, 3-hour scheduler, `story_plan.json` + `final_short.mp4` + `final_long.mp4` |
| 7 | Deployment | `Dockerfile` | slim multi-stage image: ffmpeg, OpenCV, ImageMagick (relaxed policy), Deva fonts, non-root |
| 8 | Compose | `docker-compose.yml` | env-file, output/downloads volumes, host network, `--schedule` restart |

## Quick start

```bash
cp .env.example .env     # then fill in YOUTUBE_API_KEY and an LLM key
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# Plan only (no TTS / download / edit):
.venv/bin/python main.py --dry-run

# Full pipeline (searches English thriller/romantic/horror, new movie each run):
.venv/bin/python main.py

# Schedule forever, every 3 hours:
.venv/bin/python main.py --schedule

# Start over with a fresh set of movies:
.venv/bin/python main.py --reset-used

# Or target a specific video directly:
.venv/bin/python main.py --video-id dQw4w9WgXcQ
```

Output: `output/story_plan.json`, `output/narration.mp3`, `output/final_short.mp4`,
and `output/final_long.mp4` (16:9 ~5-minute version, when `MAKE_LONG_VIDEO=true`).
Downloaded clips live in `downloads/` and are removed unless `KEEP_CLIPS=true`.
Movies already used are tracked in `output/used_movies.json` so every run picks
a new one. Completed runs are also logged to `output/movie_history.json` and
reconstructed from `output/backups/*/story_plan.json`, so `--reset-used` only
clears the live registry — finished movies never repeat.

## Docker

```bash
docker compose build
docker compose up -d        # starts the 3-hour scheduler
docker compose run --rm agent --dry-run   # one-off plan-only run
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
| `GENRES` | `thriller,horror,romantic,survival` | genres searched (one query each) |
| `SEARCH_LANGUAGE` | `en` | preferred audio language filter |
| `TTS_VOICE` | `hi-IN-SwaraNeural` | edge-tts Hindi voice |
| `TARGET_DURATION_SECONDS` | `120` | target Short length; narration fits the distinct clip timeline, so clips are never repeated |
| `MAX_SCENES` | `8` | max LLM-chosen clips |
| `MIN/MAX_SCENE_SECONDS` | `10`/`18` | allowed clip lengths |
| `USED_MOVIES_PATH` | `output/used_movies.json` | live registry of movies already used (cleared by `--reset-used`) |
| `MOVIE_HISTORY_PATH` | `output/movie_history.json` | append-only log of completed runs; never cleared, so finished movies don't repeat |
| `SEARCH_MAX_RESULTS` | `20` | candidates fetched per genre query so the pool doesn't run dry |
| `SCHEDULE_INTERVAL_HOURS` | `3` | delay between scheduled runs |
| `HINDI_FONT` | Lohit-Devanagari | caption font path |
| `KEEP_CLIPS` | `false` | keep downloaded clip files |
| `MIN_CLIP_MOTION` | `2.5` | drop clips below this motion score (still photos / title cards) |
| `HOOK_SECONDS` | `3` | length of the most-viewed moment prepended to every output |
| `MAKE_LONG_VIDEO` | `true` | also render a 16:9 ~5-minute `final_long.mp4` |
| `LONG_DURATION_SECONDS` | `300` | target long-video length |
| `LONG_SCRIPT_WORDS` | `700` | narration word count for the long video |
| `LONG_OUTPUT_WIDTH/HEIGHT` | `1920`/`1080` | long-video resolution |
| `BACKUP_DIR` | `output/backups` | timestamped folder per run holding the output artifacts |
| `BACKUP_RETENTION_HOURS` | `48` | delete backup folders older than this many hours |

## Notes / limitations

- Requires a *transcript* (manual or auto-generated) — speech is matched to the
  movie timeline; candidates are walked in view-count order until one works.
- Search is restricted to English audio (`SEARCH_LANGUAGE=en`) and the genres
  in `GENRES` (thriller, horror, romantic, survival); candidates without a
  declared language are kept as fallback.
- Scene timings are clamped to `[MIN_SCENE_SECONDS, MAX_SCENE_SECONDS]` and to
  the actual video duration, and are tied to the narration beats.
- The 16:9→9:16 crop centers on the in-focus subject (Sobel sharpness map via
  OpenCV) instead of always the frame centre.
- Narrations tell the full story including the climax/ending; the LLM also
  picks at least one scene from the climax and avoids static title cards.
- Every scene plays exactly once: the narration is fitted to the total length
  of the downloaded clips, so no clip is ever repeated in the final Short.
- The 16:9 long video (`final_long.mp4`) reuses the Short's clips and loops
  them to cover its longer narration, so it needs no extra downloads. Its
  narration is a separate, longer LLM script (`LONG_SCRIPT_WORDS` words).
- Every output opens with a 3-second hook — the movie's single most-viewed,
  most eye-catching beat, chosen by the LLM and clamped to `HOOK_SECONDS`.
- YouTube does not expose per-timestamp view counts, so the "most-viewed
  portion" is approximated by having the LLM pick the highest-impact, most
  viral beat (usually the peak of the climax) from the whole film.
- Captions use ImageMagick `TextClip`; if the policy/font is unavailable the
  editor transparently falls back to Pillow-rendered frames.