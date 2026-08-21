"""Central settings for the Movie Shorts agent.

Environment variables (see ``.env.example``) are loaded via pydantic-settings.
Use ``get_settings()`` to obtain a cached, validated settings object.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration, populated from environment / ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API credentials -------------------------------------------------
    youtube_api_key: str = Field(default="", description="YouTube Data API v3 key")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    deepseek_api_key: str = Field(default="", description="DeepSeek API key")
    deepseek_base_url: str = Field(default="https://api.deepseek.com")
    opencode_api_key: str = Field(default="", description="OpenCode Zen API key (optional on free tier)")
    opencode_base_url: str = Field(default="https://opencode.ai/zen/v1")
    opencode_model: str = Field(default="big-pickle")
    ollama_api_key: str = Field(default="", description="Ollama Cloud API key")
    ollama_base_url: str = Field(default="https://ollama.com/v1")
    ollama_model: str = Field(default="deepseek-v4-flash:0731")

    # --- LLM provider (Module 2) -----------------------------------------
    llm_provider: str = Field(
        default="gemini",
        description="'gemini', 'openai', 'deepseek', 'opencode' or 'ollama'",
    )
    gemini_model: str = Field(default="gemini-2.0-flash")
    openai_model: str = Field(default="gpt-4o-mini")
    deepseek_model: str = Field(default="deepseek-chat")

    # --- TTS (Module 3) ---------------------------------------------------
    tts_voice: str = Field(default="hi-IN-SwaraNeural")
    tts_rate: str = Field(default="+0%", description="initial edge-tts rate")

    # --- Story plan (Module 2) --------------------------------------------
    target_script_words: int = Field(default=360, ge=50, le=2000)
    max_scenes: int = Field(default=30, ge=1, le=40)
    min_scene_seconds: float = Field(default=5.0, ge=1.0)
    max_scene_seconds: float = Field(default=5.0, ge=1.0)
    max_transcript_chars: int = Field(default=250_000, description="truncate transcript for the LLM")

    # --- Search (Module 1) ------------------------------------------------
    movie_genres: str = Field(
        default="romantic drama,romance,steamy romance,erotic thriller",
        description="comma-separated genres to search (one query each)",
    )
    search_language: str = Field(
        default="en",
        description="preferred audio language; only candidates whose language matches are kept",
    )

    # --- Used-movie tracking (Module 6) -----------------------------------
    used_movies_path: Path = Field(
        default=Path("output/used_movies.json"),
        description="JSON list of video_ids already turned into Shorts",
    )
    movie_history_path: Path = Field(
        default=Path("output/movie_history.json"),
        description="append-only log of video_ids that completed a run; "
        "never cleared by --reset-used so finished movies never repeat",
    )
    search_max_results: int = Field(
        default=20,
        ge=1,
        description="candidates to fetch per genre query, so the pool "
        "does not run dry after filtering already-used movies",
    )

    # --- Output backup (retain each run's files for a while) --------------
    backup_dir: Path = Field(
        default=Path("output/backups"),
        description="timestamped subfolder per run holding the output artifacts",
    )
    backup_retention_hours: float = Field(
        default=48.0, ge=0.0,
        description="delete backup folders older than this many hours",
    )

    # --- Schedule (Module 6) ----------------------------------------------
    schedule_interval_hours: float = Field(default=3.0, ge=0.1)

    # --- Download / edit (Modules 4-5) ------------------------------------
    target_duration_seconds: int = Field(default=150, ge=10)
    output_dir: Path = Field(default=Path("output"))
    downloads_dir: Path = Field(default=Path("downloads"))
    keep_clips: bool = Field(default=False, description="keep downloaded clip files")
    max_download_retries: int = Field(default=5, ge=0)
    min_clip_motion: float = Field(
        default=2.5,
        ge=0.0,
        description="min mean frame-difference (0..255) for a clip to be kept; "
        "clips below this are still photos/title cards and are dropped",
    )

    # --- Hook + long-form video (Module 5) --------------------------------
    hook_seconds: float = Field(
        default=3.0, ge=0.5, le=15.0,
        description="length of the 'most viewed' moment prepended to every output",
    )
    make_long_video: bool = Field(
        default=True,
        description="also render a 16:9 ~5-minute long video (final_long.mp4)",
    )
    long_duration_seconds: int = Field(default=300, ge=60)
    long_script_words: int = Field(default=700, ge=100, le=3000)
    long_output_width: int = Field(default=1920)
    long_output_height: int = Field(default=1080)

    # --- Captions (Module 5) ----------------------------------------------
    hindi_font: str = Field(
        default="/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
        description="ImageMagick/PIL font for Devanagari captions",
    )
    caption_font_size: int = Field(default=52)
    caption_stroke: int = Field(default=3)
    output_width: int = Field(default=1080)
    output_height: int = Field(default=1920)
    output_fps: int = Field(default=30)

    # --- Tools ------------------------------------------------------------
    ffmpeg_binary: str = Field(default="ffmpeg")
    ffprobe_binary: str = Field(default="ffprobe")

    # --- Logging ----------------------------------------------------------
    log_level: str = Field(default="INFO")

    @property
    def llm_api_key(self) -> str:
        if self.llm_provider == "openai":
            return self.openai_api_key
        if self.llm_provider == "deepseek":
            return self.deepseek_api_key
        if self.llm_provider == "opencode":
            return self.opencode_api_key
        if self.llm_provider == "ollama":
            return self.ollama_api_key
        return self.gemini_api_key

    @property
    def llm_model(self) -> str:
        if self.llm_provider == "openai":
            return self.openai_model
        if self.llm_provider == "deepseek":
            return self.deepseek_model
        if self.llm_provider == "opencode":
            return self.opencode_model
        if self.llm_provider == "ollama":
            return self.ollama_model
        return self.gemini_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()