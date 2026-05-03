"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- API keys ---
    openai_api_key: str = ""

    # --- Database ---
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/visionrag"

    # --- Models ---
    embedding_model: str = "text-embedding-3-large"
    vision_model: str = "gpt-4o"
    llm_model: str = "gpt-4o-mini"
    summary_model: str = "gpt-4o-mini"

    # --- Paths ---
    vision_cache_dir: str = ".vision_cache"
    upload_dir: str = "uploads"

    # --- Retrieval ---
    default_top_k: int = 10
    rrf_k: int = 60

    # --- Concurrency ---
    vision_concurrency: int = 5

    # --- Enrichment ---
    enable_concept_split: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
