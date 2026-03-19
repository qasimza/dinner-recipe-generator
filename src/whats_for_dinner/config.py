from functools import lru_cache
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env file."""

    openai_api_key: str
    database_url: str
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o"
    top_k: int = 3
    rag_cache_similarity_threshold: float = 0.75
    generated_recipes_dir: str = "data/generated_recipes"

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
