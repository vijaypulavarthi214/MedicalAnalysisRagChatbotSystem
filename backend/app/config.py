from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-v3.5"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chroma_persist_directory: str = "./data/chroma"

    hybrid_alpha: float = 0.7

    max_file_size_mb: int = 20

    cors_origins: str = "*"

    debug_rag: bool = False

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip().rstrip("/") for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
