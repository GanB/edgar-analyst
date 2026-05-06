"""Application settings loaded from environment variables.

All configuration is read once at process start. Production secrets
(Anthropic, Voyage, LangSmith API keys) are optional at boot so that
the spine can run without them; downstream components that need a
key fail loudly when they reach for it.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model providers
    anthropic_api_key: str | None = None
    voyage_api_key: str | None = None

    # Postgres / pgvector
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "edgar_analyst"
    postgres_user: str = "edgar_analyst"
    postgres_password: str = "change_me"

    # LangSmith tracing (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "edgar-analyst"

    # SEC EDGAR fair-use header. Required by SEC; default is the
    # project owner's contact and is sufficient for local dev.
    sec_edgar_user_agent: str = Field(
        default="Ganesh Babu ganesh@ganeshbabu.dev",
        description="User-Agent header value for SEC EDGAR requests.",
    )

    # Runtime
    app_env: str = "development"
    log_level: str = "info"

    @property
    def postgres_dsn(self) -> str:
        """SQLAlchemy-style DSN (psycopg dialect) for langchain-postgres."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_dsn_psycopg(self) -> str:
        """Plain psycopg DSN, no dialect prefix."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


def get_settings() -> Settings:
    return Settings()
