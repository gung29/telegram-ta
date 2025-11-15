from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised configuration loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )

    bot_token: str = Field(..., alias="BOT_TOKEN")
    api_key: str = Field(..., alias="API_KEY")
    secret_key: str = Field("supersecret", alias="SECRET_KEY")

    inference_api_host: str = Field("0.0.0.0", alias="INFERENCE_API_HOST")
    inference_api_port: int = Field(8000, alias="INFERENCE_API_PORT")
    inference_api_url: str = Field("http://localhost:8000", alias="INFERENCE_API_URL")

    webhook_url: str = Field("", alias="WEBHOOK_URL")
    webhook_host: str = Field("0.0.0.0", alias="WEBHOOK_HOST")
    webhook_port: int = Field(8081, alias="WEBHOOK_PORT")
    webhook_path: str = Field("/webhook", alias="WEBHOOK_PATH")
    webhook_secret: str = Field("hate-guard-secret", alias="WEBHOOK_SECRET")

    database_url: str = Field("sqlite:///./data/db.sqlite3", alias="DATABASE_URL")
    retention_days: int = Field(30, alias="RETENTION_DAYS")
    default_threshold: float = Field(0.62, alias="DEFAULT_THRESHOLD")

    model_path: str = Field("./data/model/model.onnx", alias="MODEL_PATH")
    tokenizer_path: str = Field("./data/model/tokenizer.json", alias="TOKENIZER_PATH")

    mini_app_base_url: str = Field("https://mini.gungzy.xyz", alias="MINI_APP_BASE_URL")
    mini_app_dev_mode: bool = Field(False, alias="MINI_APP_DEV_MODE")
    admin_ids_raw: str | int | List[int] | None = Field(None, alias="ADMIN_IDS")
    allowed_origins_raw: str | List[str] | None = Field(None, alias="ALLOWED_ORIGINS")

    telemetry_enabled: bool = Field(True, alias="TELEMETRY_ENABLED")

    @staticmethod
    def _parse_ints(value: str | Sequence[int] | int | None) -> List[int]:
        if value in (None, "", []):
            return []
        if isinstance(value, int):
            return [value]
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        return [int(v.strip()) for v in str(value).split(",") if v.strip()]

    @staticmethod
    def _parse_strs(value: str | Sequence[str] | None) -> List[str]:
        if value in (None, "", []):
            return []
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        return [item.strip() for item in str(value).split(",") if item.strip()]

    @computed_field
    def admin_ids(self) -> List[int]:
        return self._parse_ints(self.admin_ids_raw)  # type: ignore[arg-type]

    @computed_field
    def allowed_origins(self) -> List[str]:
        return self._parse_strs(self.allowed_origins_raw)  # type: ignore[arg-type]

    @computed_field
    def cors_origins(self) -> List[str]:
        """Expose CORS origins while ensuring uniqueness."""
        origins = list(dict.fromkeys(self.allowed_origins))
        if self.mini_app_base_url and self.mini_app_base_url not in origins:
            origins.append(self.mini_app_base_url)
        return origins


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[arg-type]


settings = get_settings()
