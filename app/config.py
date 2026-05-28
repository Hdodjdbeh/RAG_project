"""
Конфигурация приложения с Pydantic Settings.
Управляет переменными окружения и настройками сервиса.
"""
import os
from pathlib import Path
from typing import Optional

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения."""

    # YandexGPT API
    yandex_api_key: str = Field(..., env="YANDEX_API_KEY")
    yandex_folder_id: str = Field(..., env="YANDEX_FOLDER_ID")
    yandex_api_url: str = Field(
        "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        env="YANDEX_API_URL"
    )
    yandex_model_uri: str = Field(
        "gpt://<folder_id>/yandexgpt-lite/latest",
        env="YANDEX_MODEL_URI"
    )
    yandex_max_tokens: int = Field(500, env="YANDEX_MAX_TOKENS")
    yandex_temperature: float = Field(0.7, env="YANDEX_TEMPERATURE")

    # Пути к данным
    chroma_path: Path = Field(Path("./chroma_db"), env="CHROMA_PATH")
    rules_path: Path = Field(
        Path("data/knowledge_base/rules.json"),
        env="RULES_PATH"
    )

    # Модель эмбеддингов
    embedding_model_name: str = Field(
        "cointegrated/rubert-tiny2",
        env="EMBEDDING_MODEL_NAME"
    )

    # Настройки RAG
    rag_top_k: int = Field(3, env="RAG_TOP_K")
    rag_similarity_threshold: float = Field(0.5, env="RAG_SIMILARITY_THRESHOLD")

    # Настройки сервера
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    cors_origins: list[str] = Field(["*"], env="CORS_ORIGINS")

    # Логирование
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Корневая директория проекта
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @field_validator("yandex_model_uri", mode="before")
    @classmethod
    def validate_model_uri(cls, v: str, info) -> str:
        """Подставляет folder_id в URI модели."""
        if "<folder_id>" in v and info.data.get("yandex_folder_id"):
            return v.replace("<folder_id>", info.data["yandex_folder_id"])
        return v

    @field_validator("chroma_path", mode="after")
    @classmethod
    def resolve_chroma_path(cls, v: Path, info) -> Path:
        """Разрешает путь к chroma_db относительно корня проекта."""
        if not v.is_absolute():
            project_root = info.data.get("project_root", Path.cwd())
            resolved = project_root / v
            return resolved
        return v

    @field_validator("rules_path", mode="after")
    @classmethod
    def resolve_rules_path(cls, v: Path, info) -> Path:
        """Разрешает путь к rules.json относительно корня проекта."""
        if not v.is_absolute():
            project_root = info.data.get("project_root", Path.cwd())
            resolved = project_root / v
            return resolved
        return v

    @field_validator("rules_path", mode="after")
    @classmethod
    def validate_rules_path(cls, v: Path) -> Path:
        """Проверяет существование файла правил."""
        if not v.exists():
            # Пробуем найти альтернативные пути
            alternatives = [
                Path("rules.json"),
                Path("../rules.json"),
                Path("data/knowledge_base/rules.json"),
                Path("../data/knowledge_base/rules.json"),
            ]

            for alt in alternatives:
                if alt.exists():
                    return alt

            raise FileNotFoundError(
                f"Rules file not found: {v}\n"
                f"Please ensure rules.json exists in data/knowledge_base/ or in project root"
            )
        return v

    def model_post_init(self, __context):
        """Пост-инициализация: создает директории если нужно."""
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        # Логируем пути для отладки
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Rules path: {self.rules_path}")
        logger.info(f"Chroma path: {self.chroma_path}")


# Глобальный экземпляр настроек
settings = Settings()