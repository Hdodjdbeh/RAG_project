"""
Pydantic схемы для валидации данных и сериализации.
Определяет структуру запросов и ответов API.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class RatingEnum(int, Enum):
    """Оценки отзыва."""
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


class ReviewInput(BaseModel):
    """
    Входные данные для генерации ответа.
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Текст отзыва на Wildberries"
    )
    rating: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Оценка отзыва (1-5 звезд)"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Очистка и валидация текста отзыва."""
        v = v.strip()
        if not v:
            raise ValueError("Review text cannot be empty")
        if len(v) < 3:
            raise ValueError("Review text is too short")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Товар пришел с браком, прошу вернуть деньги",
                "rating": 1
            }
        }


class UsedRuleInfo(BaseModel):
    """Информация об использованном правиле."""
    id: str = Field(..., description="ID правила из базы знаний")
    similarity: float = Field(..., ge=0, le=1, description="Степень схожести отзыва с правилом")
    category: Optional[str] = Field(None, description="Категория правила")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "rule_001",
                "similarity": 0.85,
                "category": "complaint"
            }
        }


class ReplyResponse(BaseModel):
    """
    Ответ генератора с готовым текстом и метаданными.
    """
    reply_text: str = Field(..., description="Сгенерированный ответ на отзыв")
    used_rules: List[UsedRuleInfo] = Field(
        default_factory=list,
        description="Правила, использованные для генерации"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Уверенность модели в качестве ответа"
    )
    tone: str = Field(
        ...,
        description="Тон ответа (positive, neutral, negative, apology, gratitude)"
    )
    is_fallback: bool = Field(
        False,
        description="Является ли ответ fallback (при ошибке или отсутствии правил)"
    )
    generation_time_ms: Optional[float] = Field(
        None,
        description="Время генерации в миллисекундах"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "reply_text": "Здравствуйте! Приносим извинения за бракованный товар. Пожалуйста, оформите возврат в личном кабинете, и мы вернем деньги в течение 3-5 дней.",
                "used_rules": [
                    {
                        "id": "rule_001",
                        "similarity": 0.85,
                        "category": "complaint"
                    }
                ],
                "confidence": 0.82,
                "tone": "apology",
                "is_fallback": False,
                "generation_time_ms": 1250.5
            }
        }


class HealthResponse(BaseModel):
    """Ответ на healthcheck запрос."""
    status: str = Field(..., description="Статус сервиса")
    version: str = Field("1.0.0", description="Версия API")
    timestamp: datetime = Field(..., description="Время проверки")
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Статус компонентов системы"
    )


class ErrorResponse(BaseModel):
    """Стандартный ответ об ошибке."""
    error: str = Field(..., description="Краткое описание ошибки")
    detail: Optional[str] = Field(None, description="Детали ошибки")
    status_code: int = Field(..., description="HTTP статус код")
    timestamp: datetime = Field(..., description="Время возникновения ошибки")


class BatchReviewInput(BaseModel):
    """Пакетный режим для нескольких отзывов."""
    reviews: List[ReviewInput] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Список отзывов для генерации"
    )


class BatchReplyResponse(BaseModel):
    """Ответ для пакетного режима."""
    results: List[ReplyResponse] = Field(..., description="Сгенерированные ответы")
    total_time_ms: float = Field(..., description="Общее время обработки")
    successful_count: int = Field(..., description="Количество успешных генераций")
    failed_count: int = Field(..., description="Количество неудачных генераций")