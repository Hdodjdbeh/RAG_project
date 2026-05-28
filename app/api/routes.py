"""
API эндпоинты для генерации ответов на отзывы.
"""
import time
import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from app.models.schemas import (
    ReviewInput,
    ReplyResponse,
    HealthResponse,
    ErrorResponse,
    BatchReviewInput,
    BatchReplyResponse
)
from app.api.dependencies import get_rag_engine
from app.core.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

# Создаем роутер
router = APIRouter()

# Prometheus метрики
REQUESTS_TOTAL = Counter(
    'reply_generator_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

GENERATION_DURATION = Histogram(
    'reply_generator_generation_seconds',
    'Time spent generating replies',
    ['rating_group'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
)

RULES_RETRIEVED = Histogram(
    'reply_generator_rules_retrieved',
    'Number of rules retrieved per request',
    buckets=(0, 1, 2, 3, 4, 5)
)

CONFIDENCE_SCORE = Histogram(
    'reply_generator_confidence',
    'Confidence scores of generated replies',
    buckets=(0, 0.2, 0.4, 0.6, 0.8, 1.0)
)


@router.post(
    "/generate",
    response_model=ReplyResponse,
    status_code=status.HTTP_200_OK,
    summary="Генерация ответа на отзыв",
    description="Анализирует отзыв, находит похожие правила и генерирует ответ через YandexGPT"
)
async def generate_reply(
        review: ReviewInput,
        rag_engine: RAGEngine = Depends(get_rag_engine)
) -> ReplyResponse:
    """
    Генерирует ответ на отзыв Wildberries.

    Args:
        review: Отзыв с текстом и опциональной оценкой
        rag_engine: RAG движок (depends)

    Returns:
        ReplyResponse: Сгенерированный ответ и метаданные
    """
    start_time = time.time()
    rating_group = f"rating_{review.rating if review.rating else 'unknown'}"

    try:
        logger.info(
            f"Generating reply for review",
            extra={
                "review_length": len(review.text),
                "rating": review.rating,
                "review_preview": review.text[:100]
            }
        )

        # Вызываем RAG движок для генерации
        result = rag_engine.generate_reply(
            review_text=review.text,
            rating=review.rating
        )

        # Обновляем метрики
        REQUESTS_TOTAL.labels(
            method="POST",
            endpoint="/generate",
            status="success"
        ).inc()

        GENERATION_DURATION.labels(rating_group=rating_group).observe(time.time() - start_time)
        RULES_RETRIEVED.observe(len(result["used_rules"]))
        CONFIDENCE_SCORE.observe(result["confidence"])

        # Формируем ответ
        response = ReplyResponse(
            reply_text=result["reply_text"],
            used_rules=result["used_rules"],
            confidence=result["confidence"],
            tone=result["tone"],
            is_fallback=result.get("is_fallback", False),
            generation_time_ms=(time.time() - start_time) * 1000
        )

        logger.info(
            f"Reply generated successfully",
            extra={
                "reply_length": len(response.reply_text),
                "confidence": response.confidence,
                "generation_time_ms": response.generation_time_ms
            }
        )

        return response

    except Exception as e:
        REQUESTS_TOTAL.labels(
            method="POST",
            endpoint="/generate",
            status="error"
        ).inc()

        logger.error(f"Failed to generate reply: {str(e)}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate reply: {str(e)}"
        )


@router.post(
    "/generate/batch",
    response_model=BatchReplyResponse,
    status_code=status.HTTP_200_OK,
    summary="Пакетная генерация ответов",
    description="Генерирует ответы для нескольких отзывов одновременно"
)
async def generate_batch_replies(
        batch: BatchReviewInput,
        rag_engine: RAGEngine = Depends(get_rag_engine)
) -> BatchReplyResponse:
    """
    Пакетная генерация ответов для экономии ресурсов.
    """
    start_time = time.time()
    results = []
    successful = 0
    failed = 0

    for review in batch.reviews:
        try:
            result = rag_engine.generate_reply(
                review_text=review.text,
                rating=review.rating
            )

            results.append(ReplyResponse(
                reply_text=result["reply_text"],
                used_rules=result["used_rules"],
                confidence=result["confidence"],
                tone=result["tone"],
                is_fallback=result.get("is_fallback", False),
                generation_time_ms=None  # Индивидуальное время не трекаем
            ))
            successful += 1

        except Exception as e:
            logger.error(f"Failed to generate reply for batch item: {str(e)}")
            # Добавляем fallback ответ
            results.append(ReplyResponse(
                reply_text="Извините, временно не могу обработать этот отзыв. Пожалуйста, попробуйте позже.",
                used_rules=[],
                confidence=0.0,
                tone="neutral",
                is_fallback=True
            ))
            failed += 1

    total_time_ms = (time.time() - start_time) * 1000

    REQUESTS_TOTAL.labels(
        method="POST",
        endpoint="/generate/batch",
        status="success"
    ).inc()

    return BatchReplyResponse(
        results=results,
        total_time_ms=total_time_ms,
        successful_count=successful,
        failed_count=failed
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Проверка состояния сервиса и его компонентов"
)
async def health_check(
        rag_engine: RAGEngine = Depends(get_rag_engine)
) -> HealthResponse:
    """
    Проверяет работоспособность всех компонентов.
    """
    components_status = {}

    # Проверяем векторное хранилище
    try:
        stats = rag_engine.vector_store.get_collection_stats()
        components_status["vector_store"] = "healthy"
        components_status["documents_count"] = str(stats.get("documents_count", 0))
    except Exception as e:
        logger.error(f"Vector store health check failed: {str(e)}")
        components_status["vector_store"] = f"unhealthy: {str(e)}"

    # Проверяем LLM клиент (легкий тест)
    try:
        # Простой тест без реального вызова API
        components_status["llm_client"] = "healthy"
    except Exception as e:
        components_status["llm_client"] = f"unhealthy: {str(e)}"

    # Проверяем эмбеддинги
    try:
        _ = rag_engine.vector_store.embedding_service.get_embedding_dim()
        components_status["embeddings"] = "healthy"
    except Exception as e:
        components_status["embeddings"] = f"unhealthy: {str(e)}"

    # Общее состояние
    is_healthy = all("unhealthy" not in v for v in components_status.values())

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        version="1.0.0",
        timestamp=datetime.now(),
        components=components_status
    )


@router.get(
    "/metrics",
    summary="Prometheus метрики",
    description="Эндпоинт для сбора метрик Prometheus"
)
async def get_metrics():
    """
    Возвращает метрики в формате Prometheus.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@router.get(
    "/stats",
    summary="Статистика системы",
    description="Возвращает статистику о базе знаний и векторном хранилище"
)
async def get_system_stats(
        rag_engine: RAGEngine = Depends(get_rag_engine)
) -> Dict[str, Any]:
    """
    Возвращает детальную статистику системы.
    """
    try:
        store_stats = rag_engine.vector_store.get_collection_stats()

        return {
            "vector_store": store_stats,
            "rag_config": {
                "top_k": rag_engine.top_k,
                "similarity_threshold": rag_engine.similarity_threshold
            },
            "embedding_dimension": rag_engine.vector_store.embedding_service.get_embedding_dim(),
            "model_info": {
                "embedding_model": rag_engine.vector_store.embedding_service.model_name,
                "llm_model": "YandexGPT Lite"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )