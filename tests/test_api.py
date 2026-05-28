"""
API тесты для проверки эндпоинтов.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from app.main import app
from app.core.rag_engine import RAGEngine


@pytest.fixture
def client():
    """Создает тестовый клиент FastAPI."""
    return TestClient(app)


@pytest.fixture
def mock_rag_engine():
    """Мок RAG движка для изолированного тестирования."""
    mock = Mock(spec=RAGEngine)

    # Мокаем успешный ответ
    mock.generate_reply.return_value = {
        "reply_text": "Спасибо за ваш отзыв! Мы учтем ваше замечание.",
        "used_rules": [
            {
                "id": "rule_001",
                "similarity": 0.85,
                "category": "complaint"
            }
        ],
        "confidence": 0.82,
        "tone": "neutral",
        "is_fallback": False
    }

    return mock


def test_health_endpoint(client):
    """Тест healthcheck эндпоинта."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["version"] == "1.0.0"
    assert "timestamp" in data
    assert "components" in data


def test_root_endpoint(client):
    """Тест корневого эндпоинта."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["service"] == "Reply Generator for Wildberries"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"


def test_generate_endpoint_success(client, mock_rag_engine):
    """Тест успешной генерации ответа."""
    with patch("app.api.routes.get_rag_engine", return_value=mock_rag_engine):
        response = client.post(
            "/api/v1/generate",
            json={
                "text": "Товар пришел с браком",
                "rating": 1
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "reply_text" in data
        assert "used_rules" in data
        assert "confidence" in data
        assert "tone" in data
        assert "generation_time_ms" in data

        assert data["confidence"] == 0.82
        assert len(data["used_rules"]) == 1


def test_generate_endpoint_invalid_rating(client):
    """Тест с невалидным рейтингом."""
    response = client.post(
        "/api/v1/generate",
        json={
            "text": "Отличный товар",
            "rating": 6  # Невалидный рейтинг
        }
    )

    assert response.status_code == 422  # Validation error


def test_generate_endpoint_empty_text(client):
    """Тест с пустым текстом отзыва."""
    response = client.post(
        "/api/v1/generate",
        json={
            "text": "",
            "rating": 5
        }
    )

    assert response.status_code == 422


def test_generate_endpoint_too_short_text(client):
    """Тест со слишком коротким текстом."""
    response = client.post(
        "/api/v1/generate",
        json={
            "text": "Ok",
            "rating": 4
        }
    )

    assert response.status_code == 422


def test_generate_endpoint_without_rating(client, mock_rag_engine):
    """Тест без указания рейтинга."""
    with patch("app.api.routes.get_rag_engine", return_value=mock_rag_engine):
        response = client.post(
            "/api/v1/generate",
            json={
                "text": "Хороший магазин, быстрая доставка"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["reply_text"] is not None


def test_batch_generate_endpoint(client, mock_rag_engine):
    """Тест пакетной генерации."""
    with patch("app.api.routes.get_rag_engine", return_value=mock_rag_engine):
        response = client.post(
            "/api/v1/generate/batch",
            json={
                "reviews": [
                    {"text": "Отличный товар", "rating": 5},
                    {"text": "Плохое качество", "rating": 1},
                    {"text": "Нормально", "rating": 3}
                ]
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "total_time_ms" in data
        assert "successful_count" in data
        assert "failed_count" in data

        assert len(data["results"]) == 3
        assert data["successful_count"] == 3


def test_metrics_endpoint(client):
    """Тест эндпоинта метрик."""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"

    # Проверяем наличие ключевых метрик
    content = response.text
    assert "reply_generator_requests_total" in content
    assert "reply_generator_generation_seconds" in content


def test_stats_endpoint(client, mock_rag_engine):
    """Тест эндпоинта статистики."""
    # Мокаем метод получения статистики
    mock_rag_engine.vector_store.get_collection_stats.return_value = {
        "collection_name": "rules_kb",
        "documents_count": 30,
        "embedding_dimension": 312,
        "storage_path": "./chroma_db"
    }

    with patch("app.api.routes.get_rag_engine", return_value=mock_rag_engine):
        response = client.get("/api/v1/stats")

        assert response.status_code == 200
        data = response.json()

        assert "vector_store" in data
        assert "rag_config" in data
        assert "embedding_dimension" in data
        assert data["embedding_dimension"] == 312


def test_generate_endpoint_error_handling(client):
    """Тест обработки ошибок при генерации."""
    # Создаем мок с ошибкой
    mock_engine = Mock(spec=RAGEngine)
    mock_engine.generate_reply.side_effect = Exception("Test error")

    with patch("app.api.routes.get_rag_engine", return_value=mock_engine):
        response = client.post(
            "/api/v1/generate",
            json={
                "text": "Тестовый отзыв",
                "rating": 3
            }
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


def test_docs_endpoint(client):
    """Тест документации."""
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/redoc")
    assert response.status_code == 200

    response = client.get("/openapi.json")
    assert response.status_code == 200