"""
Тестирование качества RAG с использованием библиотеки RAGAS.
Оценивает faithfulness, answer_relevancy и другие метрики.
"""
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

import pytest
from datasets import Dataset

# Для RAGAS метрик
try:
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision
    )
    from ragas import evaluate

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("RAGAS not available. Install with: pip install ragas")

from app.core.rag_engine import RAGEngine
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_client import YandexGPTClient
from app.config import settings


@pytest.fixture(scope="module")
def rag_engine():
    """Создает реальный RAG движок для тестирования."""
    embedding_service = EmbeddingService()
    vector_store = VectorStore(embedding_service)
    llm_client = YandexGPTClient()
    engine = RAGEngine(vector_store, llm_client)
    return engine


@pytest.fixture
def sample_reviews():
    """Загружает sample отзывов для тестирования."""
    reviews_path = Path("data/processed/reviews_clean.json")
    if not reviews_path.exists():
        pytest.skip("Sample reviews not found")

    with open(reviews_path, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    # Берем 20 отзывов для тестирования (по 4 на каждый рейтинг)
    test_reviews = []
    for rating in range(1, 6):
        rating_reviews = [r for r in reviews if r.get("rating") == rating][:4]
        test_reviews.extend(rating_reviews)

    return test_reviews


@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="RAGAS not installed")
class TestRAGASMetrics:
    """Тестирование метрик RAGAS."""

    def test_faithfulness(self, rag_engine, sample_reviews):
        """
        Тест faithfulness: насколько ответ соответствует найденным правилам.
        """
        test_data = []

        for review in sample_reviews[:10]:  # Ограничиваем для скорости
            review_text = review["text"]
            rating = review.get("rating")

            # Получаем правила и ответ
            rules = rag_engine.retrieve_rules(review_text, rating)
            if not rules:
                continue

            result = rag_engine.generate_reply(review_text, rating, rules)

            # Формируем контекст из правил
            context = [rule["text"] for rule in rules[:2]]

            test_data.append({
                "question": f"Ответь на отзыв: {review_text[:200]}",
                "answer": result["reply_text"],
                "contexts": context
            })

        if not test_data:
            pytest.skip("No valid test data")

        # Создаем датасет
        dataset = Dataset.from_list(test_data)

        # Оцениваем
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy]
        )

        # Проверяем что метрики не нулевые
        assert result["faithfulness"] > 0.5, "Faithfulness too low"
        assert result["answer_relevancy"] > 0.5, "Answer relevancy too low"

        logging.info(f"RAGAS Results: {result}")

    def test_context_metrics(self, rag_engine, sample_reviews):
        """
        Тест контекстных метрик: recall и precision.
        """
        test_data = []

        for review in sample_reviews[:10]:
            review_text = review["text"]
            rating = review.get("rating")

            # Получаем правила
            rules = rag_engine.retrieve_rules(review_text, rating, top_k=3)

            if rules:
                test_data.append({
                    "question": review_text[:300],
                    "contexts": [rule["text"] for rule in rules],
                    "answer": rag_engine.generate_reply(review_text, rating, rules)["reply_text"]
                })

        if len(test_data) < 5:
            pytest.skip("Not enough test data")

        dataset = Dataset.from_list(test_data)

        result = evaluate(
            dataset,
            metrics=[context_recall, context_precision]
        )

        assert result["context_recall"] > 0.4, "Context recall too low"
        assert result["context_precision"] > 0.4, "Context precision too low"

    def test_response_quality_by_rating(self, rag_engine, sample_reviews):
        """
        Тест качества ответов для разных рейтингов.
        """
        rating_quality = {}

        for rating in range(1, 6):
            rating_reviews = [r for r in sample_reviews if r.get("rating") == rating][:5]

            if not rating_reviews:
                continue

            qualities = []
            for review in rating_reviews:
                result = rag_engine.generate_reply(review["text"], rating)
                qualities.append(result["confidence"])

            if qualities:
                rating_quality[rating] = sum(qualities) / len(qualities)

        # Проверяем что confidence не падает критически для негативных отзывов
        if 1 in rating_quality and 5 in rating_quality:
            # Негативные отзывы могут иметь чуть меньший confidence
            assert rating_quality[1] > 0.3, "Low confidence for negative reviews"
            assert rating_quality[5] > 0.3, "Low confidence for positive reviews"


class TestManualQualityChecks:
    """Ручные проверки качества (не автоматические)."""

    def test_no_hallucinations(self, rag_engine, sample_reviews):
        """
        Проверяет, что ответы не содержат галлюцинаций.
        Запрещенные фразы, не связанные с правилами.
        """
        forbidden_phrases = [
            "я не знаю",
            "я не могу ответить",
            "извините, я не понимаю",
            "как ИИ",
            "как искусственный интеллект"
        ]

        violations = []

        for review in sample_reviews[:20]:
            result = rag_engine.generate_reply(review["text"], review.get("rating"))
            reply = result["reply_text"].lower()

            for phrase in forbidden_phrases:
                if phrase in reply:
                    violations.append({
                        "review": review["text"][:100],
                        "reply": reply[:100],
                        "forbidden_phrase": phrase
                    })

        if violations:
            logging.warning(f"Found {len(violations)} potential hallucinations")
            # Не делаем assert, только логируем для ручного анализа
            for v in violations[:5]:
                logging.warning(f"Hallucination: {v}")

        # Допускаем небольшое количество, но не более 10%
        assert len(violations) < len(sample_reviews) * 0.1

    def test_reply_length(self, rag_engine, sample_reviews):
        """
        Проверяет, что ответы имеют адекватную длину.
        """
        lengths = []

        for review in sample_reviews[:20]:
            result = rag_engine.generate_reply(review["text"], review.get("rating"))
            reply_len = len(result["reply_text"])
            lengths.append(reply_len)

            # Ответ не должен быть пустым
            assert reply_len > 0, "Empty reply generated"

            # Ответ не должен быть слишком длинным
            assert reply_len <= 1000, f"Reply too long: {reply_len}"

        avg_length = sum(lengths) / len(lengths)
        logging.info(f"Average reply length: {avg_length:.0f} chars")

        # Средняя длина должна быть в разумных пределах
        assert 30 <= avg_length <= 500, f"Average length unusual: {avg_length}"

    def test_politeness(self, rag_engine, sample_reviews):
        """
        Проверяет вежливость ответов.
        """
        polite_phrases = [
            "спасиб", "благодар", "пожалуйста",
            "извинит", "простит", "поможем", "рады помочь"
        ]

        scores = []

        for review in sample_reviews[:20]:
            result = rag_engine.generate_reply(review["text"], review.get("rating"))
            reply_lower = result["reply_text"].lower()

            # Считаем количество вежливых фраз
            polite_count = sum(1 for phrase in polite_phrases if phrase in reply_lower)
            scores.append(polite_count > 0)

        politeness_rate = sum(scores) / len(scores)
        logging.info(f"Politeness rate: {politeness_rate:.2%}")

        # Как минимум 70% ответов должны быть вежливыми
        assert politeness_rate >= 0.7, f"Politeness rate too low: {politeness_rate}"