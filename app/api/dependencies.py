"""
DI контейнеры для внедрения зависимостей.
Использует гибридный ретривер.
"""
import logging
from functools import lru_cache

from app.services.embedding_service import EmbeddingService, create_embedding_service
from app.services.hybrid_retriever import HybridRetriever, create_hybrid_retriever
from app.services.llm_client import YandexGPTClient, create_llm_client
from app.core.rag_engine import RAGEngine
from app.config import settings

logger = logging.getLogger(__name__)

# Глобальные переменные
_embedding_service = None
_hybrid_retriever = None
_llm_client = None
_rag_engine = None


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Возвращает синглтон сервиса эмбеддингов."""
    global _embedding_service
    if _embedding_service is None:
        logger.info("Creating EmbeddingService singleton")
        _embedding_service = create_embedding_service()
    return _embedding_service


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    """Возвращает синглтон гибридного ретривера."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        logger.info("Creating HybridRetriever singleton")
        embedding_service = get_embedding_service()
        _hybrid_retriever = create_hybrid_retriever(embedding_service)
    return _hybrid_retriever


@lru_cache(maxsize=1)
def get_llm_client() -> YandexGPTClient:
    """Возвращает синглтон LLM клиента."""
    global _llm_client
    if _llm_client is None:
        logger.info("Creating YandexGPTClient singleton")
        _llm_client = create_llm_client()
    return _llm_client


@lru_cache(maxsize=1)
def get_rag_engine() -> RAGEngine:
    """Возвращает синглтон RAG движка с гибридным поиском."""
    global _rag_engine
    if _rag_engine is None:
        logger.info("Creating RAGEngine singleton with hybrid search")
        hybrid_retriever = get_hybrid_retriever()
        llm_client = get_llm_client()
        _rag_engine = RAGEngine(
            hybrid_retriever=hybrid_retriever,
            llm_client=llm_client,
            top_k=settings.rag_top_k,
            similarity_threshold=settings.rag_similarity_threshold,
            alpha=0.6  # 60% семантика, 40% ключевые слова
        )
    return _rag_engine


async def reset_dependencies():
    """Сброс зависимостей (для тестов)."""
    global _embedding_service, _hybrid_retriever, _llm_client, _rag_engine
    _embedding_service = None
    _hybrid_retriever = None
    _llm_client = None
    _rag_engine = None
    get_embedding_service.cache_clear()
    get_hybrid_retriever.cache_clear()
    get_llm_client.cache_clear()
    get_rag_engine.cache_clear()
    logger.info("Dependencies reset")