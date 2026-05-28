"""
FastAPI приложение для генерации ответов на отзывы Wildberries.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.config import settings
from app.api.routes import router
from app.utils.logger import setup_logging
from app.api.dependencies import get_rag_engine

# Настройка логирования
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan менеджер для инициализации и очистки ресурсов.
    """
    # Startup
    logger.info("Starting Reply Generator Service...")
    logger.info(f"Configuration: host={settings.api_host}, port={settings.api_port}")

    # Инициализируем RAG Engine при старте
    try:
        rag_engine = get_rag_engine()
        # Используем hybrid_retriever вместо vector_store
        stats = rag_engine.hybrid_retriever.get_collection_stats()
        logger.info(
            "RAG Engine initialized successfully with hybrid search",
            extra={
                "documents_count": stats["documents_count"],
                "embedding_dim": stats["embedding_dimension"],
                "bm25_initialized": stats.get("bm25_initialized", False),
                "alpha": stats.get("alpha", 0.6)
            }
        )
    except Exception as e:
        logger.error(f"Failed to initialize RAG Engine: {str(e)}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down Reply Generator Service...")

# Создаем FastAPI приложение
app = FastAPI(
    title="Reply Generator API for Wildberries",
    description="""
    Сервис для автоматической генерации ответов на отзывы Wildberries 
    с использованием RAG (Retrieval-Augmented Generation) и YandexGPT.

    ## Возможности:
    - Анализ отзывов и поиск релевантных правил
    - Генерация персонализированных ответов
    - Поддержка пакетной обработки
    - Метрики для Prometheus

    ## Как это работает:
    1. Отзыв анализируется и преобразуется в эмбеддинг
    2. Поиск похожих правил в векторной БД
    3. Генерация ответа через YandexGPT с учетом найденных правил
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Регистрируем роуты
app.include_router(router, prefix="/api/v1", tags=["replies"])


# Root эндпоинт для информации
@app.get("/", tags=["info"])
async def root():
    """Информация о сервисе."""
    return {
        "service": "Reply Generator for Wildberries",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "status": "running"
    }


# Кастомная OpenAPI схема
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Добавляем примеры авторизации если нужно
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )