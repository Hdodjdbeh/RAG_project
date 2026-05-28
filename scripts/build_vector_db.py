"""
Скрипт для создания векторной базы данных ChromaDB из правил.
"""
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_vector_db():
    """Создает векторную базу данных."""
    try:
        # Проверяем существование файла с правилами
        if not settings.rules_path.exists():
            logger.error(f"Rules file not found: {settings.rules_path}")
            return False

        # Загружаем правила
        with open(settings.rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)

        logger.info(f"Loaded {len(rules)} rules from {settings.rules_path}")

        # Инициализируем сервисы
        logger.info("Initializing embedding service...")
        embedding_service = EmbeddingService()

        logger.info("Initializing vector store...")
        vector_store = VectorStore(
            embedding_service=embedding_service,
            chroma_path=settings.chroma_path
        )

        # Получаем статистику
        stats = vector_store.get_collection_stats()
        logger.info(f"Vector store ready: {stats}")

        logger.info("Vector DB built successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to build vector DB: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = build_vector_db()
    sys.exit(0 if success else 1)