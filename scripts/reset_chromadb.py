"""
Скрипт для сброса и пересоздания ChromaDB.
"""
import shutil
import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reset_chromadb():
    """Удаляет и пересоздает ChromaDB."""
    chroma_path = settings.chroma_path

    if chroma_path.exists():
        logger.info(f"Removing existing ChromaDB at {chroma_path}")
        shutil.rmtree(chroma_path)

    logger.info(f"Creating fresh ChromaDB at {chroma_path}")
    chroma_path.mkdir(parents=True, exist_ok=True)

    logger.info("ChromaDB reset complete. Run build_vector_db.py to recreate collections.")


if __name__ == "__main__":
    reset_chromadb()