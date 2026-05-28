"""
Сервис для работы с эмбеддингами текста.
Использует локальную модель rubert-tiny2 для генерации векторных представлений.
"""
import logging
from typing import List, Union, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Сервис для генерации эмбеддингов текста."""

    def __init__(self, model_name: Optional[str] = None):
        """
        Инициализация сервиса эмбеддингов.

        Args:
            model_name: Название модели sentence-transformers
        """
        self.model_name = model_name or settings.embedding_model_name
        self._model: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None

        self._initialize_model()

    def _initialize_model(self):
        """Загружает модель эмбеддингов."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)

            # Проверяем размерность эмбеддингов на тестовом тексте
            test_embedding = self._model.encode(["test"], show_progress_bar=False)
            self._embedding_dim = test_embedding.shape[1] if len(test_embedding.shape) > 1 else test_embedding.shape[0]

            logger.info(
                f"Embedding model loaded successfully",
                extra={
                    "model_name": self.model_name,
                    "embedding_dim": self._embedding_dim,
                    "device": str(self._model.device)
                }
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}", exc_info=True)
            raise

    def encode(
            self,
            texts: Union[str, List[str]],
            normalize: bool = True,
            show_progress: bool = False
    ) -> np.ndarray:
        """
        Преобразует текст(ы) в эмбеддинги.

        Args:
            texts: Текст или список текстов для энкодинга
            normalize: Нормализовать ли эмбеддинги (для косинусного расстояния)
            show_progress: Показывать прогресс-бар

        Returns:
            numpy массив с эмбеддингами размерности (n_texts, embedding_dim)
        """
        if self._model is None:
            raise RuntimeError("Embedding model not initialized")

        # Приводим к списку для единообразия
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        if not texts:
            raise ValueError("Empty texts list provided")

        try:
            logger.debug(f"Encoding {len(texts)} text(s)")

            embeddings = self._model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )

            logger.debug(
                f"Encoding completed",
                extra={
                    "num_texts": len(texts),
                    "embedding_shape": embeddings.shape,
                    "normalized": normalize
                }
            )

            # Если был один текст, возвращаем 1D массив
            if single_input:
                return embeddings[0]

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode texts: {str(e)}", exc_info=True)
            raise

    def encode_batch(
            self,
            texts: List[str],
            batch_size: int = 32,
            normalize: bool = True
    ) -> np.ndarray:
        """
        Кодирует большой батч текстов с оптимизацией памяти.

        Args:
            texts: Список текстов
            batch_size: Размер батча для кодирования
            normalize: Нормализовать ли эмбеддинги

        Returns:
            numpy массив с эмбеддингами
        """
        if self._model is None:
            raise RuntimeError("Embedding model not initialized")

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Encoding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

            batch_embeddings = self._model.encode(
                batch,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def get_embedding_dim(self) -> int:
        """Возвращает размерность эмбеддингов."""
        if self._embedding_dim is None:
            raise RuntimeError("Embedding dimension not initialized")
        return self._embedding_dim

    def similarity_score(
            self,
            embedding1: np.ndarray,
            embedding2: np.ndarray
    ) -> float:
        """
        Вычисляет косинусное сходство между двумя эмбеддингами.

        Args:
            embedding1: Первый эмбеддинг
            embedding2: Второй эмбеддинг

        Returns:
            Значение косинусного сходства (0-1 для нормализованных)
        """
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)

        similarity = np.dot(embedding1, embedding2.T)

        # Для нормализованных векторов косинусное расстояние уже в [-1, 1]
        # Клиппируем к [0, 1] для удобства
        return float(np.clip(similarity[0, 0], 0, 1))


# Фабрика для DI
def create_embedding_service() -> EmbeddingService:
    """Создает экземпляр EmbeddingService."""
    return EmbeddingService()