"""
Сервис для работы с векторной базой данных ChromaDB.
Обеспечивает поиск похожих правил по эмбеддингам.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.types import QueryResult

from app.config import settings
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStore:
    """Обертка над ChromaDB для работы с векторным хранилищем."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        chroma_path: Optional[Path] = None,
        collection_name: str = "rules_kb"
    ):
        """
        Инициализация векторного хранилища.

        Args:
            embedding_service: Сервис для генерации эмбеддингов
            chroma_path: Путь к persistent хранилищу ChromaDB
            collection_name: Имя коллекции с правилами
        """
        self.embedding_service = embedding_service
        self.chroma_path = chroma_path or settings.chroma_path
        self.collection_name = collection_name

        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None

        self._initialize_store()

    def _initialize_store(self):
        """Инициализирует соединение с ChromaDB."""
        try:
            logger.info(f"Initializing ChromaDB at {self.chroma_path}")

            # Создаем директорию если её нет
            self.chroma_path.mkdir(parents=True, exist_ok=True)

            # Инициализируем клиент с правильными настройками
            self._client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,  # Разрешаем сброс для совместимости
                    is_persistent=True,
                    persist_directory=str(self.chroma_path)
                )
            )

            # Получаем список коллекций
            try:
                existing_collections = self._client.list_collections()
                collection_names = [col.name for col in existing_collections]
            except Exception as e:
                logger.warning(f"Failed to list collections: {e}, trying alternative method")
                collection_names = []
                # Пробуем получить коллекцию напрямую
                try:
                    test_col = self._client.get_collection(self.collection_name)
                    if test_col:
                        collection_names = [self.collection_name]
                except:
                    pass

            # Проверяем существование коллекции
            if self.collection_name not in collection_names:
                logger.warning(
                    f"Collection '{self.collection_name}' not found, attempting to create",
                    extra={
                        "available_collections": collection_names,
                        "expected_collection": self.collection_name
                    }
                )

                # Создаем коллекцию заново
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )

                # Загружаем правила из JSON
                self._load_rules_to_collection()
            else:
                self._collection = self._client.get_collection(self.collection_name)

            # Получаем информацию о коллекции
            collection_count = self._collection.count()
            logger.info(
                f"Connected to ChromaDB collection",
                extra={
                    "collection_name": self.collection_name,
                    "documents_count": collection_count
                }
            )

        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {str(e)}", exc_info=True)
            raise

    def _load_rules_to_collection(self):
        """Загружает правила из JSON в коллекцию ChromaDB."""
        try:
            rules_file = settings.rules_path
            if not rules_file.exists():
                logger.error(f"Rules file not found: {rules_file}")
                return

            with open(rules_file, 'r', encoding='utf-8') as f:
                rules = json.load(f)

            logger.info(f"Loading {len(rules)} rules to ChromaDB")

            # Подготавливаем данные
            ids = []
            documents = []
            metadatas = []

            for rule in rules:
                rule_id = rule.get('id', f"rule_{len(ids)}")
                rule_text = rule.get('text', '')

                ids.append(rule_id)
                documents.append(rule_text)
                metadatas.append({
                    'category': rule.get('category', 'general'),
                    'tone': rule.get('tone', 'neutral'),
                    'priority': rule.get('priority', 0),
                    'original_id': rule_id
                })

            # Генерируем эмбеддинги
            embeddings = self.embedding_service.encode_batch(documents)

            # Добавляем в коллекцию
            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings.tolist()
            )

            logger.info(f"Successfully loaded {len(ids)} rules to collection")

        except Exception as e:
            logger.error(f"Failed to load rules to collection: {str(e)}", exc_info=True)
            raise

    def search(
        self,
        query_text: str,
        top_k: int = 3,
        similarity_threshold: Optional[float] = None,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск похожих правил по текстовому запросу.

        Args:
            query_text: Текст запроса (отзыв)
            top_k: Количество результатов
            similarity_threshold: Минимальный порог сходства (0-1)
            where_filter: Фильтр метаданных (например, {'category': 'complaint'})

        Returns:
            Список найденных правил с метаданными
        """
        if self._collection is None:
            raise RuntimeError("Vector store not initialized")

        try:
            # Генерируем эмбеддинг для запроса
            query_embedding = self.embedding_service.encode(query_text)

            # Выполняем поиск
            results: QueryResult = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            # Парсим результаты
            similarities = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    document = results['documents'][0][i] if results['documents'] else ""
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    # ChromaDB возвращает distance = 1 - similarity для нормализованных векторов
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    similarity = 1.0 - distance

                    # Фильтруем по порогу
                    if similarity_threshold and similarity < similarity_threshold:
                        continue

                    similarities.append({
                        "id": doc_id,
                        "text": document,
                        "similarity": similarity,
                        "category": metadata.get("category", "unknown"),
                        "tone": metadata.get("tone", "neutral"),
                        "priority": metadata.get("priority", 0),
                        "metadata": metadata
                    })

            logger.debug(
                f"Search completed",
                extra={
                    "query_length": len(query_text),
                    "top_k": top_k,
                    "results_count": len(similarities),
                    "threshold": similarity_threshold
                }
            )

            return similarities

        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            raise

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 3,
        similarity_threshold: Optional[float] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Пакетный поиск для множества запросов.

        Args:
            queries: Список текстов запросов
            top_k: Количество результатов на запрос
            similarity_threshold: Порог сходства

        Returns:
            Список результатов для каждого запроса
        """
        results = []
        for query in queries:
            query_results = self.search(
                query_text=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            results.append(query_results)

        return results

    def get_rule_by_id(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """
        Получает правило по его ID.

        Args:
            rule_id: ID правила (строка)

        Returns:
            Правило с метаданными или None
        """
        if self._collection is None:
            raise RuntimeError("Vector store not initialized")

        try:
            results = self._collection.get(
                ids=[rule_id],
                include=["documents", "metadatas"]
            )

            if results['ids'] and len(results['ids']) > 0:
                return {
                    "id": results['ids'][0],
                    "text": results['documents'][0] if results['documents'] else "",
                    "metadata": results['metadatas'][0] if results['metadatas'] else {}
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get rule by ID {rule_id}: {str(e)}")
            return None

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику по коллекции.

        Returns:
            Словарь со статистикой
        """
        if self._collection is None:
            raise RuntimeError("Vector store not initialized")

        try:
            count = self._collection.count()
        except:
            count = 0

        return {
            "collection_name": self.collection_name,
            "documents_count": count,
            "embedding_dimension": self.embedding_service.get_embedding_dim(),
            "storage_path": str(self.chroma_path)
        }

    def reset_connection(self):
        """Сбрасывает соединение с хранилищем."""
        self._client = None
        self._collection = None
        logger.info("Vector store connection reset")


# Фабрика для DI
def create_vector_store(
    embedding_service: EmbeddingService
) -> VectorStore:
    """Создает экземпляр VectorStore."""
    return VectorStore(embedding_service=embedding_service)