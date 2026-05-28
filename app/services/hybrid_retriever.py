"""
Гибридный ретривер для поиска правил: семантический (векторный) + BM25 (ключевой)
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

from app.config import settings
from app.services.embedding_service import EmbeddingService

# Скачиваем NLTK данные при первом импорте
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Гибридный ретривер: объединяет семантический поиск через ChromaDB
    и ключевой поиск через BM25.
    """

    def __init__(
            self,
            embedding_service: EmbeddingService,
            chroma_path: Optional[Path] = None,
            collection_name: str = "rules_kb",
            alpha: float = 0.6  # вес семантического поиска (0-1)
    ):
        """
        Инициализация гибридного ретривера.

        Args:
            embedding_service: Сервис для эмбеддингов
            chroma_path: Путь к ChromaDB
            collection_name: Имя коллекции
            alpha: Баланс между семантикой (alpha) и BM25 (1-alpha)
        """
        self.embedding_service = embedding_service
        self.chroma_path = chroma_path or settings.chroma_path
        self.collection_name = collection_name
        self.alpha = alpha

        self._client = None
        self._collection = None
        self._bm25_index = None
        self._documents_texts = []
        self._documents_metadata = []
        self._documents_ids = []

        self._initialize_store()
        self._initialize_bm25()

        logger.info(
            f"HybridRetriever initialized",
            extra={
                "alpha": alpha,
                "documents_count": len(self._documents_texts)
            }
        )

    def _initialize_store(self):
        """Инициализирует ChromaDB."""
        try:
            self.chroma_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )

            # Получаем или создаем коллекцию
            try:
                self._collection = self._client.get_collection(self.collection_name)
            except:
                logger.warning(f"Collection {self.collection_name} not found, creating...")
                self._collection = self._client.create_collection(self.collection_name)
                self._load_rules_to_collection()

            # Сохраняем все документы для BM25
            all_data = self._collection.get(include=["documents", "metadatas"])
            self._documents_ids = all_data['ids']
            self._documents_texts = all_data['documents']
            self._documents_metadata = all_data['metadatas']

            logger.info(f"Loaded {len(self._documents_texts)} documents from ChromaDB")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}", exc_info=True)
            raise

    def _load_rules_to_collection(self):
        """Загружает правила из JSON в коллекцию."""
        try:
            rules_file = settings.rules_path
            if not rules_file.exists():
                logger.error(f"Rules file not found: {rules_file}")
                return

            with open(rules_file, 'r', encoding='utf-8') as f:
                rules = json.load(f)

            logger.info(f"Loading {len(rules)} rules to ChromaDB")

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
                    'priority': rule.get('priority', 0)
                })

            # Генерируем эмбеддинги
            embeddings = self.embedding_service.encode_batch(documents)

            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings.tolist()
            )

            logger.info(f"Successfully loaded {len(ids)} rules")

        except Exception as e:
            logger.error(f"Failed to load rules: {str(e)}", exc_info=True)
            raise

    def _initialize_bm25(self):
        """Инициализирует BM25 индекс."""
        if not self._documents_texts:
            logger.warning("No documents for BM25 initialization")
            return

        # Токенизируем документы
        tokenized_docs = []
        for doc in self._documents_texts:
            tokens = word_tokenize(doc.lower())
            tokenized_docs.append(tokens)

        self._bm25_index = BM25Okapi(tokenized_docs)
        logger.info(f"BM25 index initialized with {len(self._documents_texts)} documents")

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Нормализует список скорингов в диапазон [0, 1].

        Args:
            scores: Список сырых скорингов

        Returns:
            Нормализованные скоринги
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [0.5] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def search(
            self,
            query_text: str,
            top_k: int = 5,
            similarity_threshold: Optional[float] = None,
            alpha: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Гибридный поиск: семантический + BM25.
        """
        if self._collection is None:
            raise RuntimeError("Vector store not initialized")

        alpha = alpha if alpha is not None else self.alpha

        try:
            # Шаг 1: Семантический поиск (векторный)
            query_embedding = self.embedding_service.encode(query_text)

            vector_results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k * 2,
                include=["documents", "metadatas", "distances"]
            )

            # Шаг 2: BM25 поиск (ключевой)
            tokenized_query = word_tokenize(query_text.lower())
            bm25_scores = None
            if self._bm25_index:
                bm25_scores = self._bm25_index.get_scores(tokenized_query)

            # Шаг 3: Объединение результатов
            combined_results = {}

            # Обрабатываем векторные результаты
            if vector_results['ids'] and vector_results['ids'][0]:
                vector_scores = []
                for i, doc_id in enumerate(vector_results['ids'][0]):
                    distance = vector_results['distances'][0][i] if vector_results['distances'] else 1.0
                    vector_score = 1.0 - distance
                    vector_scores.append(vector_score)

                # Нормализуем векторные скоры
                if vector_scores:
                    min_vec = min(vector_scores)
                    max_vec = max(vector_scores)
                    if max_vec > min_vec:
                        normalized_vector_scores = [(s - min_vec) / (max_vec - min_vec) for s in vector_scores]
                    else:
                        normalized_vector_scores = [0.5] * len(vector_scores)
                else:
                    normalized_vector_scores = []

                for i, doc_id in enumerate(vector_results['ids'][0]):
                    # Находим индекс документа в общем списке
                    try:
                        doc_index = self._documents_ids.index(doc_id)
                    except ValueError:
                        continue

                    vector_score = normalized_vector_scores[i] if i < len(normalized_vector_scores) else 0

                    # Получаем BM25 скор
                    bm25_score_normalized = 0
                    if bm25_scores is not None and doc_index < len(bm25_scores):
                        bm25_score = bm25_scores[doc_index]
                        # Нормализуем BM25 скор
                        max_bm25 = float(np.max(bm25_scores)) if bm25_scores.size > 0 else 1
                        if max_bm25 > 0:
                            bm25_score_normalized = bm25_score / max_bm25

                    # Комбинированный скор
                    combined_score = alpha * vector_score + (1 - alpha) * bm25_score_normalized

                    combined_results[doc_id] = {
                        "id": doc_id,
                        "text": vector_results['documents'][0][i],
                        "similarity": combined_score,
                        "vector_score": vector_score,
                        "bm25_score": bm25_score_normalized,
                        "category": vector_results['metadatas'][0][i].get("category", "unknown"),
                        "tone": vector_results['metadatas'][0][i].get("tone", "neutral"),
                        "priority": vector_results['metadatas'][0][i].get("priority", 0),
                    }

            # Шаг 4: Добавляем результаты из BM25, которых нет в векторных
            if bm25_scores is not None and len(bm25_scores) == len(self._documents_ids):
                # Находим топ-k по BM25
                bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
                max_bm25 = float(np.max(bm25_scores)) if bm25_scores.size > 0 else 1

                for idx in bm25_indices:
                    doc_id = self._documents_ids[idx]
                    if doc_id in combined_results:
                        continue

                    bm25_score = bm25_scores[idx]
                    bm25_score_normalized = bm25_score / max_bm25 if max_bm25 > 0 else 0

                    combined_results[doc_id] = {
                        "id": doc_id,
                        "text": self._documents_texts[idx],
                        "similarity": (1 - alpha) * bm25_score_normalized,
                        "vector_score": 0,
                        "bm25_score": bm25_score_normalized,
                        "category": self._documents_metadata[idx].get("category", "unknown"),
                        "tone": self._documents_metadata[idx].get("tone", "neutral"),
                        "priority": self._documents_metadata[idx].get("priority", 0),
                    }

            # Шаг 5: Сортируем и фильтруем
            results = sorted(
                combined_results.values(),
                key=lambda x: x["similarity"],
                reverse=True
            )[:top_k]

            # Фильтруем по порогу
            if similarity_threshold:
                results = [r for r in results if r["similarity"] >= similarity_threshold]

            logger.debug(
                f"Hybrid search completed",
                extra={
                    "query": query_text[:50],
                    "results_count": len(results),
                    "alpha": alpha
                }
            )

            return results

        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}", exc_info=True)
            raise


    def semantic_search_only(
            self,
            query_text: str,
            top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Только семантический поиск (для сравнения)."""
        return self.search(query_text, top_k, alpha=1.0)

    def bm25_search_only(
            self,
            query_text: str,
            top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Только BM25 поиск (для сравнения)."""
        return self.search(query_text, top_k, alpha=0.0)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Возвращает статистику по коллекции."""
        return {
            "collection_name": self.collection_name,
            "documents_count": len(self._documents_texts),
            "embedding_dimension": self.embedding_service.get_embedding_dim(),
            "storage_path": str(self.chroma_path),
            "bm25_initialized": self._bm25_index is not None,
            "alpha": self.alpha
        }


# Фабрика для DI
def create_hybrid_retriever(
        embedding_service: EmbeddingService
) -> HybridRetriever:
    """Создает экземпляр HybridRetriever."""
    return HybridRetriever(
        embedding_service=embedding_service,
        alpha=0.6  # 60% семантика, 40% ключевые слова
    )