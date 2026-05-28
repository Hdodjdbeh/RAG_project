"""
RAG движок для генерации ответов на отзывы.
Объединяет поиск по векторной БД и генерацию через LLM.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple

from app.services.hybrid_retriever import HybridRetriever
from app.services.llm_client import YandexGPTClient
from app.config import settings

logger = logging.getLogger(__name__)


class RAGEngine:
    """Движок для Retrieval-Augmented Generation ответов на отзывы."""

    def __init__(
            self,
            hybrid_retriever: HybridRetriever,
            llm_client: YandexGPTClient,
            top_k: int = 3,
            similarity_threshold: float = 0.5,
            alpha: float = 0.6  # баланс семантики и ключевых слов
    ):
        """
        Инициализация RAG движка.

        Args:
            hybrid_retriever: Гибридный ретривер
            llm_client: Клиент YandexGPT
            top_k: Количество правил для ретривала
            similarity_threshold: Порог сходства для правил
            alpha: Баланс между семантикой (alpha) и BM25 (1-alpha)
        """
        self.hybrid_retriever = hybrid_retriever
        self.llm_client = llm_client
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha

        logger.info(
            "RAGEngine initialized with hybrid search",
            extra={
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "alpha": alpha
            }
        )

    def retrieve_rules(
            self,
            review_text: str,
            rating: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск релевантных правил для отзыва с гибридным подходом.

        Args:
            review_text: Текст отзыва
            rating: Оценка отзыва (1-5) для дополнительной фильтрации

        Returns:
            Список найденных правил
        """
        # Гибридный поиск
        rules = self.hybrid_retriever.search(
            query_text=review_text,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold,
            alpha=self.alpha
        )

        logger.info(
            f"Retrieved {len(rules)} rules for review",
            extra={
                "review_length": len(review_text),
                "rating": rating,
                "rules_count": len(rules),
                "top_similarity": rules[0]["similarity"] if rules else None,
                "top_vector_score": rules[0]["vector_score"] if rules else None,
                "top_bm25_score": rules[0]["bm25_score"] if rules else None,
            }
        )

        return rules

    def build_prompt(
            self,
            review_text: str,
            rules: List[Dict[str, Any]],
            rating: Optional[int] = None
    ) -> Tuple[str, str]:
        """
        Формирует system и user промпты для LLM.

        Args:
            review_text: Текст отзыва
            rules: Найденные правила
            rating: Оценка отзыва

        Returns:
            Кортеж (system_prompt, user_message)
        """
        # Системный промпт (роль ассистента)
        system_prompt = """Ты - оператор службы поддержки Wildberries. 
Твоя задача - отвечать на отзывы клиентов, соблюдая следующие принципы:
1. Будь вежливым и эмпатичным
2. Отвечай по существу, используя предоставленные правила
3. Если проблема описана в отзыве - предложи решение
4. Не обещай того, что не можешь выполнить
5. Ответ должен быть на русском языке
6. Благодари за положительные отзывы
7. Для негативных отзывов приноси извинения и предлагай помощь

НЕ ИСПОЛЬЗУЙ общие фразы без конкретики. Опирайся на правила."""

        # Форматируем правила для промпта
        rules_text = ""
        for i, rule in enumerate(rules, 1):
            rules_text += f"\nПравило {i} (категория: {rule.get('category', 'general')}, тон: {rule.get('tone', 'neutral')}):\n{rule['text']}\n"

        # Формируем сообщение пользователя
        rating_text = f"Оценка: {rating} звезд(ы)\n" if rating else ""

        user_message = f"""Клиент оставил отзыв на Wildberries:

Оценка: {rating_text}   Текст отзыва: "{review_text}"

На основе правил выше, напиши ответ клиенту.
Ответ должен:
1. Быть конкретным и полезным
2. Соответствовать тону правил
3. Адресовать проблему из отзыва (если есть)
4. Быть не короче 20 и не длиннее 500 символов

Твой ответ:"""

        # Проверяем длину контекста
        total_tokens = self.llm_client.count_tokens(system_prompt + user_message + rules_text)
        if total_tokens > 1900:  # Оставляем 100 токенов на ответ
            logger.warning(f"Context length too high: {total_tokens} tokens, truncating rules")
            # Урезаем правила, оставляя самые релевантные
            rules_text = "\n".join(rules_text.split("\n")[:len(rules) * 2])

        user_message_with_rules = rules_text + "\n" + user_message

        return system_prompt, user_message_with_rules

    def generate_reply(
            self,
            review_text: str,
            rating: Optional[int] = None,
            rules: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Генерирует ответ на отзыв.

        Args:
            review_text: Текст отзыва
            rating: Оценка отзыва
            rules: Правила (если не переданы, будут найдены автоматически)

        Returns:
            Словарь с ответом и метаданными
        """
        try:
            # Шаг 1: Поиск правил
            if rules is None:
                rules = self.retrieve_rules(review_text, rating)

            if not rules:
                logger.warning(f"No rules found for review: {review_text[:100]}")
                # Fallback: используем стандартный ответ
                fallback_reply = self._generate_fallback_reply(review_text, rating)
                return {
                    "reply_text": fallback_reply,
                    "used_rules": [],
                    "confidence": 0.3,
                    "tone": "neutral",
                    "is_fallback": True
                }

            # Шаг 2: Формирование промпта
            system_prompt, user_message = self.build_prompt(review_text, rules, rating)

            # Шаг 3: Генерация через YandexGPT
            generated_reply = self.llm_client.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                max_tokens=500,
                temperature=0.85
            )

            # Шаг 4: Пост-обработка ответа
            final_reply = self._postprocess_reply(generated_reply)

            # Шаг 5: Вычисление confidence
            confidence = self._calculate_confidence(rules, review_text)

            # Определяем тон ответа на основе правил
            tones = [r.get("tone", "neutral") for r in rules[:2]]
            primary_tone = tones[0] if tones else "neutral"

            logger.info(
                "Successfully generated reply",
                extra={
                    "review_length": len(review_text),
                    "reply_length": len(final_reply),
                    "used_rules_count": len(rules),
                    "confidence": confidence,
                    "tone": primary_tone
                }
            )

            return {
                "reply_text": final_reply,
                "used_rules": [
                    {
                        "id": rule.get("id"),
                        "similarity": rule.get("similarity"),
                        "category": rule.get("category")
                    }
                    for rule in rules[:3]
                ],
                "confidence": confidence,
                "tone": primary_tone,
                "is_fallback": False
            }

        except Exception as e:
            logger.error(f"Failed to generate reply: {str(e)}", exc_info=True)
            # В случае ошибки возвращаем fallback ответ
            fallback_reply = self._generate_fallback_reply(review_text, rating)
            return {
                "reply_text": fallback_reply,
                "used_rules": [],
                "confidence": 0.1,
                "tone": "neutral",
                "is_fallback": True,
                "error": str(e)
            }

    def _postprocess_reply(self, reply: str) -> str:
        """
        Пост-обработка сгенерированного ответа.

        Args:
            reply: Сырой ответ от LLM

        Returns:
            Очищенный ответ
        """
        # Удаляем лишние пробелы и переносы строк
        reply = reply.strip()

        # Удаляем возможные префиксы типа "Ответ:", "Ответ оператора:"
        prefixes = ["Ответ:", "Ответ оператора:", "Ответ:", "Оператор:"]
        for prefix in prefixes:
            if reply.startswith(prefix):
                reply = reply[len(prefix):].strip()

        # Убираем кавычки в начале и конце
        reply = reply.strip('"\'')

        # Ограничиваем длину
        if len(reply) > 500:
            reply = reply[:497] + "..."

        return reply

    def _calculate_confidence(self, rules: List[Dict[str, Any]], review_text: str) -> float:
        """
        Вычисляет уверенность в ответе на основе правил.

        Args:
            rules: Найденные правила
            review_text: Текст отзыва

        Returns:
            Confidence score от 0 до 1
        """
        if not rules:
            return 0.1

        # Средняя схожесть правил
        similarity_score = sum(r.get("similarity", 0) for r in rules) / len(rules)

        # Бонус за высокий приоритет правил
        priority_bonus = sum(r.get("priority", 0) for r in rules) / (len(rules) * 10) if rules else 0

        # Итоговый confidence
        confidence = min(0.95, similarity_score * 0.8 + priority_bonus * 0.2)

        return round(confidence, 2)

    def _generate_fallback_reply(
            self,
            review_text: str,
            rating: Optional[int] = None
    ) -> str:
        """
        Генерирует fallback ответ при ошибках.

        Args:
            review_text: Текст отзыва
            rating: Оценка

        Returns:
            Стандартный ответ
        """
        if rating is not None and rating >= 4:
            return "Благодарим вас за высокую оценку! Мы рады, что вам понравился товар. Спасибо, что выбираете Wildberries!"
        elif rating is not None and rating <= 2:
            return "Приносим извинения за доставленные неудобства. Пожалуйста, свяжитесь с нашей службой поддержки, и мы поможем решить вашу проблему."
        else:
            return "Спасибо за ваш отзыв! Каждое мнение важно для нас. Если у вас есть вопросы или предложения, пожалуйста, обратитесь в службу поддержки Wildberries."


# Фабрика для DI
def create_rag_engine(
    hybrid_retriever: HybridRetriever,  # ← меняем с vector_store на hybrid_retriever
    llm_client: YandexGPTClient
) -> RAGEngine:
    """Создает экземпляр RAGEngine с гибридным поиском."""
    return RAGEngine(
        hybrid_retriever=hybrid_retriever,  # ← меняем имя параметра
        llm_client=llm_client,
        top_k=settings.rag_top_k,
        similarity_threshold=settings.rag_similarity_threshold,
        alpha=0.6  # баланс между семантикой и BM25
    )