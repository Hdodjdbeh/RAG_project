"""
Клиент для работы с YandexGPT API.
Обеспечивает генерацию текста через API Yandex Foundation Models.
"""
import json
import logging
from typing import Optional, Dict, Any, List

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings

logger = logging.getLogger(__name__)


class YandexGPTClient:
    """Клиент для взаимодействия с YandexGPT API."""

    def __init__(
            self,
            api_key: Optional[str] = None,
            folder_id: Optional[str] = None,
            api_url: Optional[str] = None,
            model_uri: Optional[str] = None,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None
    ):
        """
        Инициализация клиента YandexGPT.

        Args:
            api_key: API ключ Yandex Cloud
            folder_id: ID каталога в Yandex Cloud
            api_url: URL API эндпоинта
            model_uri: URI модели (например, gpt://<folder_id>/yandexgpt-lite/latest)
            max_tokens: Максимальное количество токенов в ответе
            temperature: Температура генерации (0-1)
        """
        self.api_key = api_key or settings.yandex_api_key
        self.folder_id = folder_id or settings.yandex_folder_id
        self.api_url = api_url or settings.yandex_api_url
        self.model_uri = model_uri or settings.yandex_model_uri
        self.max_tokens = max_tokens or settings.yandex_max_tokens
        self.temperature = temperature or settings.yandex_temperature

        if not self.api_key:
            raise ValueError("Yandex API key is required")
        if not self.folder_id:
            raise ValueError("Yandex folder ID is required")

        self.headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }

        logger.info(
            "YandexGPTClient initialized",
            extra={
                "api_url": self.api_url,
                "model_uri": self.model_uri,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        )

    def _build_request_payload(
            self,
            system_prompt: str,
            user_message: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Формирует payload для API запроса.

        Args:
            system_prompt: Системный промпт (роль ассистента)
            user_message: Сообщение пользователя
            max_tokens: Лимит токенов для ответа
            temperature: Температура генерации

        Returns:
            Dict с payload для API
        """
        return {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature or self.temperature,
                "maxTokens": max_tokens or self.max_tokens
            },
            "messages": [
                {
                    "role": "system",
                    "text": system_prompt
                },
                {
                    "role": "user",
                    "text": user_message
                }
            ]
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (requests.RequestException, ConnectionError, TimeoutError)
        )
    )
    def generate(
            self,
            system_prompt: str,
            user_message: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            timeout: int = 30
    ) -> str:
        """
        Генерирует ответ на основе промптов.

        Args:
            system_prompt: Системный промпт (роль ассистента)
            user_message: Сообщение пользователя
            max_tokens: Лимит токенов для ответа
            temperature: Температура генерации
            timeout: Таймаут запроса в секундах

        Returns:
            Сгенерированный текст ответа

        Raises:
            Exception: При ошибке API или превышении лимитов
        """
        try:
            payload = self._build_request_payload(
                system_prompt=system_prompt,
                user_message=user_message,
                max_tokens=max_tokens,
                temperature=temperature
            )

            logger.debug(
                "Sending request to YandexGPT",
                extra={
                    "system_prompt_length": len(system_prompt),
                    "user_message_length": len(user_message),
                    "max_tokens": max_tokens or self.max_tokens
                }
            )

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=timeout
            )

            response.raise_for_status()

            result = response.json()

            # Извлекаем текст ответа из структуры YandexGPT
            if "result" in result and "alternatives" in result["result"]:
                alternatives = result["result"]["alternatives"]
                if alternatives and len(alternatives) > 0:
                    generated_text = alternatives[0].get("message", {}).get("text", "")
                    if generated_text:
                        logger.info(
                            "Successfully generated response",
                            extra={
                                "response_length": len(generated_text),
                                "tokens_used": result.get("result", {}).get("usage", {})
                            }
                        )
                        return generated_text.strip()

            # Fallback для другой структуры ответа
            if "response" in result and "text" in result["response"]:
                return result["response"]["text"].strip()

            logger.error(f"Unexpected response structure: {result}")
            raise ValueError("Could not extract generated text from response")

        except requests.Timeout:
            logger.error(f"Request timeout after {timeout} seconds")
            raise Exception(f"YandexGPT API timeout after {timeout} seconds")
        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response body: {e.response.text}")
            raise Exception(f"YandexGPT API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in generate: {str(e)}", exc_info=True)
            raise

    def generate_with_history(
            self,
            messages: List[Dict[str, str]],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None
    ) -> str:
        """
        Генерирует ответ на основе истории сообщений.

        Args:
            messages: Список сообщений с полями "role" и "text"
            max_tokens: Лимит токенов
            temperature: Температура

        Returns:
            Сгенерированный текст
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        # Извлекаем system сообщение если есть
        system_messages = [m for m in messages if m.get("role") == "system"]
        user_messages = [m for m in messages if m.get("role") == "user"]

        system_prompt = system_messages[0]["text"] if system_messages else ""
        # Берем последнее user сообщение как основной запрос
        user_message = user_messages[-1]["text"] if user_messages else ""

        # TODO: Для полной поддержки истории нужно использовать другой API эндпоинт
        # Сейчас YandexGPT v1 completion не поддерживает многооборотные диалоги
        return self.generate(system_prompt, user_message, max_tokens, temperature)

    def count_tokens(self, text: str) -> int:
        """
        Примерная оценка количества токенов в тексте.

        Args:
            text: Входной текст

        Returns:
            Приблизительное количество токенов
        """
        # Грубая оценка: для русского языка ~1 токен на символ
        # YandexGPT использует примерно такое соотношение
        return len(text)

    def validate_context_length(
            self,
            system_prompt: str,
            user_message: str,
            max_context_tokens: int = 2000
    ) -> bool:
        """
        Проверяет, влезает ли промпт в контекст.

        Args:
            system_prompt: Системный промпт
            user_message: Сообщение пользователя
            max_context_tokens: Максимальный размер контекста

        Returns:
            True если влезает, False если нет
        """
        total_length = len(system_prompt) + len(user_message)
        estimated_tokens = self.count_tokens(system_prompt) + self.count_tokens(user_message)

        if estimated_tokens > max_context_tokens:
            logger.warning(
                "Context length limit exceeded",
                extra={
                    "estimated_tokens": estimated_tokens,
                    "max_tokens": max_context_tokens,
                    "total_chars": total_length
                }
            )
            return False

        return True


# Фабрика для создания клиента (для DI)
def create_llm_client() -> YandexGPTClient:
    """Создает экземпляр YandexGPTClient с настройками из окружения."""
    return YandexGPTClient()