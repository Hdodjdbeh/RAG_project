"""
Расширенные метрики для Prometheus.
"""
from prometheus_client import Counter, Histogram, Gauge, Info
import psutil
import platform

# Метрики приложения
APP_INFO = Info("reply_generator", "Application info")
CONFIDENCE_SCORE = Histogram(
    "reply_generator_confidence_score",
    "Confidence score of generated replies",
    buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
)

# Системные метрики
CPU_USAGE = Gauge("system_cpu_usage_percent", "CPU usage percentage")
MEMORY_USAGE = Gauge("system_memory_usage_bytes", "Memory usage in bytes")
RULES_CACHE_HITS = Counter("reply_generator_cache_hits", "Number of cache hits")
RULES_CACHE_MISSES = Counter("reply_generator_cache_misses", "Number of cache misses")

# Метрики LLM
LLM_REQUESTS = Counter("llm_requests_total", "Total LLM requests", ["status"])
LLM_REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Метрики векторного поиска
VECTOR_SEARCH_DURATION = Histogram(
    "vector_search_duration_seconds",
    "Vector search duration",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)
VECTOR_SEARCH_RESULTS = Histogram(
    "vector_search_results_count",
    "Number of results from vector search",
    buckets=[0, 1, 2, 3, 4, 5]
)


def update_system_metrics():
    """Обновляет системные метрики."""
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    MEMORY_USAGE.set(psutil.virtual_memory().used)


def set_app_info():
    """Устанавливает информацию о приложении."""
    APP_INFO.info({
        "version": "1.0.0",
        "python_version": platform.python_version(),
        "platform": platform.platform()
    })