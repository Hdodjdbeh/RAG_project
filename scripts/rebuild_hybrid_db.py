"""
Пересоздание базы для гибридного поиска.
"""
import sys
import os
from pathlib import Path

# Добавляем корень проекта в PATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Загружаем .env файл
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

# Проверяем что переменные загрузились
if not os.getenv('YANDEX_API_KEY'):
    print("❌ YANDEX_API_KEY not found in .env")
    print(f"Looking for .env at: {env_path}")
    sys.exit(1)

from app.services.embedding_service import EmbeddingService
from app.services.hybrid_retriever import HybridRetriever

def rebuild():
    """Пересоздает коллекцию для гибридного поиска."""
    print("✓ Environment loaded")
    print("Initializing embedding service...")
    emb = EmbeddingService()

    print("Creating hybrid retriever...")
    retriever = HybridRetriever(
        embedding_service=emb,
        alpha=0.6
    )

    print("Checking collection...")
    stats = retriever.get_collection_stats()
    print(f"Stats: {stats}")

    # Тестируем поиск
    test_query = "Товар пришел с браком, прошу вернуть деньги"
    print(f"\nTesting hybrid search with: '{test_query}'")

    results = retriever.search(test_query, top_k=3)

    if results:
        print(f"\n✅ Found {len(results)} results:")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['id']} (hybrid: {r['similarity']:.3f}, "
                  f"vector: {r['vector_score']:.3f}, bm25: {r['bm25_score']:.3f})")
            print(f"   {r['text'][:100]}...")
    else:
        print("\n⚠️ No results found")

    print("\n✅ Hybrid search ready!")

if __name__ == "__main__":
    rebuild()