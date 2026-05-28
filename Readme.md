reply_generator/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI приложение
│   ├── config.py            # Settings (pydantic)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py        # эндпоинты
│   │   └── dependencies.py  # DI контейнеры
│   ├── core/
│   │   ├── rag_engine.py    # RAG поиск
│   │   ├── react_agent.py   # ReAct логика
│   │   └── validator.py     # пост-обработка
│   ├── services/
│   │   ├── embedding_service.py
│   │   ├── llm_client.py    # твой YandexGPT API
│   │   └── vector_store.py  # ChromaDB
│   ├── models/
│   │   ├── schemas.py       # Pydantic модели
│   │   └── review_class.py  # классификация
│   └── utils/
│       ├── logger.py        # structured logging
│       └── metrics.py       # prometheus
├── data/
│   ├── raw/                 # спарсенные отзывы
│   ├── processed/           # размеченные данные
│   └── knowledge_base/      # твои правила (создадим сами)
├── tests/
│   ├── test_ragas.py        # RAGAS метрики
│   └── test_api.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env