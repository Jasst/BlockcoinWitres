"""
Централизованная конфигурация для всей AI-системы (когнитивная архитектура + GCN)
"""
import os
from pathlib import Path

# -------------------------------
# Общие пути
# -------------------------------
MEMORY_BASE_DIR = Path(__file__).resolve().parent.parent / "ai_memory_v3"
MEMORY_BASE_DIR.mkdir(exist_ok=True)

# -------------------------------
# LLM (LM Studio)
# -------------------------------
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_TIMEOUT = 300
LM_STUDIO_STREAM_TIMEOUT = 600
LM_STUDIO_USE_STREAM = True
LM_STUDIO_VISION_SUPPORTED = True

# -------------------------------
# Параметры памяти (GCN)
# -------------------------------
WORKING_MEMORY_SIZE = 20
SENSORY_BUFFER_SIZE = 5
EPISODIC_MAX_SIZE = 500
SEMANTIC_MAX_FACTS = 10000
ASSOCIATIVE_GRAPH_MAX_NODES = 20000
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IMPORTANCE = 1.0

# -------------------------------
# Гибридный поиск (GCN)
# -------------------------------
HYBRID_WEIGHT_SEMANTIC = 0.40      # вес семантического (эмбеддинги)
HYBRID_WEIGHT_GRAPH = 0.30         # вес графовых связей
HYBRID_WEIGHT_FRESHNESS = 0.15     # вес свежести
HYBRID_WEIGHT_EVIDENCE = 0.10      # вес количества доказательств
HYBRID_WEIGHT_CONFIDENCE = 0.05    # вес доверия
DYNAMIC_WEIGHTS_ENABLED = True

# -------------------------------
# Эмбеддинги (заглушка – можно подключить SentenceTransformer)
# -------------------------------
EMBEDDING_DIM = 128                # размерность векторов (для демо)
USE_EMBEDDINGS = True
GLOBAL_FACT_CONFIDENCE_THRESHOLD = 0.75

# -------------------------------
# GCN-специфичные параметры
# -------------------------------
GCN_STATE_FILENAME = "gcn_state.json"
GCN_AUTO_VERIFY = True
GCN_EVIDENCE_THRESHOLD = 0.6

# -------------------------------
# Консолидация и сон
# -------------------------------
CONSOLIDATION_INTERVAL = 3600 * 2
DEEP_CONSOLIDATION_INTERVAL = 3600 * 8
REPLAY_BATCH_SIZE = 20
REPLAY_MIX_RATIO = (0.4, 0.3, 0.2, 0.1)

# -------------------------------
# Любопытство и автономность
# -------------------------------
CURIOSITY_UNCERTAINTY_THRESHOLD = 0.7
CURIOSITY_NOVELTY_THRESHOLD = 0.6
CURIOSITY_RESEARCH_INTERVAL = 600
RESOURCE_BUDGET_LLM_CALLS = 100
AUTO_RESEARCH_ENABLED = True

# -------------------------------
# Веб-поиск
# -------------------------------
MAX_SEARCH_ITERATIONS = 3
SEARCH_CACHE_TTL = 300
PAGE_CONTENT_MAX_CHARS = 6000
MAX_PAGES_TO_FETCH = 7
MIN_RELEVANCE_THRESHOLD = 0.28
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
PARALLEL_FETCH_LIMIT = 8
DDG_MIN_INTERVAL = 1.2
DDG_MAX_RETRIES = 3
SEARCH_CACHE_MAX_SIZE = 200
AUTO_SEARCH_ENABLED = True
MAX_SEARCH_ATTEMPTS = 3
ENABLE_QUERY_REWRITE = True
EXTRACT_FACTS_FROM_SEARCH = True
EXTRACT_FACTS_WITH_LLM = True
DEEP_SEARCH_TOTAL_BUDGET = 15

# -------------------------------
# Планировщик целей
# -------------------------------
LONG_TERM_PLANNER_INTERVAL = 3600 * 6
GOAL_MAX_ACTIVE = 5

# -------------------------------
# Команды управления памятью (дополнительные триггеры для гибкости)
# -------------------------------
MEMORY_CONTROL_COMMANDS = {
    "запомни": "store",
    "забудь": "forget",
    "что ты знаешь о": "recall",
    "напомни о": "recall",           # новый синоним
    "сохрани": "store",              # новый синоним
    "удали": "forget",               # новый синоним
}

# -------------------------------
# Генерация изображений
# -------------------------------
EASYDIFFUSION_ENABLED = True
EASYDIFFUSION_URL = "http://localhost:7860"
EASYDIFFUSION_ENDPOINT = "/v1/sdapi/v1/txt2img"
EASYDIFFUSION_TIMEOUT = 120
EASYDIFFUSION_DEFAULT_STEPS = 20
EASYDIFFUSION_DEFAULT_WIDTH = 512
EASYDIFFUSION_DEFAULT_HEIGHT = 512

EASYDIFFUSION_DEFAULT_LORA_USE = True
EASYDIFFUSION_DEFAULT_LORA = "E:/Easy-Diffusion/models/lora/Realism Lora By Stable Yogi_V3_Lite.safetensors"
EASYDIFFUSION_DEFAULT_LORA_WEIGHT = 0.8

EASYDIFFUSION_MODEL = "realismByStableYogi_ponyV65.safetensors"   # <-- добавьте эту строку
# Раньше был жёстко зашит Windows-путь "E:/BlockcoinWitres/generated_images" —
# на любой другой машине (в т.ч. на Linux self-hosted сервере) mkdir() на
# этом пути падал бы сразу. По умолчанию кладём рядом с MEMORY_BASE_DIR,
# путь можно переопределить через переменную окружения GENERATED_IMAGES_DIR.
GENERATED_IMAGES_DIR = Path(
    os.environ.get("GENERATED_IMAGES_DIR", str(MEMORY_BASE_DIR.parent / "generated_images"))
)
GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Тайм-ауты и надёжность вызова инструментов (internal + MCP)
# -------------------------------
# Раньше вызов инструмента (особенно внешнего MCP-сервера) ничем не был
# ограничен по времени — если сервер/локальная LLM внутри него подвиснет,
# весь ответ чата зависал без вариантов восстановления.
TOOL_CALL_TIMEOUT_SECONDS = 45
# Периодичность повторных попыток подключиться к упавшим при старте MCP-серверам.
MCP_RECONNECT_INTERVAL = 120

# -------------------------------
# Прочее
# -------------------------------
STREAM_CHAR_BY_CHAR = False
STREAM_CHAR_DELAY = 0.02
MAX_IMAGE_SIZE_BASE64 = 5 * 1024 * 1024
MAX_MESSAGE_LENGTH = 10000
MIN_MESSAGE_LENGTH = 1

# -------------------------------
# Кэш памяти
# -------------------------------
MEMORY_CACHE_TTL = 60
MEMORY_CACHE_MAX_SIZE = 1000

# -------------------------------
# Дубликаты (light consolidation)
# -------------------------------
DUPLICATE_SIMILARITY_THRESHOLD = 0.92

# -------------------------------
# Рефлексия (самообучение)
# -------------------------------
REFLECTION_INTERVAL = 3600 * 4
REFLECTION_ERROR_THRESHOLD = 0.6
REFLECTION_HISTORY_SIZE = 100
REFLECTION_LLM_TEMP = 0.5
REFLECTION_LLM_MAX_TOKENS = 300

# -------------------------------
# Когнитивные дефолты фактов (memory_graph.Fact)
# -------------------------------
DEFAULT_NOVELTY = 0.0
DEFAULT_SALIENCE = 0.0
DEFAULT_STABILITY = 0.5
DEFAULT_PLASTICITY = 0.5
DEFAULT_PREDICTION_ERROR = 0.0

# -------------------------------
# Синапсы / Hebbian / STDP (memory_graph.Synapse)
# -------------------------------
SYNAPSE_INITIAL_WEIGHT = 0.1
SYNAPSE_MAX_WEIGHT = 1.0
SYNAPSE_MIN_WEIGHT = 0.01
SYNAPSE_DECAY_RATE = 0.001
SYNAPSE_PLASTICITY_RATE = 0.01
SYNAPSE_COACTIVATION_THRESHOLD = 0.3
HEBBIAN_LEARNING_RATE = 0.02
STDP_LEARNING_RATE = 0.03
STDP_TIME_WINDOW = 5.0
STDP_LONG_TERM_POTENTIATION = 0.01
STDP_LONG_TERM_DEPRESSION = 0.005

# -------------------------------
# Spreading activation
# -------------------------------
SPREADING_MAX_DEPTH = 3
SPREADING_MAX_NODES = 50
SPREADING_DECAY = 0.5
SPREADING_THRESHOLD = 0.05

# -------------------------------
# Предсказательная матрица
# -------------------------------
PREDICTIVE_MATRIX_MAX_SIZE = 5000
PREDICTIVE_LEARNING_RATE = 0.1
PREDICTION_ERROR_THRESHOLD = 0.3

# -------------------------------
# Эмбеддинги для CognitiveMemory (SentenceTransformer + FAISS)
# -------------------------------
# ИСПРАВЛЕНИЕ (пункт №1 из анализа интеллекта): all-mpnet-base-v2 —
# англоцентричная модель. Основной язык диалогов — русский, а семантический
# канал (HYBRID_WEIGHT_SEMANTIC/HYBRID_WEIGHT_COSINE) — самый весомый канал
# гибридного поиска, поэтому слабое качество эмбеддингов на русском напрямую
# било по релевантности всего, что подмешивается в промпт как "самый
# надёжный источник". multilingual-e5-large — сильная мультиязычная модель
# с хорошим покрытием русского. Если ресурсы (VRAM/RAM) ограничены — можно
# заменить на более лёгкую "intfloat/multilingual-e5-base" (тот же протокол
# query:/passage:, размерность вектора меньше, качество чуть ниже).
MEMORY_USE_EMBEDDINGS = True
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
# Модели семейства e5 обучены с асимметричными префиксами: текст запроса и
# текст сохраняемого факта эмбеддятся по-разному ("query: "/"passage: "),
# это даёт заметно более точный поиск, чем эмбеддинг без префиксов. Для
# моделей, которые в этом не нуждаются, можно оставить обе строки пустыми.
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_PASSAGE_PREFIX = "passage: "
FAISS_NLIST = 200
FAISS_NPROBE = 30
FAISS_REBUILD_THRESHOLD = 300
FAISS_MIN_TRAIN_VECTORS = 500

# -------------------------------
# Гибридный поиск (memory_graph.retrieve_hybrid)
# -------------------------------
HYBRID_WEIGHT_BM25 = 0.25
HYBRID_WEIGHT_COSINE = 0.40
FACTUAL_WEIGHTS = (0.35, 0.30, 0.15, 0.20)
GENERAL_WEIGHTS = (HYBRID_WEIGHT_BM25, HYBRID_WEIGHT_COSINE, HYBRID_WEIGHT_FRESHNESS, HYBRID_WEIGHT_GRAPH)

# -------------------------------
# FAISS адаптивные пороги
# -------------------------------
FAISS_SMALL_THRESHOLD = 50
FAISS_MEDIUM_THRESHOLD = 500
FAISS_HNSW_EF_CONSTRUCTION = 80
FAISS_HNSW_M = 32

# ===== НОВЫЕ ПАРАМЕТРЫ ДЛЯ УЛУЧШЕНИЙ =====

# Классификация намерений (если False, используем старую логику)
ENABLE_INTENT_CLASSIFICATION = True

# Автоматическое извлечение фактов из сообщений (без команды)
AUTO_EXTRACT_FACTS = True
AUTO_EXTRACT_CONFIDENCE = 0.5

# Порог уверенности для автоматического запоминания
AUTO_EXTRACT_THRESHOLD = 0.6

# Время бездействия для запуска консолидации (секунды)
IDLE_CONSOLIDATION_DELAY = 900  # 15 минут

# -------------------------------
# Формирование концептов (абстрагирование, шаг эмерджентности)
# -------------------------------
CONCEPT_MIN_CLUSTER_SIZE = 3        # минимум фактов в кластере, чтобы сформировать концепт
CONCEPT_SIMILARITY_THRESHOLD = 0.6  # порог косинусной близости для объединения в кластер
CONCEPT_MAX_SCAN = 2000             # лимит сканируемых фактов за один запуск (O(n^2) по эмбеддингам)
CONCEPT_MAX_PER_RUN = 5             # не больше стольких новых концептов за один цикл консолидации

# -------------------------------
# Кросс-слойное заземление (PRIVATE/SHARED -> GLOBAL через GROUNDS_IN)
# -------------------------------
CROSS_LAYER_GROUNDING_THRESHOLD = 0.75

# -------------------------------
# LLM-реранкинг результатов памяти (пункт №2 из анализа интеллекта)
# -------------------------------
# hybrid_retrieve/retrieve_hybrid отдают top_k по линейной взвешенной сумме
# (семантика+граф+свежесть+confidence+evidence) — это хорошо отсеивает явно
# нерелевантное, но плохо разруливает "похожее по вектору, но не по сути"
# кандидатов. RERANK_ENABLED включает лёгкий LLM-судью поверх уже
# найденных кандидатов GCNMemoryRouter.retrieve(): берём с запасом
# (top_k * RERANK_CANDIDATE_MULTIPLIER, но не больше RERANK_MAX_CANDIDATES),
# просим модель выбрать и упорядочить только реально релевантные — и уже
# из этого списка обрезаем до top_k. При сбое LLM или пустом ответе всегда
# безопасно откатываемся на исходный порядок (см. GCNMemoryRouter._llm_rerank).
RERANK_ENABLED = True
RERANK_CANDIDATE_MULTIPLIER = 3
RERANK_MAX_CANDIDATES = 20

# -------------------------------
# Верификация финального ответа (критик) — пункт №3
# -------------------------------
# Дешёвый второй проход LLM после генерации ответа: проверяет, нет ли в
# ответе конкретных фактических утверждений, не подтверждённых тем, что
# реально было передано модели (память/поиск/результаты инструментов).
# Не переписывает ответ — только добавляет короткую пометку, если находит
# подозрительные утверждения. Короткие/светские реплики не проверяются
# (VERIFICATION_MIN_WORDS), чтобы не тратить лишний вызов зря.
RESPONSE_VERIFICATION_ENABLED = True
VERIFICATION_MIN_WORDS = 12
VERIFICATION_MAX_TOKENS = 150

# -------------------------------
# Планирование подзадач перед ReAct-циклом инструментов — пункт №4
# -------------------------------
# ToolRouter.run() раньше был полностью реактивным: на каждом раунде модель
# заново решала, какой СЛЕДУЮЩИЙ инструмент вызвать, не имея явного плана на
# составной запрос ("сравни X и Y, потом посчитай Z"). Для запросов, которые
# эвристически выглядят как составные (см. TOOL_PLANNING_MIN_LEN и маркеры
# в tool_router.py), делаем один дешёвый предварительный вызов, который
# раскладывает запрос на список подзадач, и передаём этот список как
# ориентир в промпт выбора инструмента на каждом раунде.
TOOL_PLANNING_ENABLED = True
TOOL_PLANNING_MIN_LEN = 140
MAX_SUBTASKS = 4

# -------------------------------
# Параллельное выполнение инструментов внутри одного ReAct-раунда — пункт №5
# -------------------------------
# Если модель одним раундом решила вызвать несколько независимых
# инструментов (например native tool_calls с 2-3 вызовами), они раньше
# выполнялись строго по очереди (await в цикле) — впустую тратя время на
# ожидание, если инструменты не зависят друг от друга. TOOL_PARALLEL_EXECUTION
# включает asyncio.gather для вызовов одного раунда.
TOOL_PARALLEL_EXECUTION = True

# -------------------------------
# Автоинжекция памяти в промпт браузерного чата (ai_assistant.py)
# -------------------------------
# router.retrieve() всегда возвращает top_k результатов (ближайшие соседи по
# эмбеддингам почти всегда находятся, даже для нерелевантных сообщений вроде
# "нарисуй кота"). Без порога эти слаборелевантные факты безусловно попадали
# в блок "КОНТЕКСТ ИЗ ДОЛГОСРОЧНОЙ ПАМЯТИ", который системный промпт называет
# самым надёжным источником — отсюда "путаница" в чате, которой нет в MCP
# (там recall — явный инструмент, вызывается по решению внешнего клиента, а
# не подмешивается в каждое сообщение). Ниже этого порога факт всё ещё
# участвует в predict_next/uncertainty/working memory, но не попадает в текст
# промпта. Подберите под свою модель эмбеддингов, если понадобится.
MEMORY_CONTEXT_MIN_SCORE = 0.15