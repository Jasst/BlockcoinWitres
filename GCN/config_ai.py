"""
Централизованная конфигурация для всей AI-системы (когнитивная архитектура + GCN)
"""
from pathlib import Path

# -------------------------------
# Общие пути
# -------------------------------
MEMORY_BASE_DIR = Path("ai_memory_v3")
MEMORY_BASE_DIR.mkdir(exist_ok=True)

# -------------------------------
# LLM (LM Studio)
# -------------------------------
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_TIMEOUT = 160
LM_STUDIO_STREAM_TIMEOUT = 500
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

# -------------------------------
# GCN-специфичные параметры
# -------------------------------
GCN_STATE_FILENAME = "gcn_state.json"  # имя файла для сохранения состояния
GCN_AUTO_VERIFY = True                 # автоматически проверять противоречия при связывании
GCN_EVIDENCE_THRESHOLD = 0.6           # минимальное доверие для использования как evidence

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
# Команды управления памятью
# -------------------------------
MEMORY_CONTROL_COMMANDS = {
    "запомни": "store",
    "забудь": "forget",
    "что ты знаешь о": "recall"
}

# -------------------------------
# Генерация изображений
# -------------------------------
EASYDIFFUSION_ENABLED = True
EASYDIFFUSION_URL = "http://localhost:9000"
EASYDIFFUSION_TIMEOUT = 120
EASYDIFFUSION_DEFAULT_STEPS = 20
EASYDIFFUSION_DEFAULT_WIDTH = 512
EASYDIFFUSION_DEFAULT_HEIGHT = 512

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
REFLECTION_INTERVAL = 3600 * 4          # запуск рефлексии каждые 4 часа
REFLECTION_ERROR_THRESHOLD = 0.6        # ошибка выше этого значения считается значимой
REFLECTION_HISTORY_SIZE = 100           # сколько последних предсказаний хранить
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
MEMORY_USE_EMBEDDINGS = True
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
FAISS_NLIST = 200
FAISS_NPROBE = 30
FAISS_REBUILD_THRESHOLD = 300
FAISS_MIN_TRAIN_VECTORS = 500

# -------------------------------
# Гибридный поиск (memory_graph.retrieve_hybrid)
# Отдельная схема весов от HYBRID_WEIGHT_SEMANTIC/GRAPH/... выше (те теперь
# реально используются в GCN.hybrid_retrieve) — эта используется
# для BM25/cosine/freshness/graph ранжирования по фактам в CognitiveMemory.
# -------------------------------
HYBRID_WEIGHT_BM25 = 0.25
HYBRID_WEIGHT_COSINE = 0.40
FACTUAL_WEIGHTS = (0.35, 0.30, 0.15, 0.20)   # (bm25, cosine, freshness, graph) для запросов с числами/единицами
GENERAL_WEIGHTS = (HYBRID_WEIGHT_BM25, HYBRID_WEIGHT_COSINE, HYBRID_WEIGHT_FRESHNESS, HYBRID_WEIGHT_GRAPH)

# -------------------------------
# FAISS адаптивные пороги
# -------------------------------
FAISS_SMALL_THRESHOLD = 50      # при числе векторов < 50 – точный поиск
FAISS_MEDIUM_THRESHOLD = 500    # при числе < 500 – HNSW, иначе IVF
FAISS_HNSW_EF_CONSTRUCTION = 80
FAISS_HNSW_M = 32