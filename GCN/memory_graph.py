"""
Когнитивная память: семантическая, эпизодическая, ассоциативный граф с Hebbian/STDP,
spreading activation, predictive transitions, противоречия, консолидация, replay.
Дополнительно интегрирован GCN (Global Cognitive Network) как основное хранилище.
Здесь CognitiveMemory – тонкая обёртка над MemoryStore, обеспечивающая совместимость
с существующим интерфейсом (ai_assistant.py) и синхронизацию кэшей.
"""

import json
import logging
import time
import re
import asyncio
import math
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Set, Optional, Any, Tuple, DefaultDict
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss
import uuid

# Импорт GCN-компонентов (папка GCN, файл GCN.py)
from GCN.GCN import (
    KnowledgeObject, KnowledgeType, KnowledgeEvent, EventType,
    MemoryStore, KnowledgeGraph as GCNKnowledgeGraph,
    AIAdapter, Provenance, MemoryHierarchy,
    MemoryScope, KnowledgeIngestion   # добавить эти два
)

logger = logging.getLogger(__name__)

try:
    from GCN.config_ai import *
except ImportError:
    # fallback значения (критические)
    WORKING_MEMORY_SIZE = 20
    SENSORY_BUFFER_SIZE = 5
    EPISODIC_MAX_SIZE = 500
    SEMANTIC_MAX_FACTS = 10000
    ASSOCIATIVE_GRAPH_MAX_NODES = 20000
    DEFAULT_IMPORTANCE = 1.0
    DEFAULT_CONFIDENCE = 0.5
    DEFAULT_NOVELTY = 0.0
    DEFAULT_SALIENCE = 0.0
    DEFAULT_STABILITY = 0.5
    DEFAULT_PLASTICITY = 0.5
    DEFAULT_PREDICTION_ERROR = 0.0
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
    SPREADING_MAX_DEPTH = 3
    SPREADING_MAX_NODES = 50
    SPREADING_DECAY = 0.5
    SPREADING_THRESHOLD = 0.05
    PREDICTIVE_MATRIX_MAX_SIZE = 5000
    PREDICTIVE_LEARNING_RATE = 0.1
    PREDICTION_ERROR_THRESHOLD = 0.3
    CONSOLIDATION_INTERVAL = 7200
    DEEP_CONSOLIDATION_INTERVAL = 28800
    REPLAY_BATCH_SIZE = 20
    REPLAY_MIX_RATIO = (0.4, 0.3, 0.2, 0.1)
    MEMORY_USE_EMBEDDINGS = True
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
    EMBEDDING_QUERY_PREFIX = "query: "
    EMBEDDING_PASSAGE_PREFIX = "passage: "
    FAISS_NLIST = 200
    FAISS_NPROBE = 30
    FAISS_REBUILD_THRESHOLD = 300
    FAISS_MIN_TRAIN_VECTORS = 500
    HYBRID_WEIGHT_BM25 = 0.25
    HYBRID_WEIGHT_COSINE = 0.40
    HYBRID_WEIGHT_FRESHNESS = 0.15
    HYBRID_WEIGHT_GRAPH = 0.20
    DYNAMIC_WEIGHTS_ENABLED = True
    FACTUAL_WEIGHTS = (0.35, 0.30, 0.15, 0.20)
    GENERAL_WEIGHTS = (HYBRID_WEIGHT_BM25, HYBRID_WEIGHT_COSINE, HYBRID_WEIGHT_FRESHNESS, HYBRID_WEIGHT_GRAPH)
    MEMORY_CACHE_TTL = 60
    MEMORY_CACHE_MAX_SIZE = 1000
    DUPLICATE_SIMILARITY_THRESHOLD = 0.92
    GCN_STATE_FILENAME = "gcn_state.json"
    GCN_AUTO_VERIFY = True
    GCN_EVIDENCE_THRESHOLD = 0.6
    HYBRID_WEIGHT_SEMANTIC = 0.40
    HYBRID_WEIGHT_CONFIDENCE = 0.05
    HYBRID_WEIGHT_EVIDENCE = 0.10
    CONCEPT_MIN_CLUSTER_SIZE = 3
    CONCEPT_SIMILARITY_THRESHOLD = 0.6
    CONCEPT_MAX_SCAN = 2000
    CONCEPT_MAX_PER_RUN = 5
    CROSS_LAYER_GROUNDING_THRESHOLD = 0.75
    RERANK_ENABLED = True
    RERANK_CANDIDATE_MULTIPLIER = 3
    RERANK_MAX_CANDIDATES = 20


# =====================================================================
# ДАТАКЛАССЫ (заглушки для совместимости с ai_assistant.py)
# =====================================================================
@dataclass
class Fact:
    """Семантический факт (заглушка, реальные данные в GCN)."""
    id: int
    text: str
    type: str
    timestamp: float
    keywords: List[str]
    importance: float = DEFAULT_IMPORTANCE
    confidence: float = DEFAULT_CONFIDENCE
    novelty: float = DEFAULT_NOVELTY
    salience: float = DEFAULT_SALIENCE
    stability: float = DEFAULT_STABILITY
    plasticity: float = DEFAULT_PLASTICITY
    prediction_error: float = DEFAULT_PREDICTION_ERROR
    access_count: int = 0
    last_accessed: float = 0.0
    activation: float = 0.0
    contradicts: Set[int] = field(default_factory=set)
    gcn_id: Optional[str] = None


@dataclass
class Synapse:
    """Синаптическая связь (заглушка)."""
    source_id: int
    target_id: int
    weight: float = SYNAPSE_INITIAL_WEIGHT
    last_activation: float = 0.0
    plasticity: float = 0.5
    confidence: float = 0.5
    coactivation_count: int = 0
    last_coactivation: float = 0.0
    pre_time: float = 0.0
    post_time: float = 0.0


@dataclass
class Episode:
    """Эпизод (заглушка)."""
    id: int
    user_msg: str
    assistant_msg: str
    timestamp: float
    importance: float = 1.0
    salience: float = 0.0
    prediction_error: float = 0.0
    accessed_count: int = 0
    # ДОБАВЛЕНО (см. deep_consolidation): раньше Episode не хранил, каким
    # именно локальным Fact соответствуют его реплики — deep_consolidation
    # находил их линейным сканированием ВСЕХ semantic_facts по точному
    # совпадению текста (без фильтра по типу), на каждый эпизод из
    # REPLAY_BATCH_SIZE при каждом deep_consolidation. Два последствия:
    # (1) O(n) скан на эпизод вместо O(1) — при большом числе фактов заметно
    # медленнее; (2) при повторяющемся тексте сообщения ("привет", "спасибо")
    # или при совпадении с фактом другого типа (например, извлечённым из
    # этого же сообщения глобальным фактом) брался facts[0] — произвольный, не
    # обязательно тот факт, что реально относится к этому эпизоду. Теперь ID
    # проставляются напрямую при создании (add_episode) или один раз при
    # восстановлении из GCN (_rebuild_caches_from_gcn), и Hebbian-обновление
    # обращается к facts_by_id за O(1). None — для старых данных без этих
    # полей; тогда используется прежний textual fallback (см. deep_consolidation).
    user_fact_id: Optional[int] = None
    assistant_fact_id: Optional[int] = None


@dataclass
class Goal:
    """Цель (заглушка)."""
    id: int
    description: str
    priority: float = 0.5
    confidence: float = 0.5
    progress: float = 0.0
    deadline: Optional[float] = None
    related_memory: List[int] = field(default_factory=list)
    dependencies: List[int] = field(default_factory=list)
    status: str = "active"
    created_at: float = field(default_factory=time.time)
    gcn_id: Optional[str] = None  # <--- добавить


# =====================================================================
# ОСНОВНОЙ КЛАСС КОГНИТИВНОЙ ПАМЯТИ (обёртка над GCN)
# =====================================================================
class CognitiveMemory:
    """
    Когнитивная память, использующая GCN (MemoryStore) как единственное хранилище.
    Кэши (semantic_facts, episodic_memory, graph, synapses, goals) синхронизируются
    с GCN для быстрого доступа и совместимости с ai_assistant.py.
    """

    # ---- Общий на процесс кэш моделей эмбеддингов ----
    # Раньше каждый CognitiveMemory (а это один инстанс на КАЖДОГО
    # пользователя + ещё global + shared синглтоны) грузил свой собственный
    # SentenceTransformer("all-mpnet-base-v2") — это ~420МБ весов и
    # инициализация модели на КАЖДОГО активного юзера одновременно, хотя
    # сама модель не имеет пользовательского состояния и полностью
    # потокобезопасна для .encode(). Теперь модель грузится один раз на
    # имя (EMBEDDING_MODEL) и переиспользуется всеми инстансами.
    _embedder_cache: Dict[str, "SentenceTransformer"] = {}
    _embedder_cache_lock = None  # инициализируется лениво (threading.Lock)

    @classmethod
    def _get_shared_embedder(cls, model_name: str) -> "SentenceTransformer":
        if cls._embedder_cache_lock is None:
            import threading
            cls._embedder_cache_lock = threading.Lock()
        with cls._embedder_cache_lock:
            embedder = cls._embedder_cache.get(model_name)
            if embedder is None:
                embedder = SentenceTransformer(model_name)
                cls._embedder_cache[model_name] = embedder
            return embedder

    def __init__(self, user_id: str, base_dir: Path):
        self.user_id = user_id
        self.base_dir = base_dir / user_id / "cognitive_memory"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # ---- Эмбеддинги (для генерации векторов) ----
        # ВАЖНО: инициализируем эмбеддер ДО MemoryStore, чтобы передать туда
        # реальную размерность вектора. Раньше MemoryStore() создавался с
        # захардкоженным EMBEDDING_DIM=128 из конфига, а
        # all-mpnet-base-v2 отдаёт 768 — из-за этого FAISS-индекс в GCN
        # никогда не строился на реальных векторах (см. комментарий в
        # MemoryStore.__init__ в GCN.py).
        self.use_embeddings = MEMORY_USE_EMBEDDINGS
        if self.use_embeddings:
            try:
                self.embedder = self._get_shared_embedder(EMBEDDING_MODEL)
                self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
            except Exception as e:
                logger.error(f"Embeddings init failed: {e}. Disabling.")
                self.use_embeddings = False
                self.embedder = None
                self.embedding_dim = 0
        else:
            self.embedder = None
            self.embedding_dim = 0

        # ---- GCN-слой (основное хранилище) ----
        self.gcn_store = MemoryStore(embedding_dim=self.embedding_dim or None)
        self.hierarchy = MemoryHierarchy(self.gcn_store, size=WORKING_MEMORY_SIZE)

        gcn_state_path = self.base_dir / GCN_STATE_FILENAME
        if gcn_state_path.exists():
            try:
                self.gcn_store.load(str(gcn_state_path))
                logger.info("GCN state loaded")
            except Exception as e:
                logger.warning(f"Failed to load GCN state: {e}")

        # ---- Кэши (синхронизируются с GCN) ----
        self.semantic_facts: List[Fact] = []
        self.facts_by_id: Dict[int, Fact] = {}
        self.episodic_memory: List[Episode] = []
        self.graph: DefaultDict[int, Set[int]] = defaultdict(set)
        self.synapses: Dict[Tuple[int, int], Synapse] = {}
        self.keyword_index: DefaultDict[str, List[int]] = defaultdict(list)
        self.goals: List[Goal] = []
        self._next_fact_id = 0
        self._next_episode_id = 0
        self._next_goal_id = 0

        # ---- Прогностическая модель (локально) ----
        self.predictive_matrix: DefaultDict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.concept_examples: Dict[str, str] = {}
        self.prediction_cache: Dict[str, List[int]] = {}

        # ---- Динамические веса (для гибридного поиска) ----
        self._dynamic_weights = {
            "bm25": HYBRID_WEIGHT_BM25,
            "cosine": HYBRID_WEIGHT_COSINE,
            "freshness": HYBRID_WEIGHT_FRESHNESS,
            "graph": HYBRID_WEIGHT_GRAPH,
            "semantic": HYBRID_WEIGHT_SEMANTIC,  # добавить
            "confidence": HYBRID_WEIGHT_CONFIDENCE,  # добавить
            "evidence": HYBRID_WEIGHT_EVIDENCE,  # добавить
        }

        # ---- Вспомогательные структуры ----
        self._lock = asyncio.Lock()
        self._dirty = False
        self._save_task = None
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = MEMORY_CACHE_TTL
        self._cache_maxsize = MEMORY_CACHE_MAX_SIZE

        # Строим кэши из GCN
        self._rebuild_caches_from_gcn()
        self._rebuild_predictive_from_episodes()

        logger.info(f"CognitiveMemory (GCN-backed) initialized for {user_id[:16]}")

    # ==================== СВОЙСТВО ДЛЯ AIAdapter ====================
    @property
    def store(self) -> MemoryStore:
        return self.gcn_store

    # ==================== ОБНОВЛЕНИЕ ИЗ ДРУГОГО ПРОЦЕССА ====================
    def reload_if_stale(self) -> bool:
        """
        Перечитывает состояние с диска, если его успел изменить другой
        процесс (например, MCP-сервер запущен отдельным процессом от
        основного FastAPI-приложения — см. mcp_server_blockcoin.py — и у
        каждого процесса своя копия в памяти).

        Полезно вызывать перед операциями чтения (recall и т.п.), чтобы не
        отдавать заведомо устаревшие данные. Возвращает True, если
        состояние было перечитано.
        """
        gcn_state_path = self.base_dir / GCN_STATE_FILENAME
        if not gcn_state_path.exists():
            return False
        try:
            disk_mtime = gcn_state_path.stat().st_mtime
        except OSError:
            return False
        if self.gcn_store._loaded_mtime is not None and disk_mtime <= self.gcn_store._loaded_mtime:
            return False
        try:
            self.gcn_store.load(str(gcn_state_path))
            self._rebuild_caches_from_gcn()
            self._rebuild_predictive_from_episodes()
            logger.info(f"[{self.user_id[:16]}] Память перечитана с диска (изменена другим процессом)")
            return True
        except Exception as e:
            logger.warning(f"reload_if_stale failed: {e}")
            return False

    # ==================== ПОСТРОЕНИЕ КЭШЕЙ ИЗ GCN ====================
    def _rebuild_caches_from_gcn(self):
        self.semantic_facts = []
        self.facts_by_id = {}
        self.keyword_index.clear()
        max_id = 0
        for obj in self.gcn_store.get_all_facts(self.user_id):
            meta = obj.object if isinstance(obj.object, dict) else {}
            fid = meta.get("local_id")
            if fid is None:
                fid = max_id + 1
            max_id = max(max_id, fid)
            fact = Fact(
                id=fid,
                text=obj.subject,
                type=meta.get("fact_type", "unknown"),
                timestamp=obj.created.timestamp(),
                keywords=list(self._extract_keywords(obj.subject)),
                importance=meta.get("importance", DEFAULT_IMPORTANCE),
                confidence=obj.confidence,
                novelty=meta.get("salience", DEFAULT_NOVELTY),
                salience=meta.get("salience", DEFAULT_SALIENCE),
                stability=meta.get("stability", DEFAULT_STABILITY),
                plasticity=meta.get("plasticity", DEFAULT_PLASTICITY),
                prediction_error=meta.get("prediction_error", DEFAULT_PREDICTION_ERROR),
                access_count=meta.get("access_count", 0),
                last_accessed=meta.get("last_accessed", 0.0),
                gcn_id=obj.id
            )
            self.semantic_facts.append(fact)
            self.facts_by_id[fid] = fact
            for kw in fact.keywords:
                self.keyword_index[kw].append(fid)
        # --- НОВОЕ: заполняем противоречия из графа ---
        for f in self.semantic_facts:
            if f.gcn_id:
                neighbors = self.gcn_store._graph.get_neighbors(f.gcn_id, "CONTRADICTS")
                for _, target_id in neighbors:
                    target_fact = self._find_fact_by_gcn_id(target_id)
                    if target_fact:
                        f.contradicts.add(target_fact.id)

        self._next_fact_id = max_id + 1

        self.episodic_memory = []
        # Восстанавливаем эпизоды из GCN
        # ДОБАВЛЕНО: индекс (текст, тип) -> id локального факта, строится один
        # раз за весь rebuild (а не по одному скану на эпизод в deep_consolidation,
        # см. Episode.user_fact_id/assistant_fact_id) — используется ниже, чтобы
        # восстановленные из GCN эпизоды тоже получили привязку к фактам.
        text_type_to_fid: Dict[Tuple[str, str], int] = {}
        for f in self.semantic_facts:
            if f.type in ('user', 'assistant'):
                text_type_to_fid.setdefault((f.text, f.type), f.id)

        user_events = [obj for obj in self.gcn_store.get_all_episodes(self.user_id) if obj.predicate == "user_message"]
        for user_obj in user_events:
            neighbors = self.gcn_store._graph.get_neighbors(user_obj.id, "replied_with")
            for rel, ass_id in neighbors:
                ass_obj = self.gcn_store.get(ass_id)
                if ass_obj and ass_obj.type == KnowledgeType.MEMORY_EVENT and ass_obj.predicate == "assistant_message":
                    episode = Episode(
                        id=self._next_episode_id,
                        user_msg=user_obj.subject,
                        assistant_msg=ass_obj.subject,
                        timestamp=user_obj.created.timestamp(),
                        salience=user_obj.object.get("salience", 0.0) if isinstance(user_obj.object, dict) else 0.0,
                        user_fact_id=text_type_to_fid.get((user_obj.subject, 'user')),
                        assistant_fact_id=text_type_to_fid.get((ass_obj.subject, 'assistant')),
                    )
                    self.episodic_memory.append(episode)
                    self._next_episode_id += 1
                    break
        self.episodic_memory.sort(key=lambda e: e.timestamp)

        self.graph.clear()
        self.synapses.clear()
        for src_id, edges in self.gcn_store._graph._edges.items():
            src_fact = self._find_fact_by_gcn_id(src_id)
            if not src_fact:
                continue
            src_int = src_fact.id
            for rel, tgt_id in edges:
                tgt_fact = self._find_fact_by_gcn_id(tgt_id)
                if not tgt_fact:
                    continue
                tgt_int = tgt_fact.id
                weight = self.gcn_store._graph.get_relation_weight(src_id, rel, tgt_id) or 0.5
                if rel == "synapse":
                    key = (src_int, tgt_int)
                    if key not in self.synapses:
                        self.synapses[key] = Synapse(
                            source_id=src_int,
                            target_id=tgt_int,
                            weight=weight,
                            last_activation=time.time(),
                            plasticity=0.5,
                            confidence=0.5
                        )
                    self.graph[src_int].add(tgt_int)

        self.goals = []
        max_goal_id = 0
        for obj in self.gcn_store.get_active_goals(self.user_id):
            meta = obj.object if isinstance(obj.object, dict) else {}
            gid = max_goal_id + 1
            goal = Goal(
                id=gid,
                description=obj.subject,
                priority=meta.get("priority", 0.5),
                confidence=obj.confidence,
                progress=meta.get("progress", 0.0),
                status=meta.get("status", "active"),
                created_at=obj.created.timestamp(),
                gcn_id = obj.id

            )
            self.goals.append(goal)
            max_goal_id = max(max_goal_id, gid)
        self._next_goal_id = max_goal_id + 1
        self._next_episode_id = len(self.episodic_memory) + 1

    def _find_fact_by_gcn_id(self, gcn_id: str) -> Optional[Fact]:
        for f in self.semantic_facts:
            if f.gcn_id == gcn_id:
                return f
        return None

    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================
    @staticmethod
    def _extract_keywords(text: str) -> Set[str]:
        words = re.findall(r'\b[a-zA-Zа-яА-ЯёЁ]{3,}\b', text.lower())
        stopwords = {'это', 'все', 'так', 'вот', 'да', 'нет', 'или', 'и', 'с', 'на', 'по', 'для', 'из', 'о', 'к', 'у',
                     'же', 'бы', 'то', 'не', 'что', 'как', 'за', 'от', 'до', 'при', 'через', 'без', 'между', 'тоже',
                     'также', 'очень', 'можно', 'нужно', 'будет', 'если', 'тогда', 'потом', 'который', 'какой'}
        return {w for w in words if w not in stopwords}

    @staticmethod
    def _compute_similarity(text1: str, text2: str) -> float:
        kw1 = CognitiveMemory._extract_keywords(text1)
        kw2 = CognitiveMemory._extract_keywords(text2)
        if not kw1 or not kw2:
            return 0.0
        return len(kw1 & kw2) / (len(kw1 | kw2) + 1e-6)

    def _get_embedding(self, text: str, is_query: bool = False) -> np.ndarray:
        # ИСПРАВЛЕНИЕ (пункт №1): модели семейства e5 (в т.ч. новый дефолт
        # EMBEDDING_MODEL="intfloat/multilingual-e5-large") обучены с
        # асимметричными префиксами — текст запроса и текст сохраняемого
        # факта/пассажа должны эмбеддиться по-разному, иначе теряется
        # заметная часть качества поиска, ради которой модель вообще
        # переключалась. Для моделей, не нуждающихся в префиксах, обе
        # константы в конфиге можно оставить пустыми строками — тогда
        # поведение не меняется.
        if not self.use_embeddings or self.embedder is None:
            return np.zeros(self.embedding_dim if self.embedding_dim else 128)
        prefix = EMBEDDING_QUERY_PREFIX if is_query else EMBEDDING_PASSAGE_PREFIX
        text_to_encode = f"{prefix}{text}" if prefix else text
        return self.embedder.encode(text_to_encode, convert_to_numpy=True)

    def embed_text(self, text: str, is_query: bool = False) -> Optional[List[float]]:
        """
        Публичная обёртка над _get_embedding() для внешних вызывающих
        (GCNMemoryRouter). Возвращает None, если эмбеддинги отключены —
        вызывающий код должен уметь работать без вектора (fallback на
        keyword-поиск), а не падать.

        is_query=True — использовать префикс запроса (EMBEDDING_QUERY_PREFIX)
        вместо префикса пассажа/факта; передавайте True для текста, по
        которому идёт поиск (semantic_search и т.п.), False (по умолчанию)
        — для текста, который сохраняется как знание/концепт.
        """
        if not self.use_embeddings or self.embedder is None:
            return None
        return self._get_embedding(text, is_query=is_query).tolist()

    # ==================== ДОБАВЛЕНИЕ ФАКТОВ ====================
    def _add_fact(self, text: str, ftype: str, importance: float = 1.0,
                  confidence: float = 0.5, novelty: float = 0.0,
                  salience: float = 0.0) -> int:
        """Добавляет факт в GCN и кэш, возвращает целочисленный id."""
        fid = self._next_fact_id
        self._next_fact_id += 1

        emb = None
        if self.use_embeddings:
            emb = self._get_embedding(text).tolist()

        gcn_id = self.gcn_store.add_fact(
            text=text,
            fact_type=ftype,
            author=self.user_id,
            confidence=confidence,
            importance=importance,
            embedding=emb,
            local_id=fid
        )

        # --- НОВОЕ: запись доступа и пересчёт confidence ---
        self.gcn_store.record_access(gcn_id, self.user_id)
        updated_conf = self.gcn_store.compute_confidence(gcn_id)
        if updated_conf != confidence:
            self.gcn_store.update(gcn_id, {"confidence": updated_conf}, self.user_id)

        fact = Fact(
            id=fid,
            text=text,
            type=ftype,
            timestamp=time.time(),
            keywords=list(self._extract_keywords(text)),
            importance=importance,
            confidence=updated_conf,  # используем обновлённое значение
            novelty=novelty,
            salience=salience,
            stability=0.5,
            plasticity=0.5,
            prediction_error=0.0,
            gcn_id=gcn_id
        )
        self.semantic_facts.append(fact)
        self.facts_by_id[fid] = fact
        for kw in fact.keywords:
            self.keyword_index[kw].append(fid)

        # Поиск похожих для создания синапсов
        if emb is not None:
            similar = self._find_similar_by_embedding(np.array(emb), k=20, exclude_id=fid)
        else:
            similar = self._find_similar_keyword(fid, top_k=20)

        # ИСПРАВЛЕНИЕ (серьёзный баг синаптического графа): оба хелпера выше
        # (_find_similar_by_embedding/_find_similar_keyword) возвращают пары
        # (fact.id, sim) — то есть ID факта, а НЕ индекс в списке
        # self.semantic_facts. Код ниже раньше трактовал это значение как
        # индекс списка (`self.semantic_facts[other_idx]`) и даже содержал
        # проверку "пропустить последний индекс" — по всей видимости, попытку
        # исключить только что добавленный факт себя, в предположении, что он
        # всегда последний в списке.
        #
        # Пока список только пополняется и ID совпадает с позицией (свежая
        # сессия, ни одного удаления), баг случайно не проявлялся. Но:
        #   - после удаления любого факта (light_consolidation/deep_consolidation
        #     чистят дубликаты, есть explicit forget) список сдвигается, и
        #     ID перестаёт совпадать с индексом для всех фактов после удалённого;
        #   - после перезапуска процесса кэш восстанавливается в
        #     _rebuild_caches_from_gcn() в порядке итерации GCN-хранилища,
        #     который не гарантированно совпадает с порядком по ID.
        # В обоих случаях self.semantic_facts[other_idx] тихо возвращал СОВСЕМ
        # ДРУГОЙ факт (не IndexError, а неверный результат) — синапсы Hebbian/
        # STDP создавались между случайными парами фактов, а
        # _detect_contradictions проверяла противоречие не с теми кандидатами.
        # Это без исключений «работало», просто наполняло ассоциативный граф
        # шумом — отсюда и вопрос, есть ли слабо работающие места.
        #
        # Исправление: искать факт по ID через facts_by_id (O(1), как и
        # везде в остальном файле), а не по индексу списка.
        similar_ids = []
        for other_id, sim in similar:
            other = self.facts_by_id.get(other_id)
            if other is None or other_id == fid:
                continue
            similar_ids.append(other.id)
            if sim > 0.45:
                self._create_synapse(fid, other.id, weight=sim * 0.5)
                self._create_synapse(other.id, fid, weight=sim * 0.5)

        self._detect_contradictions(fid, candidate_ids=similar_ids)
        self._dirty = True
        return fid

    def get_working_memory(self) -> List[Dict]:
        """Возвращает объекты рабочей памяти в формате словарей (для ai_assistant)."""
        objects = self.hierarchy.get_working()
        result = []
        for obj in objects:
            if obj.type == KnowledgeType.CLAIM:
                meta = obj.object if isinstance(obj.object, dict) else {}
                fact = self._find_fact_by_gcn_id(obj.id)
                result.append({
                    "id": fact.id if fact else None,
                    "text": obj.subject,
                    "type": meta.get("fact_type", "unknown"),
                    "timestamp": obj.created.timestamp(),
                    "confidence": obj.confidence,
                    "importance": meta.get("importance", 1.0),
                    "gcn_id": obj.id,
                    "scope": obj.scope.value,
                })
        return result

    def _find_similar_by_embedding(self, emb: np.ndarray, k: int = 20,
                                   exclude_id: Optional[int] = None) -> List[Tuple[int, float]]:
        """Ищет похожие факты по эмбеддингу (локально, используя FAISS или косинус)."""
        if not self.semantic_facts or len(self.semantic_facts) < 2:
            return []
        # Используем GCN поиск для получения кандидатов
        # Но GCN требует вектора, а у нас numpy, преобразуем
        vec = emb.tolist()
        results = self.gcn_store.semantic_search(vec, top_k=k*3)
        # Сопоставляем с локальными id
        out = []
        for gcn_id, sim in results:
            fact = self._find_fact_by_gcn_id(gcn_id)
            if fact and fact.id != exclude_id:
                out.append((fact.id, sim))
        return out[:k]

    def _find_similar_keyword(self, fact_id: int, top_k: int = 20) -> List[Tuple[int, float]]:
        """Поиск похожих по ключевым словам (локально)."""
        fact = self.facts_by_id.get(fact_id)
        if not fact:
            return []
        candidates = []
        for other in self.semantic_facts:
            if other.id == fact_id:
                continue
            sim = self._compute_similarity(fact.text, other.text)
            if sim > 0:
                candidates.append((other.id, sim))
        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]

    def _create_synapse(self, src: int, tgt: int, weight: float = SYNAPSE_INITIAL_WEIGHT):
        key = (src, tgt)
        if key in self.synapses:
            syn = self.synapses[key]
            syn.weight = min(SYNAPSE_MAX_WEIGHT,
                             max(SYNAPSE_MIN_WEIGHT, syn.weight + HEBBIAN_LEARNING_RATE * (weight - syn.weight)))
        else:
            self.synapses[key] = Synapse(
                source_id=src,
                target_id=tgt,
                weight=max(SYNAPSE_MIN_WEIGHT, min(SYNAPSE_MAX_WEIGHT, weight)),
                last_activation=time.time(),
                plasticity=0.5,
                confidence=0.5
            )
        self.graph[src].add(tgt)
        # Синхронизация через GCN link
        fact_src = self.facts_by_id.get(src)
        fact_tgt = self.facts_by_id.get(tgt)
        if fact_src and fact_tgt and fact_src.gcn_id and fact_tgt.gcn_id:
            try:
                self.gcn_store.link(fact_src.gcn_id, fact_tgt.gcn_id, "synapse", self.user_id,
                                    weight=self.synapses[key].weight)
            except Exception as e:
                logger.debug(f"GCN link failed: {e}")
        self._dirty = True

    def _detect_contradictions(self, fact_id: int, candidate_ids: Optional[List[int]] = None):
        fact = self.facts_by_id.get(fact_id)
        if not fact:
            return
        neg_words = {'не', 'нет', 'без', 'против', 'отрицает', 'опровергает'}
        has_neg = any(w in fact.text.lower() for w in neg_words)
        if not has_neg or not candidate_ids:
            return
        fact_kw = set(fact.keywords)
        for other_id in candidate_ids:
            other = self.facts_by_id.get(other_id)
            if not other or other_id == fact_id:
                continue
            common = fact_kw & set(other.keywords)
            if len(common) > 0 and len(common) / max(1, len(fact_kw)) > 0.5:
                # Регистрируем противоречие через GCN
                try:
                    self.gcn_store.register_contradiction(fact.gcn_id, other.gcn_id, self.user_id)
                    # Обновляем локальные confidence
                    fact.confidence = self.gcn_store.compute_confidence(fact.gcn_id)
                    other.confidence = self.gcn_store.compute_confidence(other.gcn_id)
                except Exception as e:
                    logger.debug(f"Contradiction registration failed: {e}")

    # ==================== ВЕРИФИКАЦИЯ ПРОТИВОРЕЧИЙ ====================
    def get_unverified_contradictions(self, limit: int = 5) -> List[Tuple['Fact', 'Fact']]:
        """Возвращает пары фактов с необработанным (LLM-непроверенным) противоречием."""
        pairs = []
        seen = set()
        for f in self.semantic_facts:
            for other_id in f.contradicts:
                key = tuple(sorted((f.id, other_id)))
                if key in seen:
                    continue
                other = self.facts_by_id.get(other_id)
                if other:
                    seen.add(key)
                    pairs.append((f, other))
                if len(pairs) >= limit:
                    return pairs
        return pairs

    # ==================== УДАЛЕНИЕ ФАКТОВ ====================
    def _remove_facts(self, ids: Set[int]) -> int:
        """Удаляет факты из GCN и кэша, возвращает количество удалённых."""
        removed = 0
        for fid in list(ids):
            fact = self.facts_by_id.get(fid)
            if fact and fact.gcn_id:
                try:
                    if self.gcn_store.retract(fact.gcn_id, self.user_id, reason="user_delete"):
                        removed += 1
                    else:
                        continue
                except Exception as e:
                    logger.debug(f"GCN retract failed for {fact.gcn_id}: {e}")
                    continue
            # Удаляем из локальных структур
            if fid in self.facts_by_id:
                del self.facts_by_id[fid]
            # Удаляем из списка
            self.semantic_facts = [f for f in self.semantic_facts if f.id != fid]
            # Удаляем из keyword_index
            for kw, lst in self.keyword_index.items():
                if fid in lst:
                    lst.remove(fid)
            # Удаляем синапсы
            self.synapses = {(s, t): syn for (s, t), syn in self.synapses.items() if s != fid and t != fid}
            self.graph = defaultdict(set)
            for (src, tgt) in self.synapses:
                self.graph[src].add(tgt)
            # Удаляем из contradicts
            for f in self.semantic_facts:
                f.contradicts.discard(fid)
            self._dirty = True
        return removed

    # ==================== ЭПИЗОДЫ ====================
    async def add_episode(self, user_msg: str, assistant_msg: str, salience: float = 0.0):
        """Добавляет эпизод в GCN и локальный кэш."""
        user_id_gcn, ass_id_gcn = self.gcn_store.add_episode(user_msg, assistant_msg, self.user_id, salience)

        # Создаём локальные факты (как раньше) для совместимости
        user_fid = self._add_fact(user_msg, 'user', importance=1.0, salience=salience)
        assistant_fid = self._add_fact(assistant_msg, 'assistant', importance=1.2, salience=salience)
        self._create_synapse(user_fid, assistant_fid, weight=0.8)
        self._create_synapse(assistant_fid, user_fid, weight=0.6)

        # Обновляем предиктивную модель
        self._update_predictive(user_msg, assistant_msg)

        # Локальный эпизод (не храним отдельно, только для совместимости)
        episode = Episode(
            id=self._next_episode_id,
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            timestamp=time.time(),
            importance=1.0,
            salience=salience,
            user_fact_id=user_fid,
            assistant_fact_id=assistant_fid,
        )
        self._next_episode_id += 1
        self.episodic_memory.append(episode)
        if len(self.episodic_memory) > EPISODIC_MAX_SIZE:
            self.episodic_memory.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)
            self.episodic_memory = self.episodic_memory[:EPISODIC_MAX_SIZE]

        await self._schedule_save()

    # ==================== ПРЕДСКАЗАТЕЛЬНАЯ МОДЕЛЬ ====================
    def _concept_key(self, text: str, top_n: int = 5) -> Optional[str]:
        kws = self._extract_keywords(text)
        if not kws:
            return None
        top = sorted(kws, key=lambda w: (-len(w), w))[:top_n]
        return "|".join(sorted(top))

    def _update_predictive(self, source_text: str, target_text: str):
        src_key = self._concept_key(source_text)
        tgt_key = self._concept_key(target_text)
        if not src_key or not tgt_key:
            return
        self.predictive_matrix[src_key][tgt_key] += PREDICTIVE_LEARNING_RATE
        total = sum(self.predictive_matrix[src_key].values())
        if total > 0:
            for k in self.predictive_matrix[src_key]:
                self.predictive_matrix[src_key][k] /= total
        self.concept_examples[src_key] = source_text[:200]
        self.concept_examples[tgt_key] = target_text[:200]
        if len(self.predictive_matrix) > PREDICTIVE_MATRIX_MAX_SIZE:
            weakest = min(self.predictive_matrix.items(), key=lambda kv: sum(kv[1].values()))[0]
            del self.predictive_matrix[weakest]
            self.concept_examples.pop(weakest, None)

    def _rebuild_predictive_from_episodes(self):
        self.predictive_matrix.clear()
        self.concept_examples.clear()
        # Используем эпизоды из GCN (MEMORY_EVENT)
        episodes = self.gcn_store.get_all_episodes(self.user_id)
        # Сортируем по времени
        episodes.sort(key=lambda obj: obj.created)
        # Группируем по парам: пользователь -> ассистент
        # Для простоты берём подряд идущие: user -> assistant
        # Но структура GCN позволяет найти связи, здесь упростим
        # Лучше строить на основе локальных эпизодов, которые мы обновляем
        for ep in sorted(self.episodic_memory, key=lambda e: e.timestamp):
            self._update_predictive(ep.user_msg, ep.assistant_msg)

    async def predict_next(self, current_texts: List[str], top_k: int = 5) -> List[str]:
        candidates = defaultdict(float)
        for text in current_texts:
            key = self._concept_key(text)
            if key and key in self.predictive_matrix:
                for nxt_key, prob in self.predictive_matrix[key].items():
                    candidates[nxt_key] += prob
        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])[:top_k]
        return [self.concept_examples.get(k, k) for k, _ in sorted_candidates]

    # ==================== HEBBIAN / STDP ====================
    def _hebbian_update(self, source_id: int, target_id: int, coactivation_time: float):
        """Обновляет вес синапса source->target по правилу STDP/Hebbian.

        ИСПРАВЛЕНИЕ (мёртвые параметры конфигурации): STDP_LONG_TERM_POTENTIATION /
        STDP_LONG_TERM_DEPRESSION, SYNAPSE_PLASTICITY_RATE и
        SYNAPSE_COACTIVATION_THRESHOLD были объявлены в config_ai.py, но нигде не
        читались — обе ветки STDP использовали один и тот же STDP_LEARNING_RATE
        (депрессия отличалась только произвольным множителем 0.5), а
        Synapse.plasticity выставлялось при создании и больше никогда не менялось:
        метапластичности не было — часто подтверждаемые и совсем новые связи
        учились с одинаковой скоростью.
        Теперь: LTP и LTD используют раздельные константы; величина изменения
        масштабируется текущей пластичностью синапса; синапсы, накопившие много
        значимых совместных активаций, постепенно теряют пластичность
        (стабилизируются) — так базовые, многократно подтверждённые ассоциации не
        размываются каждым новым шумным сигналом, а действительно новые связи
        остаются пластичными и быстро обучаются.
        """
        key = (source_id, target_id)
        if key not in self.synapses:
            return
        syn = self.synapses[key]
        dt = coactivation_time - syn.last_activation
        plastic = max(0.05, syn.plasticity)
        if dt > 0 and dt < STDP_TIME_WINDOW:
            # Пост-событие следует за пре-событием в разумном окне — потенциация.
            delta = STDP_LONG_TERM_POTENTIATION * (1.0 - dt / STDP_TIME_WINDOW) * plastic
        elif dt < 0 and abs(dt) < STDP_TIME_WINDOW:
            # Обратный порядок — депрессия (в норме слабее и медленнее LTP).
            delta = -STDP_LONG_TERM_DEPRESSION * (1.0 - abs(dt) / STDP_TIME_WINDOW) * plastic
        else:
            delta = HEBBIAN_LEARNING_RATE * 0.1 * plastic
        syn.weight = min(SYNAPSE_MAX_WEIGHT, max(SYNAPSE_MIN_WEIGHT, syn.weight + delta))
        syn.last_activation = coactivation_time
        syn.coactivation_count += 1
        syn.last_coactivation = coactivation_time
        syn.confidence = min(1.0, syn.confidence + 0.01)
        # Метапластичность: значимое по модулю изменение постепенно "остужает" синапс.
        if abs(delta) >= SYNAPSE_COACTIVATION_THRESHOLD * STDP_LONG_TERM_POTENTIATION:
            syn.plasticity = max(0.05, syn.plasticity - SYNAPSE_PLASTICITY_RATE)
        self._sync_synapse_to_gcn(source_id, target_id)
        self._dirty = True

    def _get_effective_weight(self, syn: Synapse, now: Optional[float] = None) -> float:
        if now is None:
            now = time.time()
        age = now - syn.last_activation
        decay = 1.0 - SYNAPSE_DECAY_RATE * min(1.0, age / 86400)
        return max(SYNAPSE_MIN_WEIGHT, syn.weight * decay)

    def _apply_decay(self):
        now = time.time()
        changed = 0
        for (src, tgt), syn in self.synapses.items():
            age = now - syn.last_activation
            if age < 3600:
                continue
            decay = 1.0 - SYNAPSE_DECAY_RATE * min(1.0, age / 86400)
            new_weight = max(SYNAPSE_MIN_WEIGHT, syn.weight * decay)
            if abs(new_weight - syn.weight) > 1e-6:
                syn.weight = new_weight
                syn.confidence = max(0.1, syn.confidence * decay)
                self._sync_synapse_to_gcn(src, tgt)
                changed += 1
        if changed:
            self._dirty = True

    def _sync_synapse_to_gcn(self, src: int, tgt: int):
        syn = self.synapses.get((src, tgt))
        if not syn:
            return
        fact_src = self.facts_by_id.get(src)
        fact_tgt = self.facts_by_id.get(tgt)
        if fact_src and fact_tgt and fact_src.gcn_id and fact_tgt.gcn_id:
            try:
                self.gcn_store.set_relation_weight(
                    fact_src.gcn_id, fact_tgt.gcn_id, "synapse", syn.weight, self.user_id
                )
            except Exception as e:
                logger.debug(f"GCN synapse sync failed: {e}")

    # ==================== SPREADING ACTIVATION ====================
    async def spread_activation(self, seed_ids: List[int], max_depth: int = SPREADING_MAX_DEPTH,
                               max_nodes: int = SPREADING_MAX_NODES) -> Dict[int, float]:
        """Распространение активации по графу синапсов (локально)."""
        now = time.time()
        # Обнуляем активации
        for f in self.semantic_facts:
            f.activation = 0.0

        valid_seeds = [sid for sid in seed_ids if sid in self.facts_by_id]
        for sid in valid_seeds:
            self.facts_by_id[sid].activation = 1.0

        visited = set(valid_seeds)
        frontier = [(sid, 1.0, 0) for sid in valid_seeds]
        activation_map = {sid: 1.0 for sid in valid_seeds}

        while frontier and len(activation_map) < max_nodes:
            new_frontier = []
            for fid, act, depth in frontier:
                if depth >= max_depth:
                    continue
                for neighbor in self.graph.get(fid, set()):
                    if neighbor in visited:
                        continue
                    if neighbor not in self.facts_by_id:
                        continue
                    syn = self.synapses.get((fid, neighbor))
                    weight = self._get_effective_weight(syn, now) if syn else 0.5
                    new_act = act * weight * SPREADING_DECAY
                    if new_act < SPREADING_THRESHOLD:
                        continue
                    visited.add(neighbor)
                    activation_map[neighbor] = activation_map.get(neighbor, 0.0) + new_act
                    new_frontier.append((neighbor, new_act, depth + 1))
            frontier = new_frontier

        for fid, act in activation_map.items():
            if fid in self.facts_by_id:
                self.facts_by_id[fid].activation = act
        return activation_map

    # ==================== ГИБРИДНЫЙ ПОИСК ====================
    async def retrieve_hybrid(self, query: str, top_k: int = 5, use_graph: bool = True) -> List[Dict]:
        """
        Гибридный поиск через GCN с преобразованием результата в старый формат.
        Добавлена адаптивная корректировка весов в зависимости от типа запроса.
        """
        cache_key = f"hybrid_{query}_{top_k}_{use_graph}"
        if cache_key in self._cache:
            result, ts = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return result

        # Генерируем вектор запроса (is_query=True — асимметричный префикс e5)
        query_vector = None
        if self.use_embeddings:
            query_vector = self._get_embedding(query, is_query=True).tolist()

        start_node = self.hierarchy.working_memory[-1] if use_graph and self.hierarchy.working_memory else None

        # ===== АДАПТИВНАЯ КОРРЕКТИРОВКА ВЕСОВ =====
        weights = self._dynamic_weights.copy()

        # Определяем тип запроса
        if self._is_time_sensitive(query):
            # Временные запросы (курсы, новости) – повышаем свежесть
            weights['freshness'] = min(0.8, weights.get('freshness', 0.15) + 0.3)
            weights['semantic'] = max(0.2, weights.get('semantic', 0.40) - 0.1)
            weights['graph'] = max(0.1, weights.get('graph', 0.20) - 0.05)
        elif self._is_factual(query):
            # Фактологические запросы – повышаем семантику и граф
            weights['semantic'] = min(0.7, weights.get('semantic', 0.40) + 0.2)
            weights['graph'] = min(0.5, weights.get('graph', 0.20) + 0.1)
            weights['freshness'] = max(0.05, weights.get('freshness', 0.15) - 0.05)

        # Нормируем веса
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        # =========================================

        # Выполняем поиск в GCN с модифицированными весами
        gcn_results = self.gcn_store.hybrid_retrieve(
            query_vector=query_vector,
            start_node=start_node,
            top_k=top_k * 3,
            weights=weights
        )

        # Применяем spreading activation для графового буста
        activation_map: Dict[int, float] = {}
        if use_graph and self.hierarchy.working_memory:
            seed_local_ids = []
            for gcn_id in self.hierarchy.working_memory:
                fact = self._find_fact_by_gcn_id(gcn_id)
                if fact:
                    seed_local_ids.append(fact.id)
            if seed_local_ids:
                try:
                    activation_map = await self.spread_activation(seed_local_ids)
                except Exception as e:
                    logger.debug(f"spread_activation failed in retrieve_hybrid: {e}")

        result = []
        for obj in gcn_results:
            meta = obj.object if isinstance(obj.object, dict) else {}
            fid = None
            activation_boost = 0.0

            if obj.type == KnowledgeType.CLAIM:
                fact = self._find_fact_by_gcn_id(obj.id)
                if fact:
                    fid = fact.id
                    activation_boost = activation_map.get(fact.id, 0.0)
                text = obj.subject
                item_type = meta.get("fact_type", "claim")
                importance = meta.get("importance", 1.0)
            elif obj.type == KnowledgeType.MEMORY_EVENT:
                text = f"{obj.subject} → {obj.predicate}" if obj.predicate else obj.subject
                item_type = "episode"
                importance = meta.get("importance", 1.0)
            elif obj.type == KnowledgeType.HYPOTHESIS:
                text = obj.subject
                item_type = "goal"
                importance = meta.get("priority", 0.5)
            elif obj.type == KnowledgeType.CONCEPT:
                text = obj.subject
                item_type = "concept"
                importance = meta.get("importance", 1.2)
            else:
                text = obj.subject
                item_type = obj.type.value
                importance = meta.get("importance", 1.0)

            base_score = obj.confidence
            final_score = (min(1.0, base_score + HYBRID_WEIGHT_GRAPH * activation_boost)
                           if activation_boost else base_score)

            result.append({
                "id": fid,
                "text": text,
                "type": item_type,
                "timestamp": obj.created.timestamp(),
                "score": final_score,
                "confidence": obj.confidence,
                "importance": importance,
                "activation": activation_boost,
                "gcn_id": obj.id,
                "scope": obj.scope.value,
            })

        result.sort(key=lambda x: x["score"], reverse=True)
        result = result[:top_k]

        self._cache[cache_key] = (result, time.time())
        if len(self._cache) > self._cache_maxsize:
            oldest = sorted(self._cache.items(), key=lambda x: x[1][1])[0][0]
            del self._cache[oldest]
        return result

    def _is_time_sensitive(self, query: str) -> bool:
        """Эвристика для запросов, требующих актуальных данных."""
        time_markers = [
            'сегодня', 'сейчас', 'курс', 'погода', 'новости', 'свежие',
            'завтра', 'вчера', 'актуальные', 'последние', '2024', '2025', '2026'
        ]
        return any(m in query.lower() for m in time_markers)

    def _is_factual(self, query: str) -> bool:
        """Эвристика для фактических запросов с числами или единицами."""
        import re
        patterns = [
            r'\b\d+[.,]?\d*\s*(?:USD|EUR|RUB|₽|$|€|%|кг|км|г|м|см|мм|MB|GB|TB)\b',
            r'\b(?:курс|цена|стоимость|тариф|скорость|температура|вес|рост|расстояние)\b'
        ]
        return any(re.search(p, query, re.IGNORECASE) for p in patterns)

    def _sync_goal_from_gcn(self, gcn_id: str):
        """Обновляет локальный Goal по данным из GCN."""
        obj = self.gcn_store.get(gcn_id)
        if not obj or obj.type != KnowledgeType.HYPOTHESIS:
            return
        for g in self.goals:
            if g.gcn_id == gcn_id:
                g.description = obj.subject
                g.confidence = obj.confidence
                meta = obj.object if isinstance(obj.object, dict) else {}
                g.status = meta.get("status", g.status)
                g.priority = meta.get("priority", g.priority)
                g.progress = meta.get("progress", g.progress)
                break

    # ==================== КОНСОЛИДАЦИЯ ====================
    async def _find_duplicates_via_faiss(self, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD) -> Set[int]:
        """Находит дубликаты через FAISS (используя GCN векторы)."""
        # Используем GCN для поиска похожих, но проще локально
        if len(self.semantic_facts) < 2:
            return set()
        # Строим векторы из GCN
        vectors = []
        fact_ids = []
        for f in self.semantic_facts:
            vec = self.gcn_store.get_embedding(f.gcn_id)
            if vec:
                vectors.append(vec)
                fact_ids.append(f.id)
        if len(vectors) < 2:
            return set()
        vectors_np = np.array(vectors).astype('float32')
        norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
        vectors_norm = vectors_np / (norms + 1e-8)

        index = faiss.IndexFlatIP(vectors_norm.shape[1])
        index.add(vectors_norm)
        k = min(10, len(vectors_norm))
        _, idxs = index.search(vectors_norm, k)

        to_remove = set()
        for i, neighbors in enumerate(idxs):
            fi_id = fact_ids[i]
            if fi_id in to_remove:
                continue
            for j in neighbors[1:]:
                if j == -1 or j <= i:
                    continue
                fj_id = fact_ids[j]
                if fj_id in to_remove:
                    continue
                sim = float(vectors_norm[i] @ vectors_norm[j])
                if sim > threshold:
                    if self.facts_by_id[fi_id].confidence <= self.facts_by_id[fj_id].confidence:
                        to_remove.add(fi_id)
                    else:
                        to_remove.add(fj_id)
        return to_remove

    def _find_duplicates_keyword(self, threshold: float = 0.8) -> Set[int]:
        to_remove = set()
        n = len(self.semantic_facts)
        if n > 2000:
            return to_remove
        for i, f1 in enumerate(self.semantic_facts):
            if f1.id in to_remove:
                continue
            for j in range(i + 1, n):
                f2 = self.semantic_facts[j]
                if f2.id in to_remove:
                    continue
                if self._compute_similarity(f1.text, f2.text) > threshold:
                    if f2.confidence > f1.confidence:
                        to_remove.add(f1.id)
                        break
                    else:
                        to_remove.add(f2.id)
        return to_remove

    async def light_consolidation(self):
        async with self._lock:
            if self.use_embeddings and len(self.semantic_facts) >= 100:
                to_remove = await self._find_duplicates_via_faiss(DUPLICATE_SIMILARITY_THRESHOLD)
            else:
                to_remove = self._find_duplicates_keyword(0.8)

            removed = self._remove_facts(to_remove)
            self._apply_decay()
            await self._schedule_save()
            logger.info(f"Light consolidation done for {self.user_id[:16]}: removed {removed} duplicates")

    async def deep_consolidation(self):
        async with self._lock:
            # Обновляем веса синапсов на основе эпизодов (локально)
            if self.episodic_memory:
                self.episodic_memory.sort(key=lambda e: (e.importance * (1 + e.salience), e.timestamp), reverse=True)
                replay_candidates = self.episodic_memory[:REPLAY_BATCH_SIZE]
                for ep in replay_candidates:
                    # ИСПРАВЛЕНИЕ (см. Episode.user_fact_id/assistant_fact_id):
                    # раньше здесь на КАЖДЫЙ эпизод делался линейный скан всех
                    # semantic_facts по точному совпадению текста, вообще без
                    # фильтра по типу — при повторяющемся тексте сообщения
                    # ("привет", "спасибо") или совпадении с фактом другого типа
                    # брался facts[0], произвольный и не обязательно тот, что
                    # реально относится к этому эпизоду. Теперь сперва O(1)
                    # обращение по id; textual-скан остаётся только как fallback
                    # для эпизодов, восстановленных из данных без этих полей, и
                    # теперь ЯВНО фильтрует по типу факта.
                    user_fact = self.facts_by_id.get(ep.user_fact_id) if ep.user_fact_id is not None else None
                    ass_fact = self.facts_by_id.get(ep.assistant_fact_id) if ep.assistant_fact_id is not None else None
                    if user_fact is None:
                        user_fact = next((f for f in self.semantic_facts
                                           if f.text == ep.user_msg and f.type == 'user'), None)
                    if ass_fact is None:
                        ass_fact = next((f for f in self.semantic_facts
                                          if f.text == ep.assistant_msg and f.type == 'assistant'), None)
                    if user_fact and ass_fact:
                        # ИСПРАВЛЕНИЕ: Episode хранит один timestamp на весь эпизод,
                        # поэтому раньше оба вызова (user->assistant и assistant->user)
                        # получали ОДИНАКОВОЕ coactivation_time: dt всегда было равно 0,
                        # оба направления попадали в одну и ту же общую хеббовскую ветку,
                        # и направленная асимметрия STDP — усиление именно причинного
                        # направления "user сказал X -> assistant ответил Y", которое
                        # реально использует predict_next(), — никогда не проявлялась.
                        # Разносим "пре" (user) и "пост" (assistant) на 1 секунду —
                        # реальный порядок реплик внутри эпизода нам известен, даже без
                        # точных временных меток на каждую реплику в отдельности.
                        self._hebbian_update(user_fact.id, ass_fact.id, ep.timestamp + 1.0)
                        self._hebbian_update(ass_fact.id, user_fact.id, ep.timestamp)

            # Обновляем confidence и importance
            now = time.time()
            for f in self.semantic_facts:
                age = now - f.timestamp
                recency = 1.0 / (1.0 + age / 86400)
                f.importance = 0.5 * (f.importance + recency + f.access_count / 10)
                f.importance = min(2.0, f.importance)
                # Обновляем в GCN
                if f.gcn_id:
                    try:
                        self.gcn_store.update(f.gcn_id, {"confidence": f.confidence}, self.user_id)
                        # Обновляем метаданные importance
                        meta = self.gcn_store.get(f.gcn_id).object if self.gcn_store.get(f.gcn_id) else {}
                        if isinstance(meta, dict):
                            meta["importance"] = f.importance
                            self.gcn_store.update(f.gcn_id, {"object": meta}, self.user_id)
                    except Exception as e:
                        logger.debug(f"GCN update failed for {f.gcn_id}: {e}")

            # Ограничиваем количество фактов
            if len(self.semantic_facts) > SEMANTIC_MAX_FACTS:
                self.semantic_facts.sort(key=lambda f: (f.importance, f.confidence, f.timestamp), reverse=True)
                keep = self.semantic_facts[:SEMANTIC_MAX_FACTS]
                removed_ids = {f.id for f in self.semantic_facts[SEMANTIC_MAX_FACTS:]}
                self.semantic_facts = keep
                self._remove_facts(removed_ids)

            await self._schedule_save()
            logger.info(f"Deep consolidation done for {self.user_id[:16]}")

    # ==================== РАБОТА С ЦЕЛЯМИ ====================
    async def add_goal(self, description: str, priority: float = 0.5, related_memory: List[int] = None):
        gcn_id = self.gcn_store.add_goal(description, self.user_id, priority=priority)
        gid = self._next_goal_id
        self._next_goal_id += 1
        goal = Goal(
            id=gid,
            description=description,
            priority=priority,
            confidence=0.5,
            related_memory=related_memory or [],
            status='active',
            created_at=time.time(),
            gcn_id=gcn_id  # <--- добавить
        )
        self.goals.append(goal)
        await self._schedule_save()
        return gid

    async def update_goal(self, goal_id: int, **kwargs):
        # ищем локальную цель по id
        for g in self.goals:
            if g.id == goal_id:
                for k, v in kwargs.items():
                    if hasattr(g, k):
                        setattr(g, k, v)
                # обновляем в GCN по gcn_id
                if g.gcn_id:
                    obj = self.gcn_store.get(g.gcn_id)
                    if obj:
                        # обновляем confidence и статус
                        if 'confidence' in kwargs:
                            self.gcn_store.update(obj.id, {"confidence": g.confidence}, self.user_id)
                        if 'status' in kwargs:
                            meta = obj.object if isinstance(obj.object, dict) else {}
                            meta["status"] = g.status
                            self.gcn_store.update(obj.id, {"object": meta}, self.user_id)
                        # если нужно обновить другие поля, добавьте аналогично
                await self._schedule_save()
                return

    async def get_active_goals(self) -> List[Goal]:
        return [g for g in self.goals if g.status == 'active']

    # ==================== СОХРАНЕНИЕ ====================
    async def _save_async(self):
        async with self._lock:
            # --- НОВОЕ: перестраиваем FAISS индекс перед сохранением ---
            self.gcn_store.build_faiss_index(force=True)
            gcn_state_path = self.base_dir / GCN_STATE_FILENAME
            await self.gcn_store.async_save(str(gcn_state_path))
            # Сохраняем локальные счётчики (опционально)
            meta_path = self.base_dir / "meta.json"
            meta = {
                "next_fact_id": self._next_fact_id,
                "next_episode_id": self._next_episode_id,
                "next_goal_id": self._next_goal_id,
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            self._dirty = False

    async def _schedule_save(self):
        self._dirty = True
        if self._save_task is None or self._save_task.done():
            self._save_task = asyncio.create_task(self._periodic_save())

    async def _periodic_save(self):
        await asyncio.sleep(5)
        if self._dirty:
            await self._save_async()

    # ==================== СТАТИСТИКА ====================
    def get_stats(self) -> Dict:
        return {
            "semantic_facts": len(self.semantic_facts),
            "episodes": len(self.episodic_memory),
            "graph_edges": sum(len(v) for v in self.graph.values()) // 2,
            "synapses": len(self.synapses),
            "goals": len(self.goals),
            "active_goals": len([g for g in self.goals if g.status == 'active']),
            "working_memory": 0,  # не используется
            "faiss_trained": self.gcn_store.faiss_index is not None,
            "gcn_objects": len(self.gcn_store._objects),
        }

    # ==================== ЗАКРЫТИЕ ====================
    async def shutdown(self):
        if self._save_task:
            self._save_task.cancel()
        await self._save_async()

class GCNMemoryRouter:
    """
    Управляет тремя слоями памяти: личный (PRIVATE), общий (SHARED), глобальный (GLOBAL).
    Обеспечивает:
    - унифицированный поиск с учётом scope и весов
    - маршрутизацию добавления знаний (в зависимости от scope)
    - извлечение с ранжированием
    - инжекшн в глобальную память (дедупликация, агрегация, противоречия)
    - извлечение фактов из диалогов через LLM
    """
    _global_instance: Optional[CognitiveMemory] = None
    _shared_instance: Optional[CognitiveMemory] = None

    def __init__(self, user_id: str, base_dir: Path):
        self.user_id = user_id
        self.base_dir = base_dir

        # Личная память – всегда своя
        self.private_memory = CognitiveMemory(user_id, base_dir)

        # Глобальная память – единый экземпляр для всех (синглтон)
        self.global_memory = self._get_global_memory(base_dir)

        # Общая память – единый экземпляр для всех (можно расширить до групповой)
        self.shared_memory = self._get_shared_memory(base_dir)

        # Инжекторы дедупликации/агрегации/противоречий.
        # УЛУЧШЕНИЕ: раньше инжектор был только у глобальной памяти — PRIVATE
        # и SHARED никогда не дедуплицировались и не усиливались по смыслу
        # (только точное совпадение текста в GCNMemoryRouter.retrieve и
        # decay-дубликаты по эмбеддингам в light_consolidation). Теперь у
        # каждого слоя свой инжектор (KnowledgeIngestion теперь параметризован
        # по scope — см. GCN.py), и вся логика merge/reinforce/contradiction
        # доступна везде, не только в глобальной памяти.
        self.global_ingestion = KnowledgeIngestion(self.global_memory.store, scope=MemoryScope.GLOBAL)
        self.shared_ingestion = KnowledgeIngestion(self.shared_memory.store, scope=MemoryScope.SHARED)
        self.private_ingestion = KnowledgeIngestion(self.private_memory.store, scope=MemoryScope.PRIVATE)

        # Функция вызова LLM (будет установлена из контроллера)
        self._llm_caller = None

    @classmethod
    def _get_global_memory(cls, base_dir: Path) -> CognitiveMemory:
        """Возвращает глобальную память как синглтон."""
        if cls._global_instance is None:
            cls._global_instance = CognitiveMemory("global", base_dir)
        return cls._global_instance

    @classmethod
    def _get_shared_memory(cls, base_dir: Path) -> CognitiveMemory:
        """Возвращает общую (shared) память как синглтон."""
        if cls._shared_instance is None:
            cls._shared_instance = CognitiveMemory("shared", base_dir)
        return cls._shared_instance

    def set_llm_caller(self, llm_caller):
        """Передаёт функцию вызова LLM для извлечения фактов."""
        self._llm_caller = llm_caller

    def refresh(self, include_private: bool = True):
        """Подтягивает изменения, сделанные другим процессом (чат-процесс
        и MCP-процесс держат независимые копии памяти в RAM). Вызывать
        перед чтением, если процесс живёт долго и в фоне мог писать другой
        процесс — например, в начале каждого MCP tool call."""
        if include_private:
            self.private_memory.reload_if_stale()
        self.shared_memory.reload_if_stale()
        self.global_memory.reload_if_stale()

    async def retrieve(self, query: str, top_k: int = 7, include_private: bool = True) -> List[Dict]:
        """
        Объединённый поиск по всем доступным слоям с ранжированием.
        """
        self.refresh(include_private=include_private)
        private_results = []
        shared_results = []
        global_results = []

        if include_private:
            private_results = await self.private_memory.retrieve_hybrid(query, top_k=top_k)
        shared_results = await self.shared_memory.retrieve_hybrid(query, top_k=top_k)
        global_results = await self.global_memory.retrieve_hybrid(query, top_k=top_k)

        # Применяем веса к скорам в зависимости от scope
        # Приватные выше, глобальные чуть ниже, общие посередине
        for item in private_results:
            item["_score"] = item.get("score", 0.5) * 1.2  # +20%
        for item in shared_results:
            item["_score"] = item.get("score", 0.5) * 1.0  # базовый
        for item in global_results:
            item["_score"] = item.get("score", 0.5) * 0.9  # -10%

        # УЛУЧШЕНИЕ (кросс-слойная связность / "глобальный мозг"): раньше три
        # слоя памяти были полностью изолированы друг от друга — три
        # независимых retrieve_hybrid, объединённых только конкатенацией и
        # статичным весом по scope. Личный факт пользователя A никак не мог
        # "дотянуться" через граф до факта пользователя B, даже если оба
        # физически обосновывают один и тот же глобальный концепт. Теперь,
        # если личный/общий факт был явно заземлён на глобальный концепт
        # через ребро GROUNDS_IN (см. add_knowledge ниже), подтягиваем этот
        # концепт в выдачу — так запрос в приватной памяти может всплыть
        # коллективной абстракцией, а не только изолированным личным фактом.
        grounded_extra = (self._pull_grounded_concepts(private_results, self.private_memory.store)
                          + self._pull_grounded_concepts(shared_results, self.shared_memory.store))

        combined = private_results + shared_results + global_results + grounded_extra

        # Убираем дубликаты по тексту (можно по id, но для надёжности по тексту)
        seen_texts = set()
        unique = []
        for item in combined:
            text = item.get("text", "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique.append(item)

        # Сортируем по _score
        unique.sort(key=lambda x: x.get("_score", 0.0), reverse=True)

        # ИСПРАВЛЕНИЕ (пункт №2 из анализа интеллекта): раньше здесь сразу
        # обрезалось до top_k по линейной взвешенной сумме скоров — это
        # хорошо отсеивает явно нерелевантное, но плохо разруливает
        # "похожее по вектору, но не по сути" (близкая по эмбеддингу, но
        # для данного вопроса бесполезная формулировка могла обойти в счёте
        # действительно нужный факт). Даём LLM-судье посмотреть на candidate
        # pool целиком (с запасом) и выбрать + упорядочить только реально
        # релевантные — при сбое/пустом ответе всегда безопасно
        # откатываемся на исходный порядок по _score.
        reranked = await self._llm_rerank(query, unique, top_k)
        return reranked if reranked is not None else unique[:top_k]

    async def _llm_rerank(self, query: str, candidates: List[Dict], top_k: int) -> Optional[List[Dict]]:
        """
        Лёгкий LLM-реранкер поверх уже найденных гибридным поиском
        кандидатов (см. пункт №2 в комментарии в retrieve()). Возвращает
        None, если реранкинг не выполнялся или не удался — в этом случае
        вызывающий код обязан откатиться на исходный порядок по _score,
        чтобы одиночный сбой LLM никогда не приводил к пустому результату
        поиска по памяти.
        """
        if not RERANK_ENABLED or self._llm_caller is None:
            return None
        if len(candidates) <= top_k:
            # Реранкинг имеет смысл только когда есть из чего выбирать —
            # если кандидатов и так не больше top_k, LLM-вызов ничего не даст.
            return None

        pool = candidates[:min(len(candidates), RERANK_MAX_CANDIDATES,
                                max(top_k * RERANK_CANDIDATE_MULTIPLIER, top_k))]
        listing = "\n".join(f"{i}. {c.get('text', '')[:300]}" for i, c in enumerate(pool))
        prompt = (
            "Ниже — вопрос пользователя и пронумерованный список фактов/эпизодов/концептов "
            "из памяти ассистента, найденных по этому вопросу.\n\n"
            f"Вопрос: {query}\n\n"
            f"Список:\n{listing}\n\n"
            f"Выбери из списка только те пункты, которые РЕАЛЬНО помогают ответить на вопрос "
            f"(не более {top_k} штук), и упорядочи их по убыванию релевантности. Не выбирай "
            "пункты, которые лишь формально похожи по теме, но не отвечают на вопрос.\n"
            "Ответь ТОЛЬКО JSON-массивом номеров, без пояснений, без markdown. "
            "Пример: [3, 0, 7]. Если ни один пункт не релевантен — верни []."
        )
        try:
            raw = await self._llm_caller([{"role": "user", "content": prompt}], temp=0.0, max_tokens=120)
        except Exception as e:
            logger.debug(f"LLM-реранкинг памяти не удался, откат на исходный порядок: {e}")
            return None

        indices = self._parse_index_list(raw, max_index=len(pool) - 1)
        if indices is None:
            return None
        if not indices:
            # LLM явно сказала "ничего релевантного" — доверяем этому вместо
            # того, чтобы молча подсунуть модели произвольные топ-факты по
            # эмбеддингу (см. исходную проблему MEMORY_CONTEXT_MIN_SCORE).
            return []
        return [pool[i] for i in indices[:top_k]]

    @staticmethod
    def _parse_index_list(raw: str, max_index: int) -> Optional[List[int]]:
        """Строгий разбор ответа LLM-реранкера: ожидаем JSON-массив целых
        чисел, ничего больше. Возвращает None при любой неоднозначности —
        вызывающий код должен трактовать это как "реранкинг не удался"."""
        if not raw:
            return None
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        match = re.search(r"\[[^\[\]]*\]", text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, list):
            return None
        seen = set()
        result = []
        for item in parsed:
            if isinstance(item, bool) or not isinstance(item, int):
                continue
            if 0 <= item <= max_index and item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _pull_grounded_concepts(self, results: List[Dict], source_store: "MemoryStore") -> List[Dict]:
        """Для топовых результатов данного слоя подтягивает связанные GLOBAL-концепты
        через ребро GROUNDS_IN (см. add_knowledge). Часть кросс-слойного заземления
        памяти — полноценный spreading activation через границы store пока не
        реализован (это отдельный шаг), но точечное "притягивание" уже связанных
        концептов работает и является дешёвой первой версией этой идеи."""
        extra = []
        seen = set()
        for item in results[:5]:
            gid = item.get("gcn_id")
            if not gid:
                continue
            for relation, target_id in source_store._graph.get_neighbors(gid, "GROUNDS_IN"):
                if target_id in seen:
                    continue
                seen.add(target_id)
                gobj = self.global_memory.store.get(target_id)
                if gobj is None:
                    continue
                score = gobj.confidence * 0.85  # чуть ниже "родного" глобального результата
                extra.append({
                    "id": None,
                    "text": gobj.subject,
                    "type": "concept",
                    "timestamp": gobj.created.timestamp(),
                    "score": score,
                    "confidence": gobj.confidence,
                    "importance": 1.2,
                    "activation": 0.0,
                    "gcn_id": gobj.id,
                    "_score": score,
                    "scope": gobj.scope.value,
                })
        return extra

    def add_knowledge(self, subject: str, predicate: str, obj: Any,
                      scope: MemoryScope = MemoryScope.PRIVATE,
                      confidence: float = 0.5, author: Optional[str] = None,
                      source_type: str = "user_input") -> str:
        """
        Добавляет знание в соответствующий слой с учётом scope.
        Для GLOBAL использует инжекшн (дедупликация, агрегация).
        """
        if author is None:
            author = self.user_id

        # Создаём объект
        ko = KnowledgeObject(
            id=f"fact_{uuid.uuid4()}",
            type=KnowledgeType.CLAIM,
            subject=subject,
            predicate=predicate,
            object={"value": obj},
            author=author,
            created=datetime.now(timezone.utc),
            confidence=confidence,
            scope=scope,
            source_type=source_type
        )

        # ---- Проставляем эмбеддинг (было добавлено ранее) ----
        dest_memory = {
            MemoryScope.GLOBAL: self.global_memory,
            MemoryScope.SHARED: self.shared_memory,
            MemoryScope.PRIVATE: self.private_memory,
        }.get(scope)
        if dest_memory is not None:
            emb = dest_memory.embed_text(subject)
            if emb is not None:
                dest_memory.store.set_embedding(ko.id, emb)

        # ---- Логика сохранения с учётом scope ----
        # УЛУЧШЕНИЕ: раньше через инжектор (дедуп/усиление/противоречия) шла
        # только GLOBAL-ветка, PRIVATE и SHARED просто делали store.create()
        # без всякой консолидации по смыслу. Теперь у каждого scope свой
        # инжектор (см. __init__), так что личная и общая память тоже
        # дедуплицируются и усиливаются повторными подтверждениями, а не
        # только глобальная.
        ingestion_map = {
            MemoryScope.GLOBAL: (self.global_ingestion, self.global_memory),
            MemoryScope.SHARED: (self.shared_ingestion, self.shared_memory),
            MemoryScope.PRIVATE: (self.private_ingestion, self.private_memory),
        }
        entry = ingestion_map.get(scope)
        if entry is None:
            raise ValueError(f"Unknown scope: {scope}")
        ingestion, dest = entry

        result = ingestion.submit_candidate(ko, author)
        # Помечаем FAISS-индекс как "грязный", чтобы при следующем поиске перестроился
        dest.store._faiss_dirty = True

        # УЛУЧШЕНИЕ (кросс-слойное заземление / "глобальный мозг"): личные и
        # общие факты раньше физически существовали только в своём store и
        # никогда не соединялись с глобальным графом — три отдельных "мозга"
        # вместо одной ткани. Теперь при добавлении PRIVATE/SHARED-знания
        # ищем семантически близкие GLOBAL-концепты/факты и проводим ребро
        # GROUNDS_IN от нового объекта к ним. Это даёт retrieve() точку
        # входа для подтягивания коллективной абстракции (см.
        # _pull_grounded_concepts) и в перспективе — путь для spreading
        # activation через границы слоёв.
        if scope in (MemoryScope.PRIVATE, MemoryScope.SHARED):
            emb = dest.store.get_embedding(result)
            if emb is not None:
                try:
                    global_matches = self.global_memory.store.semantic_search(emb, top_k=3)
                    for gid, sim in global_matches:
                        if sim >= CROSS_LAYER_GROUNDING_THRESHOLD:
                            dest.store._graph.add_relation(result, "GROUNDS_IN", gid, weight=sim)
                except Exception as e:
                    logger.debug(f"Cross-layer grounding failed for {result}: {e}")

        return result

    async def add_episode(self, user_msg: str, assistant_msg: str, salience: float = 0.0,
                          scope: MemoryScope = MemoryScope.PRIVATE,
                          extract_facts: bool = False):
        """
        Сохраняет эпизод в личную память, а также, если extract_facts=True,
        извлекает факты с помощью LLM и отправляет в глобальную память.
        """
        # Всегда сохраняем эпизод в личную память
        await self.private_memory.add_episode(user_msg, assistant_msg, salience)

        # Если включено извлечение фактов для глобальной памяти
        if extract_facts and self._llm_caller is not None:
            extracted = await self._extract_facts_with_llm(user_msg, assistant_msg)
            for fact in extracted:
                self.add_knowledge(
                    subject=fact,
                    predicate="is_fact",
                    obj="true",
                    scope=MemoryScope.GLOBAL,
                    confidence=0.75,
                    source_type="dialogue_extraction"
                )

    async def _extract_facts_with_llm(self, user_msg: str, assistant_msg: str) -> List[str]:
        """Извлекает факты из диалога с помощью LLM с улучшенной фильтрацией."""
        if not self._llm_caller:
            return []
        combined = f"User: {user_msg}\nAssistant: {assistant_msg}"
        prompt = (
            "Извлеки из диалога только объективные, проверяемые факты. "
            "Факт должен быть кратким утверждением, содержащим конкретную информацию "
            "(числа, даты, имена, определения). "
            "НЕ включай: мнения, прогнозы, инструкции, общие фразы. "
            "Каждый факт — отдельное предложение. "
            "Верни только факты, каждый с новой строки, без нумерации.\n\n"
            f"Диалог:\n{combined}"
        )
        try:
            raw = await self._llm_caller([{"role": "user", "content": prompt}], temp=0.2, max_tokens=300)
            if not raw:
                return []
            # Разбиваем по строкам и фильтруем
            lines = [line.strip().strip('-•*').strip() for line in raw.split('\n') if line.strip()]
            facts = []
            for line in lines:
                # Длина и базовые фильтры
                if not (20 < len(line) < 400):
                    continue
                # Исключаем субъективные начала.
                # ИСПРАВЛЕНИЕ: было `line[0].lower() in (...)` — line[0] это ОДИН
                # символ строки, а не первое слово, поэтому сравнение с 'ты', 'мы',
                # 'давайте', 'попробуйте' (длиннее одного символа) не могло сработать
                # никогда — реально отфильтровывались только строки, начинавшиеся
                # ровно с буквы "я". Сравниваем первое слово целиком.
                first_word = line.split(maxsplit=1)[0].lower().strip('.,!?:;-—') if line.split() else ''
                if first_word in ('я', 'ты', 'мы', 'давайте', 'попробуйте'):
                    continue
                # Должен содержать ключевой глагол или цифры
                if not re.search(r'(является|составляет|равен|находится|имеет|был|стал|\d)', line):
                    continue
                facts.append(line[:300])
            return facts[:5]  # максимум 5 фактов
        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")
            return []

    # ==================== ФОРМИРОВАНИЕ КОНЦЕПТОВ ====================
    # НОВОЕ: KnowledgeType.CONCEPT существовал в схеме, но нигде не создавался —
    # у системы не было шага абстрагирования/обобщения. Консолидация (light/deep)
    # только дедуплицировала и пересчитывала confidence/importance сырых CLAIM.
    # Именно на этом шаге в такой архитектуре рождается эмерджентность:
    # периодическая кластеризация связанных фактов в узлы более высокого уровня
    # (CONCEPT), с обратными рёбрами ABSTRACTS_FROM к источникам. Это даёт:
    # (1) компактный, осмысленный контекст в промпт вместо кучи мелких фактов,
    # (2) узлы графа, через которые spreading activation начинает "перепрыгивать"
    # между темами, а не только между дословно связанными синапсами.
    async def form_concepts(self, scope: MemoryScope = MemoryScope.GLOBAL,
                            min_cluster_size: int = CONCEPT_MIN_CLUSTER_SIZE,
                            similarity_threshold: float = CONCEPT_SIMILARITY_THRESHOLD,
                            max_scan: int = CONCEPT_MAX_SCAN,
                            max_concepts_per_run: int = CONCEPT_MAX_PER_RUN) -> List[str]:
        """Кластеризует семантически близкие CLAIM-объекты указанного слоя и
        формулирует для каждого достаточно большого кластера обобщающий CONCEPT
        через LLM. Вызывать из периодической глубокой консолидации (не на
        каждый запрос — операция O(n^2) по эмбеддингам в пределах max_scan)."""
        target = {
            MemoryScope.GLOBAL: self.global_memory,
            MemoryScope.SHARED: self.shared_memory,
            MemoryScope.PRIVATE: self.private_memory,
        }.get(scope)
        if target is None or self._llm_caller is None:
            return []

        store = target.store
        candidates = [obj for obj in store._objects.values()
                     if obj.type == KnowledgeType.CLAIM and obj.scope == scope][:max_scan]
        vectors, objs = [], []
        for obj in candidates:
            vec = store.get_embedding(obj.id)
            if vec:
                vectors.append(vec)
                objs.append(obj)
        if len(objs) < min_cluster_size:
            return []

        vectors_np = np.array(vectors, dtype='float32')
        norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
        vectors_norm = vectors_np / (norms + 1e-8)
        sim_matrix = vectors_norm @ vectors_norm.T

        # Простая union-find кластеризация по порогу косинусной близости —
        # без внешних зависимостей (sklearn/hdbscan не гарантированы в окружении)
        parent = list(range(len(objs)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        n = len(objs)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= similarity_threshold:
                    union(i, j)

        clusters: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(i)

        # Уже абстрагированные факты (есть ABSTRACTS_FROM входящее ребро) не переобобщаем
        already_abstracted = set()
        for obj in objs:
            if store._graph.get_incoming(obj.id, "ABSTRACTS_FROM"):
                already_abstracted.add(obj.id)

        formed: List[str] = []
        for idxs in clusters.values():
            if len(formed) >= max_concepts_per_run:
                break
            if len(idxs) < min_cluster_size:
                continue
            members = [objs[i] for i in idxs if objs[i].id not in already_abstracted]
            if len(members) < min_cluster_size:
                continue

            facts_text = "\n".join(f"- {m.subject}" for m in members[:12])
            prompt = (
                "Ниже — набор связанных фактов из памяти AI-ассистента. "
                "Сформулируй ОДНО краткое обобщающее утверждение (концепт), которое "
                "связывает их общий смысл. Не перечисляй факты, а обобщи их суть в "
                "одном предложении на русском языке. Ответь только этим предложением, "
                "без пояснений и без кавычек.\n\n" + facts_text
            )
            try:
                concept_text = await self._llm_caller([{"role": "user", "content": prompt}],
                                                       temp=0.3, max_tokens=120)
                concept_text = (concept_text or "").strip().strip('"').strip()
            except Exception as e:
                logger.warning(f"Concept formation LLM call failed: {e}")
                continue
            if not (10 < len(concept_text) < 400):
                continue

            centroid = np.mean([vectors_norm[i] for i in idxs if objs[i].id not in already_abstracted], axis=0)
            concept_id = f"concept_{uuid.uuid4()}"
            concept_obj = KnowledgeObject(
                id=concept_id,
                type=KnowledgeType.CONCEPT,
                subject=concept_text,
                predicate="abstracts",
                object={"member_count": len(members)},
                author=f"consolidation:{scope.value}",
                created=datetime.now(timezone.utc),
                confidence=0.6,
                scope=scope,
                source_type="concept_formation",
            )
            store.create(concept_obj, actor="system:consolidation")
            store.set_embedding(concept_id, centroid.tolist())
            for m in members:
                store._graph.add_relation(concept_id, "ABSTRACTS_FROM", m.id, weight=1.0)
            formed.append(concept_id)

        if formed:
            store._faiss_dirty = True
            await target._schedule_save()
            logger.info(f"[ConceptFormation] scope={scope.value}: сформировано {len(formed)} концептов")
        return formed