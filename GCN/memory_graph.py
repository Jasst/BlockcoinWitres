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
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
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
                        salience=user_obj.object.get("salience", 0.0) if isinstance(user_obj.object, dict) else 0.0
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

    def _get_embedding(self, text: str) -> np.ndarray:
        if not self.use_embeddings or self.embedder is None:
            return np.zeros(self.embedding_dim if self.embedding_dim else 128)
        return self.embedder.encode(text, convert_to_numpy=True)

    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Публичная обёртка над _get_embedding() для внешних вызывающих
        (GCNMemoryRouter). Возвращает None, если эмбеддинги отключены —
        вызывающий код должен уметь работать без вектора (fallback на
        keyword-поиск), а не падать.
        """
        if not self.use_embeddings or self.embedder is None:
            return None
        return self._get_embedding(text).tolist()

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

        similar_ids = []
        for other_idx, sim in similar:
            if other_idx >= len(self.semantic_facts) or other_idx == len(self.semantic_facts) - 1:
                continue
            other = self.semantic_facts[other_idx]
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
        key = (source_id, target_id)
        if key not in self.synapses:
            return
        syn = self.synapses[key]
        dt = coactivation_time - syn.last_activation
        if dt > 0 and dt < STDP_TIME_WINDOW:
            delta = STDP_LEARNING_RATE * (1.0 - dt / STDP_TIME_WINDOW)
        elif dt < 0 and abs(dt) < STDP_TIME_WINDOW:
            delta = -STDP_LEARNING_RATE * 0.5 * (1.0 - abs(dt) / STDP_TIME_WINDOW)
        else:
            delta = HEBBIAN_LEARNING_RATE * 0.1
        syn.weight = min(SYNAPSE_MAX_WEIGHT, max(SYNAPSE_MIN_WEIGHT, syn.weight + delta))
        syn.last_activation = coactivation_time
        syn.coactivation_count += 1
        syn.last_coactivation = coactivation_time
        syn.confidence = min(1.0, syn.confidence + 0.01)
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
        """Гибридный поиск через GCN с преобразованием результата в старый формат."""
        cache_key = f"hybrid_{query}_{top_k}_{use_graph}"
        if cache_key in self._cache:
            result, ts = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return result

        # Генерируем вектор запроса
        query_vector = None
        if self.use_embeddings:
            query_vector = self._get_embedding(query).tolist()

        # start_node для графового компонента гибридного поиска.
        # Раньше сюда всегда передавался None — весь граф Hebbian/STDP
        # синапсов, который старательно строится в _create_synapse()/
        # _hebbian_update(), никогда не участвовал в retrieve_hybrid():
        # вес HYBRID_WEIGHT_GRAPH существовал только на бумаге. Берём
        # последний активный элемент рабочей памяти как точку старта
        # обхода графа — так соседи по синапсам реально попадают в
        # кандидаты поиска.
        start_node = self.hierarchy.working_memory[-1] if use_graph and self.hierarchy.working_memory else None

        # Выполняем поиск в GCN
        gcn_results = self.gcn_store.hybrid_retrieve(
            query_vector=query_vector,
            start_node=start_node,
            top_k=top_k * 2,
            weights = self._dynamic_weights
        )

        # Преобразуем в формат, ожидаемый ai_assistant.py
        result = []
        for obj in gcn_results:
            # Определяем, является ли объект фактом (CLAIM)
            if obj.type == KnowledgeType.CLAIM:
                fid = None
                # Пытаемся извлечь id из obj.id
                try:
                    fid = int(obj.id.split("_")[-1])
                except:
                    pass
                # Находим локальный Fact по gcn_id
                fact = self._find_fact_by_gcn_id(obj.id)
                if fact:
                    fid = fact.id
                if fid is None:
                    continue
                # Формируем словарь
                meta = obj.object if isinstance(obj.object, dict) else {}
                result.append({
                    "id": fid,
                    "text": obj.subject,
                    "type": meta.get("fact_type", "unknown"),
                    "timestamp": obj.created.timestamp(),
                    "score": obj.confidence,  # GCN возвращает confidence как score
                    "confidence": obj.confidence,
                    "importance": meta.get("importance", 1.0),
                    "activation": 0.0,  # не вычисляем
                    "gcn_id": obj.id,
                })
            # Можно также включить эпизоды, если нужно
        # Сортируем по score и ограничиваем top_k
        result.sort(key=lambda x: x["score"], reverse=True)
        result = result[:top_k]

        # Сохраняем в кэш
        self._cache[cache_key] = (result, time.time())
        if len(self._cache) > self._cache_maxsize:
            oldest = sorted(self._cache.items(), key=lambda x: x[1][1])[0][0]
            del self._cache[oldest]
        return result

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
                    user_facts = [f for f in self.semantic_facts if f.text == ep.user_msg]
                    ass_facts = [f for f in self.semantic_facts if f.text == ep.assistant_msg]
                    if user_facts and ass_facts:
                        self._hebbian_update(user_facts[0].id, ass_facts[0].id, ep.timestamp)
                        self._hebbian_update(ass_facts[0].id, user_facts[0].id, ep.timestamp)

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

        # Инжектор для глобальной памяти (отвечает за дедупликацию, агрегацию, противоречия)
        self.global_ingestion = KnowledgeIngestion(self.global_memory.store)

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

    async def retrieve(self, query: str, top_k: int = 7, include_private: bool = True) -> List[Dict]:
        """
        Объединённый поиск по всем доступным слоям с ранжированием.
        """
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

        combined = private_results + shared_results + global_results

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
        return unique[:top_k]

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

        # --- КРИТИЧНО: эмбеддинг объекта ---
        # Раньше объекты, добавленные через add_knowledge() (весь путь
        # GLOBAL/SHARED и часть PRIVATE), создавались БЕЗ вызова
        # store.set_embedding(). MemoryStore.hybrid_retrieve() ищет
        # кандидатов только через semantic_search(), который смотрит
        # исключительно в _embedding_index — а туда объект без
        # set_embedding() никогда не попадал. На практике это значило,
        # что любой факт, сохранённый как "глобально" или "общее",
        # физически лежал в сторе, но был НЕВИДИМ для router.retrieve():
        # ни разу не мог быть найден обратно. Плюс KnowledgeIngestion
        # (дедуп для GLOBAL) искал похожие кандидаты тем же способом —
        # get_embedding(candidate.id) всегда возвращал None, и дедуп
        # реально работал только через грубый keyword-fallback.
        # Теперь эмбеддинг считается той же моделью, что и слой-назначение,
        # и проставляется ДО create()/submit_candidate().
        dest_memory = {
            MemoryScope.GLOBAL: self.global_memory,
            MemoryScope.SHARED: self.shared_memory,
            MemoryScope.PRIVATE: self.private_memory,
        }.get(scope)
        if dest_memory is not None:
            emb = dest_memory.embed_text(subject)
            if emb is not None:
                dest_memory.store.set_embedding(ko.id, emb)

        if scope == MemoryScope.GLOBAL:
            return self.global_ingestion.submit_candidate(ko, author)
        elif scope == MemoryScope.PRIVATE:
            return self.private_memory.store.create(ko, author)
        elif scope == MemoryScope.SHARED:
            return self.shared_memory.store.create(ko, author)
        else:
            raise ValueError(f"Unknown scope: {scope}")

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
                    confidence=0.6,
                    source_type="dialogue_extraction"
                )

    async def _extract_facts_with_llm(self, user_msg: str, assistant_msg: str) -> List[str]:
        """Извлекает факты из диалога с помощью LLM."""
        if not self._llm_caller:
            return []
        combined = f"User: {user_msg}\nAssistant: {assistant_msg}"
        prompt = (
            "Извлеки из диалога ниже все фактические утверждения (не мнения, не общие фразы). "
            "Верни только факты, каждый с новой строки, без нумерации и пояснений.\n\n"
            f"Диалог:\n{combined}"
        )
        try:
            raw = await self._llm_caller([{"role": "user", "content": prompt}], temp=0.2, max_tokens=300)
            if not raw:
                return []
            # Разбиваем по строкам и фильтруем
            lines = [line.strip().strip('-•*').strip() for line in raw.split('\n') if line.strip()]
            # Оставляем только предложения длиной > 20 символов
            facts = [line for line in lines if len(line) > 20]
            return facts[:5]
        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")
            return []