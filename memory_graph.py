"""
Когнитивная память: семантическая, эпизодическая, ассоциативный граф с Hebbian/STDP,
spreading activation, predictive transitions, противоречия, консолидация, replay.
Дополнительно интегрирован GCN (Global Cognitive Network) для структурированного
хранения, версионирования, событий и гибридного поиска.
"""
import json
import logging
import time
import re
import asyncio
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Set, Optional, Any, Tuple, DefaultDict
from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss

# Импорт GCN-компонентов (папка GCN, файл GCN.py)
from GCN.GCN import (
    KnowledgeObject, KnowledgeType, KnowledgeEvent, EventType,
    MemoryStore, KnowledgeGraph as GCNKnowledgeGraph,
    AIAdapter, Provenance, MemoryHierarchy
)

logger = logging.getLogger(__name__)

try:
    from config_ai import *
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


# =====================================================================
# ДАТАКЛАССЫ (оригинальные, без изменений)
# =====================================================================
@dataclass
class Fact:
    """Семантический факт с когнитивными состояниями."""
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

    # Для совместимости с GCN – храним ссылку на KnowledgeObject id (если создан)
    gcn_id: Optional[str] = None


@dataclass
class Synapse:
    """Синаптическая связь между двумя фактами."""
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
    """Эпизод – диалог или событие с временной привязкой."""
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
    """Цель с состоянием."""
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


# =====================================================================
# ОСНОВНОЙ КЛАСС КОГНИТИВНОЙ ПАМЯТИ (с GCN-слоем)
# =====================================================================
class CognitiveMemory:
    """
    Многоуровневая память с ассоциативным графом, Hebbian/STDP,
    spreading activation, предсказаниями, консолидацией и replay.
    Дополнительно интегрирован GCN как структурированное хранилище
    с версионированием, событиями и гибридным поиском.
    """

    def __init__(self, user_id: str, base_dir: Path):
        self.user_id = user_id
        self.base_dir = base_dir / user_id / "cognitive_memory"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # ---- Оригинальные уровни памяти ----
        self.sensory_buffer: deque = deque(maxlen=SENSORY_BUFFER_SIZE)
        self.working_memory: deque = deque(maxlen=WORKING_MEMORY_SIZE)
        self.episodic_memory: List[Episode] = []
        self.semantic_facts: List[Fact] = []
        self.facts_by_id: Dict[int, Fact] = {}            # id -> Fact
        self.graph: DefaultDict[int, Set[int]] = defaultdict(set)  # граф синапсов
        self.synapses: Dict[Tuple[int, int], Synapse] = {}
        self.keyword_index: DefaultDict[str, List[int]] = defaultdict(list)  # fact.id

        # ---- Прогностическая модель ----
        self.predictive_matrix: DefaultDict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.concept_examples: Dict[str, str] = {}
        self.prediction_cache: Dict[str, List[int]] = {}

        # ---- Очередь пар-кандидатов на LLM-проверку противоречий ----
        self.pending_contradiction_checks: List[Tuple[int, int]] = []

        # ---- Цели ----
        self.goals: List[Goal] = []
        self._next_goal_id = 0

        # ---- Счётчики ----
        self._next_fact_id = 0
        self._next_episode_id = 0
        self._lock = asyncio.Lock()
        self._dirty = False
        self._save_task = None

        # ---- Динамические веса для гибридного поиска ----
        self._dynamic_weights = {
            "bm25": HYBRID_WEIGHT_BM25,
            "cosine": HYBRID_WEIGHT_COSINE,
            "freshness": HYBRID_WEIGHT_FRESHNESS,
            "graph": HYBRID_WEIGHT_GRAPH,
        }

        # ---- GCN-слой (дополнительное хранилище) ----
        self.gcn_store = MemoryStore()
        self.gcn_graph = self.gcn_store._graph  # для доступа
        # Загружаем состояние GCN (если есть)
        gcn_state_path = self.base_dir / GCN_STATE_FILENAME
        if gcn_state_path.exists():
            try:
                self.gcn_store.load(str(gcn_state_path))
                logger.info("GCN state loaded")
            except Exception as e:
                logger.warning(f"Failed to load GCN state: {e}")

        # ---- Эмбеддинги (оригинальные) ----
        self.use_embeddings = MEMORY_USE_EMBEDDINGS
        if self.use_embeddings:
            try:
                self.embedder = SentenceTransformer(EMBEDDING_MODEL)
                self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, FAISS_NLIST)
                self.index.set_direct_map_type(faiss.INDIRECT)
                self.fact_embeddings: List[np.ndarray] = []
                self._emb_added_since_train = 0
                self._load_embeddings()
            except Exception as e:
                logger.error(f"Embeddings init failed: {e}. Disabling.")
                self.use_embeddings = False
                self.embedder = None
                self.index = None
        else:
            self.embedder = None
            self.index = None

        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        self._tfidf_dirty = True
        self._tfidf_matrix = None
        self._tfidf_texts_hash = None

        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = MEMORY_CACHE_TTL
        self._cache_maxsize = MEMORY_CACHE_MAX_SIZE

        # Загрузка сохранённого состояния (оригинальные файлы)
        self._load_sync()
        logger.info(f"CognitiveMemory initialized for {user_id[:16]}")

    @property
    def store(self) -> MemoryStore:
        """Алиас на gcn_store. ai_assistant.py (AIAdapter, CognitiveController) обращается
        к нему как self.memory.store — без этого алиаса конструктор падает с AttributeError,
        так как единственный реальный атрибут называется gcn_store."""
        return self.gcn_store

    # ==================== ЗАГРУЗКА / СОХРАНЕНИЕ (оригинальные) ====================
    def _load_sync(self):
        facts_path = self.base_dir / "facts.json"
        if facts_path.exists():
            try:
                with open(facts_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    raw = data.get('facts', [])
                    self.semantic_facts = []
                    for fd in raw:
                        fact = Fact(
                            id=fd['id'],
                            text=fd['text'],
                            type=fd['type'],
                            timestamp=fd['timestamp'],
                            keywords=fd.get('keywords', []),
                            importance=fd.get('importance', DEFAULT_IMPORTANCE),
                            confidence=fd.get('confidence', DEFAULT_CONFIDENCE),
                            novelty=fd.get('novelty', DEFAULT_NOVELTY),
                            salience=fd.get('salience', DEFAULT_SALIENCE),
                            stability=fd.get('stability', DEFAULT_STABILITY),
                            plasticity=fd.get('plasticity', DEFAULT_PLASTICITY),
                            prediction_error=fd.get('prediction_error', DEFAULT_PREDICTION_ERROR),
                            access_count=fd.get('access_count', 0),
                            last_accessed=fd.get('last_accessed', 0.0),
                            contradicts=set(fd.get('contradicts', [])),
                            gcn_id=fd.get('gcn_id', None),
                        )
                        self.semantic_facts.append(fact)
                    self._next_fact_id = data.get('next_id', len(self.semantic_facts))
            except Exception as e:
                logger.warning(f"Load facts error: {e}")

        graph_path = self.base_dir / "graph.json"
        if graph_path.exists():
            try:
                with open(graph_path, 'r', encoding='utf-8') as f:
                    gdata = json.load(f)
                    self.graph = defaultdict(set, {int(k): set(v) for k, v in gdata.items()})
            except Exception as e:
                logger.warning(f"Load graph error: {e}")

        synapses_path = self.base_dir / "synapses.json"
        if synapses_path.exists():
            try:
                with open(synapses_path, 'r', encoding='utf-8') as f:
                    sdata = json.load(f)
                    for key, val in sdata.items():
                        src, tgt = map(int, key.split(','))
                        self.synapses[(src, tgt)] = Synapse(
                            source_id=src,
                            target_id=tgt,
                            weight=val.get('weight', SYNAPSE_INITIAL_WEIGHT),
                            last_activation=val.get('last_activation', 0.0),
                            plasticity=val.get('plasticity', 0.5),
                            confidence=val.get('confidence', 0.5),
                            coactivation_count=val.get('coactivation_count', 0),
                            last_coactivation=val.get('last_coactivation', 0.0),
                            pre_time=val.get('pre_time', 0.0),
                            post_time=val.get('post_time', 0.0),
                        )
            except Exception as e:
                logger.warning(f"Load synapses error: {e}")

        episodes_path = self.base_dir / "episodes.json"
        if episodes_path.exists():
            try:
                with open(episodes_path, 'r', encoding='utf-8') as f:
                    edata = json.load(f)
                    self.episodic_memory = [Episode(**ep) for ep in edata.get('episodes', [])]
                    self._next_episode_id = edata.get('next_id', len(self.episodic_memory))
            except Exception as e:
                logger.warning(f"Load episodes error: {e}")

        goals_path = self.base_dir / "goals.json"
        if goals_path.exists():
            try:
                with open(goals_path, 'r', encoding='utf-8') as f:
                    gdata = json.load(f)
                    self.goals = [Goal(**g) for g in gdata.get('goals', [])]
                    self._next_goal_id = gdata.get('next_id', len(self.goals))
            except Exception as e:
                logger.warning(f"Load goals error: {e}")

        # Восстанавливаем маппинг id -> Fact
        self.facts_by_id = {f.id: f for f in self.semantic_facts}
        self._build_keyword_index()
        self._rebuild_predictive_from_episodes()

    def _embeddings_path(self) -> Path:
        return self.base_dir / "embeddings.npy"

    def _load_embeddings(self):
        path = self._embeddings_path()
        if path.exists() and self.use_embeddings:
            try:
                arr = np.load(path)
                if arr.shape[0] > 0:
                    self.fact_embeddings = list(arr)
                    if len(self.fact_embeddings) != len(self.semantic_facts):
                        logger.warning(
                            f"Embeddings count ({len(self.fact_embeddings)}) != facts count ({len(self.semantic_facts)}). Resetting."
                        )
                        self.fact_embeddings = []
                        return
                    if not self.index.is_trained and len(arr) >= FAISS_MIN_TRAIN_VECTORS:
                        self.index.train(arr.astype('float32'))
                        self.index.add(arr.astype('float32'))
                        self._emb_added_since_train = 0
                    elif self.index.is_trained:
                        self.index.add(arr.astype('float32'))
                else:
                    self.fact_embeddings = []
            except Exception as e:
                logger.warning(f"Load embeddings error: {e}")
                self.fact_embeddings = []

    async def _save_async(self):
        async with self._lock:
            facts_path = self.base_dir / "facts.json"
            graph_path = self.base_dir / "graph.json"
            synapses_path = self.base_dir / "synapses.json"
            episodes_path = self.base_dir / "episodes.json"
            goals_path = self.base_dir / "goals.json"
            try:
                facts_data = []
                for f in self.semantic_facts:
                    facts_data.append({
                        'id': f.id,
                        'text': f.text,
                        'type': f.type,
                        'timestamp': f.timestamp,
                        'keywords': f.keywords,
                        'importance': f.importance,
                        'confidence': f.confidence,
                        'novelty': f.novelty,
                        'salience': f.salience,
                        'stability': f.stability,
                        'plasticity': f.plasticity,
                        'prediction_error': f.prediction_error,
                        'access_count': f.access_count,
                        'last_accessed': f.last_accessed,
                        'contradicts': list(f.contradicts),
                        'gcn_id': f.gcn_id,
                    })
                with open(facts_path, 'w', encoding='utf-8') as f:
                    json.dump({'facts': facts_data, 'next_id': self._next_fact_id}, f, ensure_ascii=False, indent=2)

                with open(graph_path, 'w', encoding='utf-8') as f:
                    json.dump({k: list(v) for k, v in self.graph.items()}, f, ensure_ascii=False, indent=2)

                synapses_data = {}
                for (src, tgt), syn in self.synapses.items():
                    synapses_data[f"{src},{tgt}"] = {
                        'weight': syn.weight,
                        'last_activation': syn.last_activation,
                        'plasticity': syn.plasticity,
                        'confidence': syn.confidence,
                        'coactivation_count': syn.coactivation_count,
                        'last_coactivation': syn.last_coactivation,
                        'pre_time': syn.pre_time,
                        'post_time': syn.post_time,
                    }
                with open(synapses_path, 'w', encoding='utf-8') as f:
                    json.dump(synapses_data, f, ensure_ascii=False, indent=2)

                episodes_data = [{'id': e.id, 'user_msg': e.user_msg, 'assistant_msg': e.assistant_msg,
                                  'timestamp': e.timestamp, 'importance': e.importance, 'salience': e.salience,
                                  'prediction_error': e.prediction_error, 'accessed_count': e.accessed_count}
                                 for e in self.episodic_memory]
                with open(episodes_path, 'w', encoding='utf-8') as f:
                    json.dump({'episodes': episodes_data, 'next_id': self._next_episode_id}, f, ensure_ascii=False, indent=2)

                goals_data = [{'id': g.id, 'description': g.description, 'priority': g.priority,
                               'confidence': g.confidence, 'progress': g.progress, 'deadline': g.deadline,
                               'related_memory': g.related_memory, 'dependencies': g.dependencies,
                               'status': g.status, 'created_at': g.created_at} for g in self.goals]
                with open(goals_path, 'w', encoding='utf-8') as f:
                    json.dump({'goals': goals_data, 'next_id': self._next_goal_id}, f, ensure_ascii=False, indent=2)

                if self.use_embeddings and self.fact_embeddings:
                    np.save(self._embeddings_path(), np.array(self.fact_embeddings))

                # Сохраняем GCN состояние
                gcn_state_path = self.base_dir / GCN_STATE_FILENAME
                await self.gcn_store.async_save(str(gcn_state_path))

                self._dirty = False
            except Exception as e:
                logger.error(f"Save error: {e}")

    async def _schedule_save(self):
        self._dirty = True
        if self._save_task is None or self._save_task.done():
            self._save_task = asyncio.create_task(self._periodic_save())

    async def _periodic_save(self):
        await asyncio.sleep(5)
        if self._dirty:
            await self._save_async()

    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ (оригинальные + _compute_similarity) ====================
    def _build_keyword_index(self):
        self.keyword_index.clear()
        for fact in self.semantic_facts:
            for word in fact.keywords:
                self.keyword_index[word].append(fact.id)

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
        if not self.use_embeddings:
            return np.zeros(self.embedding_dim)
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        emb = self.embedder.encode(text, convert_to_numpy=True)
        self._embedding_cache[text] = emb
        if len(self._embedding_cache) > 1024:
            first = next(iter(self._embedding_cache))
            del self._embedding_cache[first]
        return emb

    def _train_index_if_needed(self):
        if not self.use_embeddings or self.index is None:
            return
        if len(self.fact_embeddings) < FAISS_MIN_TRAIN_VECTORS:
            return
        if self.index.is_trained and self._emb_added_since_train < FAISS_REBUILD_THRESHOLD:
            return
        vectors = np.array(self.fact_embeddings).astype('float32')
        if vectors.shape[0] < FAISS_MIN_TRAIN_VECTORS:
            return
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        new_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, FAISS_NLIST)
        new_index.train(vectors)
        new_index.set_direct_map_type(faiss.INDIRECT)
        new_index.add(vectors)
        self.index = new_index
        self._emb_added_since_train = 0
        logger.info(f"FAISS retrained on {vectors.shape[0]} vectors")

    def _rebuild_faiss_index(self):
        if not self.use_embeddings or self.index is None:
            return
        if not self.fact_embeddings:
            self.index.reset()
            self._emb_added_since_train = 0
            return
        vectors = np.array(self.fact_embeddings).astype('float32')
        if len(vectors) < FAISS_MIN_TRAIN_VECTORS:
            self.index.reset()
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(vectors)
            self._emb_added_since_train = 0
            return
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        new_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, FAISS_NLIST)
        new_index.train(vectors)
        new_index.set_direct_map_type(faiss.INDIRECT)
        new_index.add(vectors)
        self.index = new_index
        self._emb_added_since_train = 0
        logger.info(f"FAISS index rebuilt with {len(vectors)} vectors")

    def _invalidate_tfidf_cache(self):
        self._tfidf_dirty = True
        self._tfidf_matrix = None
        self._tfidf_texts_hash = None

    def _ensure_tfidf(self):
        texts = [f.text for f in self.semantic_facts]
        texts_hash = hashlib.md5("\n".join(texts[:100]).encode()).hexdigest()
        if not self._tfidf_dirty and self._tfidf_matrix is not None and texts_hash == self._tfidf_texts_hash:
            return
        if not texts:
            self._tfidf_matrix = None
            self._tfidf_dirty = False
            return
        try:
            self._tfidf_matrix = self.tfidf.fit_transform(texts)
            self._tfidf_texts_hash = texts_hash
            self._tfidf_dirty = False
        except Exception as e:
            logger.warning(f"TF-IDF rebuild failed: {e}")
            self._tfidf_matrix = None

    # ==================== УДАЛЕНИЕ ФАКТОВ (оригинальное) ====================
    def _remove_facts(self, ids: Set[int]) -> int:
        if not ids:
            return 0
        indices_to_remove = {i for i, f in enumerate(self.semantic_facts) if f.id in ids}

        # Раньше удалённые факты оставляли осиротевшие KnowledgeObject'ы в gcn_store —
        # retract() снимает их из активных индексов, сохраняя событие в логе.
        for f in self.semantic_facts:
            if f.id in ids and f.gcn_id:
                try:
                    self.gcn_store.retract(f.gcn_id, self.user_id, reason="duplicate_removed")
                except Exception as e:
                    logger.debug(f"GCN retract failed for {f.gcn_id}: {e}")

        self.semantic_facts = [f for f in self.semantic_facts if f.id not in ids]
        self.facts_by_id = {f.id: f for f in self.semantic_facts}

        if self.use_embeddings and self.fact_embeddings:
            self.fact_embeddings = [emb for i, emb in enumerate(self.fact_embeddings) if i not in indices_to_remove]
            self._rebuild_faiss_index()

        self.synapses = {(s, t): syn for (s, t), syn in self.synapses.items()
                         if s not in ids and t not in ids}
        self.graph = defaultdict(set)
        for (src, tgt) in self.synapses:
            self.graph[src].add(tgt)

        for f in self.semantic_facts:
            f.contradicts -= ids

        self._build_keyword_index()
        self._invalidate_tfidf_cache()
        self._dirty = True
        return len(indices_to_remove)

    # ==================== ДОБАВЛЕНИЕ ФАКТОВ (оригинальное + синхронизация с GCN) ====================
    def _add_fact(self, text: str, ftype: str, importance: float = 1.0,
                  confidence: float = 0.5, novelty: float = 0.0,
                  salience: float = 0.0) -> int:
        fid = self._next_fact_id
        self._next_fact_id += 1
        fact = Fact(
            id=fid,
            text=text,
            type=ftype,
            timestamp=time.time(),
            keywords=list(self._extract_keywords(text)),
            importance=importance,
            confidence=confidence,
            novelty=novelty,
            salience=salience,
            stability=0.5,
            plasticity=0.5,
            prediction_error=0.0,
        )
        self.semantic_facts.append(fact)
        self.facts_by_id[fid] = fact
        fact_idx = len(self.semantic_facts) - 1

        for kw in fact.keywords:
            self.keyword_index[kw].append(fid)

        # ---- GCN синхронизация: создаём KnowledgeObject ----
        gcn_obj = KnowledgeObject(
            id=f"fact_gcn_{fid}",
            type=KnowledgeType.CLAIM,
            subject=text,
            predicate="",
            object="",
            author=self.user_id,
            created=datetime.now(timezone.utc),
            confidence=confidence,
            evidence=[],
            version=1,
        )
        try:
            self.gcn_store.create(gcn_obj, self.user_id)
            fact.gcn_id = gcn_obj.id
        except Exception as e:
            logger.warning(f"Failed to create GCN object for fact {fid}: {e}")

        # ---- Оригинальная логика эмбеддингов и синапсов ----
        emb = None
        if self.use_embeddings:
            try:
                emb = self._get_embedding(text)
                self.fact_embeddings.append(emb)
                if self.index is not None and self.index.is_trained:
                    self.index.add(np.array([emb]).astype('float32'))
                self._emb_added_since_train += 1
                self._train_index_if_needed()
                # ---- Синхронизация эмбеддинга с GCN ----
                if self.use_embeddings and emb is not None and fact.gcn_id:
                    try:
                        self.gcn_store.set_embedding(fact.gcn_id, emb.tolist())
                    except Exception as e:
                        logger.debug(f"GCN embedding sync failed: {e}")
            except Exception:
                emb = None

        similar: List[Tuple[int, float]] = []
        if emb is not None:
            similar = self._find_similar_by_embedding(emb, k=20, exclude_idx=fact_idx)
        else:
            recent_window_start = max(0, len(self.semantic_facts) - 300)
            for other_idx in range(recent_window_start, len(self.semantic_facts)):
                if other_idx == fact_idx:
                    continue
                sim = self._compute_similarity(text, self.semantic_facts[other_idx].text)
                if sim > 0.0:
                    similar.append((other_idx, sim))
            similar.sort(key=lambda x: -x[1])
            similar = similar[:20]

        similar_ids: List[int] = []
        for other_idx, sim in similar:
            if other_idx >= len(self.semantic_facts) or other_idx == fact_idx:
                continue
            other = self.semantic_facts[other_idx]
            similar_ids.append(other.id)
            if sim > 0.45:
                self._create_synapse(fid, other.id, weight=sim * 0.5)
                self._create_synapse(other.id, fid, weight=sim * 0.5)

        self._detect_contradictions(fid, candidate_ids=similar_ids)
        self._dirty = True
        self._invalidate_tfidf_cache()
        return fid

    def _find_similar_by_embedding(self, emb: np.ndarray, k: int = 20,
                                   exclude_idx: Optional[int] = None) -> List[Tuple[int, float]]:
        if not self.fact_embeddings:
            return []
        n = len(self.fact_embeddings)
        q = emb.astype('float32')
        if self.index is not None and self.index.is_trained and n > 200:
            try:
                self.index.nprobe = FAISS_NPROBE
                _, idxs = self.index.search(q.reshape(1, -1), min(k * 3, n))
                candidates = [i for i in idxs[0].tolist() if 0 <= i < n and i != exclude_idx]
            except Exception:
                candidates = [i for i in range(n) if i != exclude_idx]
        else:
            candidates = [i for i in range(n) if i != exclude_idx]

        if not candidates:
            return []
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        cand_matrix = np.array([self.fact_embeddings[i] for i in candidates])
        cand_norm = cand_matrix / (np.linalg.norm(cand_matrix, axis=1, keepdims=True) + 1e-8)
        sims = cand_norm @ q_norm
        order = np.argsort(-sims)[:k]
        return [(candidates[i], float(sims[i])) for i in order]

    def _create_synapse(self, src: int, tgt: int, weight: float = SYNAPSE_INITIAL_WEIGHT):
        key = (src, tgt)
        if key in self.synapses:
            syn = self.synapses[key]
            syn.weight = min(SYNAPSE_MAX_WEIGHT, syn.weight + HEBBIAN_LEARNING_RATE * (weight - syn.weight))
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
        self._sync_synapse_to_gcn(src, tgt)
        self._dirty = True

    def _sync_synapse_to_gcn(self, src: int, tgt: int):
        """Отражает текущий вес синапса src->tgt в GCN-графе. Раньше связь в GCN
        создавалась только один раз (при первом _create_synapse) и больше никогда не
        обновлялась, из-за чего gcn_store расходился с реальными весами по мере
        Hebbian-обучения и затухания. set_relation_weight обновляет вес существующего
        ребра, не плодя дубликаты."""
        syn = self.synapses.get((src, tgt))
        if syn is None:
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
                if other_id not in fact.contradicts:
                    fact.contradicts.add(other_id)
                    other.contradicts.add(fact_id)
                    fact.confidence *= 0.97
                    other.confidence *= 0.97
                    self.pending_contradiction_checks.append((fact_id, other_id))
                    self._dirty = True

    def confirm_contradiction(self, fact_id1: int, fact_id2: int):
        f1 = self.facts_by_id.get(fact_id1)
        f2 = self.facts_by_id.get(fact_id2)
        if f1 and f2:
            f1.confidence = max(0.05, f1.confidence * 0.85)
            f2.confidence = max(0.05, f2.confidence * 0.85)
            self._dirty = True

    def clear_contradiction(self, fact_id1: int, fact_id2: int):
        f1 = self.facts_by_id.get(fact_id1)
        f2 = self.facts_by_id.get(fact_id2)
        if f1 and f2:
            f1.contradicts.discard(fact_id2)
            f2.contradicts.discard(fact_id1)
            f1.confidence = min(1.0, f1.confidence / 0.97)
            f2.confidence = min(1.0, f2.confidence / 0.97)
            self._dirty = True

    # ==================== ЭПИЗОДЫ (оригинальные) ====================
    async def add_episode(self, user_msg: str, assistant_msg: str, salience: float = 0.0):
        episode = Episode(
            id=self._next_episode_id,
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            timestamp=time.time(),
            importance=1.0,
            salience=salience,
            prediction_error=0.0,
        )
        self._next_episode_id += 1
        self.episodic_memory.append(episode)
        if len(self.episodic_memory) > EPISODIC_MAX_SIZE:
            self.episodic_memory.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)
            self.episodic_memory = self.episodic_memory[:EPISODIC_MAX_SIZE]

        user_id = self._add_fact(user_msg, 'user', importance=1.0, salience=salience)
        assistant_id = self._add_fact(assistant_msg, 'assistant', importance=1.2, salience=salience)
        self._create_synapse(user_id, assistant_id, weight=0.8)
        self._create_synapse(assistant_id, user_id, weight=0.6)
        self._update_predictive(user_msg, assistant_msg)

        # Сохраняем эпизод также в GCN как MEMORY_EVENT
        gcn_ep = KnowledgeObject(
            id=f"ep_gcn_{episode.id}",
            type=KnowledgeType.MEMORY_EVENT,
            subject=user_msg,
            predicate="assistant_replied",
            object=assistant_msg,
            author=self.user_id,
            created=datetime.now(timezone.utc),
            confidence=0.8,
            evidence=[],
        )
        try:
            self.gcn_store.create(gcn_ep, self.user_id)
        except Exception as e:
            logger.warning(f"GCN episode creation failed: {e}")

        await self._schedule_save()

    # ==================== ПРЕДСКАЗАТЕЛЬНАЯ МОДЕЛЬ (оригинальная) ====================
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

    # ==================== HEBBIAN / STDP (оригинальные) ====================
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

    # ==================== SPREADING ACTIVATION (оригинальная) ====================
    async def spread_activation(self, seed_ids: List[int], max_depth: int = SPREADING_MAX_DEPTH,
                               max_nodes: int = SPREADING_MAX_NODES) -> Dict[int, float]:
        now = time.time()
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

    # ==================== ГИБРИДНЫЙ ПОИСК (оригинальный, но с возможностью использовать GCN) ====================
    async def retrieve_hybrid(self, query: str, top_k: int = 5, use_graph: bool = True) -> List[Dict]:
        # Если включено использование GCN, можно использовать gcn_store.hybrid_retrieve
        # Но для совместимости оставляем оригинальную логику, но добавим опцию использовать GCN
        # Здесь я оставлю оригинальную логику, но можно добавить параметр use_gcn=False
        cache_key = f"hybrid_{query}_{top_k}_{use_graph}"
        if cache_key in self._cache:
            result, ts = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return result

        if not self.semantic_facts:
            return []

        # 1. Кандидаты через FAISS или keywords
        candidate_ids: Set[int] = set()
        if self.use_embeddings and len(self.fact_embeddings) > 200 and self.index is not None and self.index.is_trained:
            try:
                q_emb = self._get_embedding(query).reshape(1, -1).astype('float32')
                self.index.nprobe = FAISS_NPROBE
                dist, idxs = self.index.search(q_emb, min(200, len(self.fact_embeddings)))
                for i in idxs[0].tolist():
                    if 0 <= i < len(self.semantic_facts):
                        candidate_ids.add(self.semantic_facts[i].id)
            except Exception:
                pass

        if not candidate_ids:
            q_keywords = self._extract_keywords(query)
            for kw in q_keywords:
                for fid in self.keyword_index.get(kw, []):
                    if fid in self.facts_by_id:
                        candidate_ids.add(fid)
            if len(candidate_ids) < 3:
                candidate_ids = set(self.facts_by_id.keys())

        candidate_list = [fid for fid in candidate_ids if fid in self.facts_by_id]
        id_to_pos = {fid: i for i, fid in enumerate(candidate_list)}
        n_cand = len(candidate_list)
        if n_cand == 0:
            return []

        # 2. BM25
        bm25_scores = np.ones(n_cand) * 0.5
        if n_cand > 1:
            try:
                self._ensure_tfidf()
                if self._tfidf_matrix is not None:
                    texts = [self.facts_by_id[fid].text for fid in candidate_list]
                    tfidf_local = self.tfidf.fit_transform(texts + [query])
                    vectors = tfidf_local[:-1]
                    q_vec = tfidf_local[-1]
                    bm25_scores = (vectors * q_vec.T).toarray().flatten()
            except Exception:
                bm25_scores = np.ones(n_cand) * 0.5

        # 3. Косинус
        cosine_scores = np.zeros(n_cand)
        if self.use_embeddings and self.fact_embeddings:
            try:
                q_emb = self._get_embedding(query).reshape(1, -1)
                q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
                cand_embs = []
                for fid in candidate_list:
                    idx = next((i for i, f in enumerate(self.semantic_facts) if f.id == fid), None)
                    if idx is not None and idx < len(self.fact_embeddings):
                        cand_embs.append(self.fact_embeddings[idx])
                    else:
                        cand_embs.append(np.zeros(self.embedding_dim))
                cand_embs_arr = np.array(cand_embs)
                if cand_embs_arr.size > 0:
                    cand_norm = cand_embs_arr / (np.linalg.norm(cand_embs_arr, axis=1, keepdims=True) + 1e-8)
                    cosine_scores = np.dot(cand_norm, q_norm.T).flatten()
            except Exception:
                cosine_scores = np.zeros(n_cand)

        # 4. Свежесть
        now = time.time()
        freshness = np.array([
            max(0.0, 1.0 - (now - self.facts_by_id[fid].timestamp) / (86400 * 30))
            for fid in candidate_list
        ])

        # 5. Графовая активация
        graph_scores = np.ones(n_cand) * 0.5
        activation_map: Dict[int, float] = {}
        if use_graph:
            seed_ids = candidate_list[:10]
            activation_map = await self.spread_activation(seed_ids, max_depth=2, max_nodes=50)
            for i, fid in enumerate(candidate_list):
                if fid in activation_map:
                    graph_scores[i] = activation_map[fid]

        # 6. Динамические веса
        if DYNAMIC_WEIGHTS_ENABLED:
            is_factual = bool(re.search(r'\b\d+[.,]?\d*\s*(?:USD|EUR|RUB|%|кг|км|г)\b', query))
            if is_factual:
                w_bm25, w_cos, w_fresh, w_graph = FACTUAL_WEIGHTS
            else:
                w_bm25, w_cos, w_fresh, w_graph = GENERAL_WEIGHTS
        else:
            w_bm25, w_cos, w_fresh, w_graph = (HYBRID_WEIGHT_BM25, HYBRID_WEIGHT_COSINE,
                                               HYBRID_WEIGHT_FRESHNESS, HYBRID_WEIGHT_GRAPH)

        w_bm25 = self._dynamic_weights.get("bm25", w_bm25)
        w_cos = self._dynamic_weights.get("cosine", w_cos)
        w_fresh = self._dynamic_weights.get("freshness", w_fresh)
        w_graph = self._dynamic_weights.get("graph", w_graph)

        # 7. Итоговый рейтинг
        final_scores: List[Tuple[float, int]] = []
        scored_ids: Set[int] = set()
        for i, fid in enumerate(candidate_list):
            score = (w_bm25 * bm25_scores[i] +
                     w_cos * cosine_scores[i] +
                     w_fresh * freshness[i] +
                     w_graph * graph_scores[i])
            final_scores.append((score, fid))
            scored_ids.add(fid)

        if use_graph and activation_map:
            for fid, act in activation_map.items():
                if fid in scored_ids or fid not in self.facts_by_id:
                    continue
                fact_age = now - self.facts_by_id[fid].timestamp
                fresh = max(0.0, 1.0 - fact_age / (86400 * 30))
                score = 0.8 * (w_graph * act + w_fresh * fresh)
                final_scores.append((score, fid))
                scored_ids.add(fid)

        final_scores.sort(reverse=True, key=lambda x: x[0])
        top_scored = final_scores[:top_k * 2]

        max_score = final_scores[0][0] if final_scores else 1.0
        result = []
        for score, fid in top_scored:
            if fid not in self.facts_by_id:
                continue
            fact = self.facts_by_id[fid]
            fact.access_count += 1
            fact.last_accessed = now
            fact.importance = min(2.0, fact.importance + 0.01)
            confidence = min(1.0, score / (max_score + 1e-6))
            result.append({
                "id": fact.id,
                "text": fact.text,
                "type": fact.type,
                "timestamp": fact.timestamp,
                "score": round(score, 4),
                "confidence": round(confidence, 3),
                "importance": fact.importance,
                "activation": fact.activation,
                "gcn_id": fact.gcn_id,
            })

        # ---- ДОБАВЛЕНО: Дополнение результатов из GCN ----
        if self.use_embeddings and hasattr(self, 'gcn_store'):
            try:
                q_emb = self._get_embedding(query).tolist() if self.use_embeddings else None
                gcn_results = self.gcn_store.hybrid_retrieve(
                    query_vector=q_emb,
                    top_k=top_k
                )
                existing_ids = {r["id"] for r in result}
                for gobj in gcn_results:
                    if gobj.id.startswith("fact_gcn_"):
                        try:
                            fid = int(gobj.id.split("_")[-1])
                            if fid not in existing_ids and fid in self.facts_by_id:
                                f = self.facts_by_id[fid]
                                result.append({
                                    "id": f.id,
                                    "text": f.text,
                                    "type": f.type,
                                    "timestamp": f.timestamp,
                                    "score": gobj.confidence * 0.8,
                                    "confidence": gobj.confidence,
                                    "importance": f.importance,
                                    "activation": f.activation,
                                    "gcn_id": f.gcn_id,
                                    "source": "gcn"
                                })
                        except:
                            pass
                result.sort(key=lambda x: x["score"], reverse=True)
                result = result[:top_k]
            except Exception as e:
                logger.debug(f"GCN hybrid search fallback: {e}")
        # ---- КОНЕЦ ДОБАВЛЕННОГО БЛОКА ----

        self._cache[cache_key] = (result, time.time())
        if len(self._cache) > self._cache_maxsize:
            oldest = sorted(self._cache.items(), key=lambda x: x[1][1])[0][0]
            del self._cache[oldest]
        return result

    # ==================== КОНСОЛИДАЦИЯ (оригинальная + GCN) ====================
    async def _find_duplicates_via_faiss(self, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD) -> Set[int]:
        if not self.use_embeddings or len(self.fact_embeddings) < 100:
            return set()
        try:
            vectors = np.array(self.fact_embeddings).astype('float32')
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors_norm = vectors / (norms + 1e-8)

            index = faiss.IndexFlatIP(self.embedding_dim)
            index.add(vectors_norm)
            k = min(10, len(vectors))
            _, idxs = index.search(vectors_norm, k)

            to_remove: Set[int] = set()
            for i, neighbors in enumerate(idxs):
                fi = self.semantic_facts[i]
                if fi.id in to_remove:
                    continue
                for j in neighbors[1:]:
                    if j == -1 or j <= i:
                        continue
                    fj = self.semantic_facts[j]
                    if fj.id in to_remove:
                        continue
                    sim = float(vectors_norm[i] @ vectors_norm[j])
                    if sim > threshold:
                        if fi.confidence <= fj.confidence:
                            to_remove.add(fi.id)
                        else:
                            to_remove.add(fj.id)
            return to_remove
        except Exception as e:
            logger.warning(f"FAISS duplicate search failed: {e}")
            return set()

    def _find_duplicates_keyword(self, threshold: float = 0.8) -> Set[int]:
        to_remove: Set[int] = set()
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
            if self.use_embeddings and len(self.fact_embeddings) >= 100:
                to_remove = await self._find_duplicates_via_faiss(DUPLICATE_SIMILARITY_THRESHOLD)
            else:
                to_remove = self._find_duplicates_keyword(0.8)

            removed = self._remove_facts(to_remove)
            self._apply_decay()
            await self._schedule_save()
            logger.info(f"Light consolidation done for {self.user_id[:16]}: removed {removed} duplicates")

    async def deep_consolidation(self):
        async with self._lock:
            if self.episodic_memory:
                self.episodic_memory.sort(key=lambda e: (e.importance * (1 + e.salience), e.timestamp), reverse=True)
                replay_candidates = self.episodic_memory[:REPLAY_BATCH_SIZE]
                for ep in replay_candidates:
                    user_facts = [f for f in self.semantic_facts if f.text == ep.user_msg]
                    ass_facts = [f for f in self.semantic_facts if f.text == ep.assistant_msg]
                    if user_facts and ass_facts:
                        self._hebbian_update(user_facts[0].id, ass_facts[0].id, ep.timestamp)
                        self._hebbian_update(ass_facts[0].id, user_facts[0].id, ep.timestamp)

            for f in self.semantic_facts:
                if f.contradicts:
                    for cid in list(f.contradicts):
                        other = self.facts_by_id.get(cid)
                        if other and other.confidence > 0.3:
                            f.confidence *= 0.95
                            other.confidence *= 0.95

            now = time.time()
            for f in self.semantic_facts:
                age = now - f.timestamp
                recency = 1.0 / (1.0 + age / 86400)
                f.importance = 0.5 * (f.importance + recency + f.access_count / 10)
                f.importance = min(2.0, f.importance)

            if len(self.semantic_facts) > SEMANTIC_MAX_FACTS:
                self.semantic_facts.sort(key=lambda f: (f.importance, f.confidence, f.timestamp), reverse=True)
                keep = self.semantic_facts[:SEMANTIC_MAX_FACTS]
                removed_ids = {f.id for f in self.semantic_facts[SEMANTIC_MAX_FACTS:]}
                self.semantic_facts = keep
                self._remove_facts(removed_ids)

            # Также можно провести консолидацию в GCN (например, обновить confidence)
            for f in self.semantic_facts:
                if f.gcn_id:
                    try:
                        self.gcn_store.update(f.gcn_id, {"confidence": f.confidence}, self.user_id)
                    except Exception as e:
                        logger.debug(f"GCN update failed for {f.gcn_id}: {e}")

            await self._schedule_save()
            logger.info(f"Deep consolidation done for {self.user_id[:16]}")

    # ==================== РАБОТА С ЦЕЛЯМИ (оригинальная) ====================
    async def add_goal(self, description: str, priority: float = 0.5, related_memory: List[int] = None):
        goal = Goal(
            id=self._next_goal_id,
            description=description,
            priority=priority,
            confidence=0.5,
            related_memory=related_memory or [],
            status='active'
        )
        self._next_goal_id += 1
        self.goals.append(goal)
        await self._schedule_save()
        return goal.id

    async def update_goal(self, goal_id: int, **kwargs):
        for g in self.goals:
            if g.id == goal_id:
                for k, v in kwargs.items():
                    if hasattr(g, k):
                        setattr(g, k, v)
                await self._schedule_save()
                return

    async def get_active_goals(self) -> List[Goal]:
        return [g for g in self.goals if g.status == 'active']

    # ==================== СТАТИСТИКА ====================
    def get_stats(self) -> Dict:
        return {
            "semantic_facts": len(self.semantic_facts),
            "episodes": len(self.episodic_memory),
            "graph_edges": sum(len(v) for v in self.graph.values()) // 2,
            "synapses": len(self.synapses),
            "goals": len(self.goals),
            "active_goals": len([g for g in self.goals if g.status == 'active']),
            "working_memory": len(self.working_memory),
            "faiss_trained": self.index.is_trained if self.index else False,
            "gcn_objects": len(self.gcn_store._objects),
        }

    # ==================== ЗАКРЫТИЕ ====================
    async def shutdown(self):
        if self._save_task:
            self._save_task.cancel()
        await self._save_async()