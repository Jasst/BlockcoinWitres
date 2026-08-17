# memory_graph.py
# Когнитивная память: семантическая, эпизодическая, ассоциативный граф с Hebbian/STDP,
# spreading activation, predictive transitions, противоречия, консолидация, replay.

import json
import logging
import time
import re
import asyncio
import math
from pathlib import Path
from typing import List, Dict, Set, Optional, Any, Tuple, DefaultDict
from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss

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

# =====================================================================
# ДАТАКЛАССЫ
# =====================================================================

@dataclass
class Fact:
    """Семантический факт с когнитивными состояниями."""
    id: int
    text: str
    type: str                     # 'user', 'assistant', 'knowledge', 'command'
    timestamp: float
    keywords: List[str]
    # Когнитивные состояния
    importance: float = DEFAULT_IMPORTANCE
    confidence: float = DEFAULT_CONFIDENCE
    novelty: float = DEFAULT_NOVELTY
    salience: float = DEFAULT_SALIENCE
    stability: float = DEFAULT_STABILITY
    plasticity: float = DEFAULT_PLASTICITY
    prediction_error: float = DEFAULT_PREDICTION_ERROR
    access_count: int = 0
    last_accessed: float = 0.0
    activation: float = 0.0       # текущая активация (для spreading)
    # Конфликты
    contradicts: Set[int] = field(default_factory=set)  # id фактов, которым противоречит

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
    # STDP временные метки
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
    status: str = "active"  # active, completed, blocked, abandoned
    created_at: float = field(default_factory=time.time)

# =====================================================================
# ОСНОВНОЙ КЛАСС КОГНИТИВНОЙ ПАМЯТИ
# =====================================================================

class CognitiveMemory:
    """
    Многоуровневая память с ассоциативным графом, Hebbian/STDP,
    spreading activation, предсказаниями, консолидацией и replay.
    """
    def __init__(self, user_id: str, base_dir: Path):
        self.user_id = user_id
        self.base_dir = base_dir / user_id / "cognitive_memory"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # ---- Уровни памяти ----
        self.sensory_buffer: deque = deque(maxlen=SENSORY_BUFFER_SIZE)      # последние сырые входы
        self.working_memory: deque = deque(maxlen=WORKING_MEMORY_SIZE)      # текущие активные элементы (id фактов)
        self.episodic_memory: List[Episode] = []                            # эпизоды
        self.semantic_facts: List[Fact] = []                                # семантические факты
        self.graph: DefaultDict[int, Set[int]] = defaultdict(set)           # связи (source -> set targets)
        self.synapses: Dict[Tuple[int, int], Synapse] = {}                  # детальные синапсы
        self.keyword_index: DefaultDict[str, List[int]] = defaultdict(list)

        # ---- Прогностическая модель (переходы) ----
        self.predictive_matrix: DefaultDict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.prediction_cache: Dict[str, List[int]] = {}                    # кэш предсказаний

        # ---- Цели ----
        self.goals: List[Goal] = []
        self._next_goal_id = 0

        # ---- Счётчики ----
        self._next_fact_id = 0
        self._next_episode_id = 0
        self._lock = asyncio.Lock()
        self._dirty = False
        self._save_task = None

        # ---- Эмбеддинги ----
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
        self._embedding_cache = {}
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = MEMORY_CACHE_TTL
        self._cache_maxsize = MEMORY_CACHE_MAX_SIZE

        # Загрузка сохранённого состояния
        self._load_sync()
        logger.info(f"CognitiveMemory initialized for {user_id[:16]}")

    # ---------- ЗАГРУЗКА / СОХРАНЕНИЕ ----------
    def _load_sync(self):
        facts_path = self.base_dir / "facts.json"
        if facts_path.exists():
            try:
                with open(facts_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    raw = data.get('facts', [])
                    self.semantic_facts = []
                    for fd in raw:
                        self.semantic_facts.append(Fact(
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
                        ))
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

    # ---------- ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ----------
    def _build_keyword_index(self):
        self.keyword_index.clear()
        for idx, fact in enumerate(self.semantic_facts):
            for word in fact.keywords:
                self.keyword_index[word].append(idx)

    @staticmethod
    def _extract_keywords(text: str) -> Set[str]:
        words = re.findall(r'\b[a-zA-Zа-яА-ЯёЁ]{3,}\b', text.lower())
        stopwords = {'это','все','так','вот','да','нет','или','и','с','на','по','для','из','о','к','у',
                     'же','бы','то','не','что','как','за','от','до','при','через','без','между','тоже',
                     'также','очень','можно','нужно','будет','если','тогда','потом','который','какой'}
        return {w for w in words if w not in stopwords}

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

    def _compute_similarity(self, text1: str, text2: str) -> float:
        kw1 = self._extract_keywords(text1)
        kw2 = self._extract_keywords(text2)
        if not kw1 or not kw2:
            return 0.0
        return len(kw1 & kw2) / (len(kw1 | kw2) + 1e-6)

    # ---------- ДОБАВЛЕНИЕ ФАКТОВ И СВЯЗЕЙ ----------
    def _add_fact(self, text: str, ftype: str, importance: float = 1.0,
                  confidence: float = 0.5, novelty: float = 0.0,
                  salience: float = 0.0) -> int:
        """Добавляет семантический факт и возвращает его id."""
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
        for kw in fact.keywords:
            self.keyword_index[kw].append(len(self.semantic_facts)-1)

        # Эмбеддинг
        if self.use_embeddings:
            try:
                emb = self._get_embedding(text)
                self.fact_embeddings.append(emb)
                if self.index is not None and self.index.is_trained:
                    self.index.add(np.array([emb]).astype('float32'))
                self._emb_added_since_train += 1
                self._train_index_if_needed()
            except Exception:
                pass

        # Проверка противоречий (упрощённо – по keywords)
        self._detect_contradictions(fid)

        # Добавляем связи с похожими фактами (семантическая близость)
        for other_idx, other in enumerate(self.semantic_facts):
            if other.id == fid:
                continue
            sim = self._compute_similarity(text, other.text)
            if sim > 0.4:   # порог для начальной связи
                self._create_synapse(fid, other.id, weight=sim * 0.5)
                self._create_synapse(other.id, fid, weight=sim * 0.5)

        self._dirty = True
        return fid

    def _create_synapse(self, src: int, tgt: int, weight: float = SYNAPSE_INITIAL_WEIGHT):
        """Создаёт или обновляет синапс."""
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
        self._dirty = True

    def _detect_contradictions(self, fact_id: int):
        """Находит факты, которые могут противоречить новому, по ключевым словам."""
        fact = self.semantic_facts[fact_id]
        # Простейшая эвристика: если есть отрицание или противопоставление
        neg_words = {'не', 'нет', 'без', 'против', 'отрицает', 'опровергает'}
        has_neg = any(w in fact.text.lower() for w in neg_words)
        if not has_neg:
            return
        for other in self.semantic_facts:
            if other.id == fact_id:
                continue
            # Если пересечение ключевых слов > 50%, считаем потенциальным противоречием
            common = set(fact.keywords) & set(other.keywords)
            if len(common) > 0 and len(common) / max(1, len(set(fact.keywords))) > 0.5:
                fact.contradicts.add(other.id)
                other.contradicts.add(fact.id)
                # Понижаем уверенность обоих
                fact.confidence *= 0.9
                other.confidence *= 0.9
                self._dirty = True

    # ---------- ЭПИЗОДЫ ----------
    async def add_episode(self, user_msg: str, assistant_msg: str, salience: float = 0.0):
        """Сохраняет диалог как эпизод."""
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
            # забываем самые старые с низкой важностью
            self.episodic_memory.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)
            self.episodic_memory = self.episodic_memory[:EPISODIC_MAX_SIZE]

        # Добавляем факты из диалога
        user_id = self._add_fact(user_msg, 'user', importance=1.0, salience=salience)
        assistant_id = self._add_fact(assistant_msg, 'assistant', importance=1.2, salience=salience)
        # Связываем их
        self._create_synapse(user_id, assistant_id, weight=0.8)
        self._create_synapse(assistant_id, user_id, weight=0.6)

        # Обновляем предсказательную модель на основе последовательности
        self._update_predictive(user_id, assistant_id)

        await self._schedule_save()

    # ---------- ПРЕДСКАЗАТЕЛЬНАЯ МОДЕЛЬ ----------
    def _update_predictive(self, src_id: int, tgt_id: int):
        """Обновляет матрицу переходов."""
        self.predictive_matrix[src_id][tgt_id] += PREDICTIVE_LEARNING_RATE
        # Нормализация
        total = sum(self.predictive_matrix[src_id].values())
        if total > 0:
            for k in self.predictive_matrix[src_id]:
                self.predictive_matrix[src_id][k] /= total
        # Ограничение размера
        if len(self.predictive_matrix) > PREDICTIVE_MATRIX_MAX_SIZE:
            # удаляем наименее используемые
            pass

    def _rebuild_predictive_from_episodes(self):
        """Перестраивает predictive matrix из эпизодов."""
        self.predictive_matrix.clear()
        # Проходим по эпизодам и строим переходы между фактами (упрощённо)
        # Для каждого эпизода ищем факты с типом user и assistant, создаём переходы
        # (реализация зависит от структуры)
        # Пока оставим заглушку
        pass

    async def predict_next(self, current_fact_ids: List[int]) -> List[int]:
        """Возвращает наиболее вероятные следующие факты."""
        candidates = defaultdict(float)
        for fid in current_fact_ids:
            if fid in self.predictive_matrix:
                for nxt, prob in self.predictive_matrix[fid].items():
                    candidates[nxt] += prob
        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
        return [fid for fid, _ in sorted_candidates[:5]]

    # ---------- HEBBian / STDP ОБУЧЕНИЕ ----------
    def _hebbian_update(self, source_id: int, target_id: int, coactivation_time: float):
        """Обновляет синапс по правилу Хебба с учётом временной задержки (STDP)."""
        key = (source_id, target_id)
        if key not in self.synapses:
            return
        syn = self.synapses[key]
        dt = coactivation_time - syn.last_activation
        # STDP: если source активировался до target (dt>0) – потенциация, иначе депрессия
        if dt > 0 and dt < STDP_TIME_WINDOW:
            delta = STDP_LEARNING_RATE * (1.0 - dt / STDP_TIME_WINDOW)
        elif dt < 0 and abs(dt) < STDP_TIME_WINDOW:
            delta = -STDP_LEARNING_RATE * 0.5 * (1.0 - abs(dt) / STDP_TIME_WINDOW)
        else:
            delta = HEBBIAN_LEARNING_RATE * 0.1  # слабое хеббовское усиление

        syn.weight = min(SYNAPSE_MAX_WEIGHT, max(SYNAPSE_MIN_WEIGHT, syn.weight + delta))
        syn.last_activation = coactivation_time
        syn.coactivation_count += 1
        syn.last_coactivation = coactivation_time
        # Обновляем уверенность
        syn.confidence = min(1.0, syn.confidence + 0.01)
        self._dirty = True

    def _apply_decay(self):
        """Ослабляет все синапсы со временем."""
        now = time.time()
        for syn in self.synapses.values():
            age = now - syn.last_activation
            decay = 1.0 - SYNAPSE_DECAY_RATE * min(1.0, age / 86400)  # за сутки ~0.001
            syn.weight = max(SYNAPSE_MIN_WEIGHT, syn.weight * decay)
            syn.confidence = max(0.1, syn.confidence * decay)
        self._dirty = True

    # ---------- SPREADING ACTIVATION ----------
    async def spread_activation(self, seed_ids: List[int], max_depth: int = SPREADING_MAX_DEPTH,
                                max_nodes: int = SPREADING_MAX_NODES) -> Dict[int, float]:
        """
        Распространяет активацию от seed-узлов по графу.
        Возвращает словарь {id: activation_score}.
        """
        # Сброс активаций
        for f in self.semantic_facts:
            f.activation = 0.0

        # Инициализация seed
        for sid in seed_ids:
            if sid < len(self.semantic_facts):
                self.semantic_facts[sid].activation = 1.0

        visited = set(seed_ids)
        frontier = [(sid, 1.0, 0) for sid in seed_ids]  # (id, activation, depth)
        activation_map = {sid: 1.0 for sid in seed_ids}

        while frontier and len(activation_map) < max_nodes:
            new_frontier = []
            for fid, act, depth in frontier:
                if depth >= max_depth:
                    continue
                for neighbor in self.graph.get(fid, set()):
                    if neighbor in visited:
                        continue
                    syn = self.synapses.get((fid, neighbor))
                    weight = syn.weight if syn else 0.5
                    # decay и ослабление
                    new_act = act * weight * SPREADING_DECAY
                    if new_act < SPREADING_THRESHOLD:
                        continue
                    visited.add(neighbor)
                    # Добавляем активацию (суммируем)
                    activation_map[neighbor] = activation_map.get(neighbor, 0.0) + new_act
                    new_frontier.append((neighbor, new_act, depth+1))
            frontier = new_frontier

        # Сохраняем активации в факты
        for fid, act in activation_map.items():
            if fid < len(self.semantic_facts):
                self.semantic_facts[fid].activation = act

        return activation_map

    # ---------- ГИБРИДНЫЙ ПОИСК С РАСПРОСТРАНЕНИЕМ ----------
    async def retrieve_hybrid(self, query: str, top_k: int = 5, use_graph: bool = True) -> List[Dict]:
        """
        Гибридный поиск: BM25 + косинус + свежесть + графовая активация.
        """
        cache_key = f"hybrid_{query}_{top_k}_{use_graph}"
        if cache_key in self._cache:
            result, ts = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return result

        # 1. Кандидаты через FAISS или keywords
        candidate_indices = set()
        if self.use_embeddings and len(self.fact_embeddings) > 200 and self.index is not None and self.index.is_trained:
            try:
                q_emb = self._get_embedding(query).reshape(1, -1).astype('float32')
                self.index.nprobe = FAISS_NPROBE
                dist, idxs = self.index.search(q_emb, min(200, len(self.fact_embeddings)))
                candidate_indices = set(idxs[0].tolist())
            except Exception:
                pass

        if not candidate_indices:
            q_keywords = self._extract_keywords(query)
            for kw in q_keywords:
                for idx in self.keyword_index.get(kw, []):
                    candidate_indices.add(idx)
            if len(candidate_indices) < 3:
                candidate_indices = set(range(len(self.semantic_facts)))

        # 2. BM25
        if candidate_indices:
            texts = [self.semantic_facts[i].text for i in candidate_indices]
            if len(texts) > 1:
                try:
                    tfidf = self.tfidf.fit_transform(texts + [query])
                    vectors = tfidf[:-1]
                    q_vec = tfidf[-1]
                    bm25_scores = (vectors * q_vec.T).toarray().flatten()
                except Exception:
                    bm25_scores = np.ones(len(candidate_indices)) * 0.5
            else:
                bm25_scores = np.ones(len(candidate_indices)) * 0.5
        else:
            bm25_scores = np.zeros(0)

        # 3. Косинус
        if self.use_embeddings and self.fact_embeddings:
            try:
                q_emb = self._get_embedding(query).reshape(1, -1)
                cand_embs = np.array([self.fact_embeddings[i] for i in candidate_indices])
                if cand_embs.size > 0:
                    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
                    cand_norm = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-8)
                    cosine_scores = np.dot(cand_norm, q_norm.T).flatten()
                else:
                    cosine_scores = np.zeros(len(candidate_indices))
            except Exception:
                cosine_scores = np.zeros(len(candidate_indices))
        else:
            cosine_scores = np.zeros(len(candidate_indices))

        # 4. Свежесть
        now = time.time()
        freshness = []
        for idx in candidate_indices:
            age = now - self.semantic_facts[idx].timestamp
            freshness.append(max(0.0, 1.0 - age / (86400 * 30)))

        # 5. Графовая активация (если включена)
        graph_scores = np.ones(len(candidate_indices)) * 0.5
        if use_graph:
            # Инициируем активацию от seed – семантические ближайшие
            seed_ids = list(candidate_indices)[:10]  # возьмём топ-10 кандидатов как seeds
            activation_map = await self.spread_activation(seed_ids, max_depth=2, max_nodes=50)
            # для каждого кандидата берём активацию
            for i, idx in enumerate(candidate_indices):
                graph_scores[i] = activation_map.get(idx, 0.0)

        # 6. Динамические веса
        if DYNAMIC_WEIGHTS_ENABLED:
            # классифицируем запрос
            is_factual = bool(re.search(r'\b\d+[.,]?\d*\s*(?:USD|EUR|RUB|%|кг|км|г)\b', query))
            if is_factual:
                w_bm25, w_cos, w_fresh, w_graph = FACTUAL_WEIGHTS
            else:
                w_bm25, w_cos, w_fresh, w_graph = GENERAL_WEIGHTS
        else:
            w_bm25, w_cos, w_fresh, w_graph = (HYBRID_WEIGHT_BM25, HYBRID_WEIGHT_COSINE,
                                               HYBRID_WEIGHT_FRESHNESS, HYBRID_WEIGHT_GRAPH)

        # 7. Итоговый рейтинг
        final_scores = []
        for i, idx in enumerate(candidate_indices):
            score = (w_bm25 * bm25_scores[i] +
                     w_cos * cosine_scores[i] +
                     w_fresh * freshness[i] +
                     w_graph * graph_scores[i])
            final_scores.append((score, idx))

        final_scores.sort(reverse=True, key=lambda x: x[0])
        top_indices = [idx for _, idx in final_scores[:top_k]]

        # Расширяем соседями по графу
        expanded = set(top_indices)
        for idx in top_indices:
            expanded.update(self.graph.get(idx, set()))

        # Формируем результат
        max_score = final_scores[0][0] if final_scores else 1.0
        result = []
        for idx in expanded:
            fact = self.semantic_facts[idx]
            fact.access_count += 1
            fact.last_accessed = now
            fact.importance = min(2.0, fact.importance + 0.01)
            score = next((s for s, i in final_scores if i == idx), 0.0)
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
            })
        result.sort(key=lambda x: x["score"], reverse=True)
        result = result[:top_k*2]

        self._cache[cache_key] = (result, time.time())
        if len(self._cache) > self._cache_maxsize:
            oldest = sorted(self._cache.items(), key=lambda x: x[1][1])[0][0]
            del self._cache[oldest]

        return result

    # ---------- КОНСОЛИДАЦИЯ ----------
    async def light_consolidation(self):
        """Лёгкая консолидация: удаление дубликатов, ослабление связей."""
        async with self._lock:
            # Удаление дубликатов по тексту (совпадение > 0.8)
            to_remove = set()
            for i, f1 in enumerate(self.semantic_facts):
                if i in to_remove:
                    continue
                for j in range(i+1, len(self.semantic_facts)):
                    if j in to_remove:
                        continue
                    if self._compute_similarity(f1.text, self.semantic_facts[j].text) > 0.8:
                        # Оставляем с большей уверенностью
                        if self.semantic_facts[j].confidence > f1.confidence:
                            to_remove.add(i)
                            break
                        else:
                            to_remove.add(j)
            if to_remove:
                self.semantic_facts = [f for i, f in enumerate(self.semantic_facts) if i not in to_remove]
                self._rebuild_graph_after_removal(to_remove)
                self._build_keyword_index()
                # перестроим эмбеддинги
                if self.use_embeddings:
                    self.fact_embeddings = [emb for i, emb in enumerate(self.fact_embeddings) if i not in to_remove]
                    self.index.reset()
                    if len(self.fact_embeddings) >= FAISS_MIN_TRAIN_VECTORS:
                        self._train_index_if_needed()
                    elif self.fact_embeddings:
                        # добавляем без обучения
                        self.index.add(np.array(self.fact_embeddings).astype('float32'))

            # Ослабление синапсов (decay)
            self._apply_decay()
            await self._schedule_save()
            logger.info(f"Light consolidation done for {self.user_id[:16]}")

    async def deep_consolidation(self):
        """Глубокая консолидация: replay, обобщение, обнаружение противоречий."""
        async with self._lock:
            # Replay: выбираем эпизоды для воспроизведения
            if self.episodic_memory:
                # Сортировка по важности + салиенс
                self.episodic_memory.sort(key=lambda e: (e.importance * (1 + e.salience), e.timestamp), reverse=True)
                replay_candidates = self.episodic_memory[:REPLAY_BATCH_SIZE]
                # Для каждого эпизода – повторная активация связей
                for ep in replay_candidates:
                    # Находим факты, соответствующие сообщениям (упрощённо)
                    user_facts = [f for f in self.semantic_facts if f.text == ep.user_msg]
                    ass_facts = [f for f in self.semantic_facts if f.text == ep.assistant_msg]
                    if user_facts and ass_facts:
                        self._hebbian_update(user_facts[0].id, ass_facts[0].id, ep.timestamp)
                        self._hebbian_update(ass_facts[0].id, user_facts[0].id, ep.timestamp)

            # Обобщение: если несколько фактов очень похожи, объединяем (уже сделано в light)
            # Проверка противоречий
            for f in self.semantic_facts:
                if f.contradicts:
                    # Снижаем уверенность конфликтующих
                    for cid in f.contradicts:
                        if cid < len(self.semantic_facts):
                            other = self.semantic_facts[cid]
                            if other.confidence > 0.3:
                                f.confidence *= 0.95
                                other.confidence *= 0.95

            # Пересчёт важности
            now = time.time()
            for f in self.semantic_facts:
                age = now - f.timestamp
                recency = 1.0 / (1.0 + age / 86400)
                f.importance = 0.5 * (f.importance + recency + f.access_count / 10)
                f.importance = min(2.0, f.importance)

            # Удаление очень старых и неважных фактов (если превышен лимит)
            if len(self.semantic_facts) > SEMANTIC_MAX_FACTS:
                self.semantic_facts.sort(key=lambda f: (f.importance, f.confidence, f.timestamp), reverse=True)
                self.semantic_facts = self.semantic_facts[:SEMANTIC_MAX_FACTS]
                self._rebuild_graph_after_removal(set())
                self._build_keyword_index()

            await self._schedule_save()
            logger.info(f"Deep consolidation done for {self.user_id[:16]}")

    def _rebuild_graph_after_removal(self, removed_indices: Set[int]):
        """Перестраивает граф и синапсы после удаления фактов."""
        # Переиндексация
        new_id_map = {}
        new_facts = []
        for i, f in enumerate(self.semantic_facts):
            if i not in removed_indices:
                new_id_map[i] = len(new_facts)
                new_facts.append(f)
        self.semantic_facts = new_facts
        # Обновляем граф
        new_graph = defaultdict(set)
        new_synapses = {}
        for (src, tgt), syn in self.synapses.items():
            if src in removed_indices or tgt in removed_indices:
                continue
            new_src = new_id_map[src]
            new_tgt = new_id_map[tgt]
            new_graph[new_src].add(new_tgt)
            new_synapses[(new_src, new_tgt)] = syn
        self.graph = new_graph
        self.synapses = new_synapses

    # ---------- РАБОТА С ЦЕЛЯМИ ----------
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

    # ---------- СТАТИСТИКА ----------
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
        }

    # ---------- ЗАКРЫТИЕ ----------
    async def shutdown(self):
        if self._save_task:
            self._save_task.cancel()
        await self._save_async()