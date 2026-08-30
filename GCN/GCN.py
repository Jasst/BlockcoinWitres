"""
GCN Core Implementation
------------------------
Улучшенная версия с интеграцией FAISS, методами для работы с фактами/эпизодами/целями.
Служит единым хранилищем для когнитивного ассистента.
"""

from __future__ import annotations
import hashlib
import json
import logging
import math
import os
import threading
import uuid
import copy
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from enum import Enum
from collections import defaultdict
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Попытка импорта FAISS (опционально)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    from GCN.config_ai import (
        HYBRID_WEIGHT_SEMANTIC, HYBRID_WEIGHT_GRAPH, HYBRID_WEIGHT_FRESHNESS,
        HYBRID_WEIGHT_EVIDENCE, HYBRID_WEIGHT_CONFIDENCE,
        FAISS_NLIST, FAISS_NPROBE, FAISS_MIN_TRAIN_VECTORS,
        EMBEDDING_DIM,
        WORKING_MEMORY_SIZE
    )
except ImportError:
    HYBRID_WEIGHT_SEMANTIC = 0.40
    HYBRID_WEIGHT_GRAPH = 0.30
    HYBRID_WEIGHT_FRESHNESS = 0.15
    HYBRID_WEIGHT_EVIDENCE = 0.10
    HYBRID_WEIGHT_CONFIDENCE = 0.05
    FAISS_NLIST = 200
    FAISS_NPROBE = 30
    FAISS_MIN_TRAIN_VECTORS = 500
    EMBEDDING_DIM = 128
    WORKING_MEMORY_SIZE = 20        # <-- добавить fallback


class KnowledgeType(Enum):
    CLAIM = "claim"
    CONCEPT = "concept"
    ENTITY = "entity"
    OBSERVATION = "observation"
    RELATIONSHIP = "relationship"
    PROCEDURE = "procedure"
    EVIDENCE = "evidence"
    HYPOTHESIS = "hypothesis"
    GOAL = "goal"          # новый тип
    MEMORY_EVENT = "memory_event"


# GCN.py — после импортов, до KnowledgeType

class MemoryScope(Enum):
    PRIVATE = "private"   # личное, видно только владельцу
    SHARED = "shared"     # доступно группе/друзьям (можно расширить)
    GLOBAL = "global"     # общедоступное, коллективное знание

@dataclass
class KnowledgeObject:
    id: str
    type: KnowledgeType
    subject: str
    predicate: str
    object: Any
    author: str
    created: datetime
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.5
    version: int = 1
    content_hash: Optional[str] = None
    signature: Optional[str] = None
    provenance: Optional[Provenance] = None
    scope: MemoryScope = MemoryScope.GLOBAL      # новое поле
    source_type: Optional[str] = None            # "agent_observation", "user_input", "web_search" и т.д.

    def __post_init__(self):
        if self.content_hash is None:
            self.content_hash = self.compute_hash()

    def compute_hash(self) -> str:
        data = {
            "type": self.type.value,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "author": self.author,
            "evidence": sorted(self.evidence),
            "confidence": self.confidence,
            "version": self.version,
            "scope": self.scope.value,           # добавить в хэш
            "source_type": self.source_type,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.version += 1
        self.content_hash = self.compute_hash()
        return self

class KnowledgeIngestion:
    """
    Обрабатывает поступление новых знаний в GCN:
    - дедупликация
    - обнаружение противоречий
    - агрегация свидетельств
    - обновление confidence
    - слияние канонических утверждений
    """
    def __init__(self, store: MemoryStore, similarity_threshold: float = 0.85,
                 scope: Optional["MemoryScope"] = None):
        self.store = store
        self.similarity_threshold = similarity_threshold
        # УЛУЧШЕНИЕ: раньше _find_similar/_keyword_search были жёстко зашиты на
        # MemoryScope.GLOBAL, поэтому дедуп/усиление/обнаружение противоречий
        # работали ТОЛЬКО для глобальной памяти — PRIVATE и SHARED росли без
        # консолидации знаний вообще (только decay/дубликаты по эмбеддингам в
        # light_consolidation). Теперь одна и та же реализация обслуживает
        # любой scope: если scope не передан явно, берём scope самого
        # кандидата (так один и тот же класс переиспользуется для
        # global_ingestion/shared_ingestion/private_ingestion в
        # GCNMemoryRouter — см. memory_graph.py).
        self.scope = scope

    def submit_candidate(self, candidate: KnowledgeObject, actor: str) -> str:
        """
        Предложить знание памяти соответствующего scope (глобальной/общей/личной).
        Возвращает ID объекта (нового или существующего).
        """
        with self.store._lock:
            # 1. Поиск семантически похожих утверждений (по эмбеддингу, если есть)
            similar = self._find_similar(candidate)

            # 2. Если найдены похожие, решаем, что делать
            if similar:
                return self._merge_or_support(candidate, similar, actor)
            else:
                # 3. Новое знание — просто сохраняем
                return self.store.create(candidate, actor)

    def _find_similar(self, candidate: KnowledgeObject) -> List[KnowledgeObject]:
        """Ищет похожие объекты того же scope, что и кандидат (см. __init__)."""
        target_scope = self.scope or candidate.scope
        # Получаем эмбеддинг кандидата (если есть)
        emb = self.store.get_embedding(candidate.id)
        if emb:
            results = self.store.semantic_search(emb, top_k=10)
            similar_ids = [oid for oid, _ in results]
        else:
            # fallback по ключевым словам (если нет эмбеддинга)
            similar_ids = self._keyword_search(candidate.subject, target_scope)
        return [self.store.get(oid) for oid in similar_ids
                if oid and self.store.get(oid) and self.store.get(oid).scope == target_scope]

    def _keyword_search(self, text: str, target_scope: "MemoryScope") -> List[str]:
        # Fallback-эвристика на случай, если у кандидата не оказалось
        # эмбеддинга (embed_text() вернул None — эмбеддинги отключены).
        # Ограничиваем скан, чтобы не пройтись по всем ~20k объектам
        # стора на каждую вставку факта.
        results = []
        scanned = 0
        SCAN_LIMIT = 3000
        for obj in self.store._objects.values():
            scanned += 1
            if scanned > SCAN_LIMIT:
                break
            if obj.scope == target_scope and any(w in obj.subject.lower() for w in text.lower().split()):
                results.append(obj.id)
                if len(results) >= 10:
                    break
        return results[:10]

    def _merge_or_support(self, candidate: KnowledgeObject, similar: List[KnowledgeObject], actor: str) -> str:
        """
        Решает: объединить с существующим, добавить свидетельство или создать противоречие.
        """
        # Выбираем наиболее похожий объект (по confidence или similarity)
        best = similar[0]
        similarity = self._compute_similarity(candidate.subject, best.subject)

        if similarity > 0.95:
            # Почти идентичны — усиливаем существующий
            return self._reinforce(best, candidate, actor)
        elif similarity > 0.7:
            # Похожи, но есть различия — проверяем противоречие
            if self._is_contradictory(candidate, best):
                # Регистрируем противоречие
                self.store.register_contradiction(candidate.id, best.id, actor)
                # Понижаем уверенность обоих (уже есть в register_contradiction)
                # Возвращаем ID нового кандидата
                return candidate.id
            else:
                # Поддерживаем существующий
                return self._reinforce(best, candidate, actor)
        else:
            # Недостаточно похожи — создаём новый
            return self.store.create(candidate, actor)

    def _reinforce(self, existing: KnowledgeObject, candidate: KnowledgeObject, actor: str) -> str:
        """
        Усиливает существующее знание свидетельством от нового кандидата.
        Обновляет confidence, evidence и создаёт событие REINFORCE.
        """
        # Добавляем evidence кандидата к существующему
        new_evidence = list(existing.evidence)
        for e in candidate.evidence:
            if e not in new_evidence:
                new_evidence.append(e)

        # УЛУЧШЕНИЕ (диверсификация confidence): раньше каждый submit поднимал
        # confidence на фиксированные +5% с насыщением, НЕЗАВИСИМО от того, кто
        # его прислал. Один активный автор мог за несколько сообщений продавить
        # confidence любого своего же утверждения почти до 1.0 — то есть
        # "коллективная" уверенность на деле измеряла активность одного
        # человека, а не согласие независимых источников. Теперь трекаем
        # авторов-подтвердивших как отдельные evidence-записи вида
        # "author:<id>" и даём полный прирост (+5%) только за НОВОГО автора;
        # повторный сабмит уже подтвердившего автора почти не двигает
        # уверенность (+1%) — так confidence реально отражает разнообразие
        # источников, а не число попыток одного человека.
        author_tag = f"author:{candidate.author}"
        is_new_author = author_tag not in new_evidence
        if is_new_author:
            new_evidence.append(author_tag)
        increment = 0.05 if is_new_author else 0.01
        new_conf = min(1.0, existing.confidence + increment * (1 - existing.confidence))
        # Обновляем объект
        self.store.update(existing.id, {"evidence": new_evidence, "confidence": new_conf}, actor)
        # Создаём событие REINFORCE отдельно (можно добавить в update)
        event = KnowledgeEvent(
            id=str(uuid.uuid4()),
            type=EventType.REINFORCE,
            timestamp=datetime.now(timezone.utc),
            actor=actor,
            target_id=existing.id,
            payload={"candidate_id": candidate.id, "new_confidence": new_conf}
        )
        self.store._events.append(event)

        # ---------------------- FIX ----------------------
        # Копируем эмбеддинг из кандидата в существующий, если у существующего его нет
        if self.store.get_embedding(existing.id) is None:
            cand_emb = self.store.get_embedding(candidate.id)
            if cand_emb is not None:
                self.store.set_embedding(existing.id, cand_emb)
        # -------------------------------------------------

        return existing.id

    def _is_contradictory(self, a: KnowledgeObject, b: KnowledgeObject) -> bool:
        # Простая эвристика: если одно содержит отрицание, а другое нет, и они о схожем предмете
        neg_words = {'не', 'нет', 'без', 'против', 'отрицает'}
        has_neg_a = any(w in a.subject.lower() for w in neg_words)
        has_neg_b = any(w in b.subject.lower() for w in neg_words)
        return has_neg_a != has_neg_b

    def _compute_similarity(self, text1: str, text2: str) -> float:
        # Используем существующий метод из CognitiveMemory
        from GCN.memory_graph import CognitiveMemory
        return CognitiveMemory._compute_similarity(text1, text2)


class EventType(Enum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    LINK = "LINK"
    UNLINK = "UNLINK"
    MERGE = "MERGE"
    SPLIT = "SPLIT"
    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    VERIFY = "VERIFY"
    RETRACT = "RETRACT"
    REINFORCE = "REINFORCE"
    DECAY = "DECAY"


@dataclass
class KnowledgeEvent:
    id: str
    type: EventType
    timestamp: datetime
    actor: str
    target_id: str
    payload: Dict[str, Any]
    signature: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())


@dataclass
class Provenance:
    object_id: str
    created_by: str
    created_at: datetime
    source: Optional[str] = None
    evidence_refs: List[str] = field(default_factory=list)
    version_history: List[KnowledgeEvent] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    verifications: List[Dict] = field(default_factory=list)

    def add_event(self, event: KnowledgeEvent):
        self.version_history.append(event)


class KnowledgeGraph:
    def __init__(self):
        self._edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._reverse: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._edge_meta: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def add_relation(self, source_id: str, relation: str, target_id: str, weight: float = 1.0):
        key = (source_id, relation, target_id)
        if key not in self._edge_meta:
            self._edges[source_id].append((relation, target_id))
            self._reverse[target_id].append((relation, source_id))
        self._edge_meta[key] = {"weight": weight, "updated": datetime.now(timezone.utc).isoformat()}

    def remove_relation(self, source_id: str, relation: str, target_id: str):
        self._edges[source_id] = [(r, t) for r, t in self._edges[source_id] if not (r == relation and t == target_id)]
        self._reverse[target_id] = [(r, s) for r, s in self._reverse[target_id] if not (r == relation and s == source_id)]
        self._edge_meta.pop((source_id, relation, target_id), None)

    def set_relation_weight(self, source_id: str, relation: str, target_id: str, weight: float):
        self.add_relation(source_id, relation, target_id, weight=weight)

    def get_relation_weight(self, source_id: str, relation: str, target_id: str) -> Optional[float]:
        meta = self._edge_meta.get((source_id, relation, target_id))
        if meta:
            try:
                return float(meta["weight"])
            except (ValueError, TypeError):
                return None
        return None

    def get_neighbors(self, node_id: str, relation: Optional[str] = None) -> List[Tuple[str, str]]:
        if relation is None:
            return self._edges[node_id][:]
        return [(r, t) for r, t in self._edges[node_id] if r == relation]

    def get_incoming(self, node_id: str, relation: Optional[str] = None) -> List[Tuple[str, str]]:
        if relation is None:
            return self._reverse[node_id][:]
        return [(r, s) for r, s in self._reverse[node_id] if r == relation]

    def traverse(self, start_id: str, max_depth: int = 2) -> List[str]:
        visited = set()
        frontier = [start_id]
        depth = 0
        result = []
        while frontier and depth < max_depth:
            next_frontier = []
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                result.append(node)
                for _, neighbor in self._edges[node]:
                    if neighbor not in visited:
                        next_frontier.append(neighbor)
            frontier = next_frontier
            depth += 1
        return result

    def remove_node_edges(self, node_id: str):
        """Удаляет все исходящие и входящие рёбра для указанного узла."""
        for relation, target in self._edges.pop(node_id, []):
            self._reverse[target] = [(r, s) for r, s in self._reverse[target] if not (r == relation and s == node_id)]
            self._edge_meta.pop((node_id, relation, target), None)
        for relation, source in self._reverse.pop(node_id, []):
            self._edges[source] = [(r, t) for r, t in self._edges[source] if not (r == relation and t == node_id)]
            self._edge_meta.pop((source, relation, node_id), None)


def _coerce_scope(raw: Any) -> MemoryScope:
    """Приводит значение scope, пришедшее из JSON, к enum MemoryScope.
    Раньше save() сериализовал enum через json.dump(..., default=str), что
    для plain Enum даёт 'MemoryScope.PRIVATE', а load() эту строку обратно
    в enum не конвертировал вовсе — obj.scope оставался строкой, и любой
    код вида obj.scope.value падал после перезапуска процесса. Здесь же
    на всякий случай поддерживаем и старый (битый) формат, и новый
    (obj.scope.value), чтобы не терять уже сохранённые файлы."""
    if isinstance(raw, MemoryScope):
        return raw
    if isinstance(raw, str):
        if raw.startswith("MemoryScope."):
            name = raw.split(".", 1)[1]
            try:
                return MemoryScope[name]
            except KeyError:
                pass
        try:
            return MemoryScope(raw)
        except ValueError:
            pass
    return MemoryScope.GLOBAL


import contextlib


@contextlib.contextmanager
def _cross_process_file_lock(path: str, timeout: float = 15.0, poll: float = 0.05, stale_after: float = 30.0):
    """
    Простая межпроцессная advisory-блокировка на файл через os.O_CREAT|O_EXCL —
    работает и на Windows, и на Linux без внешних зависимостей (fcntl/msvcrt
    платформозависимы и требовали бы ветвления). Раньше save() защищал только
    self._lock (threading.RLock) — это блокирует потоки ВНУТРИ одного процесса,
    но никак не мешает второму процессу (например, mcp_server_blockcoin.py,
    работающему параллельно с чатом) писать в тот же gcn_state.json одновременно.
    stale_after — если .lock-файл старше этого времени, считаем, что владевший
    им процесс упал и не снял блокировку, и забираем её сами, а не виснем вечно.
    """
    lock_path = f"{path}.lock"
    start = time.time()
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_path) > stale_after:
                    os.remove(lock_path)
                    continue
            except OSError:
                pass
            if time.time() - start > timeout:
                raise TimeoutError(f"Не удалось получить файловую блокировку {lock_path} за {timeout}с")
            time.sleep(poll)
    try:
        yield
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.remove(lock_path)
        except OSError:
            pass


class MemoryStore:
    def __init__(self, embedding_dim: Optional[int] = None):
        self._objects: Dict[str, KnowledgeObject] = {}
        self._events: List[KnowledgeEvent] = []
        self._graph = KnowledgeGraph()
        self._embedding_index: Dict[str, List[float]] = {}
        self._by_type: Dict[KnowledgeType, Set[str]] = defaultdict(set)
        self._by_author: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()

        # --- FAISS-индекс (опционально) ---
        self.faiss_index = None
        self.faiss_id_map: Dict[str, int] = {}          # obj_id -> позиция в индексе
        self.faiss_rev_map: Dict[int, str] = {}         # позиция -> obj_id
        # ВАЖНО: embedding_dim ДОЛЖЕН совпадать с реальной размерностью векторов,
        # которые вы передаёте в set_embedding()/add_fact(embedding=...).
        # Раньше здесь всегда стояла константа EMBEDDING_DIM=128 из конфига, а
        # SentenceTransformer("all-mpnet-base-v2") отдаёт 768-мерные векторы —
        # build_faiss_index() их отбрасывал (len(vec) == self.embedding_dim
        # никогда не было True), и быстрый ANN-индекс никогда не строился.
        # Поиск при этом не падал: semantic_search() молча уходил в O(n)
        # косинусный перебор по _embedding_index — корректно, но без ускорения
        # FAISS и без гарантии, что ANN-индекс вообще когда-либо появится.
        # Передавайте фактическую размерность эмбеддера при создании
        # MemoryStore (см. CognitiveMemory.__init__ в memory_graph.py).
        self.embedding_dim = embedding_dim if embedding_dim is not None else EMBEDDING_DIM
        self._faiss_dirty = False                       # флаг для перестройки

        # mtime файла состояния на момент последней загрузки/сохранения этим
        # процессом. Используется, чтобы отличить "файл не менялся" от
        # "другой процесс (например, MCP-сервер в отдельном процессе) уже
        # что-то туда записал" — раньше save() всегда писал поверх файла
        # своим снапшотом целиком, и более поздние записи другого процесса
        # молча терялись (последний записавший побеждает).
        self._loaded_mtime: Optional[float] = None

    # ---------- Основные операции (синхронные, с блокировкой) ----------
    def create(self, obj: KnowledgeObject, actor: str) -> str:
        with self._lock:
            if obj.id in self._objects:
                raise ValueError(f"Object {obj.id} already exists")
            self._objects[obj.id] = obj
            self._by_type[obj.type].add(obj.id)
            self._by_author[obj.author].add(obj.id)
            event = KnowledgeEvent(
                id=str(uuid.uuid4()),
                type=EventType.CREATE,
                timestamp=datetime.now(timezone.utc),
                actor=actor,
                target_id=obj.id,
                payload={"object": obj.__dict__}
            )
            self._events.append(event)
            if obj.provenance:
                obj.provenance.add_event(event)
            self._faiss_dirty = True
            return obj.id

    def record_access(self, obj_id: str, actor: str):
        """Запись факта обращения к объекту (для динамики памяти)."""
        with self._lock:
            obj = self._objects.get(obj_id)
            if obj is None:
                return
            meta = obj.object if isinstance(obj.object, dict) else {}
            meta["access_count"] = meta.get("access_count", 0) + 1
            meta["last_accessed"] = datetime.now(timezone.utc).isoformat()
            # Обновляем через штатный update – создаст событие UPDATE
            self.update(obj_id, {"object": meta}, actor)

    def apply_decay(self, actor: str, decay_factor: float = 0.01):
        """Применяет распад салиентности к объектам, к которым не обращались >7 дней."""
        with self._lock:
            now = datetime.now(timezone.utc)
            for obj in list(self._objects.values()):
                meta = obj.object if isinstance(obj.object, dict) else {}
                last = meta.get("last_accessed")
                if last:
                    try:
                        age_days = (now - datetime.fromisoformat(last)).total_seconds() / 86400.0
                    except (ValueError, TypeError):
                        continue
                    if age_days > 7:
                        salience = meta.get("salience", 0.0)
                        new_salience = max(0.0, salience - decay_factor * age_days)
                        if new_salience != salience:
                            meta["salience"] = new_salience
                            self.update(obj.id, {"object": meta}, actor)
                            # Создаём событие DECAY
                            event = KnowledgeEvent(
                                id=str(uuid.uuid4()),
                                type=EventType.DECAY,
                                timestamp=now,
                                actor=actor,
                                target_id=obj.id,
                                payload={"decay": decay_factor * age_days, "new_salience": new_salience}
                            )
                            self._events.append(event)

    def register_contradiction(self, id_a: str, id_b: str, actor: str):
        """Регистрирует противоречие между двумя объектами в графе и создаёт события."""
        with self._lock:
            if id_a not in self._objects or id_b not in self._objects:
                raise ValueError("Both objects must exist")
            # Добавляем двунаправленные рёбра CONTRADICTS
            self._graph.add_relation(id_a, "CONTRADICTS", id_b, weight=1.0)
            self._graph.add_relation(id_b, "CONTRADICTS", id_a, weight=1.0)
            # Создаём события
            event1 = KnowledgeEvent(
                id=str(uuid.uuid4()),
                type=EventType.CONTRADICT,
                timestamp=datetime.now(timezone.utc),
                actor=actor,
                target_id=id_a,
                payload={"contradicts": id_b}
            )
            event2 = KnowledgeEvent(
                id=str(uuid.uuid4()),
                type=EventType.CONTRADICT,
                timestamp=datetime.now(timezone.utc),
                actor=actor,
                target_id=id_b,
                payload={"contradicts": id_a}
            )
            self._events.append(event1)
            self._events.append(event2)
            # Понижаем confidence
            obj_a = self._objects.get(id_a)
            obj_b = self._objects.get(id_b)
            if obj_a and obj_b:
                obj_a.confidence *= 0.9
                obj_b.confidence *= 0.9
                self.update(id_a, {"confidence": obj_a.confidence}, actor)
                self.update(id_b, {"confidence": obj_b.confidence}, actor)

    def compute_confidence(self, obj_id: str) -> float:
        """Вычисляет итоговую уверенность с учётом evidence, verifications и противоречий."""
        obj = self._objects.get(obj_id)
        if not obj:
            return 0.0
        base = obj.confidence
        # Бонус за количество свидетельств (до 5)
        ev_count = len(obj.evidence)
        ev_bonus = min(ev_count, 5) / 5 * 0.1
        # Бонус за верификации (до 5)
        ver_count = 0
        if obj.provenance:
            ver_count = len(obj.provenance.verifications)
        ver_bonus = min(ver_count, 5) * 0.01
        # Штраф за активные противоречия
        contrad_penalty = 0.0
        for _, neighbor in self._graph.get_neighbors(obj_id, "CONTRADICTS"):
            # Проверяем, что объект-сосед всё ещё существует
            if neighbor in self._objects:
                contrad_penalty += 0.05
        return max(0.0, min(1.0, base + ev_bonus + ver_bonus - contrad_penalty))

    def get(self, obj_id: str) -> Optional[KnowledgeObject]:
        return self._objects.get(obj_id)

    def update(self, obj_id: str, new_data: Dict[str, Any], actor: str) -> Optional[KnowledgeObject]:
        with self._lock:
            old = self._objects.get(obj_id)
            if not old:
                return None
            new_obj = old.update(**new_data)
            self._objects[obj_id] = new_obj
            event = KnowledgeEvent(
                id=str(uuid.uuid4()),
                type=EventType.UPDATE,
                timestamp=datetime.now(timezone.utc),
                actor=actor,
                target_id=obj_id,
                payload={"old_version": old.version, "new_version": new_obj.version, "changes": new_data}
            )
            self._events.append(event)
            if new_obj.provenance:
                new_obj.provenance.add_event(event)
            self._faiss_dirty = True
            return new_obj

    def link(self, source_id: str, target_id: str, relation: str, actor: str, weight: float = 1.0):
        with self._lock:
            if source_id not in self._objects or target_id not in self._objects:
                raise ValueError("Both objects must exist")
            self._graph.add_relation(source_id, relation, target_id, weight=weight)
            event = KnowledgeEvent(
                id=str(uuid.uuid4()),
                type=EventType.LINK,
                timestamp=datetime.now(timezone.utc),
                actor=actor,
                target_id=source_id,
                payload={"relation": relation, "target": target_id, "weight": weight}
            )
            self._events.append(event)

    def set_relation_weight(self, source_id: str, target_id: str, relation: str, weight: float, actor: str):
        with self._lock:
            self._graph.set_relation_weight(source_id, relation, target_id, weight)
            event = KnowledgeEvent(
                id=str(uuid.uuid4()),
                type=EventType.REINFORCE if weight > 0 else EventType.DECAY,
                timestamp=datetime.now(timezone.utc),
                actor=actor,
                target_id=source_id,
                payload={"relation": relation, "target": target_id, "weight": weight}
            )
            self._events.append(event)

    def unlink(self, source_id: str, target_id: str, relation: str, actor: str):
        with self._lock:
            self._graph.remove_relation(source_id, relation, target_id)
            event = KnowledgeEvent(
                id=str(uuid.uuid4()),
                type=EventType.UNLINK,
                timestamp=datetime.now(timezone.utc),
                actor=actor,
                target_id=source_id,
                payload={"relation": relation, "target": target_id}
            )
            self._events.append(event)

    def add_evidence(self, claim_id: str, evidence_id: str, actor: str):
        with self._lock:
            claim = self._objects.get(claim_id)
            if not claim:
                raise ValueError("Claim not found")
            # Добавляем evidence
            new_evidence = claim.evidence + [evidence_id]
            self.update(claim_id, {"evidence": new_evidence}, actor)
            # Событие SUPPORT
            event = KnowledgeEvent(
                id=str(uuid.uuid4()),
                type=EventType.SUPPORT,
                timestamp=datetime.now(timezone.utc),
                actor=actor,
                target_id=claim_id,
                payload={"evidence_id": evidence_id}
            )
            self._events.append(event)

    def verify(self, obj_id: str, verifier: str, status: str, actor: str):
        with self._lock:
            obj = self._objects.get(obj_id)
            if not obj:
                raise ValueError("Object not found")
            # Обновляем provenance
            if obj.provenance is None:
                obj.provenance = Provenance(object_id=obj_id, created_by=actor, created_at=datetime.now(timezone.utc))
            obj.provenance.verifications.append({
                "verifier": verifier,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": status
            })
            # Обновляем объект (вызовет событие UPDATE)
            self.update(obj_id, {"provenance": obj.provenance}, actor)
            # Дополнительно событие VERIFY
            event = KnowledgeEvent(
                id=str(uuid.uuid4()),
                type=EventType.VERIFY,
                timestamp=datetime.now(timezone.utc),
                actor=actor,
                target_id=obj_id,
                payload={"verifier": verifier, "status": status}
            )
            self._events.append(event)

    def retract(self, obj_id: str, actor: str, reason: str = "") -> bool:
        with self._lock:
            obj = self._objects.pop(obj_id, None)
            if obj is None:
                return False
            # Удаляем все рёбра, включая CONTRADICTS
            self._graph.remove_node_edges(obj_id)
            # Также удаляем обратные рёбра CONTRADICTS от других объектов к этому
            for other_id in list(self._objects.keys()):
                self._graph.remove_relation(other_id, "CONTRADICTS", obj_id)
            # Остальная логика без изменений ...
            self._by_type[obj.type].discard(obj_id)
            self._by_author[obj.author].discard(obj_id)
            self._embedding_index.pop(obj_id, None)
            if obj_id in self.faiss_id_map:
                del self.faiss_id_map[obj_id]
            self._faiss_dirty = True
            event = KnowledgeEvent(
                id=str(uuid.uuid4()),
                type=EventType.RETRACT,
                timestamp=datetime.now(timezone.utc),
                actor=actor,
                target_id=obj_id,
                payload={"reason": reason}
            )
            self._events.append(event)
            if obj.provenance:
                obj.provenance.add_event(event)
            return True

    # ---------- Вспомогательные индексы ----------
    def set_embedding(self, obj_id: str, vector: List[float]):
        self._embedding_index[obj_id] = vector
        self._faiss_dirty = True

    def get_embedding(self, obj_id: str) -> Optional[List[float]]:
        return self._embedding_index.get(obj_id)

    def get_object_history(self, obj_id: str) -> List[KnowledgeEvent]:
        return sorted(
            [e for e in self._events if e.target_id == obj_id],
            key=lambda e: e.timestamp
        )

    # ---------- Методы для работы с фактами (добавлены) ----------
    def add_fact(self, text: str, fact_type: str, author: str,
                 confidence: float = 0.5, importance: float = 1.0,
                 embedding: Optional[List[float]] = None,
                 local_id: Optional[int] = None) -> str:  # ✅ новый параметр
        meta = {
            "value": "true",
            "importance": importance,
            "fact_type": fact_type,
            "salience": 0.0,
            "stability": 0.5,
            "plasticity": 0.5,
            "prediction_error": 0.0,
            "access_count": 0,
            "last_accessed": None,
            "local_id": local_id  # ✅ сохраняем
        }
        obj = KnowledgeObject(
            id=f"fact_{uuid.uuid4()}",
            type=KnowledgeType.CLAIM,
            subject=text,
            predicate="is_fact",
            object=meta,
            author=author,
            created=datetime.now(timezone.utc),
            confidence=confidence,
            evidence=[],
            version=1
        )
        obj_id = self.create(obj, author)
        if embedding is not None:
            self.set_embedding(obj_id, embedding)
        return obj_id

    def add_episode(self, user_msg: str, assistant_msg: str, author: str,
                    salience: float = 0.0) -> Tuple[str, str]:
        """
        Создаёт два объекта MEMORY_EVENT и связывает их.
        Возвращает (user_obj_id, assistant_obj_id).
        """
        user_obj = KnowledgeObject(
            id=f"ep_{uuid.uuid4()}",
            type=KnowledgeType.MEMORY_EVENT,
            subject=user_msg,
            predicate="user_message",
            object={"salience": salience},
            author=author,
            created=datetime.now(timezone.utc),
            confidence=1.0
        )
        user_id = self.create(user_obj, author)

        ass_obj = KnowledgeObject(
            id=f"ep_{uuid.uuid4()}",
            type=KnowledgeType.MEMORY_EVENT,
            subject=assistant_msg,
            predicate="assistant_message",
            object={"salience": salience},
            author=author,
            created=datetime.now(timezone.utc),
            confidence=1.0
        )
        ass_id = self.create(ass_obj, author)

        self.link(user_id, ass_id, "replied_with", author, weight=0.9)
        self.link(ass_id, user_id, "in_response_to", author, weight=0.7)
        return user_id, ass_id

    def add_goal(self, description: str, author: str, priority: float = 0.5,
                 confidence: float = 0.5) -> str:
        obj = KnowledgeObject(
            id=f"goal_{uuid.uuid4()}",
            type=KnowledgeType.HYPOTHESIS,
            subject=description,
            predicate="is_goal",
            object={"status": "active", "priority": priority},
            author=author,
            created=datetime.now(timezone.utc),
            confidence=confidence,
            evidence=[]
        )
        return self.create(obj, author)

    def update_goal_status(self, goal_id: str, status: str, actor: str):
        """Обновляет статус цели (active, completed, failed)."""
        obj = self.get(goal_id)
        if obj and obj.type == KnowledgeType.HYPOTHESIS:
            new_object = obj.object.copy() if isinstance(obj.object, dict) else {}
            new_object["status"] = status
            self.update(goal_id, {"object": new_object}, actor)

    def get_active_goals(self, author: Optional[str] = None) -> List[KnowledgeObject]:
        """Возвращает все цели со статусом 'active'."""
        result = []
        for obj in self._objects.values():
            if obj.type == KnowledgeType.HYPOTHESIS:
                if isinstance(obj.object, dict) and obj.object.get("status") == "active":
                    if author is None or obj.author == author:
                        result.append(obj)
        return result

    def delete_fact(self, fact_id: str, actor: str) -> bool:
        """Удаляет факт (объект типа CLAIM) и все связанные рёбра."""
        obj = self.get(fact_id)
        if obj and obj.type == KnowledgeType.CLAIM:
            return self.retract(fact_id, actor, reason="user_delete")
        return False

    # ---------- Получение всех объектов определённого типа ----------
    def get_all_facts(self, author: Optional[str] = None) -> List[KnowledgeObject]:
        result = []
        for obj in self._objects.values():
            if obj.type == KnowledgeType.CLAIM:
                if author is None or obj.author == author:
                    result.append(obj)
        return result

    def get_all_episodes(self, author: Optional[str] = None) -> List[KnowledgeObject]:
        result = []
        for obj in self._objects.values():
            if obj.type == KnowledgeType.MEMORY_EVENT:
                if author is None or obj.author == author:
                    result.append(obj)
        return result

    # ---------- FAISS-индекс (опционально) ----------
    def build_faiss_index(self, force: bool = False):
        if not FAISS_AVAILABLE:
            return
        with self._lock:
            if not self._faiss_dirty and not force:
                return
            ids_with_emb = []
            vectors = []
            for obj_id, vec in self._embedding_index.items():
                if len(vec) == self.embedding_dim:
                    ids_with_emb.append(obj_id)
                    vectors.append(vec)
            # Самокоррекция размерности, если необходимо
            if not vectors and self._embedding_index:
                dim_counts = defaultdict(int)
                for vec in self._embedding_index.values():
                    dim_counts[len(vec)] += 1
                actual_dim = max(dim_counts.items(), key=lambda kv: kv[1])[0]
                logger.warning(f"embedding_dim mismatch, switching to {actual_dim}")
                self.embedding_dim = actual_dim
                for obj_id, vec in self._embedding_index.items():
                    if len(vec) == self.embedding_dim:
                        ids_with_emb.append(obj_id)
                        vectors.append(vec)

            n_vectors = len(vectors)
            if n_vectors == 0:
                self.faiss_index = None
                self.faiss_id_map = {}
                self.faiss_rev_map = {}
                self._faiss_dirty = False
                return

            vectors_np = np.array(vectors).astype('float32')
            # Нормализуем для косинусного сходства (IndexFlatIP работает с нормализованными)
            faiss.normalize_L2(vectors_np)

            # Выбор типа индекса
            if n_vectors < 50:  # малый набор – точный поиск
                index = faiss.IndexFlatIP(self.embedding_dim)
                index.add(vectors_np)
            elif n_vectors < 500:  # средний – HNSW для скорости
                index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
                index.hnsw.efConstruction = 80
                index.add(vectors_np)
            else:  # большой – IVF
                nlist = max(1, min(FAISS_NLIST, int(math.sqrt(n_vectors) * 2)))
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                # Обучаем на подмножестве (если данных мало, берём все)
                index.train(vectors_np)
                index.add(vectors_np)
                index.nprobe = min(FAISS_NPROBE, nlist)

            self.faiss_index = index
            self.faiss_id_map = {obj_id: i for i, obj_id in enumerate(ids_with_emb)}
            self.faiss_rev_map = {i: obj_id for i, obj_id in enumerate(ids_with_emb)}
            self._faiss_dirty = False

    def _ensure_faiss(self):
        """Вызывается перед поиском, чтобы перестроить индекс при необходимости."""
        if self._faiss_dirty:
            self.build_faiss_index()

    # ---------- Поиск ----------
    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return dot / (norm_a * norm_b)

    def semantic_search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Поиск по вектору с использованием FAISS (если доступен) или косинусного сходства.
        """
        self._ensure_faiss()
        if self.faiss_index is not None and FAISS_AVAILABLE:
            try:
                q = np.array(query_vector).astype('float32').reshape(1, -1)
                self.faiss_index.nprobe = FAISS_NPROBE
                dist, idxs = self.faiss_index.search(q, min(top_k * 3, len(self.faiss_rev_map)))
                results = []
                for i, pos in enumerate(idxs[0]):
                    if pos >= 0:
                        obj_id = self.faiss_rev_map.get(pos)
                        if obj_id:
                            # dist — это L2, преобразуем в сходство (чем меньше, тем лучше)
                            sim = 1.0 / (1.0 + dist[0][i])
                            results.append((obj_id, sim))
                return results[:top_k]
            except Exception as e:
                # fallback
                pass

        # Косинусное сходство по всем векторам
        results = []
        for obj_id, vec in self._embedding_index.items():
            sim = self._cosine(query_vector, vec)
            results.append((obj_id, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def graph_search(self, start_id: str, relation: Optional[str] = None, max_depth: int = 2) -> List[str]:
        if relation is None:
            return self._graph.traverse(start_id, max_depth)
        else:
            return [t for _, t in self._graph.get_neighbors(start_id, relation)]

    def hybrid_retrieve(
            self,
            query_vector: Optional[List[float]] = None,
            query_text: Optional[str] = None,
            embedder_func: Optional[Callable[[str], List[float]]] = None,
            start_node: Optional[str] = None,
            top_k: int = 10,
            alpha: float = 0.7,
            weights: Optional[Dict[str, float]] = None
    ) -> List[KnowledgeObject]:
        # ... генерация вектора из текста ...
        if query_text and query_vector is None and embedder_func:
            try:
                query_vector = embedder_func(query_text)
            except Exception:
                pass

        # Веса с нормализацией
        if weights is None:
            weights = {}
        default_weights = {
            'semantic': HYBRID_WEIGHT_SEMANTIC,
            'graph': HYBRID_WEIGHT_GRAPH,
            'freshness': HYBRID_WEIGHT_FRESHNESS,
            'confidence': HYBRID_WEIGHT_CONFIDENCE,
            'evidence': HYBRID_WEIGHT_EVIDENCE,
        }
        for k in default_weights:
            weights.setdefault(k, default_weights[k])
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        candidates = set()
        scores = {}

        # 1. Семантический
        if query_vector is not None:
            sem_results = self.semantic_search(query_vector, top_k=top_k * 3)
            for obj_id, sim in sem_results:
                candidates.add(obj_id)
                scores[obj_id] = scores.get(obj_id, 0.0) + max(0.0, min(1.0, sim)) * weights['semantic']

        # 2. Графовый
        if start_node:
            graph_ids = self.graph_search(start_node, max_depth=2)
            for obj_id in graph_ids:
                if obj_id == start_node:
                    continue
                candidates.add(obj_id)
                edge_weight = self._graph.get_relation_weight(start_node, "synapse", obj_id) or 1.0
                scores[obj_id] = scores.get(obj_id, 0.0) + min(1.0, edge_weight) * weights['graph']

        # 3. Дополнительные факторы
        now = datetime.now(timezone.utc)
        for obj_id in list(candidates):
            obj = self._objects.get(obj_id)
            if not obj:
                candidates.remove(obj_id)
                continue
            # ИСПРАВЛЕНИЕ: раньше "свежесть" считалась только от obj.created —
            # даты первого создания факта. Из-за этого стабильное, часто
            # подтверждаемое/вспоминаемое ядро знаний (record_access пишет
            # last_accessed в obj.object) со временем неизбежно теряло весь
            # freshness-баллы просто потому, что было создано давно, даже если
            # к нему обращались вчера — а разовый шумовой факт недельной
            # давности оказывался "свежее" только по дате создания. Теперь
            # используем last_accessed из метаданных, если он есть, и только
            # при его отсутствии откатываемся на created.
            reference_time = obj.created
            meta = obj.object if isinstance(obj.object, dict) else {}
            last_accessed_raw = meta.get("last_accessed")
            if last_accessed_raw:
                try:
                    reference_time = datetime.fromisoformat(last_accessed_raw)
                except (ValueError, TypeError):
                    pass
            age_days = (now - reference_time).days
            recency = max(0.0, 1.0 - age_days / 365.0)
            scores[obj_id] = scores.get(obj_id, 0.0) + recency * weights['freshness']
            scores[obj_id] += obj.confidence * weights['confidence']
            ev = min(len(obj.evidence), 5) / 5
            scores[obj_id] += ev * weights['evidence']

        # Сортировка и возврат объектов
        sorted_ids = sorted(candidates, key=lambda x: scores.get(x, 0.0), reverse=True)
        return [self._objects[oid] for oid in sorted_ids[:top_k]]

    # ---------- Персистентность ----------
    def _merge_disk_state(self, path: str):
        """
        Если файл состояния изменился с момента нашей последней загрузки
        (значит, его записал другой процесс — например, MCP-сервер, идущий
        отдельным процессом от основного FastAPI-приложения), подтягиваем
        то, что там появилось, вместо того чтобы просто затереть его своим
        снапшотом.

        Стратегия слияния:
          - объекты: побеждает более высокая version (KnowledgeObject.version
            инкрементится на каждый update()), при равенстве — уже
            присутствующий в памяти (наш) вариант;
          - события: объединение по KnowledgeEvent.id (событие — неизменяемый
            факт, дублей по смыслу быть не может);
          - рёбра графа: объединение множеств, вес ребра — максимум из двух.

        Не идеальная CRDT-семантика, но устраняет главный риск: полную потерю
        записей одного процесса при сохранении другого.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                disk_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Не удалось прочитать {path} для слияния: {e}")
            return

        # --- объекты: побеждает больший version ---
        for k, v in disk_data.get("objects", {}).items():
            local_obj = self._objects.get(k)
            disk_version = v.get("version", 1)
            if local_obj is None or disk_version > local_obj.version:
                vv = dict(v)
                vv["type"] = KnowledgeType(vv["type"])
                vv["created"] = datetime.fromisoformat(vv["created"]) if isinstance(vv["created"], str) else vv["created"]
                vv["provenance"] = None
                vv["scope"] = _coerce_scope(vv.get("scope", MemoryScope.GLOBAL))
                obj = KnowledgeObject(**vv)
                self._objects[k] = obj
                self._by_type[obj.type].add(k)
                self._by_author[obj.author].add(k)

        # --- события: объединение по id ---
        known_event_ids = {e.id for e in self._events}
        for ed in disk_data.get("events", []):
            if ed.get("id") in known_event_ids:
                continue
            ed = dict(ed)
            ed["type"] = EventType(ed["type"])
            ed["timestamp"] = datetime.fromisoformat(ed["timestamp"]) if isinstance(ed["timestamp"], str) else ed["timestamp"]
            self._events.append(KnowledgeEvent(**ed))

        # --- граф: объединение рёбер, вес = максимум ---
        for src, edges in disk_data.get("graph_edges", {}).items():
            for relation, target in edges:
                existing_weight = self._graph.get_relation_weight(src, relation, target)
                meta_key = "|".join([src, relation, target])
                disk_weight = disk_data.get("graph_edge_meta", {}).get(meta_key, {}).get("weight", 1.0)
                weight = max(existing_weight or 0.0, disk_weight)
                self._graph.add_relation(src, relation, target, weight=weight)

        # --- эмбеддинги: подтягиваем те, которых у нас ещё нет ---
        for k, v in disk_data.get("embeddings", {}).items():
            self._embedding_index.setdefault(k, v)

        logger.info(f"Слияние с диском: {path} — учтены изменения другого процесса")

    def save(self, path: str):
        """Синхронное сохранение. Перед записью подтягивает изменения,
        сделанные другим процессом с момента нашей последней загрузки —
        см. _merge_disk_state(). Теперь также защищено межпроцессной
        файловой блокировкой и пишет через tmp+rename, чтобы падение
        процесса посреди записи не оставляло битый gcn_state.json."""
        with _cross_process_file_lock(path):
            with self._lock:
                if os.path.exists(path):
                    try:
                        disk_mtime = os.path.getmtime(path)
                        if self._loaded_mtime is None or disk_mtime > self._loaded_mtime:
                            self._merge_disk_state(path)
                    except OSError:
                        pass
                objects_data = {}
                for k, v in self._objects.items():
                    od = dict(v.__dict__)
                    od["type"] = v.type.value
                    od["created"] = v.created.isoformat()
                    od["provenance"] = None
                    od["scope"] = v.scope.value if isinstance(v.scope, MemoryScope) else v.scope
                    objects_data[k] = od
                data = {
                    "objects": objects_data,
                    "events": [
                        {**e.__dict__, "type": e.type.value, "timestamp": e.timestamp.isoformat()}
                        for e in self._events
                    ],
                    "graph_edges": dict(self._graph._edges),
                    "graph_edge_meta": {
                        "|".join(k): v for k, v in self._graph._edge_meta.items()
                    },
                    "embeddings": self._embedding_index,
                    "faiss_dirty": self._faiss_dirty,
                    "embedding_dim": self.embedding_dim,
                }
            # Атомарная запись: сначала во временный файл в той же директории
            # (чтобы rename был атомарным на одной ФС), затем os.replace().
            # Раньше писали сразу в path — конкурентный читатель (или crash
            # посреди json.dump) мог увидеть/оставить наполовину записанный JSON.
            tmp_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
            try:
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, default=str, indent=2, ensure_ascii=False)
                os.replace(tmp_path, path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
            try:
                self._loaded_mtime = os.path.getmtime(path)
            except OSError:
                self._loaded_mtime = time.time()

    def load(self, path: str):
        """Синхронная загрузка."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with self._lock:
            self._objects = {}
            self._by_type = defaultdict(set)
            self._by_author = defaultdict(set)
            for k, v in data.get("objects", {}).items():
                v = dict(v)
                v["type"] = KnowledgeType(v["type"])
                v["created"] = datetime.fromisoformat(v["created"]) if isinstance(v["created"], str) else v["created"]
                v["provenance"] = None
                v["scope"] = _coerce_scope(v.get("scope", MemoryScope.GLOBAL))
                obj = KnowledgeObject(**v)
                self._objects[k] = obj
                self._by_type[obj.type].add(k)
                self._by_author[obj.author].add(k)

            self._graph = KnowledgeGraph()
            edge_meta = data.get("graph_edge_meta", {})
            for src, edges in data.get("graph_edges", {}).items():
                for relation, target in edges:
                    meta_key = "|".join([src, relation, target])
                    weight = edge_meta.get(meta_key, {}).get("weight", 1.0)
                    self._graph.add_relation(src, relation, target, weight=weight)

            self._events = []
            for ed in data.get("events", []):
                ed = dict(ed)
                ed["type"] = EventType(ed["type"])
                ed["timestamp"] = datetime.fromisoformat(ed["timestamp"]) if isinstance(ed["timestamp"], str) else ed[
                    "timestamp"]
                self._events.append(KnowledgeEvent(**ed))

            self._embedding_index = dict(data.get("embeddings", {}))
            saved_dim = data.get("embedding_dim")
            if saved_dim and self.embedding_dim == EMBEDDING_DIM:
                self.embedding_dim = saved_dim
            self._faiss_dirty = data.get("faiss_dirty", True)
        try:
            self._loaded_mtime = os.path.getmtime(path)
        except OSError:
            self._loaded_mtime = time.time()

    # ---------- Асинхронные обёртки ----------
    async def async_save(self, path: str):
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.save, path)

    async def async_load(self, path: str):
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.load, path)


# ==================== AIAdapter (без изменений, адаптирован под новый Store) ====================
class AIAdapter:
    def __init__(self, memory_store: MemoryStore, agent_id: str,
                 embedder_func: Optional[Callable[[str], List[float]]] = None):
        self.memory = memory_store
        self.agent_id = agent_id
        # Раньше retrieve() генерировал СЛУЧАЙНЫЙ вектор вместо реального
        # эмбеддинга ("Здесь должен быть реальный embedder. Используем
        # заглушку.") — то есть semantic-часть hybrid_retrieve() на практике
        # была шумом. Сейчас не вызывается из основного пайплайна
        # ai_assistant.py (используется только .publish()), но это была
        # готовая мина для любого будущего кода (например, agent_core.py),
        # который решит воспользоваться AIAdapter.retrieve()/.query().
        # Передавайте сюда реальную функцию эмбеддинга, например:
        #   AIAdapter(store, user_id, embedder_func=lambda t: memory.embedder.encode(t).tolist())
        self.embedder_func = embedder_func

    def query(self, question: str, context: Optional[List[str]] = None) -> str:
        retrieved = self.retrieve(question)
        return f"AI {self.agent_id} отвечает на '{question}' на основе {len(retrieved)} объектов."

    def retrieve(self, query: str, top_k: int = 5) -> List[KnowledgeObject]:
        if self.embedder_func is None:
            # Без реального эмбеддера MemoryStore.hybrid_retrieve() не имеет
            # текстового (BM25/keyword) пути — только семантика по вектору и
            # обход графа от start_node. Раньше здесь подставлялся случайный
            # вектор, что тихо портило скоринг правдоподобным на вид, но
            # бессмысленным результатом. Честнее вернуть пусто и залогировать,
            # чем притворяться, что поиск отработал.
            logger.warning("AIAdapter.retrieve() called without embedder_func — no semantic path available, returning [].")
            return []
        try:
            return self.memory.hybrid_retrieve(query_text=query, embedder_func=self.embedder_func, top_k=top_k)
        except Exception as e:
            logger.warning(f"AIAdapter.retrieve: embedder_func failed ({e})")
            return []

    def publish(self, knowledge: Union[KnowledgeObject, Dict]) -> str:
        # ВАЖНО: AIAdapter пишет напрямую в self.memory — тот MemoryStore,
        # с которым он был создан (в ai_assistant.py это store ПРИВАТНОЙ
        # памяти пользователя). KnowledgeObject.scope по умолчанию =
        # MemoryScope.GLOBAL (см. dataclass выше), и раньше это поле здесь
        # не переопределялось: объект физически лежал в приватном сторе,
        # но был помечен как GLOBAL — GCNMemoryRouter/KnowledgeIngestion
        # никогда его не увидят, а любой код, ориентирующийся на obj.scope,
        # ошибочно считал бы его глобальным. Проставляем scope, реально
        # соответствующий тому, куда объект попадает физически (this
        # store), если вызывающий явно не указал иное.
        if isinstance(knowledge, dict):
            # ожидаем поля: subject, predicate, object, type (опционально), confidence
            obj = KnowledgeObject(
                id=str(uuid.uuid4()),
                type=KnowledgeType(knowledge.get("type", "claim")),
                subject=knowledge["subject"],
                predicate=knowledge.get("predicate", ""),
                object=knowledge.get("object", ""),
                author=self.agent_id,
                created=datetime.now(timezone.utc),
                evidence=knowledge.get("evidence", []),
                confidence=knowledge.get("confidence", 0.5),
                scope=knowledge.get("scope", MemoryScope.PRIVATE),
            )
        else:
            obj = knowledge
            obj.author = self.agent_id
            if "scope" not in obj.__dict__ or obj.scope is None:
                obj.scope = MemoryScope.PRIVATE
        # Эмбеддинг для publish() не проставляется автоматически — вызывающий
        # код, которому нужна семантическая находимость этого объекта,
        # должен сам вызвать self.memory.set_embedding(obj_id, vector)
        # (см. GCNMemoryRouter.add_knowledge для образца) либо использовать
        # router.add_knowledge() вместо публикации через AIAdapter.
        return self.memory.create(obj, self.agent_id)

    def verify(self, obj_id: str, status: str):
        self.memory.verify(obj_id, self.agent_id, status, self.agent_id)

    def explain(self, obj_id: str) -> Dict:
        obj = self.memory.get(obj_id)
        if not obj:
            return {"error": "not found"}
        history = self.memory.get_object_history(obj_id)
        return {
            "id": obj.id,
            "author": obj.author,
            "created": obj.created.isoformat(),
            "confidence": obj.confidence,
            "evidence": obj.evidence,
            "history_events": len(history),
            "provenance": obj.provenance.__dict__ if obj.provenance else None,
        }


class MemoryHierarchy:
    def __init__(self, store: MemoryStore, size: int = WORKING_MEMORY_SIZE):
        self.store = store
        self.working_memory: List[str] = []   # список id объектов
        self.size = size

    def add_to_working(self, obj_id: str):
        """Добавляет объект в рабочую память (или перемещает в конец)."""
        if obj_id not in self.working_memory:
            self.working_memory.append(obj_id)
        else:
            self.working_memory.remove(obj_id)
            self.working_memory.append(obj_id)
        # Ограничиваем размер
        while len(self.working_memory) > self.size:
            self.working_memory.pop(0)

    def get_working(self) -> List[KnowledgeObject]:
        """Возвращает объекты рабочей памяти (живые)."""
        result = []
        for oid in self.working_memory:
            obj = self.store.get(oid)
            if obj:
                result.append(obj)
        return result

    def clear_working(self):
        self.working_memory.clear()


# ==================== Демонстрация ====================
if __name__ == "__main__":
    store = MemoryStore()
    alice = AIAdapter(store, "Alice")
    # Добавляем факт
    fact_id = store.add_fact("Python используется для машинного обучения", "claim", "Alice", confidence=0.9)
    # Добавляем эпизод
    user_id, ass_id = store.add_episode("Что такое GCN?", "GCN это графовая свёрточная сеть", "Alice")
    # Добавляем цель
    goal_id = store.add_goal("Изучить графовые нейросети", "Alice", priority=0.8)
    # Активные целиф
    goals = store.get_active_goals()
    print("Active goals:", [g.subject for g in goals])
    # Гибридный поиск (заглушка вектора)
    results = store.hybrid_retrieve(query_vector=[0.5]*128, start_node=fact_id, top_k=3)
    print("Found objects:", [obj.subject for obj in results])