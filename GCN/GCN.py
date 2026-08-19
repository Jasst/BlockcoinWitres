"""
GCN Core Implementation
------------------------
Улучшенная версия:
- Исправлен retract (удаляет рёбра из графа)
- Добавлен embedder в hybrid_retrieve (может сам векторизовать текст)
- Добавлен get_object_history для аудита
- Исправлена загрузка графовых весов
- Все критические операции обёрнуты в RLock (быстрые) + асинхронный save/load
"""

from __future__ import annotations
import hashlib
import json
import math
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from enum import Enum
from collections import defaultdict
import copy

try:
    from config_ai import (
        HYBRID_WEIGHT_SEMANTIC, HYBRID_WEIGHT_GRAPH, HYBRID_WEIGHT_FRESHNESS,
        HYBRID_WEIGHT_EVIDENCE, HYBRID_WEIGHT_CONFIDENCE,
    )
except ImportError:
    HYBRID_WEIGHT_SEMANTIC = 0.40
    HYBRID_WEIGHT_GRAPH = 0.30
    HYBRID_WEIGHT_FRESHNESS = 0.15
    HYBRID_WEIGHT_EVIDENCE = 0.10
    HYBRID_WEIGHT_CONFIDENCE = 0.05


class KnowledgeType(Enum):
    CLAIM = "claim"
    CONCEPT = "concept"
    ENTITY = "entity"
    OBSERVATION = "observation"
    RELATIONSHIP = "relationship"
    PROCEDURE = "procedure"
    EVIDENCE = "evidence"
    HYPOTHESIS = "hypothesis"
    MEMORY_EVENT = "memory_event"


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
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def update(self, **kwargs) -> KnowledgeObject:
        new_obj = copy.deepcopy(self)
        for k, v in kwargs.items():
            if hasattr(new_obj, k):
                setattr(new_obj, k, v)
        new_obj.version += 1
        new_obj.created = datetime.now(timezone.utc)
        new_obj.content_hash = new_obj.compute_hash()
        return new_obj


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
        return meta["weight"] if meta else None

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

    # ---------- НОВЫЙ МЕТОД: удаление всех рёбер, связанных с узлом ----------
    def remove_node_edges(self, node_id: str):
        """Удаляет все исходящие и входящие рёбра для указанного узла."""
        # Исходящие
        for relation, target in self._edges.pop(node_id, []):
            self._reverse[target] = [(r, s) for r, s in self._reverse[target] if not (r == relation and s == node_id)]
            self._edge_meta.pop((node_id, relation, target), None)
        # Входящие
        for relation, source in self._reverse.pop(node_id, []):
            self._edges[source] = [(r, t) for r, t in self._edges[source] if not (r == relation and t == node_id)]
            self._edge_meta.pop((source, relation, node_id), None)


class MemoryStore:
    def __init__(self):
        self._objects: Dict[str, KnowledgeObject] = {}
        self._events: List[KnowledgeEvent] = []
        self._graph = KnowledgeGraph()
        self._embedding_index: Dict[str, List[float]] = {}
        self._by_type: Dict[KnowledgeType, Set[str]] = defaultdict(set)
        self._by_author: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()  # для быстрых операций в памяти

    # ---------- Основные операции (синхронные, но быстрые) ----------
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
            return obj.id

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
            claim.evidence.append(evidence_id)
            self.update(claim_id, {"evidence": claim.evidence}, actor)
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
            if obj.provenance:
                obj.provenance.verifications.append({
                    "verifier": verifier,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": status
                })
            event = KnowledgeEvent(
                id=str(uuid.uuid4()),
                type=EventType.VERIFY,
                timestamp=datetime.now(timezone.utc),
                actor=actor,
                target_id=obj_id,
                payload={"verifier": verifier, "status": status}
            )
            self._events.append(event)

    # ---------- ИСПРАВЛЕННЫЙ RETRACT (удаляет рёбра из графа) ----------
    def retract(self, obj_id: str, actor: str, reason: str = "") -> bool:
        with self._lock:
            obj = self._objects.pop(obj_id, None)
            if obj is None:
                return False
            # Удаляем все связи, инцидентные этому узлу
            self._graph.remove_node_edges(obj_id)
            self._by_type[obj.type].discard(obj_id)
            self._by_author[obj.author].discard(obj_id)
            self._embedding_index.pop(obj_id, None)
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

    def get_embedding(self, obj_id: str) -> Optional[List[float]]:
        return self._embedding_index.get(obj_id)

    # ---------- НОВЫЙ МЕТОД: история объекта ----------
    def get_object_history(self, obj_id: str) -> List[KnowledgeEvent]:
        """Возвращает все события, связанные с указанным объектом, отсортированные по времени."""
        return sorted(
            [e for e in self._events if e.target_id == obj_id],
            key=lambda e: e.timestamp
        )

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
        with self._lock:
            items = list(self._embedding_index.items())
        if not items:
            return []
        results = [(obj_id, self._cosine(query_vector, vec)) for obj_id, vec in items]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def graph_search(self, start_id: str, relation: Optional[str] = None, max_depth: int = 2) -> List[str]:
        return self._graph.traverse(start_id, max_depth) if relation is None else [t for _, t in self._graph.get_neighbors(start_id, relation)]

    # ---------- УЛУЧШЕННЫЙ ГИБРИДНЫЙ ПОИСК (принимает embedder) ----------
    def hybrid_retrieve(
        self,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        embedder_func: Optional[Callable[[str], List[float]]] = None,
        start_node: Optional[str] = None,
        top_k: int = 10
    ) -> List[KnowledgeObject]:
        """
        Гибридный поиск с поддержкой текста через embedder_func.
        Если передан query_text, но нет query_vector, и embedder_func задан – генерирует вектор.
        """
        # Генерируем вектор из текста, если нужно
        if query_text and query_vector is None and embedder_func:
            try:
                query_vector = embedder_func(query_text)
            except Exception as e:
                pass

        candidates = set()
        scores = {}

        # 1. Семантический поиск
        if query_vector is not None:
            sem_results = self.semantic_search(query_vector, top_k=top_k*2)
            for obj_id, score in sem_results:
                candidates.add(obj_id)
                scores[obj_id] = scores.get(obj_id, 0.0) + score * HYBRID_WEIGHT_SEMANTIC

        # 2. Графовый поиск
        if start_node:
            graph_ids = self.graph_search(start_node, max_depth=2)
            for obj_id in graph_ids:
                if obj_id == start_node:
                    continue
                candidates.add(obj_id)
                edge_weight = self._graph.get_relation_weight(start_node, "synapse", obj_id) or 1.0
                scores[obj_id] = scores.get(obj_id, 0.0) + HYBRID_WEIGHT_GRAPH * min(1.0, edge_weight)

        # 3. Дополнительные факторы
        for obj_id in list(candidates):
            obj = self._objects.get(obj_id)
            if not obj:
                candidates.remove(obj_id)
                continue
            age_days = (datetime.now(timezone.utc) - obj.created).days
            recency = max(0.0, 1.0 - age_days / 365.0) if age_days < 365 else 0.0
            scores[obj_id] = scores.get(obj_id, 0.0) + recency * HYBRID_WEIGHT_FRESHNESS
            scores[obj_id] += obj.confidence * HYBRID_WEIGHT_CONFIDENCE
            scores[obj_id] += min(len(obj.evidence), 5) / 5 * HYBRID_WEIGHT_EVIDENCE

        sorted_ids = sorted(candidates, key=lambda x: scores.get(x, 0.0), reverse=True)
        return [self._objects[oid] for oid in sorted_ids[:top_k]]

    # ---------- Персистентность (асинхронная обёртка) ----------
    def save(self, path: str):
        """Синхронное сохранение (для обратной совместимости)."""
        with self._lock:
            objects_data = {}
            for k, v in self._objects.items():
                od = dict(v.__dict__)
                od["type"] = v.type.value
                od["created"] = v.created.isoformat()
                od["provenance"] = None  # восстанавливается из событий
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
            }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, default=str, indent=2)

    def load(self, path: str):
        """Синхронная загрузка (для обратной совместимости)."""
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
                ed["timestamp"] = datetime.fromisoformat(ed["timestamp"]) if isinstance(ed["timestamp"], str) else ed["timestamp"]
                self._events.append(KnowledgeEvent(**ed))

            self._embedding_index = dict(data.get("embeddings", {}))

    # ---------- АСИНХРОННАЯ ОБЁРТКА ДЛЯ ВВОДА-ВЫВОДА ----------
    async def async_save(self, path: str):
        """Асинхронное сохранение (не блокирует event loop)."""
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.save, path)

    async def async_load(self, path: str):
        """Асинхронная загрузка (не блокирует event loop)."""
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.load, path)


# ==================== AIAdapter (без изменений, работает с новым Store) ====================
class AIAdapter:
    def __init__(self, memory_store: MemoryStore, agent_id: str):
        self.memory = memory_store
        self.agent_id = agent_id

    def query(self, question: str, context: Optional[List[str]] = None) -> str:
        retrieved = self.retrieve(question)
        return f"AI {self.agent_id} отвечает на '{question}' на основе {len(retrieved)} объектов."

    def retrieve(self, query: str, top_k: int = 5) -> List[KnowledgeObject]:
        # Здесь должен быть реальный embedder. Используем заглушку.
        import random
        vec = [random.random() for _ in range(128)]
        return self.memory.hybrid_retrieve(query_vector=vec, top_k=top_k)

    def publish(self, knowledge: Union[KnowledgeObject, Dict]) -> str:
        if isinstance(knowledge, dict):
            obj = KnowledgeObject(
                id=str(uuid.uuid4()),
                type=KnowledgeType(knowledge.get("type", "claim")),
                subject=knowledge["subject"],
                predicate=knowledge["predicate"],
                object=knowledge["object"],
                author=self.agent_id,
                created=datetime.now(timezone.utc),
                evidence=knowledge.get("evidence", []),
                confidence=knowledge.get("confidence", 0.5),
            )
        else:
            obj = knowledge
            obj.author = self.agent_id
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
    def __init__(self, store: MemoryStore):
        self.store = store

    def add_to_working(self, obj: KnowledgeObject):
        pass

    def episodic_recall(self, time_range: Tuple[datetime, datetime]) -> List[KnowledgeObject]:
        result = []
        for obj in self.store._objects.values():
            if time_range[0] <= obj.created <= time_range[1]:
                result.append(obj)
        return result


if __name__ == "__main__":
    # Демонстрация
    store = MemoryStore()
    alice = AIAdapter(store, "Alice")
    obj1 = KnowledgeObject(
        id=str(uuid.uuid4()),
        type=KnowledgeType.CLAIM,
        subject="Python",
        predicate="is_used_for",
        object="ML",
        author=alice.agent_id,
        created=datetime.now(timezone.utc),
        confidence=0.9
    )
    alice.publish(obj1)
    obj2 = KnowledgeObject(
        id=str(uuid.uuid4()),
        type=KnowledgeType.CLAIM,
        subject="ML",
        predicate="requires",
        object="Data",
        author=alice.agent_id,
        created=datetime.now(timezone.utc),
        confidence=0.8
    )
    alice.publish(obj2)
    store.link(obj1.id, obj2.id, "implies", alice.agent_id)

    # Проверка истории
    print("History for obj1:", len(store.get_object_history(obj1.id)))

    # Retract – теперь чистит граф
    store.retract(obj1.id, alice.agent_id, "test")
    print("Edges after retract:", store._graph._edges)  # Должно быть пусто