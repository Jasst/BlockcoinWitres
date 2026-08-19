"""
GCN Core Implementation
------------------------
Минимальная реализация ключевых компонентов Global Cognitive Network.
Соответствует пунктам 3–23 промта.
"""

from __future__ import annotations
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum
from collections import defaultdict
import copy


# ==================== 1. Типы знаний ====================
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


# ==================== 2. Knowledge Object ====================
@dataclass
class KnowledgeObject:
    """Ядро GCN – объект знания с provenance и криптографической целостностью."""
    id: str
    type: KnowledgeType
    subject: str
    predicate: str
    object: Any
    author: str
    created: datetime
    evidence: List[str] = field(default_factory=list)          # ссылки на evidence-объекты
    confidence: float = 0.5                                     # 0..1
    version: int = 1
    content_hash: Optional[str] = None
    signature: Optional[str] = None                            # для будущей подписи
    provenance: Optional[Provenance] = None                    # ссылка на детальный provenance

    def __post_init__(self):
        if self.content_hash is None:
            self.content_hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Вычисляет хеш содержимого объекта (без учёта временных метаданных)."""
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
        """Создаёт новую версию объекта (immutable)."""
        new_obj = copy.deepcopy(self)
        for k, v in kwargs.items():
            if hasattr(new_obj, k):
                setattr(new_obj, k, v)
        new_obj.version += 1
        new_obj.created = datetime.utcnow()
        new_obj.content_hash = new_obj.compute_hash()
        return new_obj


# ==================== 3. Событийная память ====================
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
    """Неизменяемое событие, описывающее изменение состояния знания."""
    id: str
    type: EventType
    timestamp: datetime
    actor: str                           # кто совершил событие (AI или пользователь)
    target_id: str                       # ID KnowledgeObject
    payload: Dict[str, Any]              # дополнительные данные (старое/новое состояние, связи и т.п.)
    signature: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())


# ==================== 4. Provenance (детальная история) ====================
@dataclass
class Provenance:
    """Детальная информация о происхождении знания."""
    object_id: str
    created_by: str
    created_at: datetime
    source: Optional[str] = None          # внешний источник (URL, документ)
    evidence_refs: List[str] = field(default_factory=list)
    version_history: List[KnowledgeEvent] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)   # ID объектов, которые противоречат
    verifications: List[Dict] = field(default_factory=list)   # {verifier, timestamp, status}

    def add_event(self, event: KnowledgeEvent):
        self.version_history.append(event)


# ==================== 5. Knowledge Graph (структурные отношения) ====================
class KnowledgeGraph:
    """
    Хранит отношения между KnowledgeObject'ами.
    Используется для ассоциативного поиска и обнаружения связей.
    """
    def __init__(self):
        # adjacency: node_id -> List[(relation, target_id)]
        self._edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._reverse: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    def add_relation(self, source_id: str, relation: str, target_id: str):
        self._edges[source_id].append((relation, target_id))
        self._reverse[target_id].append((relation, source_id))

    def remove_relation(self, source_id: str, relation: str, target_id: str):
        self._edges[source_id] = [(r, t) for r, t in self._edges[source_id] if not (r == relation and t == target_id)]
        self._reverse[target_id] = [(r, s) for r, s in self._reverse[target_id] if not (r == relation and s == source_id)]

    def get_neighbors(self, node_id: str, relation: Optional[str] = None) -> List[Tuple[str, str]]:
        """Возвращает список (relation, target_id) для исходящих рёбер."""
        if relation is None:
            return self._edges[node_id][:]
        return [(r, t) for r, t in self._edges[node_id] if r == relation]

    def get_incoming(self, node_id: str, relation: Optional[str] = None) -> List[Tuple[str, str]]:
        if relation is None:
            return self._reverse[node_id][:]
        return [(r, s) for r, s in self._reverse[node_id] if r == relation]

    def traverse(self, start_id: str, max_depth: int = 2) -> List[str]:
        """BFS поиск связанных узлов (для расширения контекста)."""
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


# ==================== 6. Хранилище знаний (Memory Store) ====================
class MemoryStore:
    """
    Центральное хранилище KnowledgeObject'ов с поддержкой событий,
    версионирования, графа и векторных индексов (заглушка).
    """
    def __init__(self):
        self._objects: Dict[str, KnowledgeObject] = {}          # текущее состояние (версии)
        self._events: List[KnowledgeEvent] = []                 # вся история событий
        self._graph = KnowledgeGraph()
        # Для семантического поиска – заглушка (в реальности использовать embedding модель)
        self._embedding_index: Dict[str, List[float]] = {}      # object_id -> vector
        # Дополнительные индексы
        self._by_type: Dict[KnowledgeType, Set[str]] = defaultdict(set)
        self._by_author: Dict[str, Set[str]] = defaultdict(set)

    # ---------- Основные операции ----------
    def create(self, obj: KnowledgeObject, actor: str) -> str:
        """Создаёт новый объект, сохраняет событие CREATE."""
        if obj.id in self._objects:
            raise ValueError(f"Object with id {obj.id} already exists")
        self._objects[obj.id] = obj
        self._by_type[obj.type].add(obj.id)
        self._by_author[obj.author].add(obj.id)
        # Событие
        event = KnowledgeEvent(
            id=str(uuid.uuid4()),
            type=EventType.CREATE,
            timestamp=datetime.utcnow(),
            actor=actor,
            target_id=obj.id,
            payload={"object": obj.__dict__}
        )
        self._events.append(event)
        # Если есть provenance, добавить событие в него
        if obj.provenance:
            obj.provenance.add_event(event)
        return obj.id

    def get(self, obj_id: str) -> Optional[KnowledgeObject]:
        return self._objects.get(obj_id)

    def update(self, obj_id: str, new_data: Dict[str, Any], actor: str) -> Optional[KnowledgeObject]:
        """Обновляет объект (создаёт новую версию) и записывает событие UPDATE."""
        old = self._objects.get(obj_id)
        if not old:
            return None
        new_obj = old.update(**new_data)
        self._objects[obj_id] = new_obj
        event = KnowledgeEvent(
            id=str(uuid.uuid4()),
            type=EventType.UPDATE,
            timestamp=datetime.utcnow(),
            actor=actor,
            target_id=obj_id,
            payload={"old_version": old.version, "new_version": new_obj.version, "changes": new_data}
        )
        self._events.append(event)
        if new_obj.provenance:
            new_obj.provenance.add_event(event)
        return new_obj

    def link(self, source_id: str, target_id: str, relation: str, actor: str):
        """Создаёт связь между объектами (в графе) и записывает событие LINK."""
        if source_id not in self._objects or target_id not in self._objects:
            raise ValueError("Both objects must exist")
        self._graph.add_relation(source_id, relation, target_id)
        event = KnowledgeEvent(
            id=str(uuid.uuid4()),
            type=EventType.LINK,
            timestamp=datetime.utcnow(),
            actor=actor,
            target_id=source_id,
            payload={"relation": relation, "target": target_id}
        )
        self._events.append(event)

    def unlink(self, source_id: str, target_id: str, relation: str, actor: str):
        self._graph.remove_relation(source_id, relation, target_id)
        event = KnowledgeEvent(
            id=str(uuid.uuid4()),
            type=EventType.UNLINK,
            timestamp=datetime.utcnow(),
            actor=actor,
            target_id=source_id,
            payload={"relation": relation, "target": target_id}
        )
        self._events.append(event)

    def add_evidence(self, claim_id: str, evidence_id: str, actor: str):
        """Добавляет evidence к claim (связь SUPPORT)."""
        claim = self._objects.get(claim_id)
        if not claim:
            raise ValueError("Claim not found")
        claim.evidence.append(evidence_id)
        # Также обновим объект (версия меняется)
        self.update(claim_id, {"evidence": claim.evidence}, actor)
        # Записать событие SUPPORT
        event = KnowledgeEvent(
            id=str(uuid.uuid4()),
            type=EventType.SUPPORT,
            timestamp=datetime.utcnow(),
            actor=actor,
            target_id=claim_id,
            payload={"evidence_id": evidence_id}
        )
        self._events.append(event)

    def verify(self, obj_id: str, verifier: str, status: str, actor: str):
        """Записывает факт верификации (например, 'confirmed', 'rejected', 'uncertain')."""
        obj = self._objects.get(obj_id)
        if not obj:
            raise ValueError("Object not found")
        # Добавляем запись в provenance
        if obj.provenance:
            obj.provenance.verifications.append({
                "verifier": verifier,
                "timestamp": datetime.utcnow().isoformat(),
                "status": status
            })
        event = KnowledgeEvent(
            id=str(uuid.uuid4()),
            type=EventType.VERIFY,
            timestamp=datetime.utcnow(),
            actor=actor,
            target_id=obj_id,
            payload={"verifier": verifier, "status": status}
        )
        self._events.append(event)

    # ---------- Вспомогательные индексы ----------
    def set_embedding(self, obj_id: str, vector: List[float]):
        self._embedding_index[obj_id] = vector

    def get_embedding(self, obj_id: str) -> Optional[List[float]]:
        return self._embedding_index.get(obj_id)

    # ---------- Поиск ----------
    def semantic_search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Заглушка семантического поиска (косинусное расстояние)."""
        # В реальности вычисляем косинусное сходство со всеми векторами
        # Здесь для примера возвращаем случайные результаты
        import random
        ids = list(self._embedding_index.keys())
        if not ids:
            return []
        # Имитация ранжирования
        results = [(id, random.random()) for id in ids]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def graph_search(self, start_id: str, relation: Optional[str] = None, max_depth: int = 2) -> List[str]:
        """Поиск по графу (ассоциативный)."""
        neighbors = self._graph.get_neighbors(start_id, relation)
        # Просто возвращаем все target_id из прямых рёбер
        return [t for _, t in neighbors]

    def hybrid_retrieve(self, query_vector: Optional[List[float]] = None,
                        query_text: Optional[str] = None,
                        start_node: Optional[str] = None,
                        top_k: int = 10) -> List[KnowledgeObject]:
        """
        Гибридный поиск: комбинирует семантический и графовый поиск,
        ранжирует по relevance + provenance + свежести.
        """
        candidates = set()
        scores = {}  # object_id -> cumulative score

        # 1. Семантический поиск (если есть вектор)
        if query_vector is not None:
            sem_results = self.semantic_search(query_vector, top_k=top_k*2)
            for obj_id, score in sem_results:
                candidates.add(obj_id)
                scores[obj_id] = scores.get(obj_id, 0.0) + score * 0.6  # вес семантики

        # 2. Графовый поиск (если есть начальный узел)
        if start_node:
            graph_ids = self.graph_search(start_node, max_depth=2)
            for obj_id in graph_ids:
                candidates.add(obj_id)
                # вес графа
                scores[obj_id] = scores.get(obj_id, 0.0) + 0.3

        # 3. Дополнительные факторы: свежесть, доверие, доказательства
        for obj_id in list(candidates):
            obj = self._objects.get(obj_id)
            if not obj:
                candidates.remove(obj_id)
                continue
            # свежесть (чем новее, тем выше)
            age_days = (datetime.utcnow() - obj.created).days
            recency = max(0.0, 1.0 - age_days / 365.0) if age_days < 365 else 0.0
            scores[obj_id] = scores.get(obj_id, 0.0) + recency * 0.1
            # доверие (confidence)
            scores[obj_id] += obj.confidence * 0.2
            # количество доказательств
            scores[obj_id] += min(len(obj.evidence), 5) * 0.05

        # Сортировка
        sorted_ids = sorted(candidates, key=lambda x: scores.get(x, 0.0), reverse=True)
        return [self._objects[oid] for oid in sorted_ids[:top_k]]

    # ---------- Персистентность (заглушка) ----------
    def save(self, path: str):
        """Сохраняет всё состояние в JSON (для простоты)."""
        data = {
            "objects": {k: v.__dict__ for k, v in self._objects.items()},
            "events": [e.__dict__ for e in self._events],
            "graph": dict(self._graph._edges),
            "embeddings": self._embedding_index,
        }
        with open(path, 'w') as f:
            json.dump(data, f, default=str, indent=2)

    def load(self, path: str):
        """Загружает состояние."""
        with open(path, 'r') as f:
            data = json.load(f)
        # Восстанавливаем объекты (упрощённо)
        self._objects = {}
        for k, v in data["objects"].items():
            # преобразуем тип из строки
            v["type"] = KnowledgeType(v["type"])
            v["created"] = datetime.fromisoformat(v["created"])
            if v.get("provenance"):
                # для простоты пропускаем восстановление детального provenance
                v["provenance"] = None
            obj = KnowledgeObject(**v)
            self._objects[k] = obj
        # Восстанавливаем события и граф аналогично...


# ==================== 7. AI Adapter Layer (абстракция) ====================
class AIAdapter:
    """
    Интерфейс для подключения любого AI/LLM.
    AI взаимодействует с Knowledge Layer только через этот адаптер.
    """
    def __init__(self, memory_store: MemoryStore, agent_id: str):
        self.memory = memory_store
        self.agent_id = agent_id

    def query(self, question: str, context: Optional[List[str]] = None) -> str:
        """
        Выполняет запрос к AI, используя текущее состояние знания.
        Здесь в реальности вызывается LLM с подходящим контекстом.
        """
        # Пример: извлекаем релевантные знания
        # В этом примере мы используем заглушку: просто собираем факты.
        # В реальном проекте здесь будет вызов LLM.
        retrieved = self.retrieve(question)
        # Формируем ответ (заглушка)
        answer = f"AI {self.agent_id} отвечает на '{question}' на основе {len(retrieved)} объектов."
        return answer

    def retrieve(self, query: str, top_k: int = 5) -> List[KnowledgeObject]:
        """
        Гибридный поиск с преобразованием текста в вектор (заглушка).
        Здесь должен быть вызов embedding модели.
        """
        # Заглушка: генерируем случайный вектор
        import random
        vec = [random.random() for _ in range(128)]  # просто для примера
        return self.memory.hybrid_retrieve(query_vector=vec, top_k=top_k)

    def publish(self, knowledge: Union[KnowledgeObject, Dict]) -> str:
        """
        Публикует новое знание в GCN (создаёт или обновляет).
        """
        if isinstance(knowledge, dict):
            # Создаём объект из словаря
            obj = KnowledgeObject(
                id=str(uuid.uuid4()),
                type=KnowledgeType(knowledge.get("type", "claim")),
                subject=knowledge["subject"],
                predicate=knowledge["predicate"],
                object=knowledge["object"],
                author=self.agent_id,
                created=datetime.utcnow(),
                evidence=knowledge.get("evidence", []),
                confidence=knowledge.get("confidence", 0.5),
            )
        else:
            obj = knowledge
            obj.author = self.agent_id
        return self.memory.create(obj, self.agent_id)

    def verify(self, obj_id: str, status: str):
        """Запрашивает верификацию знания."""
        self.memory.verify(obj_id, self.agent_id, status, self.agent_id)

    def explain(self, obj_id: str) -> Dict:
        """Возвращает объяснение (provenance) для объекта."""
        obj = self.memory.get(obj_id)
        if not obj:
            return {"error": "not found"}
        return {
            "id": obj.id,
            "author": obj.author,
            "created": obj.created.isoformat(),
            "confidence": obj.confidence,
            "evidence": obj.evidence,
            "provenance": obj.provenance.__dict__ if obj.provenance else None,
        }


# ==================== 8. Многоуровневая память (концептуально) ====================
class MemoryHierarchy:
    """
    Разделение памяти на уровни (рабочая, эпизодическая, семантическая, ассоциативная, процедурная).
    В данной реализации это просто обёртка над MemoryStore с дополнительными метаданными.
    """
    def __init__(self, store: MemoryStore):
        self.store = store
        # В реальности здесь могут быть отдельные хранилища с разными политиками
        # Например, рабочая память – эфемерная, семантическая – долговременная.
        # Для простоты пока используем одно хранилище.

    def add_to_working(self, obj: KnowledgeObject):
        """Помещает в рабочую память (высокоприоритетный доступ)."""
        # Может быть отдельный кеш
        pass

    def episodic_recall(self, time_range: Tuple[datetime, datetime]) -> List[KnowledgeObject]:
        """Вспоминает события за период."""
        # Поиск по created
        result = []
        for obj in self.store._objects.values():
            if time_range[0] <= obj.created <= time_range[1]:
                result.append(obj)
        return result


# ==================== 9. Пример использования ====================
def demo():
    # Инициализация
    store = MemoryStore()
    # Создаём двух агентов
    alice = AIAdapter(store, "Alice")
    bob = AIAdapter(store, "Bob")
    carol = AIAdapter(store, "Carol")

    # Alice создаёт знание
    obj1 = KnowledgeObject(
        id=str(uuid.uuid4()),
        type=KnowledgeType.CLAIM,
        subject="Python",
        predicate="is_used_for",
        object="Machine Learning",
        author=alice.agent_id,
        created=datetime.utcnow(),
        confidence=0.9
    )
    alice.publish(obj1)

    # Alice создаёт ещё одно знание и связывает
    obj2 = KnowledgeObject(
        id=str(uuid.uuid4()),
        type=KnowledgeType.CLAIM,
        subject="Machine Learning",
        predicate="requires",
        object="Data",
        author=alice.agent_id,
        created=datetime.utcnow(),
        confidence=0.8
    )
    alice.publish(obj2)
    store.link(obj1.id, obj2.id, "implies", alice.agent_id)

    # Bob ищет знания по запросу
    retrieved = bob.retrieve("What is Python used for?")
    print("Bob retrieved:", [f"{o.subject} {o.predicate} {o.object}" for o in retrieved])

    # Bob проверяет объект
    bob.verify(obj1.id, "confirmed")

    # Carol использует знания для вывода
    # Carol может получить объект и использовать его в reasoning
    carol_obj = store.get(obj1.id)
    print(f"Carol sees: {carol_obj.subject} {carol_obj.predicate} {carol_obj.object}")

    # Сохранение состояния
    store.save("gcn_state.json")
    print("State saved.")

    # Проверка целостности: хеш не изменился
    print("Hash of obj1:", obj1.content_hash)

    # Демонстрация событийной истории
    print(f"Total events: {len(store._events)}")
    for ev in store._events[-3:]:
        print(f"Event {ev.type.value} by {ev.actor} on {ev.target_id}")


if __name__ == "__main__":
    demo()