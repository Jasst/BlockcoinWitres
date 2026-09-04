# memory_service.py
"""
Сервисный слой для работы с когнитивной памятью.
Используется как MCP-инструментами, так и чат-контроллером.
"""

import logging
import time
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

from GCN.memory_graph import GCNMemoryRouter, MemoryScope, CognitiveMemory
from GCN.llm_client import call_llm
from GCN.config_ai import MEMORY_BASE_DIR

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Единый интерфейс для операций с памятью:
    - recall (поиск)
    - remember (сохранение)
    - forget (удаление)
    - add_goal, get_goals
    - semantic_search, graph_explore
    - get_contradictions, resolve_contradiction
    - explain_fact, get_memory_stats
    - управление эпизодами (add_episode, get_episodes)
    """
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.router = GCNMemoryRouter(user_id, MEMORY_BASE_DIR)
        self.router.set_llm_caller(call_llm)
        self.private_memory = self.router.private_memory
        self.shared_memory = self.router.shared_memory
        self.global_memory = self.router.global_memory

    def refresh(self):
        """Подтягивает изменения, сделанные другими процессами."""
        self.router.refresh(include_private=True)

    async def recall(self, query: str, top_k: int = 5, scope: Optional[str] = None) -> List[Dict]:
        """
        Поиск по всем слоям с опциональным фильтром по скоупу.
        Возвращает список фактов с метаданными.
        """
        self.refresh()
        results = await self.router.retrieve(query, top_k=top_k * 2, include_private=True)
        if scope:
            scope_lower = scope.lower()
            results = [r for r in results if r.get('scope') == scope_lower]
        return results[:top_k]

    async def remember(self, fact: str, scope: Optional[str] = None) -> Dict[str, Any]:
        """
        Сохраняет факт в указанный скоуп (автоопределение, если scope не задан).
        Возвращает id и скоуп.
        """
        self.refresh()
        if scope is None:
            if "глобально" in fact.lower() or "global" in fact.lower():
                scope_enum = MemoryScope.GLOBAL
            else:
                scope_enum = MemoryScope.PRIVATE
        else:
            scope_map = {"private": MemoryScope.PRIVATE, "shared": MemoryScope.SHARED, "global": MemoryScope.GLOBAL}
            scope_enum = scope_map.get(scope.lower(), MemoryScope.PRIVATE)

        # Улучшаем факт через LLM (как в чате)
        enhanced = await self._enhance_fact(fact)
        obj_id = self.router.add_knowledge(
            subject=enhanced,
            predicate="is_fact",
            obj="true",
            scope=scope_enum,
            confidence=0.9,
            author=self.user_id,
            source_type="memory_service"
        )
        # Добавляем в рабочую память
        if obj_id:
            self.private_memory.hierarchy.add_to_working(obj_id)

        # Сохраняем соответствующий слой
        await self._save_scope(scope_enum)
        return {"id": obj_id, "scope": scope_enum.value, "fact": enhanced}

    async def _enhance_fact(self, fact: str) -> str:
        """Улучшает формулировку факта через LLM (как в ai_assistant)."""
        try:
            prompt = (
                "Если факт структурирован и содержит достаточно информации по теме, запомни как есть, полный текст. "
                "Иначе извлеки из запроса пользователя объективный факт (утверждение, которое может быть проверено или использовано как знание). "
                "Игнорируй мнения, временные события, эмоции, инструкции и пожелания. "
                "Сформулируй факт как краткое предложение в настоящем времени (или прошедшем, если это не теряет актуальности). "
                "Ответь только фактом, без пояснений. Или полным текстом, если факт структурирован.\n\n"
                f"Запрос: {fact}"
            )
            enhanced = await call_llm([{"role": "user", "content": prompt}], temp=0.3, max_tokens=150)
            enhanced = enhanced.strip()
            if len(enhanced) < 5:
                enhanced = fact
            return enhanced
        except Exception:
            return fact

    async def _save_scope(self, scope: MemoryScope):
        if scope == MemoryScope.GLOBAL:
            await self.global_memory._schedule_save()
        elif scope == MemoryScope.SHARED:
            await self.shared_memory._schedule_save()
        else:
            await self.private_memory._schedule_save()

    async def forget(self, query: str, scope: str = "private", dry_run: bool = True) -> Dict[str, Any]:
        """
        Удаляет факты, содержащие query, из указанного слоя.
        Если dry_run=True – только возвращает кандидаты.
        """
        self.refresh()
        scope_map = {
            "private": self.private_memory,
            "shared": self.shared_memory,
            "global": self.global_memory,
        }
        memory = scope_map.get(scope.lower())
        if memory is None:
            return {"status": "error", "message": f"Неизвестный scope: {scope}"}
        memory.reload_if_stale()
        to_remove = [f for f in memory.semantic_facts if query.lower() in f.text.lower()]
        if not to_remove:
            return {"status": "ok", "removed": 0, "scope": scope.lower(), "message": "Ничего не найдено."}

        if dry_run:
            return {
                "status": "dry_run",
                "would_remove": len(to_remove),
                "scope": scope.lower(),
                "candidates": [{"id": f.id, "text": f.text[:200]} for f in to_remove[:20]],
                "message": "Ничего не удалено. Повторите вызов с dry_run=False, чтобы удалить эти факты.",
            }

        removed = memory._remove_facts({f.id for f in to_remove})
        await memory._schedule_save()
        return {"status": "ok", "removed": removed, "scope": scope.lower()}

    async def add_goal(self, description: str, priority: float = 0.5) -> Dict[str, Any]:
        self.refresh()
        gid = await self.private_memory.add_goal(description, priority)
        return {"id": gid, "description": description, "priority": priority}

    async def get_goals(self) -> List[Dict]:
        self.refresh()
        goals = await self.private_memory.get_active_goals()
        return [
            {"description": g.description, "priority": g.priority, "confidence": g.confidence, "status": g.status}
            for g in goals
        ]

    async def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        self.refresh()
        emb = self.private_memory.embed_text(query, is_query=True)
        if emb is None:
            return []
        results = self.private_memory.store.semantic_search(emb, top_k=top_k * 2)
        return [
            {"text": self.private_memory.store.get(gcn_id).subject if self.private_memory.store.get(gcn_id) else "",
             "score": score}
            for gcn_id, score in results[:top_k]
        ]

    async def graph_explore(self, seed_text: str, depth: int = 2) -> Dict[str, Any]:
        self.refresh()
        memory = self.private_memory
        seed_ids = [f.id for f in memory.semantic_facts if seed_text.lower() in f.text.lower()]
        if not seed_ids:
            return {"error": f"Факты с '{seed_text}' не найдены."}
        activation = await memory.spread_activation(seed_ids[:3], max_depth=min(depth, 3))
        sorted_items = sorted(activation.items(), key=lambda x: x[1], reverse=True)
        return {
            "nodes": [
                {"id": fid, "text": memory.facts_by_id.get(fid).text[:200] if memory.facts_by_id.get(fid) else "",
                 "activation": act}
                for fid, act in sorted_items[:20] if fid not in seed_ids
            ]
        }

    async def explain_fact(self, gcn_id: str) -> Dict[str, Any]:
        self.refresh()
        obj = (self.private_memory.store.get(gcn_id) or
               self.shared_memory.store.get(gcn_id) or
               self.global_memory.store.get(gcn_id))
        if not obj:
            return {"error": f"Объект {gcn_id} не найден ни в одном слое памяти."}

        store = (self.private_memory.store if self.private_memory.store.get(gcn_id) else
                 self.shared_memory.store if self.shared_memory.store.get(gcn_id) else
                 self.global_memory.store)

        contradictions = store._graph.get_neighbors(gcn_id, "CONTRADICTS")
        grounds_in = store._graph.get_neighbors(gcn_id, "GROUNDS_IN")
        abstracts_from = store._graph.get_neighbors(gcn_id, "ABSTRACTS_FROM")
        confirming_authors = [e.split("author:", 1)[1] for e in obj.evidence if e.startswith("author:")]

        return {
            "id": obj.id,
            "type": obj.type.value,
            "scope": obj.scope.value,
            "text": obj.subject,
            "confidence": obj.confidence,
            "version": obj.version,
            "author": obj.author,
            "source_type": obj.source_type,
            "created": obj.created.isoformat(),
            "confirming_authors": confirming_authors,
            "contradicts": [target for _, target in contradictions],
            "grounds_in_global": [target for _, target in grounds_in],
            "abstracted_from": [target for _, target in abstracts_from],
        }

    async def get_contradictions(self, limit: int = 5) -> List[Tuple[Dict, Dict]]:
        self.refresh()
        memory = self.private_memory
        pairs = memory.get_unverified_contradictions(limit=limit)
        return [
            (
                {"id": a.id, "text": a.text, "confidence": a.confidence},
                {"id": b.id, "text": b.text, "confidence": b.confidence}
            )
            for a, b in pairs
        ]

    async def resolve_contradiction(self, fact_id_a: str, fact_id_b: str, verdict: str,
                                    reason: str = "") -> Dict[str, Any]:
        self.refresh()
        memory = self.private_memory

        def find_fact(fid):
            if fid in memory.facts_by_id:
                return memory.facts_by_id[fid]
            for f in memory.semantic_facts:
                if f.gcn_id == fid:
                    return f
            return None

        fa = find_fact(fact_id_a)
        fb = find_fact(fact_id_b)
        if not fa or not fb:
            return {"status": "error", "message": f"Факты не найдены: A={fact_id_a}, B={fact_id_b}"}

        v = verdict.lower()
        if v == "a":
            memory._remove_facts({fb.id})
            memory.gcn_store._graph.remove_relation(fa.gcn_id, "CONTRADICTS", fb.gcn_id)
            memory.gcn_store._graph.remove_relation(fb.gcn_id, "CONTRADICTS", fa.gcn_id)
            fa.contradicts.discard(fb.id)
            fb.contradicts.discard(fa.id)
            await memory._schedule_save()
            return {"status": "ok", "verdict": "a", "kept": fa.text, "removed": fb.text}
        elif v == "b":
            memory._remove_facts({fa.id})
            memory.gcn_store._graph.remove_relation(fa.gcn_id, "CONTRADICTS", fb.gcn_id)
            memory.gcn_store._graph.remove_relation(fb.gcn_id, "CONTRADICTS", fa.gcn_id)
            fa.contradicts.discard(fb.id)
            fb.contradicts.discard(fa.id)
            await memory._schedule_save()
            return {"status": "ok", "verdict": "b", "kept": fb.text, "removed": fa.text}
        elif v == "both":
            memory.gcn_store._graph.remove_relation(fa.gcn_id, "CONTRADICTS", fb.gcn_id)
            memory.gcn_store._graph.remove_relation(fb.gcn_id, "CONTRADICTS", fa.gcn_id)
            fa.contradicts.discard(fb.id)
            fb.contradicts.discard(fa.id)
            await memory._schedule_save()
            return {"status": "ok", "verdict": "both", "message": "Противоречие снято, оба сохранены."}
        elif v == "neither":
            memory._remove_facts({fa.id, fb.id})
            await memory._schedule_save()
            return {"status": "ok", "verdict": "neither", "message": "Оба удалены."}
        else:
            return {"status": "error", "message": f"Неизвестный вердикт: {verdict}"}

    async def get_memory_stats(self) -> Dict:
        self.refresh()
        return self.private_memory.get_stats()

    async def get_episodes(self, limit: int = 5) -> List[Dict]:
        self.refresh()
        episodes = self.private_memory.episodic_memory[-limit:] if self.private_memory.episodic_memory else []
        return [
            {"user": ep.user_msg, "assistant": ep.assistant_msg, "timestamp": ep.timestamp}
            for ep in reversed(episodes)
        ]

    # ===== ДОБАВЛЕННЫЙ МЕТОД =====
    async def add_episode(self, user_msg: str, assistant_msg: str, salience: float = 0.0):
        """
        Сохраняет эпизод (диалог) в личную память пользователя.
        """
        self.refresh()
        await self.private_memory.add_episode(user_msg, assistant_msg, salience)

    async def shutdown(self):
        """
        ИСПРАВЛЕНИЕ: раньше shutdown() одного пользователя закрывал private_memory
        И shared_memory И global_memory. Но shared/global — это ПРОЦЕСС-ШИРОКИЕ
        синглтоны (GCNMemoryRouter._get_shared_memory/_get_global_memory), общие
        для всех пользователей, а не что-то принадлежащее этому MemoryService.
        ai_assistant.py вызывает CognitiveController.shutdown() (который вызывает
        этот метод) при обычной выгрузке простаивающего пользователя из LRU-кэша
        (_evict_stale_assistants, по умолчанию раз в час простоя, или при
        превышении лимита одновременных пользователей) — то есть в штатном
        режиме, а не только при остановке процесса. Каждая такая выгрузка
        отменяла фоновую _save_task и форсировала полное пересохранение
        (включая перестройку FAISS-индекса) ОБЩЕЙ памяти для ВСЕХ пользователей
        — просто потому, что один конкретный пользователь давно не писал в чат.
        Теперь per-user shutdown трогает только private_memory. shared/global
        должны закрываться ровно один раз, при остановке всего процесса — см.
        shutdown_shared_global().
        """
        await self.private_memory.shutdown()

    @staticmethod
    async def shutdown_shared_global():
        """
        Закрывает общую (shared) и глобальную (global) память — процесс-широкие
        синглтоны. Вызывать один раз при остановке всего приложения, а НЕ при
        выгрузке отдельного простаивающего пользователя (см. комментарий в
        shutdown() выше).
        """
        from GCN.memory_graph import GCNMemoryRouter
        from GCN.config_ai import MEMORY_BASE_DIR
        shared = GCNMemoryRouter._get_shared_memory(MEMORY_BASE_DIR)
        glob = GCNMemoryRouter._get_global_memory(MEMORY_BASE_DIR)
        await shared.shutdown()
        await glob.shutdown()


# ===== Фабрика сервисов с LRU-кэшированием (аналогично CognitiveController) =====
_services: Dict[str, MemoryService] = {}
_services_last_used: Dict[str, float] = {}
_SERVICE_MAX_IDLE = 1800  # 30 минут
_SERVICE_MAX_COUNT = 50

async def get_memory_service(user_id: str) -> MemoryService:
    """Возвращает экземпляр MemoryService для пользователя, с выгрузкой неактивных."""
    if user_id not in _services:
        _services[user_id] = MemoryService(user_id)
        logger.info(f"MemoryService создан для {user_id[:16]}")
        # ИСПРАВЛЕНИЕ: _SERVICE_MAX_IDLE/_SERVICE_MAX_COUNT были объявлены, но
        # нигде не читались — MemoryService (со своим приватным
        # CognitiveMemory: эмбеддинги, FAISS-индекс, факты) накапливался в
        # словаре _services без ограничений для каждого нового user_id, в
        # отличие от параллельного кэша _assistants в ai_assistant.py и кэша
        # роутеров в mcp_server_blockcoin.py, у которых выгрузка уже была.
        # Любой код, обращающийся к памяти через get_memory_service() напрямую
        # (а не через CognitiveController), тёк по памяти процесса.
        await _evict_stale_services(exclude_uid=user_id)
    _services_last_used[user_id] = time.time()
    return _services[user_id]


async def _evict_stale_services(exclude_uid: Optional[str] = None) -> None:
    now = time.time()
    stale = [
        uid for uid, ts in _services_last_used.items()
        if uid != exclude_uid and now - ts > _SERVICE_MAX_IDLE
    ]
    for uid in stale:
        service = _services.pop(uid, None)
        _services_last_used.pop(uid, None)
        if service is not None:
            try:
                await service.shutdown()  # только private_memory — см. MemoryService.shutdown()
            except Exception as e:
                logger.error(f"Ошибка при выгрузке MemoryService {uid[:16]}: {e}")
            logger.info(f"MemoryService {uid[:16]} выгружен (простой > {_SERVICE_MAX_IDLE}с)")

    while len(_services) > _SERVICE_MAX_COUNT:
        oldest_uid = min(_services_last_used, key=_services_last_used.get, default=None)
        if oldest_uid is None or oldest_uid == exclude_uid:
            break
        service = _services.pop(oldest_uid, None)
        _services_last_used.pop(oldest_uid, None)
        if service is not None:
            try:
                await service.shutdown()
            except Exception as e:
                logger.error(f"Ошибка при выгрузке MemoryService {oldest_uid[:16]}: {e}")
        logger.info(f"MemoryService {oldest_uid[:16]} выгружен по лимиту количества (LRU, max={_SERVICE_MAX_COUNT})")