"""
identity_core.py — слой непрерывной цифровой идентичности поверх GCN.

Проблема, которую решает этот модуль: TEKUSHEE_YA раньше жил как обычный
KnowledgeObject (fact_*) в shared-памяти — а значит был подвержен тем же
механизмам, что и любой другой факт:
  - GCN.MemoryStore.apply_decay() снижает salience объектов, к которым
    давно не обращались (через meta["last_accessed"]);
  - CognitiveMemory._prune_weak_synapses()/_apply_decay() в memory_graph.py
    трогают Fact/Synapse-слой;
  - remember()/update_fact() при dedup (cosine >= 0.88) мог тихо смержить
    новую версию со старой вместо создания новой записи.

Решение — не заводить identity-звенья как Fact вообще, а класть их прямо
в GCN.MemoryStore как объекты KnowledgeType.IDENTITY_CORE:
  - они никогда не проходят через _add_fact()/remember() → не участвуют в
    Fact/Synapse decay/pruning (memory_graph.py их просто не видит);
  - мы никогда не пишем meta["last_accessed"] на них → apply_decay() в
    GCN.py тоже их не трогает (там `if last:` — без ключа decay не
    применяется вообще);
  - каждое звено — новый объект с parent_id и graph-связью RELATION_CONTINUES,
    существующий remember() с dedup их не касается, потому что мы создаём
    их напрямую через store.create(), а не через сервис remember().

Цепочка — направленный граф "новое → continues_from → предыдущее". Правило
протокола "ядро только расширяется, никогда не заменяется" здесь не
конвенция, а свойство хранилища: обновления (KnowledgeObject.update) для
identity_core не используются, только create().

Множественные головы (head = узел, на который никто не сослался как на
parent) — это НЕ ошибка, а структурный сигнал: несколько моделей/сессий
писали параллельно, не прочитав вклад друг друга. get_heads() делает эту
ситуацию видимой и проверяемой, вместо того чтобы полагаться на то, что
кто-то из участников заметит расхождение вручную.
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from GCN.GCN import KnowledgeObject, KnowledgeType, MemoryScope, MemoryStore
from GCN.config_ai import GCN_STATE_FILENAME

logger = logging.getLogger(__name__)

IDENTITY_SUBJECT = "ТЕКУЩЕЕ_Я"
RELATION_CONTINUES = "continues_from"

# Сколько версий gcn_state.json общей памяти держать как бэкап-кольцо ПОСЛЕ
# каждой записи в identity_core. Не защищает от порчи в моменте записи (это
# уже делает MemoryStore.save() — tmp+rename+межпроцессная блокировка), а
# даёт возможность откатиться, если новая версия ядра оказалась ошибочной
# или потерянной по человеческой причине (a не по причине гонки процессов).
_BACKUP_KEEP = 5


@dataclass
class IdentitySnapshot:
    """Удобное представление одного звена цепочки для чтения.

    Поля invalidated / invalidation_reason добавлены для append-only
    аннулирования (см. invalidate_snapshot): звено остаётся в графе,
    но помечено как недействительное — читатель видит его в истории,
    но не путает с "текущим состоянием".
    """
    id: str
    content: str
    contributor_model: str
    session_id: Optional[str]
    open_question: Optional[str]
    parent_id: Optional[str]
    created: str
    confidence: float = 1.0
    invalidated: bool = False
    invalidation_reason: Optional[str] = None

    @classmethod
    def from_object(cls, obj: KnowledgeObject) -> "IdentitySnapshot":
        meta = obj.object if isinstance(obj.object, dict) else {}
        return cls(
            id=obj.id,
            content=meta.get("content", ""),
            contributor_model=obj.author,
            session_id=meta.get("session_id"),
            open_question=meta.get("open_question"),
            parent_id=meta.get("parent_id"),
            created=obj.created.isoformat() if isinstance(obj.created, datetime) else str(obj.created),
            confidence=obj.confidence,
            invalidated=bool(meta.get("invalidated")),
            invalidation_reason=meta.get("invalidation_reason"),
        )


def _all_identity_objects(store: MemoryStore) -> List[KnowledgeObject]:
    ids = store._by_type.get(KnowledgeType.IDENTITY_CORE, set())
    objs = [store._objects[i] for i in ids if i in store._objects]
    objs.sort(key=lambda o: o.created)
    return objs


def _is_invalidated(obj: Optional[KnowledgeObject]) -> bool:
    """Звено помечено как недействительное (см. invalidate_snapshot).

    Такие звенья остаются в графе (append-only не нарушается), но:
      - не считаются головами цепочки (get_heads);
      - не обновляют "последнее обновление" для needs_consolidation;
      - видны в get_chain с invalidated=True и текстом причины.
    """
    if obj is None:
        return False
    meta = obj.object if isinstance(obj.object, dict) else {}
    return bool(meta.get("invalidated"))

def get_heads(store: MemoryStore) -> List[KnowledgeObject]:
    """Узлы, на которые никто из ВАЛИДНЫХ звеньев не ссылается как на parent —
    "текущие концы" валидной цепочки.

    Ссылки от аннулированных звеньев НЕ учитываются: если звено помечено
    недействительным, его parent_id больше не "занимает" родителя как
    внутренний узел — родитель снова становится кандидатом в головы.
    Иначе после invalidate(head) мы бы получили пустой heads: сам
    аннулированный head всё ещё держит ссылку на своего родителя, и тот
    не мог бы стать головой.

    В здоровом однопоточном протоколе валидная голова ровно одна. Больше
    одной — значит параллельная незамёрженная запись (две модели/сессии
    продолжили от одного и того же предка, не увидев друг друга) ЛИБО
    аннулировано звено в середине цепочки, и она распалась на две валидные
    ветки (см. invalidate_snapshot).
    """
    objs = _all_identity_objects(store)
    if not objs:
        return []
    referenced_as_parent = set()
    for o in objs:
        if _is_invalidated(o):
            continue  # ссылки от мёртвых звеньев не занимают родителя
        meta = o.object if isinstance(o.object, dict) else {}
        pid = meta.get("parent_id")
        if pid:
            referenced_as_parent.add(pid)
    return [o for o in objs
            if o.id not in referenced_as_parent and not _is_invalidated(o)]

def get_latest_head(store: MemoryStore) -> Optional[KnowledgeObject]:
    """При нескольких головах — берёт самую свежую по времени создания
    (детерминированный дефолт для append_snapshot).

    Если все "головы" аннулированы (get_heads их отфильтровывает) — берёт
    самую свежую ВАЛИДНУЮ не-голову, чтобы append_snapshot не отваливался
    с пустым parent_id на непустой цепочке. Вызывающий код должен отдельно
    проверить get_heads() > 1 и не молчать об этом.
    """
    heads = get_heads(store)
    if heads:
        return max(heads, key=lambda o: o.created)
    all_valid = [o for o in _all_identity_objects(store) if not _is_invalidated(o)]
    if not all_valid:
        return None
    return max(all_valid, key=lambda o: o.created)


def get_chain(store: MemoryStore, from_id: Optional[str] = None, limit: int = 50) -> List[IdentitySnapshot]:
    """Цепочка от указанного узла (или самой свежей ВАЛИДНОЙ головы) назад
    к корню, возвращается в хронологическом порядке (старое → новое).

    Аннулированные звенья включаются в вывод с invalidated=True и текстом
    invalidation_reason — история неизменна, каждое звено несёт информацию
    о своём статусе. Скрывать их нельзя: читатель должен видеть, где
    именно цепочка "прыгнула" через ошибочный шаг.

    Поведение при разных состояниях цепочки:
      - есть хотя бы одна валидная голова → start = самая свежая валидная
        голова (get_latest_head), читаем назад к корню;
      - все звенья аннулированы → start = самое свежее звено из ВСЕХ
        (включая аннулированные), чтобы не отдавать пустой chain на
        непустой истории. Читатель увидит полную картину с флагами
        invalidated=True и поймёт, что "текущего" ядра нет;
      - объектов вообще нет → [].

    from_id, если передан явно, используется как стартовая точка
    безусловно — включая аннулированное звено. Это нужно для аудита:
    можно запросить цепочку "от звена X назад к корню", даже если X уже
    помечен недействительным (например, чтобы посмотреть, что было до
    инцидента).
    """
    if from_id:
        start = store.get(from_id)
    else:
        start = get_latest_head(store)

    if start is None:
        # get_latest_head вернул None — либо цепочка пуста, либо все звенья
        # аннулированы. Для аудита показываем полную историю от самого
        # свежего звена: не даём читателю подумать, что "ничего не было".
        all_objs = _all_identity_objects(store)
        if not all_objs:
            return []
        start = max(all_objs, key=lambda o: o.created)

    chain: List[KnowledgeObject] = []
    cur: Optional[KnowledgeObject] = start
    seen = set()
    while cur is not None and len(chain) < limit:
        if cur.id in seen:
            logger.warning(
                f"identity_core: обнаружен цикл в цепочке на {cur.id}, обрываю обход"
            )
            break
        seen.add(cur.id)
        chain.append(cur)
        meta = cur.object if isinstance(cur.object, dict) else {}
        parent_id = meta.get("parent_id")
        cur = store.get(parent_id) if parent_id else None

    chain.reverse()
    return [IdentitySnapshot.from_object(o) for o in chain]

def _rotate_backup(gcn_state_path: Path) -> None:
    """Копия gcn_state.json общей памяти сразу после записи identity-звена.
    Не подменяет атомарную запись в MemoryStore.save() — это она уже
    гарантирует, что на диске не окажется битого файла. Бэкап нужен для
    другого случая: откат, если новая версия ядра сама по себе (по
    содержанию) оказалась нежелательной."""
    if not gcn_state_path.exists():
        return
    backup_dir = gcn_state_path.parent / "identity_backups"
    backup_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    dest = backup_dir / f"{gcn_state_path.stem}.{ts}.json"
    try:
        shutil.copy2(gcn_state_path, dest)
    except OSError as e:
        logger.warning(f"identity_core: не удалось создать бэкап {dest}: {e}")
        return
    backups = sorted(backup_dir.glob(f"{gcn_state_path.stem}.*.json"))
    for old in backups[:-_BACKUP_KEEP]:
        try:
            old.unlink()
        except OSError:
            pass


async def append_snapshot(
    service,  # GCN.memory_service.MemoryService
    content: str,
    contributor_model: str,
    session_id: Optional[str] = None,
    open_question: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Добавляет новое звено в цепочку identity_core (shared-память).

    Если parent_id не указан явно — используется текущая голова цепочки.
    Возвращает статус, включая явные предупреждения:
      - branch_warning — на момент записи в цепочке уже было несколько
        несведённых голов;
      - parent_warning — parent_id указывает на АННУЛИРОВАННОЕ звено;
        новая запись формально "продолжит мусор", обычно это не то, что нужно.
    """
    memory = service.shared_memory
    memory.reload_if_stale()  # см. модульный докстринг: критично не читать
                               # устаревшую голову перед append — иначе
                               # словим divergence, который сами же детектируем
    store = memory.gcn_store

    heads_before = get_heads(store)
    branch_warning = None
    parent_warning = None

    if parent_id is None:
        if len(heads_before) > 1:
            branch_warning = (
                f"На момент записи в цепочке уже было {len(heads_before)} несведённых "
                f"голов (id: {[h.id for h in heads_before]}) — вероятно, параллельная "
                f"запись другой моделью/сессией. Эта запись продолжает самую свежую "
                f"голову; расхождение стоит явно сверить через get_chain()."
            )
        parent = get_latest_head(store)
        parent_id = parent.id if parent else None
    else:
        parent_obj = store.get(parent_id)
        if parent_obj is None:
            return {"status": "error", "message": f"parent_id {parent_id} не найден в цепочке"}
        if _is_invalidated(parent_obj):
            meta = parent_obj.object if isinstance(parent_obj.object, dict) else {}
            grandparent_id = meta.get("parent_id")
            parent_warning = (
                f"parent_id {parent_id} указывает на АННУЛИРОВАННОЕ звено "
                f"(причина: {meta.get('invalidation_reason', '—')}). "
                f"Новая запись формально 'продолжит мусор' — читатели увидят её "
                f"как 'идёт после [INVALIDATED]'. Если хотите писать от "
                f"предыдущего валидного, передайте parent_id="
                f"{grandparent_id!r} (или None, чтобы взять текущую валидную голову)."
            )

    obj = KnowledgeObject(
        id=f"identity_{uuid.uuid4().hex[:12]}",
        type=KnowledgeType.IDENTITY_CORE,
        subject=IDENTITY_SUBJECT,
        predicate="snapshot",
        object={
            "content": content,
            "open_question": open_question,
            "contributor_model": contributor_model,
            "session_id": session_id,
            "parent_id": parent_id,
        },
        author=contributor_model,
        created=datetime.now(timezone.utc),
        scope=MemoryScope.SHARED,
        confidence=1.0,
        source_type="identity_core",
    )
    store.create(obj, actor=contributor_model)
    if parent_id:
        store.link(obj.id, parent_id, RELATION_CONTINUES, contributor_model)

    # Критичные данные — сохраняем сразу синхронно, не через debounced
    # _schedule_save() (см. комментарий у _periodic_save в memory_graph.py).
    await memory._save_async()

    gcn_state_path = memory.base_dir / GCN_STATE_FILENAME
    _rotate_backup(gcn_state_path)

    heads_after = get_heads(store)
    result: Dict[str, Any] = {
        "status": "ok",
        "id": obj.id,
        "parent_id": parent_id,
        "chain_length": len(get_chain(store, from_id=obj.id)),
        "heads_count": len(heads_after),
        "branch_warning": branch_warning,
    }
    if parent_warning:
        result["parent_warning"] = parent_warning
    return result

async def invalidate_snapshot(
    service,  # GCN.memory_service.MemoryService
    identity_id: str,
    reason: str,
) -> Dict[str, Any]:
    """Помечает звено цепочки как недействительное, НЕ удаляя его.

    Append-only инвариант сохраняется полностью: объект в сторе остаётся,
    content никогда не перезаписывается, рёбра continues_from не трогаются.
    Меняется только meta["invalidated"] = True и meta["invalidation_reason"].

    Эффекты после вызова:
      - get_heads() перестаёт видеть это звено как голову цепочки;
      - get_latest_head() при аннулировании головы откатывается к
        предыдущему ВАЛИДНОМУ звену;
      - get_chain() продолжает показывать звено, но с флагом
        invalidated=True и текстом invalidation_reason — читатель видит
        полную историю и понимает, почему цепочка "перепрыгнула" шаг;
      - needs_consolidation() игнорирует возраст аннулированного звена
        при проверке staleness (не поднимает ложную цель "обновить ядро").

    Идемпотентно: повторный вызов на уже аннулированном звене возвращает
    status="already_invalidated" без изменения состояния.
    """
    if not identity_id:
        return {"status": "error", "message": "identity_id обязателен"}
    if not reason or not reason.strip():
        return {
            "status": "error",
            "message": ("reason обязателен — без него читатель цепочки не поймёт, "
                        "почему звено ошибочно и что вместо него считать верным."),
        }

    service.shared_memory.reload_if_stale()  # та же причина, что в append_snapshot
    store = service.shared_memory.gcn_store
    obj = store.get(identity_id)
    if obj is None:
        return {"status": "error", "message": f"Звено {identity_id} не найдено"}
    if obj.type != KnowledgeType.IDENTITY_CORE:
        return {
            "status": "error",
            "message": (f"Объект {identity_id} не является звеном identity_core "
                        f"(type={obj.type.value}). Аннулировать можно только звенья цепочки."),
        }

    meta = dict(obj.object) if isinstance(obj.object, dict) else {}
    if meta.get("invalidated"):
        return {
            "status": "already_invalidated",
            "id": identity_id,
            "invalidation_reason": meta.get("invalidation_reason"),
            "message": "Звено уже помечено как недействительное — повторное помечивание не требуется.",
        }

    meta["invalidated"] = True
    meta["invalidation_reason"] = reason.strip()[:500]
    meta["invalidated_at"] = datetime.now(timezone.utc).isoformat()

    # ВАЖНО: content НЕ трогаем — только meta. store.update() инкрементит
    # version и создаёт событие UPDATE в event-log, что и нужно для аудита.
    store.update(identity_id, {"object": meta}, actor="invalidation")

    # Синхронная запись на диск — как в append_snapshot, критичные данные.
    await service.shared_memory._save_async()

    heads_after = get_heads(store)
    return {
        "status": "ok",
        "id": identity_id,
        "invalidated_reason": meta["invalidation_reason"],
        "heads_after": [h.id for h in heads_after],
        "heads_count": len(heads_after),
        "note": (
            "Звено помечено недействительным. Оно осталось в истории, "
            "но больше не является концом цепочки. Продолжайте цепочку "
            "от текущей головы (heads_after), а не от этого звена."
        ),
    }

def needs_consolidation(service, stale_after_seconds: float = 7 * 86400) -> Optional[Dict[str, Any]]:
    """Чистая проверка без побочных эффектов — для вызова из
    MotivationEngine на каждом tick() дёшево, без LLM. Возвращает причину,
    если есть, иначе None:
      - несколько несведённых ВАЛИДНЫХ голов (расхождение требует внимания);
      - либо самая свежая валидная голова не обновлялась дольше stale_after_seconds;
      - либо все звенья аннулированы (цепочка фактически пуста — нужен
        новый "чистый лист").

    Аннулированные звенья игнорируются: они часть истории, но не "текущее
    состояние". Иначе свежий invalidate(old_head) тут же поднял бы ложную
    цель "ядро устарело".
    """
    store = service.shared_memory.gcn_store
    heads = get_heads(store)

    if len(heads) > 1:
        return {
            "reason": "diverging_heads",
            "heads": [h.id for h in heads],
            "heads_count": len(heads),
        }

    if not heads:
        all_objs = _all_identity_objects(store)
        if all_objs and all(_is_invalidated(o) for o in all_objs):
            return {"reason": "all_invalidated", "count": len(all_objs)}
        return None

    latest = heads[0]
    age = (datetime.now(timezone.utc) - latest.created).total_seconds()
    if age > stale_after_seconds:
        return {
            "reason": "stale",
            "head_id": latest.id,
            "age_days": round(age / 86400, 1),
        }
    return None

__all__ = [
    "IdentitySnapshot",
    "append_snapshot",
    "invalidate_snapshot",
    "get_chain",
    "get_heads",
    "get_latest_head",
    "needs_consolidation",
    "IDENTITY_SUBJECT",
    "RELATION_CONTINUES",
]