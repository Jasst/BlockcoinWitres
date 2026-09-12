"""
autonomy.py — движок автономности и проактивности BlockcoinWitres.

Зачем этот файл
---------------
В прежней архитектуре автономность была размазана по CognitiveController четырьмя
независимыми while-True циклами (_periodic_research/_periodic_planning/_periodic_reflection/
_periodic_consolidation), каждый со своим фиксированным sleep. Из этого следовало
несколько системных слабостей, которые здесь устранены:

  1. Каждый цикл сам дёргал research() напрямую — темы не имели общей очереди,
     приоритетов, дедупликации и ретраев. Три цикла могли одновременно будить
     локальную LLM, конкурируя с самим чатом пользователя.
  2. Проактивность была «одно находка — одно уведомление»: при активной
     рефлексии пользователь получал поток отдельных тостов. Теперь находки
     копятся и одним LLM-проходом отбираются лучшие (дайджест, максимум 2 за раз).
  3. Фоновые циклы не знали, активен ли пользователь в чате прямо сейчас, и
     не знали о тихих часах — LLM могла получить параллельную нагрузку в момент,
     когда человек ждёт ответа, или будить его ночью.
  4. Не было обратной связи: если пользователь каждый раз игнорировал
     находки определённого типа, система продолжала их слать с той же частотой.

Что даёт AutonomyEngine
-----------------------
  A. Единый цикл (интервал + джиттер), который по очереди выполняет:
       - обработку приоритетной очереди исследовательских тем (ResearchQueue,
         персистентна: user_dir/autonomy_queue.json);
       - декомпозицию активных целей на проверяемые подзадачи (LLM, один раз
         на цель), каждая подзадача — отдельная тема в очереди;
       - актуализацию устаревших временно-чувствительных фактов из памяти
         (курсы/цены/новости старше TTL — тема «актуализировать»);
       - постановку в очередь неразрешённых противоречий из памяти;
       - сброс дайджеста проактивных уведомлений при выполнении условий.
  B. Очередь тем собирается из пяти источников (с бустами приоритета):
       'goal' (цель), 'goal_subtask', 'reflection' (рефлексия),
       'search_failure' (поиск в чате дал пусто), 'knowledge_gap'
       (высокая неопределённость ответа), 'contradiction', 'refresh'.
     Дедуп по нормализованному ключу: повторная постановка темы лишь слегка
     поднимает её приоритет. Вытеснение самых слабых тем при переполнении.
     Ретраи с экспоненциальным бэкоффом, TTL темы — 7 дней.
  C. Дайджест проактивности: находки копятся, LLM одним вызовом выбирает
     достойные (≤2), уведомления уходят через общий push_notification → их
     видит и браузерный чат, и MCP-клиент (общий GCN-стор). Ограничения:
     минимальный интервал между дайджестами, дневной лимит, тихие часы,
     подавление пока пользователь активно переписывается.
  D. Обучение на сигналах: после уведомления тема запоминается; если
     пользователь в течение часа продолжил её (пересечение ключевых слов) —
     приоритет источника растёт, иначе — притухает. Множители источников
     персистентны (autonomy_state.json) и переживают перезапуск.
  E. Единый бюджет: каждый фоновый проход расходует весовые единицы через
     штатный _consume_autonomous_llm_budget контроллера (research=3,
     decompose=2, digest=1) — автономность никогда не вытеснит чат.
"""

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from GCN.llm_client import call_llm
from GCN.web_search import is_time_sensitive_query

try:
    from GCN.config_ai import (
        AUTONOMY_ENABLED,
        AUTONOMY_LOOP_INTERVAL,
        AUTONOMY_LOOP_JITTER,
        AUTONOMY_USER_ACTIVE_SUPPRESS_SECONDS,
        RESEARCH_QUEUE_MAX_SIZE,
        RESEARCH_TOPIC_TTL_SECONDS,
        RESEARCH_MAX_ATTEMPTS,
        RESEARCH_RETRY_BACKOFF_SECONDS,
        RESEARCH_PRIORITY_SOURCE_BOOST,
        GOAL_DECOMPOSE_INTERVAL,
        TIME_SENSITIVE_REFRESH_INTERVAL,
        TIME_SENSITIVE_REFRESH_MAX_PER_RUN,
        DIGEST_ENABLED,
        DIGEST_MAX_ITEMS,
        DIGEST_FLUSH_MIN_ITEMS,
        DIGEST_MIN_INTERVAL_SECONDS,
        DIGEST_MAX_NOTIFICATIONS_PER_DAY,
        PROACTIVE_NOTIFICATIONS_ENABLED,
        QUIET_HOURS_START,
        QUIET_HOURS_END,
        FEEDBACK_WINDOW_SECONDS,
        FEEDBACK_POSITIVE_BONUS,
        FEEDBACK_NEGATIVE_DECAY,
    )
except ImportError:
    AUTONOMY_ENABLED = True
    AUTONOMY_LOOP_INTERVAL = 45
    AUTONOMY_LOOP_JITTER = 0.3
    AUTONOMY_USER_ACTIVE_SUPPRESS_SECONDS = 300
    RESEARCH_QUEUE_MAX_SIZE = 40
    RESEARCH_TOPIC_TTL_SECONDS = 7 * 86400
    RESEARCH_MAX_ATTEMPTS = 2
    RESEARCH_RETRY_BACKOFF_SECONDS = 1800
    RESEARCH_PRIORITY_SOURCE_BOOST = {
        "goal": 0.25, "goal_subtask": 0.20, "reflection": 0.15,
        "search_failure": 0.20, "contradiction": 0.15,
        "knowledge_gap": 0.10, "refresh": 0.05,
    }
    GOAL_DECOMPOSE_INTERVAL = 6 * 3600
    TIME_SENSITIVE_REFRESH_INTERVAL = 24 * 3600
    TIME_SENSITIVE_REFRESH_MAX_PER_RUN = 3
    DIGEST_ENABLED = True
    DIGEST_MAX_ITEMS = 5
    DIGEST_FLUSH_MIN_ITEMS = 1
    DIGEST_MIN_INTERVAL_SECONDS = 45 * 60
    DIGEST_MAX_NOTIFICATIONS_PER_DAY = 6
    PROACTIVE_NOTIFICATIONS_ENABLED = True
    QUIET_HOURS_START = 23
    QUIET_HOURS_END = 8
    FEEDBACK_WINDOW_SECONDS = 3600
    FEEDBACK_POSITIVE_BONUS = 0.15
    FEEDBACK_NEGATIVE_DECAY = 0.05

logger = logging.getLogger(__name__)

# Веса LLM-вызовов для суточного бюджета (через _consume_autonomous_llm_budget).
_BUDGET_WEIGHT_RESEARCH = 3
_BUDGET_WEIGHT_DECOMPOSE = 2
_BUDGET_WEIGHT_DIGEST = 1

_WORD_RE = re.compile(r"\b[а-яa-zё]{4,}\b")

_STOPWORDS = {
    "этот", "эта", "это", "все", "уже", "какой", "какая", "когда", "если",
    "котор", "можно", "нужно", "будет", "есть", "что", "чтобы", "сейчас",
    "сегодня", "завтра", "вчера", "почему", "зачем", "скажи", "расскажи",
}


def _keywords(text: str, limit: int = 30) -> set:
    words = {w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOPWORDS}
    return set(sorted(words)[:limit])


# =====================================================================
# Очередь исследовательских тем
# =====================================================================
@dataclass
class ResearchTopic:
    topic: str
    source: str = "unknown"
    priority: float = 0.5
    related_goal: str = ""
    enqueued_at: float = field(default_factory=time.time)
    available_at: float = field(default_factory=time.time)
    attempts: int = 0
    last_error: str = ""

    @property
    def key(self) -> str:
        return re.sub(r"\s+", " ", self.topic.lower()).strip()[:160]


class ResearchQueue:
    """Приоритетная персистентная очередь тем с дедупом, TTL и ретраями."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._items: List[ResearchTopic] = []
        self._load()

    def __len__(self) -> int:
        return len(self._items)

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._items = [ResearchTopic(**item) for item in raw if isinstance(item, dict)]
        except Exception:
            self._items = []

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps([asdict(t) for t in self._items], ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug(f"ResearchQueue save failed: {e}")

    def _purge_stale(self) -> None:
        now = time.time()
        before = len(self._items)
        self._items = [t for t in self._items if now - t.enqueued_at < RESEARCH_TOPIC_TTL_SECONDS]
        if len(self._items) != before:
            self.save()

    def enqueue(self, topic: str, source: str, priority: float, related_goal: str = "") -> bool:
        """Возвращает True, если тема новая и попала в очередь."""
        if not topic or not AUTONOMY_ENABLED:
            return False
        self._purge_stale()
        key = ResearchTopic(topic=topic).key
        for existing in self._items:
            if existing.key == key:
                # Повторная постановка — лёгкий буст приоритета, не дубль.
                existing.priority = min(1.0, existing.priority + 0.05)
                self.save()
                return False
        if len(self._items) >= RESEARCH_QUEUE_MAX_SIZE:
            self._items.sort(key=lambda t: (t.priority, t.enqueued_at))
            if self._items and self._items[0].priority >= priority:
                return False  # очередь полна и всё в ней важнее
            self._items.pop(0)
        self._items.append(ResearchTopic(
            topic=topic[:300], source=source,
            priority=max(0.0, min(1.0, priority)),
            related_goal=related_goal[:200],
        ))
        self.save()
        return True

    def pop_due(self) -> Optional[ResearchTopic]:
        now = time.time()
        due = [t for t in self._items if t.available_at <= now]
        if not due:
            return None
        due.sort(key=lambda t: (-t.priority, t.enqueued_at))
        topic = due[0]
        self._items.remove(topic)
        self.save()
        return topic

    def requeue(self, topic: ResearchTopic) -> None:
        topic.available_at = time.time() + RESEARCH_RETRY_BACKOFF_SECONDS
        self._items.append(topic)
        self.save()

    def fail(self, topic: ResearchTopic, error: str) -> None:
        topic.attempts += 1
        topic.last_error = error[:200]
        if topic.attempts < RESEARCH_MAX_ATTEMPTS:
            topic.available_at = time.time() + RESEARCH_RETRY_BACKOFF_SECONDS * topic.attempts
            self._items.append(topic)
            logger.info(
                f"[Autonomy] тема '{topic.topic[:60]}' упала ({error[:80]}), "
                f"повтор {topic.attempts}/{RESEARCH_MAX_ATTEMPTS}"
            )
        else:
            logger.info(f"[Autonomy] тема '{topic.topic[:60]}' снята после {topic.attempts} неудач")
        self.save()

    def complete(self, topic: ResearchTopic) -> None:
        self.save()


# =====================================================================
# Движок
# =====================================================================
class AutonomyEngine:
    """
    Единый фоновый движок автономности для одного пользователя.

    Создаётся CognitiveController-ом, получает ссылки на контроллер (память,
    research(), бюджет, флаги активности) и общается с внешним миром только
    через push_notification (общий GCN-стор → виден и чату, и MCP-клиенту).
    """

    def __init__(self, controller):
        self.ctl = controller
        self.user_id = controller.user_id
        self.queue = ResearchQueue(controller.user_dir / "autonomy_queue.json")

        # Инициализация SelfModel и MotivationEngine
        try:
            from GCN.self_model import SelfModel
            from GCN.motivation_engine import MotivationEngine
            
            self.self_model = SelfModel(controller.user_dir)
            self.motivation = MotivationEngine(self.self_model, getattr(controller, 'memory', None))
        except ImportError as e:
            logger.warning(f"[Autonomy] не удалось загрузить модули сознания: {e}")
            self.self_model = None
            self.motivation = None

        self._task: Optional[asyncio.Task] = None
        self._stopped = False

        self._pending_findings: List[Dict[str, Any]] = []
        self._last_digest_at: float = 0.0
        self._digest_day: str = ""
        self._digest_count_today: int = 0
        self._last_goal_decompose: float = 0.0
        self._last_refresh: float = 0.0
        self._last_motivation_tick: float = 0.0

        # Обратная связь по проактивности.
        self._notified: List[Dict[str, Any]] = []       # {keywords, ts, source}
        self._source_weight: Dict[str, float] = {}      # source -> multiplier

        self._state_path = Path(controller.user_dir) / "autonomy_state.json"
        self._load_state()

    # ---------------- lifecycle ----------------
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(
                self._run(), name=f"autonomy:{self.user_id[:12]}"
            )
            logger.info(f"[Autonomy] движок запущен для {self.user_id[:16]}")

    async def shutdown(self) -> None:
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
        self.queue.save()
        self._save_state()

    # ---------------- внешние хуки (вызываются контроллером) ----------------
    def on_user_message(self, message: str) -> None:
        """Каждое сообщение пользователя — сигнал активности и возможная реакция на уведомления."""
        self._register_feedback(message or "")

    def on_search_failed(self, query: str) -> None:
        """Поиск в чате дал пусто — тема явно интересна, но данных нет."""
        if query and len(query) >= 10:
            self.enqueue_topic(query[:200], source="search_failure", priority=0.55)

    def on_high_uncertainty(self, message: str, uncertainty: float) -> None:
        """Ответ был дан с высокой неопределённостью — стоит доисследовать."""
        if uncertainty >= 0.75 and len(message) >= 15:
            self.enqueue_topic(message[:200], source="knowledge_gap", priority=0.40)

    def enqueue_topic(self, topic: str, source: str,
                      priority: float = 0.5, related_goal: str = "") -> bool:
        if not AUTONOMY_ENABLED or not topic:
            return False
        boost = RESEARCH_PRIORITY_SOURCE_BOOST.get(source, 0.0)
        weight = self._source_weight.get(source, 1.0)
        final = max(0.0, min(1.0, (priority + boost) * weight))
        ok = self.queue.enqueue(topic=topic, source=source,
                                priority=final, related_goal=related_goal)
        if ok:
            logger.info(
                f"[Autonomy] в очередь ({source}, p={final:.2f}): {topic[:80]}"
            )
        return ok

    async def submit_finding(self, finding_text: str, source: str) -> None:
        """Находка фонового исследования — в дайджест, а не сразу пользователю."""
        if not PROACTIVE_NOTIFICATIONS_ENABLED or not finding_text:
            return
        text = finding_text.strip()
        if not text:
            return
        self._pending_findings.append(
            {"text": text[:2000], "source": source, "ts": time.time()}
        )
        # Бэкпрессер: скопилось слишком много — не ждём интервала.
        if len(self._pending_findings) >= DIGEST_MAX_ITEMS * 2:
            await self._maybe_flush_digest(force=True)
        else:
            await self._maybe_flush_digest(force=False)

    # ---------------- главный цикл ----------------
    def _interval(self) -> float:
        return AUTONOMY_LOOP_INTERVAL * (1.0 + random.uniform(0.0, AUTONOMY_LOOP_JITTER))

    async def _run(self) -> None:
        while not self._stopped:
            try:
                await asyncio.sleep(self._interval())
                if not AUTONOMY_ENABLED:
                    continue
                if self._user_active():
                    continue  # человек в чате — не конкурируем за LLM
                if self._generation_running():
                    continue  # ответ сейчас генерируется — не мешаем
                
                # Запуск метакогнитивного тика (генерация внутренних целей)
                if self.motivation and time.time() - self._last_motivation_tick > 600:
                    goal = self.motivation.tick()
                    if goal:
                        logger.info(f"[Autonomy] эндогенная цель: {goal.get('goal', '')[:60]}")
                    self._last_motivation_tick = time.time()
                
                await self._pump_queue()
                await self._maybe_decompose_goals()
                await self._maybe_refresh_time_sensitive()
                self._maybe_enqueue_contradictions()
                await self._maybe_flush_digest(force=False)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"[Autonomy] ошибка цикла: {e}")

    # ---------------- защитные проверки ----------------
    def _user_active(self) -> bool:
        last = getattr(self.ctl, "_last_activity_time", 0.0) or 0.0
        return (time.time() - last) < AUTONOMY_USER_ACTIVE_SUPPRESS_SECONDS

    def _generation_running(self) -> bool:
        state = getattr(self.ctl, "_active_stream", None)
        return bool(state) and not state.get("done", True)

    def _quiet_hours(self) -> bool:
        hour = time.localtime().tm_hour
        start, end = QUIET_HOURS_START, QUIET_HOURS_END
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end  # интервал через полночь

    def _consume_budget(self, weight: int) -> bool:
        consume = getattr(self.ctl, "_consume_autonomous_llm_budget", None)
        if consume is None:
            return True
        try:
            return bool(consume(n=weight))
        except TypeError:
            return bool(consume())  # старая сигнатаура без n

    # ---------------- обработка очереди ----------------
    async def _pump_queue(self, max_per_cycle: int = 2) -> None:
        for _ in range(max_per_cycle):
            topic = self.queue.pop_due()
            if topic is None:
                return
            if not self._consume_budget(_BUDGET_WEIGHT_RESEARCH):
                self.queue.requeue(topic)
                return
            logger.info(
                f"[Autonomy] исследую ({topic.source}, p={topic.priority:.2f}): {topic.topic[:80]}"
            )
            try:
                result = await self.ctl.research(topic.topic)
            except Exception as e:
                self.queue.fail(topic, str(e))
                continue
            answer = (result or {}).get("answer", "")
            self.queue.complete(topic)
            if answer and answer.strip():
                await self.submit_finding(answer, source=topic.source)
            if topic.source in ("goal", "goal_subtask") and topic.related_goal:
                self._bump_goal_confidence(topic.related_goal, +0.10)

    def _bump_goal_confidence(self, goal_description: str, delta: float) -> None:
        try:
            for g in self.ctl.memory.goals:
                if g.description == goal_description and g.gcn_id:
                    g.confidence = max(0.0, min(1.0, g.confidence + delta))
                    obj = self.ctl.memory.store.get(g.gcn_id)
                    meta = dict(obj.object) if obj and isinstance(obj.object, dict) else {}
                    if g.confidence >= 0.9:
                        meta["status"] = "completed"
                        g.status = "completed"
                    self.ctl.memory.store.update(
                        g.gcn_id, {"object": meta, "confidence": g.confidence}, self.user_id
                    )
                    break
        except Exception as e:
            logger.debug(f"bump_goal_confidence failed: {e}")

    # ---------------- декомпозиция целей ----------------
    async def _maybe_decompose_goals(self) -> None:
        now = time.time()
        if now - self._last_goal_decompose < GOAL_DECOMPOSE_INTERVAL:
            return
        self._last_goal_decompose = now
        try:
            goals = [g for g in self.ctl.memory.goals if g.status == "active"]
        except Exception:
            return
        for goal in goals:
            try:
                obj = self.ctl.memory.store.get(goal.gcn_id) if goal.gcn_id else None
                meta = obj.object if obj and isinstance(obj.object, dict) else {}
                if meta.get("decomposed"):
                    continue
                if not self._consume_budget(_BUDGET_WEIGHT_DECOMPOSE):
                    return
                subtasks = await self._decompose_goal(goal.description)
                if not subtasks:
                    continue
                meta = dict(meta)
                meta["decomposed"] = True
                meta["subtasks"] = subtasks[:4]
                self.ctl.memory.store.update(goal.gcn_id, {"object": meta}, self.user_id)
                for st in subtasks[:4]:
                    self.enqueue_topic(
                        st, source="goal_subtask",
                        priority=goal.priority, related_goal=goal.description,
                    )
                logger.info(
                    f"[Autonomy] цель «{goal.description[:60]}» разложена на {len(subtasks[:4])} подзадач"
                )
            except Exception as e:
                logger.debug(f"decompose goal failed: {e}")

    async def _decompose_goal(self, goal: str) -> List[str]:
        prompt = (
            "Разбей цель на 2-4 конкретные проверяемые подзадачи. Каждая подзадача — "
            "отдельный вопрос, на который можно найти актуальный ответ в интернете "
            "(сравнение, актуальные данные, пошаговая инструкция). "
            "Ответь ТОЛЬКО JSON-массивом строк, без пояснений и без markdown.\n\n"
            f"Цель: {goal}"
        )
        try:
            raw = await call_llm([{"role": "user", "content": prompt}], temp=0.2, max_tokens=200)
        except Exception as e:
            logger.debug(f"goal decompose LLM call failed: {e}")
            return []
        m = re.search(r"\[[\s\S]*\]", raw or "")
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [str(x).strip() for x in data
                if isinstance(x, str) and 10 <= len(str(x).strip()) <= 200][:4]

    # ---------------- актуализация временно-чувствительных фактов ----------------
    async def _maybe_refresh_time_sensitive(self) -> None:
        now = time.time()
        if now - self._last_refresh < TIME_SENSITIVE_REFRESH_INTERVAL:
            return
        self._last_refresh = now
        count = 0
        try:
            facts = list(self.ctl.memory.semantic_facts)
        except Exception:
            return
        # Старые факты первыми — им актуализация нужнее.
        facts.sort(key=lambda f: f.timestamp)
        for fact in facts:
            if count >= TIME_SENSITIVE_REFRESH_MAX_PER_RUN:
                break
            if now - fact.timestamp < TIME_SENSITIVE_REFRESH_INTERVAL:
                continue
            if not is_time_sensitive_query(fact.text):
                continue
            if self.enqueue_topic(
                f"Актуализировать данные: {fact.text[:150]}",
                source="refresh", priority=0.30,
            ):
                count += 1

    # ---------------- противоречия ----------------
    def _maybe_enqueue_contradictions(self) -> None:
        try:
            pairs = self.ctl.memory.get_unverified_contradictions(limit=3)
        except Exception:
            return
        for a, b in pairs:
            self.enqueue_topic(
                f"Разобрать противоречие в памяти: «{a.text[:110]}» против «{b.text[:110]}»",
                source="contradiction", priority=0.50,
            )

    # ---------------- дайджест проактивных уведомлений ----------------
    def _roll_digest_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._digest_day:
            self._digest_day = today
            self._digest_count_today = 0

    async def _maybe_flush_digest(self, force: bool = False) -> None:
        if not DIGEST_ENABLED or not PROACTIVE_NOTIFICATIONS_ENABLED:
            return
        if not force:
            if len(self._pending_findings) < DIGEST_FLUSH_MIN_ITEMS:
                return
            if time.time() - self._last_digest_at < DIGEST_MIN_INTERVAL_SECONDS:
                return
            if self._quiet_hours() or self._user_active() or self._generation_running():
                return
        if not self._pending_findings:
            return

        self._roll_digest_day()
        if self._digest_count_today >= DIGEST_MAX_NOTIFICATIONS_PER_DAY:
            # Лимит на сегодня — оставляем только свежий хвост, остальное сбрасываем.
            self._pending_findings = self._pending_findings[-DIGEST_MAX_ITEMS:]
            return

        batch = self._pending_findings[:DIGEST_MAX_ITEMS]
        self._pending_findings = self._pending_findings[DIGEST_MAX_ITEMS:]
        if not batch:
            return
        if not self._consume_budget(_BUDGET_WEIGHT_DIGEST):
            self._pending_findings = batch + self._pending_findings
            return

        texts = await self._select_notifications(batch)
        self._last_digest_at = time.time()
        for text in texts[:2]:
            try:
                await self.ctl.memory_service.push_notification(
                    text, source="autonomy_digest"
                )
                self._digest_count_today += 1
                self._notified.append({
                    "keywords": _keywords(text),
                    "ts": time.time(),
                    "source": "autonomy_digest",
                })
                logger.info(
                    f"[Autonomy] дайджест для {self.user_id[:16]}: {text[:80]}"
                )
            except Exception as e:
                logger.error(f"push_notification failed: {e}")
        self._save_state()

    async def _select_notifications(self, batch: List[Dict[str, Any]]) -> List[str]:
        """Один LLM-выбор: какие из находок достойны уведомления (≤2)."""
        listing = "\n".join(
            f"[{i}] ({item['source']}) {item['text'][:400]}"
            for i, item in enumerate(batch)
        )
        prompt = (
            "Ты — фильтр проактивных уведомлений когнитивного ассистента. Ниже — "
            f"{len(batch)} находок из фоновых исследований для пользователя.\n\n"
            f"{listing}\n\n"
            "Выбери находки, которые РЕАЛЬНО стоит отправить пользователю прямо сейчас: "
            "конкретные, не тривиальные, полезные. Максимум 2. Для каждой перепиши "
            "текст одним коротким сообщением от первого лица (1-2 предложения, "
            "разговорный тон, без вступлений вроде «вот что я нашёл»).\n"
            "Ответь ТОЛЬКО JSON-объектом вида {\"notify\": [\"текст1\", \"текст2\"]}; "
            "если ничего не достойно — {\"notify\": []}."
        )
        try:
            raw = await call_llm([{"role": "user", "content": prompt}], temp=0.3, max_tokens=300)
        except Exception as e:
            logger.debug(f"digest selection failed: {e}")
            return []
        m = re.search(r"\{[\s\S]*\}", raw or "")
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        notify = data.get("notify", [])
        if not isinstance(notify, list):
            return []
        return [str(x).strip() for x in notify if isinstance(x, str) and len(str(x).strip()) >= 20]

    # ---------------- обратная связь ----------------
    def _register_feedback(self, message: str) -> None:
        now = time.time()
        msg_kw = _keywords(message)
        if not msg_kw:
            return
        still_pending: List[Dict[str, Any]] = []
        for item in self._notified:
            age = now - item["ts"]
            overlap = msg_kw & item["keywords"]
            if overlap and age < FEEDBACK_WINDOW_SECONDS:
                # Пользователь продолжил тему уведомления — проактивность попала в цель.
                w = min(1.5, self._source_weight.get(item["source"], 1.0) + FEEDBACK_POSITIVE_BONUS)
                self._source_weight[item["source"]] = w
                logger.info(
                    f"[Autonomy] положительная реакция на уведомление "
                    f"(источник {item['source']} → x{w:.2f})"
                )
                self._save_state()
                continue  # обработано, из списка убираем
            if age < FEEDBACK_WINDOW_SECONDS:
                still_pending.append(item)
            else:
                # Окно истекло, реакции не было — лёгкое притухание источника.
                w = max(0.5, self._source_weight.get(item["source"], 1.0) - FEEDBACK_NEGATIVE_DECAY)
                self._source_weight[item["source"]] = w
                self._save_state()
        self._notified = still_pending[-20:]

    # ---------------- персистентность обучения ----------------
    def _load_state(self) -> None:
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            self._source_weight = {
                str(k): float(v) for k, v in (data.get("source_weight") or {}).items()
            }
            self._digest_day = str(data.get("digest_day", ""))
            self._digest_count_today = int(data.get("digest_count_today", 0))
        except Exception:
            pass

    def _save_state(self) -> None:
        try:
            self._state_path.write_text(json.dumps({
                "source_weight": self._source_weight,
                "digest_day": self._digest_day,
                "digest_count_today": self._digest_count_today,
            }, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as e:
            logger.debug(f"autonomy state save failed: {e}")


__all__ = ["AutonomyEngine", "ResearchQueue", "ResearchTopic"]