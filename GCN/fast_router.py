"""
fast_router.py — «быстрый / медленный» контур для CognitiveController.

Три независимых компонента, без внешних зависимостей (только stdlib):

1. ComplexityRouter — решает ЗА МИЛЛИСЕКУНДЫ (без LLM), нужен ли тяжёлый путь
   (ToolRouter / ReAct) или можно сразу стримить ответ.
   Каскад:
     (а) жёсткие правила (приветствие -> FAST; URL, код, явные глаголы
         действий, самоанализ -> TOOLS);
     (б) мягкие признаки (время-зависимость, составной запрос, длина);
     (в) ОНЛАЙН-ОБУЧЕНИЕ: логистическая регрессия на хэшированных n-граммах.
         Метка берётся бесплатно из реальности: «ReAct-цикл реально вызвал
         инструмент?» (да=1, нет=0). Система учится на собственном опыте,
         какие запросы на самом деле требовали инструментов.
     (г) защита от «залипания в FAST»: небольшая доля пограничных запросов
         принудительно идёт в медленный путь (exploration), а реплики
         пользователя вида «ты не проверил / найди в сети» трактуются как
         сигнал, что предыдущий запрос надо было отправить в TOOLS.

2. ActivityGate — шлюз приоритета: пока идёт ход пользователя, фоновые LLM-
   вызовы (planning, research, reflection...) ждут; отложенные проверки
   ответа отменяются, если пришло новое сообщение.

3. StageTimer — тайминги этапов одного хода (prepare / route / tools /
   ttft / stream / finalize) одной строкой в лог.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# 1. РОУТЕР СЛОЖНОСТИ
# =====================================================================
class Route(str, Enum):
    FAST = "fast"     # сразу стрим ответа (память уже подтянута в _prepare_messages)
    TOOLS = "tools"   # ToolRouter.run (ReAct) -> затем стрим


@dataclass
class RouteDecision:
    route: Route
    score: float                       # итоговая вероятность «нужны инструменты», 0..1
    reasons: List[str] = field(default_factory=list)
    hard: bool = False                 # решение принято жёстким правилом
    explore: bool = False              # принудительно в TOOLS ради сбора метки
    rule_score: float = 0.0
    learned_score: Optional[float] = None
    elapsed_ms: float = 0.0

    def __str__(self) -> str:
        l = "-" if self.learned_score is None else f"{self.learned_score:.2f}"
        return (f"route={self.route.value} p={self.score:.2f} "
                f"(rule={self.rule_score:.2f} learned={l}) "
                f"hard={self.hard} explore={self.explore} "
                f"why={','.join(self.reasons) or '-'}")


# ---- жёсткие правила -------------------------------------------------
_GREETING_RE = re.compile(
    r"^\s*(привет\w*|здравствуй\w*|добр(ый|ое|ого)\s+\w+|хай|hello|hi|hey|"
    r"спасибо\w*|благодарю|ок|окей|ok|понял\w*|ясно|хорошо|ладно|угу|ага|"
    r"пока|до свидания|да|нет)\W*$",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_CODE_RE = re.compile(
    r"\.(py|js|ts|tsx|jsx|json|md|yml|yaml|toml|sql|html|css|cpp|java|rs|go)\b"
    r"|traceback|```|\bdef \w+\(|\bclass \w+|github\.com|stack ?trace",
    re.IGNORECASE,
)
_ACTION_VERBS = (
    "найди", "поищи", "загугли", "погугли", "прочитай", "открой", "скачай",
    "сгенерируй", "нарисуй", "запусти", "выполни", "покажи структуру",
    "проверь в интернете", "посмотри в интернете", "поиск в сети",
)
_SELF_INTROSPECTION = (
    "твой код", "свой код", "твоя архитектура", "как ты устроен",
    "прочитай свой", "структура проекта", "дерево проекта",
    "какие файлы у тебя", "исходный код проекта", "как устроена память",
)
# ---- мягкие признаки -------------------------------------------------
_COMPOUND = (
    "сначала", "потом", "затем", "после этого", "сравни", "составь план",
    "пошагово", "по шагам", "проанализируй", "разбери", "сделай обзор",
    "исследуй", "подробно расскажи", "найди и", "и затем",
)
# ---- сигнал «ты зря не пошёл в инструменты» --------------------------
_CORRECTION_RE = re.compile(
    r"ты не (посмотр|проверил|искал|поискал|нашёл|нашел|прочитал)"
    r"|не (посмотрел|проверил|поискал)"
    r"|почему не (искал|нашёл|нашел|проверил)"
    r"|проверь в (интернете|сети)|найди в (интернете|сети)"
    r"|это устарело|неактуальн|данные устарел",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[a-zа-яё0-9_]+", re.IGNORECASE)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _extract_features(message: str) -> List[str]:
    words = [w[:5] for w in _WORD_RE.findall(message.lower())]  # грубый стемминг
    feats = [f"w:{w}" for w in words]
    feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
    n = len(message)
    feats.append("len:" + ("xs" if n < 20 else "s" if n < 60 else "m" if n < 140
                           else "l" if n < 400 else "xl"))
    if "?" in message:
        feats.append("has_q")
    if re.search(r"\d", message):
        feats.append("has_digit")
    return feats


class HashedLogReg:
    """Онлайн-логрегрессия на хэшированных признаках. Разреженные веса в dict."""

    def __init__(self, dim: int = 1 << 15, lr: float = 0.35, l2: float = 1e-4):
        self.dim, self.lr, self.l2 = dim, lr, l2
        self.w: Dict[int, float] = {}
        self.b: float = -0.8        # старт: «скорее FAST»
        self.n: int = 0

    def _idx(self, feats: List[str]) -> List[int]:
        return list({
            int.from_bytes(hashlib.blake2b(f.encode("utf-8"), digest_size=4).digest(),
                           "little") % self.dim
            for f in feats
        })

    def predict_idx(self, idxs: List[int]) -> float:
        return _sigmoid(self.b + sum(self.w.get(i, 0.0) for i in idxs))

    def update_idx(self, idxs: List[int], y: float, weight: float = 1.0) -> None:
        if not idxs:
            return
        g = self.predict_idx(idxs) - y
        step = weight * self.lr / math.sqrt(len(idxs))
        for i in idxs:
            self.w[i] = self.w.get(i, 0.0) * (1.0 - self.l2) - step * g
        self.b -= 0.1 * weight * self.lr * g
        self.n += 1

    def to_json(self) -> Dict:
        return {"b": self.b, "n": self.n, "dim": self.dim,
                "w": {str(i): round(v, 5) for i, v in self.w.items() if abs(v) > 1e-3}}

    @classmethod
    def from_json(cls, data: Dict) -> "HashedLogReg":
        m = cls(dim=int(data.get("dim", 1 << 15)))
        m.b = float(data.get("b", m.b))
        m.n = int(data.get("n", 0))
        m.w = {int(k): float(v) for k, v in (data.get("w") or {}).items()}
        return m


class ComplexityRouter:
    """
    Использование:
        router = ComplexityRouter(Path(user_dir) / "router_state.json",
                                  search_hint=needs_search_heuristic)
        d = router.classify(message, force_slow=bool(web_search or reasoning))
        if d.route is Route.TOOLS:
            tool_run = await tool_router.run(...)
            router.learn(message, used_tools=bool(tool_run["tool_trace"]))
    """

    def __init__(
        self,
        state_path: Optional[Path] = None,
        search_hint: Optional[Callable[[str], bool]] = None,
        threshold: float = 0.5,
        explore_rate: float = 0.05,
        warmup_samples: int = 150,
        max_learned_weight: float = 0.6,
        save_every: int = 20,
        rng: Optional[random.Random] = None,
    ):
        self.state_path = Path(state_path) if state_path else None
        self.search_hint = search_hint
        self.threshold = threshold
        self.explore_rate = explore_rate
        self.warmup = max(1, warmup_samples)
        self.max_learned_weight = max_learned_weight
        self.save_every = save_every
        self.rng = rng or random.Random()
        self.model = HashedLogReg()
        self._unsaved = 0
        self._last: Optional[Tuple[str, List[int], Route]] = None
        self.counters = {"fast": 0, "tools": 0, "explore": 0, "corrections": 0}
        self._load()

    # ---------- публичный API ----------
    def classify(self, message: str, force_slow: bool = False) -> RouteDecision:
        t0 = time.perf_counter()
        msg = (message or "").strip()
        self._observe_followup(msg)

        if force_slow:
            return self._finish(msg, RouteDecision(
                Route.TOOLS, 1.0, ["forced(web/reasoning/image)"], hard=True), t0)

        rule_p, hard, reasons = self._rules(msg)
        idxs = self.model._idx(_extract_features(msg))

        if hard:
            route = Route.TOOLS if rule_p >= self.threshold else Route.FAST
            return self._finish(msg, RouteDecision(
                route, rule_p, reasons, hard=True, rule_score=rule_p), t0, idxs)

        learned = self.model.predict_idx(idxs)
        w = min(self.max_learned_weight, self.model.n / self.warmup)
        p = (1.0 - w) * rule_p + w * learned
        route = Route.TOOLS if p >= self.threshold else Route.FAST
        explore = False
        if (route is Route.FAST and 0.25 <= p < self.threshold
                and self.explore_rate > 0 and self.rng.random() < self.explore_rate):
            route, explore = Route.TOOLS, True
            reasons.append("explore")
        return self._finish(msg, RouteDecision(
            route, p, reasons, hard=False, explore=explore,
            rule_score=rule_p, learned_score=learned), t0, idxs)

    def learn(self, message: str, used_tools: bool, weight: float = 1.0) -> None:
        """Вызывать ПОСЛЕ медленного пути: used_tools = tool_trace непустой."""
        idxs = self.model._idx(_extract_features((message or "").strip()))
        self.model.update_idx(idxs, 1.0 if used_tools else 0.0, weight)
        self._unsaved += 1
        if self._unsaved >= self.save_every:
            self.save()

    def stats(self) -> Dict:
        return {**self.counters, "samples": self.model.n,
                "weights": len(self.model.w), "threshold": self.threshold}

    def save(self) -> None:
        if not self.state_path:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.model.to_json(), ensure_ascii=False),
                           encoding="utf-8")
            os.replace(tmp, self.state_path)
            self._unsaved = 0
        except Exception as e:  # состояние роутера — не критично
            logger.debug(f"[router] save failed: {e}")

    # ---------- внутреннее ----------
    def _finish(self, msg: str, d: RouteDecision, t0: float,
                idxs: Optional[List[int]] = None) -> RouteDecision:
        d.elapsed_ms = (time.perf_counter() - t0) * 1000
        self.counters[d.route.value] += 1
        if d.explore:
            self.counters["explore"] += 1
        self._last = (msg, idxs if idxs is not None
                      else self.model._idx(_extract_features(msg)), d.route)
        return d

    def _observe_followup(self, msg: str) -> None:
        """Если предыдущий ход ушёл в FAST, а пользователь недоволен, что не искали/не проверяли —
        это обучающий сигнал «надо было TOOLS» (двойной вес)."""
        if self._last and self._last[2] is Route.FAST and _CORRECTION_RE.search(msg):
            _, prev_idxs, _ = self._last
            self.model.update_idx(prev_idxs, 1.0, weight=2.0)
            self.counters["corrections"] += 1
            self._unsaved += 1
            logger.info("[router] корректировка пользователя: прошлый запрос надо было в TOOLS")

    def _rules(self, msg: str) -> Tuple[float, bool, List[str]]:
        low = msg.lower()
        reasons: List[str] = []

        if not msg or _GREETING_RE.match(msg):
            return 0.02, True, ["greeting/ack"]

        if _URL_RE.search(msg):
            return 0.95, True, ["url"]
        if _CODE_RE.search(msg):
            return 0.95, True, ["code"]
        if any(v in low for v in _ACTION_VERBS):
            return 0.95, True, ["action_verb"]
        if any(m in low for m in _SELF_INTROSPECTION):
            return 0.95, True, ["self_introspection"]

        p = 0.20
        if self.search_hint is not None:
            try:
                if self.search_hint(msg):
                    p += 0.45
                    reasons.append("time_sensitive/search")
            except Exception:
                pass
        if any(m in low for m in _COMPOUND):
            p += 0.25
            reasons.append("compound")
        n = len(msg)
        if n > 300:
            p += 0.15
            reasons.append("long")
        elif n > 140:
            p += 0.08
            reasons.append("medium_long")
        elif n < 25:
            p -= 0.10
            reasons.append("short")
        if msg.count("?") >= 2:
            p += 0.10
            reasons.append("multi_question")
        return max(0.0, min(1.0, p)), False, reasons

    def _load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            self.model = HashedLogReg.from_json(
                json.loads(self.state_path.read_text(encoding="utf-8")))
            logger.info(f"[router] состояние загружено: samples={self.model.n}")
        except Exception as e:
            logger.warning(f"[router] не удалось загрузить состояние: {e}")


# =====================================================================
# 2. ШЛЮЗ ПРИОРИТЕТА (пользователь > фон)
# =====================================================================
class ActivityGate:
    """
    begin_user_turn()/end_user_turn() — вокруг генерации ответа пользователю.
    wait_idle()   — фоновый код ждёт «тишины» перед LLM-вызовом.
    register(t)   — «отменяемые» задачи (отложенные проверки ответа):
                    при новом сообщении пользователя они cancel()-ятся.
    """

    def __init__(self) -> None:
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._idle_since = 0.0
        self._preemptible: Set[asyncio.Task] = set()

    @property
    def busy(self) -> bool:
        return self._active > 0

    def begin_user_turn(self) -> None:
        self._active += 1
        self._idle.clear()
        for t in list(self._preemptible):
            if not t.done():
                t.cancel()

    def end_user_turn(self) -> None:
        self._active = max(0, self._active - 1)
        if self._active == 0:
            self._idle_since = time.monotonic()
            self._idle.set()

    def register(self, task: asyncio.Task) -> asyncio.Task:
        self._preemptible.add(task)
        task.add_done_callback(self._preemptible.discard)
        return task

    async def wait_idle(self, max_wait: float = 300.0, cooldown: float = 3.0) -> bool:
        """True — дождались тишины; False — вышли по max_wait (вызывающий решает, продолжать ли)."""
        deadline = time.monotonic() + max_wait
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if self._active:
                try:
                    await asyncio.wait_for(self._idle.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return False
                continue
            quiet = time.monotonic() - self._idle_since
            if quiet >= cooldown:
                return True
            await asyncio.sleep(min(cooldown - quiet, remaining))


def make_background_llm(llm_call: Callable[..., Awaitable],
                        gate: ActivityGate,
                        max_wait: float = 300.0,
                        cooldown: float = 3.0) -> Callable[..., Awaitable]:
    """Обёртка над call_llm с той же сигнатурой: ждёт, пока пользователь не занимает модель."""
    async def _bg_llm(*args, **kwargs):
        await gate.wait_idle(max_wait=max_wait, cooldown=cooldown)
        return await llm_call(*args, **kwargs)
    return _bg_llm


async def heuristic_only_llm(*_a, **_k):
    """Подставляется как llm_caller в intellect.make_subqueries: бросает исключение ->
    make_subqueries уходит в _heuristic_subqueries без обращения к модели."""
    raise RuntimeError("LLM-декомпозиция отключена (быстрый путь)")


# =====================================================================
# 3. ТАЙМЕР ЭТАПОВ
# =====================================================================
class StageTimer:
    def __init__(self, label: str = "") -> None:
        self.label = label
        self.t0 = self._last = time.perf_counter()
        self.marks: List[Tuple[str, float]] = []

    def mark(self, name: str) -> float:
        now = time.perf_counter()
        dt = now - self._last
        self.marks.append((name, dt))
        self._last = now
        return dt

    def total(self) -> float:
        return time.perf_counter() - self.t0

    def report(self) -> str:
        parts = " ".join(f"{n}={d:.2f}s" for n, d in self.marks)
        return f"[timing] {self.label} {parts} total={self.total():.2f}s"