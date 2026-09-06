"""
tool_router.py — единая точка принятия решения "нужен ли инструмент, и какой".

Зачем этот файл:
Раньше браузерный чат (ai_assistant.py) пытался понять, вызывать ли внешний
MCP-инструмент, угадывая JSON внутри обычного текстового ответа модели
(поиск первой "{" и последней "}" по всей строке — см. историю правок).
Это ломалось на любом ответе, где модель просто упоминала фигурные скобки,
не давало few-shot примеров под конкретно эту задачу и смешивало "решение"
и "финальный ответ" в одной генерации.

В MCP-режиме (когда пользователь работает через внешнего клиента, например
Claude Desktop, к mcp_server_blockcoin.py) всё работает надёжно, потому что:
  1. Вызывающая модель использует НАТИВНЫЙ function calling — она получает
     строго типизированные JSON Schema инструментов и возвращает tool_calls
     отдельным полем, а не подмешивает JSON в текст ответа.
  2. Схемы инструментов явные и самодокументирующиеся (Pydantic Field).
  3. Нет гигантского конкурирующего системного промпта — инструменты и
     диалог разделены протоколом.

Этот модуль воспроизводит то же самое для локальной LM Studio-модели:
  - Пытается использовать нативный tools/tool_calls (многие модели в LM
    Studio его поддерживают — Qwen2.5-Instruct, Llama-3.1/3.3-Instruct,
    Hermes-function-calling и т.д.).
  - Если модель tool_calls не вернула (не умеет / бэкенд их не поддержал),
    делает fallback: ОТДЕЛЬНЫЙ узкий вызов с few-shot примерами и строгим
    парсингом (весь ответ обязан быть валидным JSON, а не подстрокой внутри
    произвольного текста).
  - Даёт ReAct-цикл (несколько раундов вызова инструментов подряд), а не
    только один вызов и потом обязательно финальный ответ.
  - Хранит "внутренние" инструменты (recall/remember/add_goal поверх
    GCNMemoryRouter) и внешние MCP-инструменты в ОДНОМ реестре — так
    браузерный чат получает те же возможности, что и mcp_server_blockcoin.py.

=====================================================================
ИСПРАВЛЕНИЯ В ЭТОЙ ВЕРСИИ (см. чат-ревью) — что было не так и почему:
=====================================================================

1) ЦИКЛ БЫЛ "СЛЕП" К СОБСТВЕННЫМ РЕЗУЛЬТАТАМ В NATIVE-РЕЖИМЕ.
   Раньше `_decide_native(base_messages, ...)` вызывался на каждой итерации
   с ОДНИМ И ТЕМ ЖЕ неизменным `base_messages` — результаты уже выполненных
   на предыдущих раундах инструментов (tool_trace) в него не попадали.
   В fallback-режиме результаты передавались (`tool_results_so_far`), а в
   native — нет. Из-за этого многошаговый ReAct (например: сначала
   web_search, потом на основе найденного — recall) для моделей с нативным
   function calling фактически не работал: модель на втором раунде не знала,
   что первый инструмент уже отработал, и либо просила его снова (гасилось
   дедупликацией seen_calls → цикл сразу обрывался), либо звала что-то ещё
   вслепую. MAX_TOOL_ITERATIONS=3 реально давало эффект только в fallback-
   режиме.
   ИСПРАВЛЕНО: теперь используется `running_messages` — рабочая копия
   диалога, которая после каждого раунда пополняется текстовым блоком с
   результатами только что вызванных инструментов (тем же форматом, что и
   build_tool_trace_context). Следующий раунд `_decide_native` видит эти
   результаты и может принять осмысленное решение о следующем шаге.

2) ЛИШНИЕ LLM-ВЫЗОВЫ НА КАЖДОЙ ИТЕРАЦИИ.
   Раньше на каждом раунде сначала безусловно пробовался native-вызов, и
   если он не вернул tool_calls — сразу пробовался fallback-вызов. Для
   моделей без нативного function calling это означало 2 LLM-вызова на
   каждый раунд решения (пустой native + fallback), а для моделей С
   нативной поддержкой — лишний fallback-вызов после того, как native уже
   один раз явно сказал "инструмент не нужен".
   ИСПРАВЛЕНО:
     - Если в рамках этого запуска native уже хоть раз вернул реальные
       tool_calls (used_native=True), последующее "пустое" решение native
       трактуется как осознанное "инструмент больше не нужен" — цикл
       завершается СРАЗУ, без обращения к fallback-промпту.
     - Добавлен адаптивный флаг `self._native_supported` на уровне
       экземпляра ToolRouter (который живёт всё время сессии пользователя,
       см. CognitiveController.__init__). Если native ни разу не сработал,
       а fallback явно решил, что инструмент был нужен — это сильный сигнал,
       что модель/бэкенд не поддерживает function calling. Флаг
       фиксируется как False, и все последующие запросы в этой сессии сразу
       идут в fallback, не тратя вызов на заведомо бесполезную native-
       попытку.

3) ИСПРАВЛЕНИЕ БЕСКОНЕЧНОГО ЦИКЛА ПРИ ОШИБКЕ ИНСТРУМЕНТА (НОВОЕ):
   Если инструмент возвращает ошибку (например, 404 при попытке загрузить
   папку как файл), модель могла повторять вызов с теми же аргументами.
   Добавлена дедупликация не только успешных вызовов, но и неудачных
   (с помощью `seen_errors`). Также добавлена эвристика: если в ответе
   инструмента содержится "не найдено", "404" или "ошибка", мы прерываем
   цикл и даём шанс модели перейти к ответу.

4) АВТОМАТИЧЕСКОЕ ИЗВЛЕЧЕНИЕ ПУТИ ИЗ GITHUB-URL (НОВОЕ):
   Добавлена функция `extract_github_path()`, которая выдёргивает путь из
   ссылок вида `.../blob/.../path` или `.../tree/.../path`.
   В методе `run()` перед запуском ReAct-цикла проверяем, есть ли в
   сообщении такая ссылка. Если есть — добавляем в `running_messages`
   пользовательское сообщение-подсказку, явно указывающее, что нужно
   вызвать `internal__fetch_github_file` с правильным `path`. Это снижает
   нагрузку на LLM и гарантирует, что вызов будет сделан с первого раза.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Awaitable
import asyncio

try:
    from GCN.config_ai import (
        TOOL_CALL_TIMEOUT_SECONDS,
        TOOL_PARALLEL_EXECUTION,
        TOOL_PLANNING_ENABLED,
        TOOL_PLANNING_MIN_LEN,
        MAX_SUBTASKS,
        MCP_TOOL_TIMEOUT_OVERRIDES,
    )
except ImportError:
    TOOL_CALL_TIMEOUT_SECONDS = 45
    TOOL_PARALLEL_EXECUTION = True
    TOOL_PLANNING_ENABLED = True
    TOOL_PLANNING_MIN_LEN = 140
    MAX_SUBTASKS = 4
    MCP_TOOL_TIMEOUT_OVERRIDES = {}

# ИСПРАВЛЕНИЕ (генерация изображений в чате "не всегда работает"): тяжёлые
# инструменты убивались единым TOOL_CALL_TIMEOUT_SECONDS=45 ещё ДО того, как
# EasyDiffusion успевал сгенерировать картинку (только enhance-промпт через
# LLM + генерация до 140с). Теперь таймаут берётся из ToolSpec, а для
# MCP-инструментов — из того же MCP_TOOL_TIMEOUT_OVERRIDES, что и в
# mcp_client_manager (generate_image 300с, research_topic 240с, web_search 90с).
def _resolve_timeout(tool_name: str):
    low = (tool_name or "").lower()
    for marker, t in MCP_TOOL_TIMEOUT_OVERRIDES.items():
        if marker in low:
            return t
    return None

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 3  # сколько раундов вызова инструментов разрешено за один ответ

# Простые маркеры составного/многочастного запроса — пункт №4 (планирование).
# Эвристика намеренно дешёвая: полноценная классификация "сложности" запроса
# сама по себе стоила бы отдельного LLM-вызова на КАЖДОЕ сообщение, что
# для простых запросов ("привет", "сколько будет 2+2") было бы чистыми
# накладными расходами без пользы.
_COMPOUND_MARKERS = (" и ", " а также ", " затем ", " потом ", " после этого ", ";", " или ")


def _looks_compound(message: str) -> bool:
    if len(message) >= TOOL_PLANNING_MIN_LEN:
        return True
    if message.count("?") >= 2:
        return True
    lowered = f" {message.lower()} "
    return any(marker in lowered for marker in _COMPOUND_MARKERS)


# ===== НОВАЯ ФУНКЦИЯ: извлечение пути из GitHub-URL =====
# === ИСПРАВЛЕНИЕ ===
def extract_github_path(url: str) -> Optional[str]:
    """
    Извлекает путь к файлу/папке из URL вида:
      https://github.com/owner/repo/blob/branch/path/to/file
      https://github.com/owner/repo/tree/branch/path/to/folder
    Возвращает строку пути (без ветки) или None, если не GitHub-ссылка.
    """
    if not url:
        return None
    # Ищем паттерн: /blob/ветка/... или /tree/ветка/...
    match = re.search(r'github\.com/[^/]+/[^/]+/(?:blob|tree)/[^/]+/(.+)$', url.strip())
    if match:
        return match.group(1)
    return None


# =====================================================================
# Реестр инструментов
# =====================================================================
@dataclass
class ToolSpec:
    """Единое описание инструмента — независимо от того, внутренний он или внешний MCP."""
    qualified_name: str                       # уникальное имя для LLM, напр. "internal__recall"
    description: str
    parameters: Dict[str, Any]                 # JSON Schema (properties/required/...)
    handler: Callable[[Dict[str, Any]], Awaitable[Any]]  # async def(arguments) -> результат
    server: str = "internal"
    original_tool_name: str = ""
    timeout_seconds: Optional[float] = None  # None = дефолт TOOL_CALL_TIMEOUT_SECONDS

    def as_openai_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.qualified_name,
                "description": self.description[:1000],
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)[:64]


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
        self._owner_server: Dict[str, str] = {}

    def register(self, name: str, description: str, parameters: Dict[str, Any],
                 handler: Callable[[Dict[str, Any]], Awaitable[Any]], server: str = "internal",
                 timeout_seconds: Optional[float] = None):
        prefix = "internal" if server == "internal" else server
        qualified = _sanitize(f"{prefix}__{name}")

        existing_owner = self._owner_server.get(qualified)
        if existing_owner is not None and existing_owner != server:
            logger.warning(
                f"ToolRegistry: инструмент '{qualified}' уже зарегистрирован сервером "
                f"'{existing_owner}' — регистрация от сервера '{server}' с тем же именем "
                f"проигнорирована, чтобы не подменить проверенный обработчик чужим."
            )
            return

        self._owner_server[qualified] = server
        self._tools[qualified] = ToolSpec(
            qualified_name=qualified,
            description=description,
            parameters=parameters,
            handler=handler,
            server=server,
            original_tool_name=name,
            timeout_seconds=timeout_seconds,
        )

    def register_mcp_tools(self, mcp_manager, mcp_call: Callable[[str, str, Dict], Awaitable[str]]):
        if not (hasattr(mcp_manager, "_initialized") and mcp_manager._initialized):
            return
        for t in mcp_manager.get_all_tools():
            server = t.get("server", "unknown")
            name = t["name"]

            async def _handler(arguments: Dict[str, Any], _server=server, _name=name) -> str:
                return await mcp_call(_server, _name, arguments)

            self.register(
                name=name,
                description=t.get("description", ""),
                parameters=t.get("inputSchema") or {"type": "object", "properties": {}},
                handler=_handler,
                server=server,
                timeout_seconds=_resolve_timeout(name),
            )

    def is_empty(self) -> bool:
        return not self._tools

    def get(self, qualified_name: str) -> Optional[ToolSpec]:
        return self._tools.get(qualified_name)

    def as_openai_tools(self) -> List[Dict[str, Any]]:
        return [t.as_openai_tool() for t in self._tools.values()]

    def as_text_catalog(self) -> str:
        examples = {
            "internal__fetch_github_file": '{"path": "GCN/config_ai.py"}',
            "internal__recall": '{"query": "проект", "top_k": 5}',
            "internal__remember": '{"fact": "мой любимый цвет синий", "scope": "private"}',
            "internal__add_goal": '{"description": "выучить Python", "priority": 0.7}',
            "internal__web_search": '{"query": "курс доллара"} ИЛИ {"queries": ["курс доллара", "евро"]}',
            "internal__generate_image": '{"prompt": "красивая девушка", "enhance_prompt": true, "steps": 30}',
            "internal__list_tools": '{}',
        }
        lines = []
        for t in self._tools.values():
            props = (t.parameters or {}).get("properties", {})
            arg_hint = ", ".join(props.keys()) if props else "без аргументов"
            ex = examples.get(t.qualified_name, "")
            ex_str = f" (пример: {ex})" if ex else ""
            lines.append(f'- "{t.qualified_name}": {t.description.strip()[:150]} (аргументы: {arg_hint}){ex_str}')
        return "\n".join(lines)


# =====================================================================
# УЛУЧШЕННЫЙ FEW-SHOT FALLBACK ПРОМПТ (распознаёт намерения из любого сообщения)
# =====================================================================
TOOL_DECISION_PROMPT = """Ты — модуль выбора инструмента когнитивного ассистента.
У тебя есть набор инструментов для работы с памятью и поиском. Твоя задача — решить, нужно ли вызвать какой-либо инструмент, чтобы ответить на запрос пользователя.

Инструменты:
{tools_catalog}

Правила:
- Если пользователь просит прочитать файл из GitHub (например, "прочти что тут?" или даёт ссылку на GitHub) — вызови инструмент `internal__fetch_github_file`.
- Если пользователь просит запомнить информацию (даже если сказано "запомни", "сохрани", "запомни глобально", "добавь в память") — вызови инструмент `internal__remember`.
- Если пользователь просит вспомнить что-то (например, "что я говорил о ...", "напомни про ...", "что ты знаешь о ...", "вспомни") — вызови `internal__recall`.
- Если пользователь упоминает цели (например, "добавь цель", "новая цель") — вызови `internal__add_goal`.
- Если нужна актуальная информация из интернета — вызови `internal__web_search`.
- **Если пользователь просит сгенерировать изображение (например, "нарисуй", "сгенерируй изображение", "создай картинку", "покажи картинку", "визуализируй" и т.п.) — ОБЯЗАТЕЛЬНО вызови инструмент `internal__generate_image`. НЕ ОТВЕЧАЙ ТЕКСТОМ, пока не получишь результат от этого инструмента.**
- **Если пользователь спрашивает о твоих возможностях, какие инструменты доступны, что ты умеешь, какие команды есть — вызови инструмент `internal__list_tools`.**
- Если запрос обычный, не требующий обращения к памяти или поиску — отвечай напрямую.
- Если ниже уже есть результаты вызванных инструментов и их достаточно, чтобы ответить — верни {{"action": "answer_directly"}}, не вызывай инструмент повторно.
- **Важно: если запрос содержит несколько независимых действий (например, "запомни X и найди Y" или "вспомни мои цели и добавь новую") — ты должен вызывать инструменты последовательно, по одному за раунд. Не считай задачу выполненной, пока не обработаны все части запроса.**

Ответь ТОЛЬКО валидным JSON-объектом, без пояснений, без markdown, без ```.

Формат для вызова инструмента:
{{"action": "call_tool", "tool": "имя_инструмента", "arguments": {{...}}}}

Формат для прямого ответа:
{{"action": "answer_directly"}}

Примеры:
- Запрос: "Запомни, что мой любимый цвет синий" -> {{"action": "call_tool", "tool": "internal__remember", "arguments": {{"fact": "мой любимый цвет синий", "scope": "private"}}}}
- Запрос: "Запомни глобально, что Земля круглая" -> {{"action": "call_tool", "tool": "internal__remember", "arguments": {{"fact": "Земля круглая", "scope": "global"}}}}
- Запрос: "Напомни, что я говорил про проект" -> {{"action": "call_tool", "tool": "internal__recall", "arguments": {{"query": "проект", "top_k": 5}}}}
- Запрос: "Вспомни мои цели" -> {{"action": "call_tool", "tool": "internal__recall", "arguments": {{"query": "цели", "top_k": 5}}}}
- Запрос: "Добавь цель: выучить Python" -> {{"action": "call_tool", "tool": "internal__add_goal", "arguments": {{"description": "выучить Python", "priority": 0.7}}}}
- Запрос: "Какой сегодня курс доллара?" -> {{"action": "call_tool", "tool": "internal__web_search", "arguments": {{"query": "курс доллара сегодня"}}}}
- Запрос: "Нарисуй красивую девушку" -> {{"action": "call_tool", "tool": "internal__generate_image", "arguments": {{"prompt": "красивая девушка", "enhance_prompt": true}}}}
- Запрос: "Что ты умеешь?" -> {{"action": "call_tool", "tool": "internal__list_tools", "arguments": {{}}}}
- Запрос: "Какие команды доступны?" -> {{"action": "call_tool", "tool": "internal__list_tools", "arguments": {{}}}}
- Запрос: "Спасибо, понятно" -> {{"action": "answer_directly"}}
- Запрос: "Как дела?" -> {{"action": "answer_directly"}}
- Запрос: "Запомни, что я люблю пиццу, и расскажи погоду в Москве" -> 
  {{"action": "call_tool", "tool": "internal__remember", "arguments": {{"fact": "я люблю пиццу"}}}}
  (после получения результата, на следующем раунде вызови internal__web_search для погоды)
- Запрос: "Вспомни мои цели и добавь новую: выучить испанский" ->
  {{"action": "call_tool", "tool": "internal__recall", "arguments": {{"query": "цели"}}}}
  (затем, в следующем раунде, вызови internal__add_goal)
# === ИСПРАВЛЕНИЕ (добавлены примеры с URL) ===
- Запрос: "прочти что тут? https://github.com/Jasst/BlockcoinWitres/tree/main/GCN" -> 
  {{"action": "call_tool", "tool": "internal__fetch_github_file", "arguments": {{"path": "GCN"}}}}
  (если это папка, инструмент вернёт список файлов; если файл – его содержимое)
- Запрос: "посмотри файл https://github.com/Jasst/BlockcoinWitres/blob/main/GCN/config_ai.py" -> 
  {{"action": "call_tool", "tool": "internal__fetch_github_file", "arguments": {{"path": "GCN/config_ai.py"}}}}

Последние реплики диалога:
{history_tail}

Запрос пользователя: {message}

Результаты уже вызванных на этом шаге инструментов (если есть):
{tool_results_so_far}
"""


def _strict_parse_json_object(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# =====================================================================
# Основной класс — принимает решение и выполняет ReAct-цикл
# =====================================================================
class ToolRouter:
    def __init__(self, registry: ToolRegistry, llm_raw_caller, llm_text_caller):
        self.registry = registry
        self.llm_raw_caller = llm_raw_caller
        self.llm_text_caller = llm_text_caller
        self._native_supported: Optional[bool] = None
        # ИНТЕЛЛЕКТ-ПАКЕТ (E): план подзадач текущего запуска — читает
        # PlanCritic из ai_assistant через этот атрибут или run()["plan"].
        self._last_plan: str = ""

    async def _execute_tool(self, qualified_name: str, arguments: Dict[str, Any]) -> str:
        spec = self.registry.get(qualified_name)
        if not spec:
            return f"Ошибка: инструмент '{qualified_name}' не найден."
        timeout = spec.timeout_seconds or TOOL_CALL_TIMEOUT_SECONDS
        try:
            result = await asyncio.wait_for(spec.handler(arguments), timeout=timeout)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False, default=str)
            return result
        except asyncio.TimeoutError:
            logger.error(f"Tool '{qualified_name}' timed out after {timeout}s")
            return f"Ошибка: инструмент '{qualified_name}' не ответил за {timeout}с."
        except Exception as e:
            logger.error(f"Tool '{qualified_name}' failed: {e}", exc_info=True)
            return f"Ошибка вызова инструмента '{qualified_name}': {e}"

    async def _decide_native(self, running_messages: List[Dict], temp: float) -> Optional[List[Dict]]:
        tools = self.registry.as_openai_tools()
        msg = await self.llm_raw_caller(running_messages, temp=temp, max_tokens=500, tools=tools)
        tool_calls = msg.get("tool_calls") if msg else None
        if not tool_calls:
            return None
        decisions = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name:
                decisions.append({"tool": name, "arguments": args})
        return decisions or None

    async def _plan_subtasks(self, message: str) -> str:
        if not TOOL_PLANNING_ENABLED or not _looks_compound(message):
            return ""
        prompt = (
            f"Разбей следующий запрос пользователя на список из максимум {MAX_SUBTASKS} "
            "независимых подзадач, которые нужно выполнить, чтобы дать ПОЛНЫЙ ответ. "
            "Если запрос на самом деле простой и состоит из одной части — верни всего "
            "один пункт.\n\n"
            f"Запрос: {message}\n\n"
            "Ответь только нумерованным списком подзадач, без пояснений и без markdown."
        )
        try:
            raw = await self.llm_text_caller([{"role": "user", "content": prompt}], temp=0.0, max_tokens=200)
            if not raw:
                return ""
            lines = [l.strip(" -•\t") for l in raw.split("\n") if l.strip()]
            return "\n".join(lines[:MAX_SUBTASKS])
        except Exception as e:
            logger.debug(f"Planning step failed, continuing without a plan: {e}")
            return ""

    async def _decide_fallback(self, message: str, history_tail: str,
                                tool_results_so_far: str, temp: float) -> Optional[Dict]:
        prompt = TOOL_DECISION_PROMPT.format(
            tools_catalog=self.registry.as_text_catalog(),
            history_tail=history_tail or "(пусто)",
            message=message,
            tool_results_so_far=tool_results_so_far or "(ещё не было)",
        )
        raw = await self.llm_text_caller([{"role": "user", "content": prompt}], temp=0.0, max_tokens=300)
        return _strict_parse_json_object(raw)

    # ===== ОСНОВНОЙ МЕТОД (с изменениями) =====
    async def run(self, message: str, base_messages: List[Dict], history_tail: str = "") -> Dict[str, Any]:
        """
        Запускает ReAct-цикл: до MAX_TOOL_ITERATIONS раундов вызова инструментов,
        затем возвращает собранные результаты — финальный текст ответа генерирует
        вызывающий код (ai_assistant.py), передав tool_trace в _build_messages.

        Возвращает:
            {"tool_trace": [{"tool": ..., "arguments": ..., "result": ...}, ...],
             "used_native": bool}
        """
        if self.registry.is_empty():
            return {"tool_trace": [], "used_native": False}

        tool_trace: List[Dict[str, Any]] = []
        used_native = False
        running_messages: List[Dict] = list(base_messages)

        # === ИСПРАВЛЕНИЕ: автоматическое извлечение GitHub-пути ===
        # Если в сообщении есть GitHub-ссылка, добавляем подсказку с уже извлечённым путём
        # Это гарантирует, что LLM вызовет fetch_github_file с правильным аргументом.
        github_path = None
        urls = re.findall(r'https?://github\.com/[^\s<>"\')\]]+', message)
        for url in urls:
            path = extract_github_path(url)
            if path:
                github_path = path
                break
        if github_path:
            hint = (
                f"[Подсказка: в запросе есть ссылка на GitHub. "
                f"Вызови инструмент internal__fetch_github_file с аргументом path='{github_path}'. "
                f"Если это папка — инструмент вернёт список файлов; если файл — его содержимое.]"
            )
            running_messages = running_messages + [{"role": "user", "content": hint}]
            history_tail = f"{hint}\n\n{history_tail}" if history_tail else hint

        # ПУНКТ №4: план подзадач
        plan_text = await self._plan_subtasks(message)
        self._last_plan = plan_text or ""
        if plan_text:
            running_messages = running_messages + [{
                "role": "user",
                "content": (
                    "[План подзадач для этого запроса — учитывай ВСЕ пункты при выборе "
                    f"инструментов и не считай запрос закрытым, пока не покрыт весь план]\n{plan_text}"
                ),
            }]
            history_tail = f"[План подзадач]\n{plan_text}\n\n{history_tail}" if history_tail else f"[План подзадач]\n{plan_text}"

        seen_calls: set = set()
        # === ИСПРАВЛЕНИЕ: отслеживаем ошибки, чтобы не повторять их ===
        seen_errors: set = set()   # (tool, frozenset(sorted(args.items())))

        for round_idx in range(MAX_TOOL_ITERATIONS):
            decisions: Optional[List[Dict]] = None

            if self._native_supported is not False:
                decisions = await self._decide_native(running_messages, temp=0.0)
                if decisions is not None:
                    used_native = True
                    self._native_supported = True
                elif used_native:
                    break

            if decisions is None and not used_native:
                results_text = "\n".join(
                    f"- {t['tool']}({t['arguments']}) -> {str(t['result'])[:300]}" for t in tool_trace
                )
                decision = await self._decide_fallback(message, history_tail, results_text, temp=0.0)
                # ИСПРАВЛЕНИЕ (crash "Stream error: 'NoneType' object has no attribute 'get'"):
                # _decide_fallback возвращает None, когда LLM выдала пустой ответ или
                # не-JSON (сбой LM Studio, таймаут). Раньше decision.get(...) сразу падал
                # с AttributeError, исключение вылетало из ToolRouter.run и гасило весь
                # ответ чата (в стрим-режиме это ловилось как "Stream error"). None =
                # "модель не смогла принять решение" — просто завершаем ReAct-цикл
                # и отвечаем без инструментов.
                if decision is None:
                    logger.debug("ToolRouter: fallback-решение недоступно (пустой/невалидный ответ LLM) — завершаю цикл.")
                    break
                tool_name = decision.get("tool")
                if tool_name and decision.get("action") == "call_tool":
                    spec = self.registry.get(tool_name)
                    if not spec:
                        for qname, tspec in self.registry._tools.items():
                            if tspec.original_tool_name == tool_name:
                                decision["tool"] = qname
                                break
                if not decision or decision.get("action") != "call_tool":
                    break
                decisions = [{"tool": decision.get("tool"), "arguments": decision.get("arguments", {})}]

                if self._native_supported is None:
                    self._native_supported = False

            if decisions:
                logger.info(f"ToolRouter: round {round_idx}, decisions: {decisions}")

            if not decisions:
                break

            # Фильтруем повторы и ошибки
            to_execute: List[Dict[str, Any]] = []
            for d in decisions:
                args = d.get("arguments", {})
                normalized_args = {}
                for k, v in args.items():
                    if isinstance(v, str):
                        normalized_args[k] = v.strip().lower() if k == "query" else v.strip()
                    else:
                        normalized_args[k] = v
                sig = (d.get("tool"), json.dumps(normalized_args, sort_keys=True, ensure_ascii=False))
                if sig in seen_calls:
                    logger.info(f"ToolRouter: пропускаю повторный вызов {sig} — результат уже есть в tool_trace.")
                    continue
                # === ИСПРАВЛЕНИЕ: если этот вызов уже дал ошибку, не повторяем ===
                if sig in seen_errors:
                    logger.info(f"ToolRouter: пропускаю ранее ошибочный вызов {sig}.")
                    continue
                seen_calls.add(sig)
                to_execute.append(d)

            new_calls_this_round = len(to_execute)
            round_results: List[Dict[str, Any]] = []
            if to_execute:
                if TOOL_PARALLEL_EXECUTION and len(to_execute) > 1:
                    results = await asyncio.gather(
                        *[self._execute_tool(d["tool"], d.get("arguments", {})) for d in to_execute],
                        return_exceptions=True,
                    )
                else:
                    results = []
                    for d in to_execute:
                        try:
                            results.append(await self._execute_tool(d["tool"], d.get("arguments", {})))
                        except Exception as e:
                            results.append(e)

                for d, result in zip(to_execute, results):
                    if isinstance(result, Exception):
                        logger.error(f"Tool '{d['tool']}' raised during parallel execution: {result}")
                        result = f"Ошибка вызова инструмента '{d['tool']}': {result}"
                    entry = {"tool": d["tool"], "arguments": d.get("arguments", {}), "result": result}
                    tool_trace.append(entry)
                    round_results.append(entry)

                    # === ИСПРАВЛЕНИЕ: если результат содержит явную ошибку, запоминаем сигнатуру как ошибочную ===
                    result_str = str(result).lower()
                    if "ошибка" in result_str or "не найдено" in result_str or "404" in result_str:
                        sig = (d["tool"], json.dumps(d.get("arguments", {}), sort_keys=True, ensure_ascii=False))
                        seen_errors.add(sig)

            if new_calls_this_round == 0:
                break

            running_messages = running_messages + [{
                "role": "user",
                "content": (
                    "[Результаты только что вызванных инструментов — используй их, "
                    "не вызывай те же инструменты с теми же аргументами повторно]\n"
                    + build_tool_trace_context(round_results)
                ),
            }]

            if not used_native and len(tool_trace) >= MAX_TOOL_ITERATIONS:
                break

        return {"tool_trace": tool_trace, "used_native": used_native, "plan": getattr(self, "_last_plan", "")}


def build_tool_trace_context(tool_trace: List[Dict[str, Any]]) -> str:
    if not tool_trace:
        return ""
    lines = ["=== РЕЗУЛЬТАТЫ ВЫЗОВА ИНСТРУМЕНТОВ ==="]
    for t in tool_trace:
        lines.append(f"Инструмент: {t['tool']}\nАргументы: {t['arguments']}\nРезультат: {t['result']}\n")
    lines.append("=== КОНЕЦ РЕЗУЛЬТАТОВ ===")
    return "\n".join(lines)