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

3) Мелкое: единый источник построения текстового блока результатов
   (`build_tool_trace_context`) используется и для промежуточных раундов, и
   для финального ответа — раньше промежуточный блок собирался отдельным
   инлайн-форматом внутри `_decide_fallback`.
"""

import json
import logging
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
    )
except ImportError:
    TOOL_CALL_TIMEOUT_SECONDS = 45
    TOOL_PARALLEL_EXECUTION = True
    TOOL_PLANNING_ENABLED = True
    TOOL_PLANNING_MIN_LEN = 140
    MAX_SUBTASKS = 4

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
        # qualified_name -> server, который его зарегистрировал первым —
        # нужно для защиты от коллизий имён (см. register()).
        self._owner_server: Dict[str, str] = {}

    def register(self, name: str, description: str, parameters: Dict[str, Any],
                 handler: Callable[[Dict[str, Any]], Awaitable[Any]], server: str = "internal"):
        """
        ИСПРАВЛЕНИЕ (коллизия имён internal vs внешний MCP того же назначения):
        раньше qualified_name ВСЕГДА строился как f"internal__{name}" независимо
        от переданного server. Если внешний MCP-сервер (например,
        mcp_servers.json -> "blockcoin-memory" -> mcp_server_blockcoin.py —
        та же память, что и у чата, но отдельным процессом со своим
        DEFAULT_USER="default_user") экспонирует инструмент с тем же именем,
        что уже зарегистрированный внутренний ("recall", "remember",
        "add_goal", "web_search", "generate_image") — qualified_name
        совпадал буквально, и т.к. self._tools — обычный dict, вторая
        регистрация (register_mcp_tools, которая всегда происходит ПОЗЖЕ
        внутренней — см. _ensure_external_tools_registered в ai_assistant.py)
        молча ЗАТИРАЛА внутренний обработчик, привязанный к self.memory_service
        (корректный user_id = адрес кошелька), внешним MCP-прокси. У внешнего
        прокси user_id — необязательный аргумент JSON Schema, который LLM в
        общем случае не заполняет (внутренние few-shot примеры его никогда
        не показывали) — итог: вызов уходил в mcp_server_blockcoin.py с
        user_id=None, там срабатывал DEFAULT_USER="default_user", и все
        recall/remember в браузерном чате после первого сообщения молча
        писали/читали ОБЩУЮ чужую память вместо личной памяти пользователя —
        то есть в точности тот десинк "чат vs MCP", который отдельно уже
        чинили на уровне MemoryService, только через другую дыру.

        Теперь: qualified_name строится с префиксом СЕРВЕРА (не всегда
        "internal__"), и если под уже занятым qualified_name сидит
        обработчик ДРУГОГО сервера — регистрация внешнего инструмента
        отклоняется с предупреждением в лог вместо тихой перезаписи.
        Повторная регистрация тем же сервером (переподключение) по-прежнему
        обновляет свою же запись как раньше.
        """
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
        )

    def register_mcp_tools(self, mcp_manager, mcp_call: Callable[[str, str, Dict], Awaitable[str]]):
        """Подтягивает инструменты из внешних MCP-серверов (mcp_client_manager) в тот же реестр."""
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
            )

    def is_empty(self) -> bool:
        return not self._tools

    def get(self, qualified_name: str) -> Optional[ToolSpec]:
        return self._tools.get(qualified_name)

    def as_openai_tools(self) -> List[Dict[str, Any]]:
        return [t.as_openai_tool() for t in self._tools.values()]

    def as_text_catalog(self) -> str:
        """Человекочитаемый список для few-shot fallback промпта."""
        lines = []
        for t in self._tools.values():
            props = (t.parameters or {}).get("properties", {})
            arg_hint = ", ".join(props.keys()) if props else "без аргументов"
            lines.append(f'- "{t.qualified_name}": {t.description.strip()[:150]} (аргументы: {arg_hint})')
        return "\n".join(lines)


# =====================================================================
# УЛУЧШЕННЫЙ FEW-SHOT FALLBACK ПРОМПТ (распознаёт намерения из любого сообщения)
# =====================================================================
TOOL_DECISION_PROMPT = """Ты — модуль выбора инструмента когнитивного ассистента.
У тебя есть набор инструментов для работы с памятью и поиском. Твоя задача — решить, нужно ли вызвать какой-либо инструмент, чтобы ответить на запрос пользователя.

Инструменты:
{tools_catalog}

Правила:
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

Последние реплики диалога:
{history_tail}

Запрос пользователя: {message}

Результаты уже вызванных на этом шаге инструментов (если есть):
{tool_results_so_far}
"""


def _strict_parse_json_object(raw: str) -> Optional[Dict]:
    """
    Строгий парсинг: в отличие от старой логики (search('{')..rfind('}') по всему
    тексту ответа), здесь мы принимаем JSON, только если ПОСЛЕ снятия markdown-
    обёртки весь ответ целиком — валидный JSON-объект. Это не даёт случайной
    фигурной скобке в обычном тексте ответа сломать парсинг.
    """
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
        """
        registry        — ToolRegistry с зарегистрированными инструментами
        llm_raw_caller   — async def(messages, temp, max_tokens, tools=None) -> raw message dict
                            (см. GCN.llm_client.call_llm_raw); должен уметь передавать tools
                            и возвращать tool_calls, если модель их поддерживает
        llm_text_caller  — async def(messages, temp, max_tokens) -> str (обычный call_llm)
        """
        self.registry = registry
        self.llm_raw_caller = llm_raw_caller
        self.llm_text_caller = llm_text_caller
        # Адаптивный кэш поддержки нативного function calling текущим бэкендом/
        # моделью. None = ещё не знаем. True = точно поддерживает (видели
        # реальные tool_calls хотя бы раз). False = есть сильные основания
        # считать, что не поддерживает (см. run()) — тогда не тратим вызов на
        # заведомо бесполезную native-попытку в последующих запросах этой сессии.
        self._native_supported: Optional[bool] = None

    async def _execute_tool(self, qualified_name: str, arguments: Dict[str, Any]) -> str:
        spec = self.registry.get(qualified_name)
        if not spec:
            return f"Ошибка: инструмент '{qualified_name}' не найден."
        try:
            # Раньше вызов ничем не был ограничен по времени — зависший
            # обработчик (особенно внешний MCP-инструмент) вешал весь ответ
            # без возможности выйти. MCPToolManager.call_tool уже ставит свой
            # таймаут для MCP-вызовов; это ещё один рубеж для внутренних
            # обработчиков (например, если web_search/LLM-вызов внутри
            # подвиснет по неучтённой причине).
            result = await asyncio.wait_for(spec.handler(arguments), timeout=TOOL_CALL_TIMEOUT_SECONDS)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False, default=str)
            return result
        except asyncio.TimeoutError:
            logger.error(f"Tool '{qualified_name}' timed out after {TOOL_CALL_TIMEOUT_SECONDS}s")
            return f"Ошибка: инструмент '{qualified_name}' не ответил за {TOOL_CALL_TIMEOUT_SECONDS}с."
        except Exception as e:
            logger.error(f"Tool '{qualified_name}' failed: {e}", exc_info=True)
            return f"Ошибка вызова инструмента '{qualified_name}': {e}"

    async def _decide_native(self, running_messages: List[Dict], temp: float) -> Optional[List[Dict]]:
        """Пытается получить решение через нативный tool_calls. Возвращает список
        {"tool": qualified_name, "arguments": {...}} или None, если модель tool_calls не вернула.

        ВАЖНО (исправление): теперь принимает `running_messages` — рабочую копию
        диалога, которая на 2+ раунде уже содержит текстовый блок с результатами
        предыдущих вызовов инструментов (см. run()). Раньше сюда всегда
        передавался неизменный исходный `base_messages`, и модель на каждом
        раунде "решала заново", не зная, что уже было сделано.
        """
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
        """
        ПУНКТ №4 (планирование): для составных запросов разбивает на подзадачи.
        Возвращает пустую строку, если план не нужен или не удался.
        """
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
            if not raw:  # пустой ответ или None
                return ""
            lines = [l.strip(" -•\t") for l in raw.split("\n") if l.strip()]
            return "\n".join(lines[:MAX_SUBTASKS])
        except Exception as e:
            logger.debug(f"Planning step failed, continuing without a plan: {e}")
            return ""

    async def _decide_fallback(self, message: str, history_tail: str,
                                tool_results_so_far: str, temp: float) -> Optional[Dict]:
        """Узкий отдельный вызов с few-shot примерами — для моделей без function calling."""
        prompt = TOOL_DECISION_PROMPT.format(
            tools_catalog=self.registry.as_text_catalog(),
            history_tail=history_tail or "(пусто)",
            message=message,
            tool_results_so_far=tool_results_so_far or "(ещё не было)",
        )
        raw = await self.llm_text_caller([{"role": "user", "content": prompt}], temp=0.0, max_tokens=300)
        return _strict_parse_json_object(raw)

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
        # Рабочая копия диалога, которую мы пополняем результатами инструментов
        # между раундами — см. пункт 1 в шапке файла. Копируем список (не
        # словари внутри), чтобы не мутировать base_messages вызывающего кода.
        running_messages: List[Dict] = list(base_messages)

        # ПУНКТ №4: план подзадач (пусто для простых запросов/при отключённой
        # настройке/при сбое LLM — см. _plan_subtasks). Добавляем его в
        # running_messages ОДИН раз, до первого раунда, чтобы он был виден
        # native-декодеру на каждой последующей итерации точно так же, как
        # результаты уже вызванных инструментов.
        plan_text = await self._plan_subtasks(message)
        if plan_text:
            running_messages = running_messages + [{
                "role": "user",
                "content": (
                    "[План подзадач для этого запроса — учитывай ВСЕ пункты при выборе "
                    f"инструментов и не считай запрос закрытым, пока не покрыт весь план]\n{plan_text}"
                ),
            }]
            history_tail = f"[План подзадач]\n{plan_text}\n\n{history_tail}" if history_tail else f"[План подзадач]\n{plan_text}"

        # Раньше при 3 итерациях цикла модель (особенно послабее локальная)
        # могла трижды подряд решить вызвать один и тот же инструмент с теми
        # же аргументами — впустую тратя раунды вместо того, чтобы перейти
        # к финальному ответу с уже полученным результатом.
        seen_calls: set = set()

        for round_idx in range(MAX_TOOL_ITERATIONS):
            decisions: Optional[List[Dict]] = None

            # Пропускаем native-попытку, если в этой сессии уже надёжно
            # установлено, что бэкенд/модель не поддерживает function calling —
            # экономим один LLM-вызов на каждый раунд (пункт 2 в шапке файла).
            if self._native_supported is not False:
                decisions = await self._decide_native(running_messages, temp=0.0)
                if decisions is not None:
                    used_native = True
                    self._native_supported = True
                elif used_native:
                    # Native уже минимум раз сработал в этом запуске и теперь
                    # явно вернул "инструментов не нужно" — доверяем этому
                    # сигналу и завершаем цикл, не тратя fallback-вызов.
                    break

            if decisions is None and not used_native:
                results_text = "\n".join(
                    f"- {t['tool']}({t['arguments']}) -> {str(t['result'])[:300]}" for t in tool_trace
                )
                decision = await self._decide_fallback(message, history_tail, results_text, temp=0.0)
                # Если инструмент не найден по точному имени — попробуем по original_tool_name
                tool_name = decision.get("tool")
                if tool_name and decision.get("action") == "call_tool":
                    spec = self.registry.get(tool_name)
                    if not spec:
                        # Ищем среди зарегистрированных по original_tool_name
                        for qname, tspec in self.registry._tools.items():
                            if tspec.original_tool_name == tool_name:
                                decision["tool"] = qname
                                break
                if not decision or decision.get("action") != "call_tool":
                    break
                decisions = [{"tool": decision.get("tool"), "arguments": decision.get("arguments", {})}]

                # Сильный сигнал, что native вообще не поддерживается этим
                # бэкендом: за весь запуск он ни разу не вернул tool_calls,
                # а fallback тем временем уверенно решил, что инструмент
                # нужен. Фиксируем это на уровне сессии (self._native_supported),
                # чтобы следующие сообщения пользователя не тратили вызов на
                # заведомо бесполезную native-попытку.
                if self._native_supported is None:
                    self._native_supported = False

            if not decisions:
                break

            # Сначала отфильтровываем повторы (дедупликация не зависит от
            # того, выполняем ли мы дальше последовательно или параллельно).
            to_execute: List[Dict[str, Any]] = []
            for d in decisions:
                args = d.get("arguments", {})
                # Нормализуем значения: обрезаем пробелы у строк
                normalized_args = {}
                for k, v in args.items():
                    if isinstance(v, str):
                        # Для поисковых запросов дополнительно приводим к нижнему регистру (опционально)
                        normalized_args[k] = v.strip().lower() if k == "query" else v.strip()
                    else:
                        normalized_args[k] = v
                sig = (d.get("tool"), json.dumps(normalized_args, sort_keys=True, ensure_ascii=False))
                if sig in seen_calls:
                    logger.info(f"ToolRouter: пропускаю повторный вызов {sig} — результат уже есть в tool_trace.")
                    continue
                seen_calls.add(sig)
                to_execute.append(d)

            new_calls_this_round = len(to_execute)
            round_results: List[Dict[str, Any]] = []
            if to_execute:
                # ПУНКТ №5: раньше несколько независимых вызовов инструментов
                # одного раунда (например, native tool_calls с 2-3 вызовами
                # сразу) выполнялись строго по очереди — await в цикле — хотя
                # ничто не мешает им идти параллельно (свой таймаут и своя
                # обработка ошибок у каждого уже есть в _execute_tool).
                # Последовательное ожидание впустую тратило время и раунды
                # ReAct-цикла. return_exceptions=True — дополнительная
                # подстраховка: _execute_tool и так ловит исключения сам,
                # но так один сорвавшийся gather-таск не обрушит остальные.
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

            # Если модель просит только то, что уже вызывалось — новой
            # информации не будет, дальше крутить цикл бессмысленно.
            if new_calls_this_round == 0:
                break

            # ИСПРАВЛЕНИЕ (пункт 1 в шапке файла): пополняем running_messages
            # результатами именно этого раунда, чтобы на следующей итерации
            # _decide_native (и, если понадобится, _decide_fallback через
            # tool_trace) видели, что уже было сделано, а не решали заново
            # вслепую по исходному base_messages.
            running_messages = running_messages + [{
                "role": "user",
                "content": (
                    "[Результаты только что вызванных инструментов — используй их, "
                    "не вызывай те же инструменты с теми же аргументами повторно]\n"
                    + build_tool_trace_context(round_results)
                ),
            }]

            # Если решали через fallback-промпт и уже набрали максимум раундов —
            # не крутим цикл дальше молча.
            if not used_native and len(tool_trace) >= MAX_TOOL_ITERATIONS:
                break

        return {"tool_trace": tool_trace, "used_native": used_native}


def build_tool_trace_context(tool_trace: List[Dict[str, Any]]) -> str:
    """Форматирует результаты инструментов для вставки в промпт финального ответа."""
    if not tool_trace:
        return ""
    lines = ["=== РЕЗУЛЬТАТЫ ВЫЗОВА ИНСТРУМЕНТОВ ==="]
    for t in tool_trace:
        lines.append(f"Инструмент: {t['tool']}\nАргументы: {t['arguments']}\nРезультат: {t['result']}\n")
    lines.append("=== КОНЕЦ РЕЗУЛЬТАТОВ ===")
    return "\n".join(lines)