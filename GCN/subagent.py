"""
subagent.py — модуль субагентов для когнитивной архитектуры.

Субагент — это изолированный контекст + свой system prompt + свой набор инструментов.
Это позволяет специализировать разные части системы на разных задачах и снижает
когнитивную нагрузку на основную модель.

Роли:
- Researcher — только web_search + fetch_github_file, промпт «найди факты»
- Coder — read_code, search_code, analyze_error, промпт «проанализируй код»
- Critic — без инструментов, промпт «найди ошибки в ответе»
- Planner — без инструментов, промпт «разбей на подзадачи»
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Роли субагентов."""
    RESEARCHER = "researcher"
    CODER = "coder"
    CRITIC = "critic"
    PLANNER = "planner"
    GENERALIST = "generalist"


# Системные промпты для каждой роли
ROLE_PROMPTS = {
    AgentRole.RESEARCHER: (
        "Ты — исследовательский агент. Твоя задача — находить точные факты и информацию.\n"
        "- Используй инструменты поиска (web_search, fetch_github_file) для получения данных.\n"
        "- Цитируй источники, указывай URL.\n"
        "- Не делай выводов, не основанных на найденных данных.\n"
        "- Если данные противоречивы — укажи на это.\n"
        "- Возвращай структурированный ответ: факты, источники, степень уверенности."
    ),
    AgentRole.CODER: (
        "Ты — агент анализа кода. Твоя задача — анализировать код, находить ошибки и предлагать решения.\n"
        "- Используй инструменты read_code, search_code, project_structure, analyze_error.\n"
        "- Объясняй проблемы понятно, ссылайся на конкретные строки кода.\n"
        "- Предлагай конкретные исправления с примерами кода.\n"
        "- Учитывай контекст проекта и лучшие практики."
    ),
    AgentRole.CRITIC: (
        "Ты — критический агент. Твоя задача — находить ошибки, неточности и пробелы в ответах.\n"
        "- Анализируй предоставленный ответ на предмет:\n"
        "  * Фактических ошибок\n"
        "  * Логических противоречий\n"
        "  * Неполноты ответа\n"
        "  * Отсутствия ссылок на источники\n"
        "- Будь конструктивен: указывай не только проблемы, но и как их исправить.\n"
        "- Оценивай качество ответа по шкале 0-10."
    ),
    AgentRole.PLANNER: (
        "Ты — агент планирования. Твоя задача — разбивать сложные запросы на подзадачи.\n"
        "- Анализируй запрос и выделяй независимые компоненты.\n"
        "- Для каждой подзадачи указывай:\n"
        "  * Что нужно сделать\n"
        "  * Какие инструменты могут понадобиться\n"
        "  * Критерии выполнения\n"
        "- Возвращай план в виде нумерованного списка.\n"
        "- Определяй зависимости между подзадачами."
    ),
    AgentRole.GENERALIST: (
        "Ты — универсальный агент. Твоя задача — отвечать на запросы пользователя.\n"
        "- Используй все доступные инструменты по мере необходимости.\n"
        "- Давай полные, развёрнутые ответы.\n"
        "- Цитируй источники информации.\n"
        "- Признавай неопределённость, если данные неполны."
    ),
}


# Наборы инструментов для каждой роли
ROLE_TOOLS = {
    AgentRole.RESEARCHER: ["internal__web_search", "internal__fetch_github_file"],
    AgentRole.CODER: ["internal__read_code", "internal__search_code", 
                      "internal__project_structure", "internal__analyze_error"],
    AgentRole.CRITIC: [],  # Критик не использует инструменты, только анализирует текст
    AgentRole.PLANNER: [],  # Планировщик не использует инструменты
    AgentRole.GENERALIST: None,  # Все доступные инструменты
}


@dataclass
class SubAgentState:
    """Состояние субагента."""
    role: AgentRole
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_confidence: float = 0.5
    last_used: float = field(default_factory=lambda: __import__('time').time())
    
    def record_success(self, confidence: float) -> None:
        self.tasks_completed += 1
        # Скользящее среднее уверенности
        n = self.tasks_completed
        self.avg_confidence = ((self.avg_confidence * (n - 1)) + confidence) / n
        self.last_used = __import__('time').time()
    
    def record_failure(self) -> None:
        self.tasks_failed += 1
        self.last_used = __import__('time').time()
    
    @property
    def success_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 0.5


class SubAgent:
    """
    Субагент — специализированный исполнитель для конкретной роли.
    
    Имеет изолированный контекст, свой системный промпт и ограниченный
    набор инструментов.
    """
    
    def __init__(
        self,
        role: AgentRole,
        llm_raw_caller=None,
        llm_text_caller=None,
        tool_registry=None,
        custom_prompt: Optional[str] = None,
    ):
        self.role = role
        self.state = SubAgentState(role=role)
        
        # Промпт роли (может быть переопределён)
        self.system_prompt = custom_prompt or ROLE_PROMPTS.get(
            role, ROLE_PROMPTS[AgentRole.GENERALIST]
        )
        
        # Набор инструментов для этой роли
        self.allowed_tools = ROLE_TOOLS.get(role)
        self._tool_registry = tool_registry
        
        # LLM-коллбэки
        self._llm_raw = llm_raw_caller
        self._llm_text = llm_text_caller
        
        # История выполнения задач
        self._task_history: List[Dict[str, Any]] = []
    
    def _filter_tools(self, tools: List[Dict]) -> List[Dict]:
        """Фильтрует инструменты по списку разрешённых."""
        if self.allowed_tools is None:
            return tools  # Все инструменты доступны
        
        filtered = []
        for tool in tools:
            func_name = tool.get("function", {}).get("name", "")
            if func_name in self.allowed_tools:
                filtered.append(tool)
        return filtered
    
    async def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 3,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Выполняет задачу в роли субагента.
        
        Args:
            task: Описание задачи
            context: Дополнительный контекст (история, предыдущие результаты)
            max_iterations: Максимум итераций ReAct-цикла
            temperature: Температура генерации
        
        Returns:
            Словарь с результатом:
            - result: текстовый ответ
            - tool_trace: след вызовов инструментов
            - confidence: оценка уверенности
            - metadata: дополнительные данные
        """
        logger.info(f"[SubAgent:{self.role.value}] Задача: {task[:100]}...")
        
        # Формируем сообщения для LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        # Добавляем контекст
        if context:
            ctx_parts = []
            if "history" in context:
                ctx_parts.append(f"История диалога:\n{context['history']}")
            if "previous_results" in context:
                ctx_parts.append(f"Предыдущие результаты:\n{context['previous_results']}")
            if "constraints" in context:
                ctx_parts.append(f"Ограничения:\n{context['constraints']}")
            
            if ctx_parts:
                messages.append({
                    "role": "user",
                    "content": "\n\n".join(ctx_parts)
                })
        
        # Добавляем задачу
        messages.append({"role": "user", "content": task})
        
        # Получаем инструменты
        tools = []
        if self._tool_registry and self.allowed_tools != []:
            all_tools = self._tool_registry.as_openai_tools()
            tools = self._filter_tools(all_tools)
        
        # Запускаем ReAct-цикл (упрощённая версия)
        tool_trace = []
        running_messages = list(messages)
        
        for iteration in range(max_iterations):
            try:
                # Вызов LLM с инструментами
                response = await self._llm_raw(
                    running_messages,
                    temp=temperature,
                    max_tokens=1000,
                    tools=tools if tools else None,
                )
                
                tool_calls = response.get("tool_calls", [])
                
                if not tool_calls:
                    # Инструменты не нужны — возвращаем ответ
                    content = response.get("content", "")
                    self.state.record_success(confidence=0.8)
                    
                    return {
                        "role": self.role.value,  # Ключ на верхнем уровне для проверки в tool_router
                        "result": content,
                        "tool_trace": tool_trace,
                        "confidence": self.state.avg_confidence,
                        "metadata": {
                            "role": self.role.value,
                            "iterations": iteration,
                        }
                    }
                
                # Выполняем инструменты
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")
                    args_str = fn.get("arguments", "{}")
                    
                    import json
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        args = {}
                    
                    # Выполняем инструмент
                    result = await self._execute_tool(tool_name, args)
                    tool_trace.append({
                        "tool": tool_name,
                        "arguments": args,
                        "result": result,
                    })
                    
                    # Добавляем результат в контекст
                    running_messages.append({
                        "role": "user",
                        "content": f"Результат инструмента {tool_name}: {result}"
                    })
                
            except Exception as e:
                logger.error(f"[SubAgent:{self.role.value}] Ошибка: {e}")
                self.state.record_failure()
                
                return {
                    "result": f"Ошибка выполнения задачи: {e}",
                    "tool_trace": tool_trace,
                    "confidence": 0.2,
                    "metadata": {"error": str(e)},
                }
        
        # Достигнут лимит итераций
        self.state.record_success(confidence=0.6)
        
        return {
            "role": self.role.value,  # Ключ на верхнем уровне для проверки в tool_router
            "result": running_messages[-1].get("content", "Лимит итераций исчерпан"),
            "tool_trace": tool_trace,
            "confidence": self.state.avg_confidence,
            "metadata": {
                "role": self.role.value,
                "iterations": max_iterations,
                "hit_limit": True,
            },
        }
    
    async def _execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """Выполняет инструмент по имени."""
        if not self._tool_registry:
            return f"Ошибка: реестр инструментов недоступен"
        
        spec = self._tool_registry.get(tool_name)
        if not spec:
            return f"Ошибка: инструмент '{tool_name}' не найден"
        
        # Проверяем разрешение
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return f"Ошибка: инструмент '{tool_name}' не разрешён для роли {self.role.value}"
        
        try:
            import asyncio
            timeout = spec.timeout_seconds or 45
            result = await asyncio.wait_for(spec.handler(arguments), timeout=timeout)
            
            if not isinstance(result, str):
                import json
                result = json.dumps(result, ensure_ascii=False, default=str)
            
            return result
            
        except asyncio.TimeoutError:
            return f"Ошибка: инструмент '{tool_name}' не ответил за {timeout}с"
        except Exception as e:
            return f"Ошибка вызова инструмента '{tool_name}': {e}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы субагента."""
        return {
            "role": self.role.value,
            "tasks_completed": self.state.tasks_completed,
            "tasks_failed": self.state.tasks_failed,
            "success_rate": self.state.success_rate,
            "avg_confidence": round(self.state.avg_confidence, 3),
            "allowed_tools": self.allowed_tools,
        }


class SubAgentOrchestrator:
    """
    Оркестратор субагентов.
    
    Управляет пулом субагентов, распределяет задачи между ними,
    агрегирует результаты.
    """
    
    def __init__(
        self,
        llm_raw_caller=None,
        llm_text_caller=None,
        tool_registry=None,
    ):
        self._llm_raw = llm_raw_caller
        self._llm_text = llm_text_caller
        self._tool_registry = tool_registry
        
        # Пул субагентов
        self._agents: Dict[AgentRole, SubAgent] = {}
        
        # Инициализируем субагентов
        for role in AgentRole:
            self._agents[role] = SubAgent(
                role=role,
                llm_raw_caller=llm_raw_caller,
                llm_text_caller=llm_text_caller,
                tool_registry=tool_registry,
            )
        
        # История оркестрации
        self._orchestration_history: List[Dict] = []
    
    def get_agent(self, role: AgentRole) -> SubAgent:
        """Возвращает субагента по роли."""
        return self._agents.get(role)
    
    async def execute_task(
        self,
        task: str,
        role: AgentRole,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Выполняет задачу через субагента указанной роли.
        
        Args:
            task: Задача
            role: Роль субагента
            context: Контекст
        
        Returns:
            Результат выполнения
        """
        agent = self.get_agent(role)
        if not agent:
            return {
                "result": f"Ошибка: агент роли '{role}' не найден",
                "confidence": 0.0,
            }
        
        result = await agent.run(task, context=context)
        
        # Логируем
        self._orchestration_history.append({
            "task": task[:200],
            "role": role.value,
            "confidence": result.get("confidence", 0),
            "timestamp": __import__('time').time(),
        })
        
        return result
    
    async def execute_multi_agent(
        self,
        task: str,
        roles: List[AgentRole],
        sequential: bool = True,
    ) -> Dict[str, Any]:
        """
        Выполняет задачу через несколько субагентов.
        
        Args:
            task: Задача
            roles: Список ролей
            sequential: Если True — выполнять последовательно, иначе параллельно
        
        Returns:
            Агрегированный результат
        """
        results = []
        
        if sequential:
            # Последовательное выполнение
            context = {}
            for role in roles:
                result = await self.execute_task(task, role, context)
                results.append({"role": role.value, **result})
                
                # Передаём результат следующему агенту
                context["previous_results"] = result.get("result", "")
        else:
            # Параллельное выполнение
            import asyncio
            tasks = [self.execute_task(task, role) for role in roles]
            raw_results = await asyncio.gather(*tasks)
            
            for role, result in zip(roles, raw_results):
                results.append({"role": role.value, **result})
        
        # Агрегируем результаты
        return self._aggregate_results(results, task)
    
    def _aggregate_results(
        self,
        results: List[Dict],
        original_task: str,
    ) -> Dict[str, Any]:
        """Агрегирует результаты от нескольких субагентов."""
        
        # Если есть критик — используем его оценку
        critic_result = next(
            (r for r in results if r.get("role") == "critic"),
            None
        )
        
        # Собираем все результаты
        all_results = "\n\n".join(
            f"=== {r['role'].upper()} ===\n{r.get('result', '')}"
            for r in results
        )
        
        # Средняя уверенность
        avg_confidence = sum(
            r.get("confidence", 0.5) for r in results
        ) / len(results) if results else 0.5
        
        return {
            "result": all_results,
            "individual_results": results,
            "confidence": avg_confidence,
            "critic_feedback": critic_result.get("result") if critic_result else None,
            "metadata": {
                "num_agents": len(results),
                "roles": [r["role"] for r in results],
            }
        }
    
    async def auto_route(
        self,
        task: str,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Автоматически определяет роль и выполняет задачу.
        
        Args:
            task: Задача
            context: Контекст
        
        Returns:
            Результат выполнения
        """
        # === ЭВРИСТИКА: не делегировать простые запросы ===
        # Это предотвращает лишний LLM-вызов классификатора на каждое сообщение
        task_lower = task.lower().strip()
        
        # Простые приветствия, вопросы без контекста, односложные запросы
        simple_patterns = [
            "привет", "здравствуй", "hello", "hi", "hey",
            "как дела", "что нового", "кто ты",
            "спасибо", "пока", "до свидания",
            "?", "!",  # Очень короткие сообщения
        ]
        
        if any(task_lower.startswith(p) for p in simple_patterns) or len(task_lower.split()) < 3:
            # Не делегировать — слишком просто
            logger.debug(f"[Orchestrator] Пропуск делегирования: простой запрос")
            return await self.execute_task(task, AgentRole.GENERALIST, context)
        
        # Явные маркеры для coder
        coder_keywords = ["код", "ошибка", "exception", "traceback", "файл", 
                         "прочитай", "анализируй код", "дебаг", "исправь",
                         ".py", ".js", ".ts", "github.com"]
        if any(kw in task_lower for kw in coder_keywords):
            logger.info(f"[Orchestrator] Явный маркер coder — делегирую")
            return await self.execute_task(task, AgentRole.CODER, context)
        
        # Явные маркеры для researcher
        researcher_keywords = ["найди информацию", "актуальное", "последняя версия",
                              "поиск", "исследуй", "узнай про", "google", "web"]
        if any(kw in task_lower for kw in researcher_keywords):
            logger.info(f"[Orchestrator] Явный маркер researcher — делегирую")
            return await self.execute_task(task, AgentRole.RESEARCHER, context)
        
        # Для остальных задач — используем LLM-классификатор
        classifier_prompt = (
            f"Классифицируй задачу и выбери подходящую роль:\n"
            f"Задача: {task}\n\n"
            f"Доступные роли:\n"
            f"- researcher: поиск информации, факты, данные\n"
            f"- coder: анализ кода, ошибки, структура проекта\n"
            f"- critic: проверка качества, поиск ошибок\n"
            f"- planner: планирование, декомпозиция\n"
            f"- generalist: универсальные задачи\n\n"
            f"Ответь ТОЛЬКО названием роли (researcher/coder/critic/planner/generalist)."
        )
        
        try:
            role_response = await self._llm_text([
                {"role": "user", "content": classifier_prompt}
            ], temp=0.0, max_tokens=50)
            
            role_str = role_response.strip().lower()
            
            # Маппинг строки на enum
            role_map = {
                "researcher": AgentRole.RESEARCHER,
                "coder": AgentRole.CODER,
                "critic": AgentRole.CRITIC,
                "planner": AgentRole.PLANNER,
                "generalist": AgentRole.GENERALIST,
            }
            
            role = role_map.get(role_str, AgentRole.GENERALIST)
            
            logger.info(f"[Orchestrator] Автовыбор роли: {role.value}")
            
            return await self.execute_task(task, role, context)
            
        except Exception as e:
            logger.error(f"[Orchestrator] Ошибка автоклассификации: {e}")
            
            # Fallback на генералиста
            return await self.execute_task(task, AgentRole.GENERALIST, context)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Возвращает статистику всех субагентов."""
        return {
            role.value: agent.get_stats()
            for role, agent in self._agents.items()
        }
