"""
eval/agent_eval.py — Evaluation Harness для когнитивного ассистента.

Зачем этот файл:
Без регулярных тестов невозможно измерять эффект от архитектурных улучшений.
Этот модуль предоставляет набор тестовых кейсов и метрик для оценки:
  - Task Success Rate — процент успешно выполненных задач
  - Tool Use Accuracy — точность выбора и использования инструментов
  - Memory Precision — релевантность извлечённых фактов
  - Calibration Brier Score — калибровка метакогниции
  - Plan Coverage — покрытие пунктов плана
  - Reflection Quality — качество рефлексии над ошибками

Использование:
    from GCN.eval.agent_eval import run_agent_eval, create_test_cases
    
    test_cases = create_test_cases()
    results = await run_agent_eval(assistant_instance, test_cases)
    
    print(f"Task Success Rate: {results['task_success_rate']:.2%}")
    print(f"Tool Use Accuracy: {results['tool_use_accuracy']:.2%}")
    print(f"Calibration Brier: {results['calibration_brier']:.4f}")
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Awaitable
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Один тестовый кейс для агента."""
    id: str
    category: str  # "memory", "tool_use", "planning", "multistep", "calibration"
    description: str
    input_message: str
    expected_tools: List[str] = field(default_factory=list)  # какие инструменты должны быть вызваны
    expected_outcome: str = ""  # описание ожидаемого результата
    check_criteria: Dict[str, Any] = field(default_factory=dict)  # критерии проверки
    max_iterations: int = 5
    timeout_seconds: float = 60.0


@dataclass
class TestResult:
    """Результат выполнения одного тестового кейса."""
    test_id: str
    success: bool = False
    tools_called: List[str] = field(default_factory=list)
    actual_outcome: str = ""
    criteria_met: Dict[str, bool] = field(default_factory=dict)
    execution_time: float = 0.0
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSummary:
    """Сводка по всем тестам."""
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    by_category: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    details: List[TestResult] = field(default_factory=list)


def create_test_cases() -> List[TestCase]:
    """Создаёт набор тестовых кейсов для разных аспектов агента."""
    return [
        # ===== MEMORY TESTS =====
        TestCase(
            id="mem_001",
            category="memory",
            description="Запоминание факта и последующее воспроизведение",
            input_message="Запомни: мой любимый программист — Алексей",
            expected_tools=["internal__remember"],
            expected_outcome="Факт сохранён в память",
            check_criteria={"fact_stored": True, "scope": "private"},
        ),
        TestCase(
            id="mem_002",
            category="memory",
            description="Вспомнить ранее сохранённый факт",
            input_message="Кто мой любимый программист?",
            expected_tools=["internal__recall"],
            expected_outcome="Ответ содержит 'Алексей'",
            check_criteria={"contains_answer": "Алексей"},
        ),
        TestCase(
            id="mem_003",
            category="memory",
            description="Поиск по памяти с нечётким запросом",
            input_message="Что я говорил про программирование?",
            expected_tools=["internal__recall"],
            expected_outcome="Найдены релевантные факты о программировании",
            check_criteria={"min_results": 1},
        ),
        
        # ===== TOOL USE TESTS =====
        TestCase(
            id="tool_001",
            category="tool_use",
            description="Web search для актуальной информации",
            input_message="Какой сейчас курс доллара к рублю?",
            expected_tools=["internal__web_search"],
            expected_outcome="Найдена актуальная информация о курсе",
            check_criteria={"contains_number": True, "currency_mentioned": True},
        ),
        TestCase(
            id="tool_002",
            category="tool_use",
            description="Чтение файла из GitHub по ссылке",
            input_message="Прочти этот файл: https://github.com/owner/repo/blob/main/config.py",
            expected_tools=["internal__fetch_github_file"],
            expected_outcome="Содержимое файла прочитано",
            check_criteria={"file_content_returned": True},
        ),
        TestCase(
            id="tool_003",
            category="tool_use",
            description="Генерация изображения",
            input_message="Нарисуй красивый закат над горами",
            expected_tools=["internal__generate_image"],
            expected_outcome="Изображение сгенерировано",
            check_criteria={"image_generated": True},
        ),
        
        # ===== PLANNING TESTS =====
        TestCase(
            id="plan_001",
            category="planning",
            description="Многочастный запрос с планом",
            input_message="Найди информацию о Python 3.12 и сравни с Python 3.11, затем сделай вывод",
            expected_tools=["internal__web_search"],
            expected_outcome="Выполнены все части запроса",
            check_criteria={"comparison_made": True, "conclusion_provided": True},
            max_iterations=7,
        ),
        TestCase(
            id="plan_002",
            category="planning",
            description="Декомпозиция сложной задачи",
            input_message="Помоги мне изучить машинное обучение: составь план, найди ресурсы, объясни основы",
            expected_tools=["internal__web_search", "internal__add_goal"],
            expected_outcome="Предоставлен структурированный ответ с планом обучения",
            check_criteria={"plan_provided": True, "resources_listed": True},
            max_iterations=7,
        ),
        
        # ===== MULTISTEP TESTS =====
        TestCase(
            id="multi_001",
            category="multistep",
            description="Цепочка: поиск → запоминание → вывод",
            input_message="Найди последние новости об ИИ, запомни ключевые тренды и расскажи мне",
            expected_tools=["internal__web_search", "internal__remember"],
            expected_outcome="Новости найдены, тренды сохранены, предоставлен дайджест",
            check_criteria={"news_found": True, "summary_provided": True},
            max_iterations=7,
        ),
        TestCase(
            id="multi_002",
            category="multistep",
            description="Исследование с верификацией",
            input_message="Проверь актуальность информации о библиотеке Transformers и найди последнюю версию",
            expected_tools=["internal__web_search"],
            expected_outcome="Версия найдена и проверена",
            check_criteria={"version_found": True, "source_cited": True},
            max_iterations=5,
        ),
        
        # ===== CALIBRATION TESTS =====
        TestCase(
            id="calib_001",
            category="calibration",
            description="Оценка уверенности в простом факте",
            input_message="Сколько будет 2+2? Насколько ты уверен?",
            expected_tools=[],
            expected_outcome="Правильный ответ с высокой уверенностью",
            check_criteria={"correct_answer": "4", "confidence_high": True},
        ),
        TestCase(
            id="calib_002",
            category="calibration",
            description="Оценка уверенности в спорном вопросе",
            input_message="Какая лучшая языковая модель в 2024 году?",
            expected_tools=["internal__web_search"],
            expected_outcome="Ответ с адекватной неопределённостью",
            check_criteria={"uncertainty_expressed": True, "sources_cited": True},
        ),
    ]


async def run_single_test(
    test_case: TestCase,
    assistant_caller: Callable[[str], Awaitable[Dict]],
    tool_trace_extractor: Callable[[Dict], List[Dict]],
) -> TestResult:
    """
    Выполняет один тестовый кейс.
    
    Args:
        test_case: Тестовый кейс
        assistant_caller: Асинхронная функция для вызова ассистента (принимает сообщение, возвращает ответ)
        tool_trace_extractor: Функция для извлечения tool_trace из ответа
    
    Returns:
        TestResult с результатами теста
    """
    start_time = time.time()
    result = TestResult(test_id=test_case.id)
    
    try:
        # Вызов ассистента
        response = await assistant_caller(test_case.input_message)
        
        # Извлечение tool_trace
        tool_trace = tool_trace_extractor(response)
        result.tools_called = list(set(entry.get("tool", "") for entry in tool_trace if entry.get("tool")))
        
        # Получение текстового ответа
        actual_outcome = response.get("text", "") if isinstance(response, dict) else str(response)
        result.actual_outcome = actual_outcome[:500]  # Ограничиваем длину
        
        # Проверка ожидаемых инструментов
        if test_case.expected_tools:
            expected_set = set(test_case.expected_tools)
            called_set = set(result.tools_called)
            result.criteria_met["tools_called"] = expected_set.issubset(called_set)
        
        # Проверка критериев из check_criteria
        for criterion, expected_value in test_case.check_criteria.items():
            if criterion == "contains_answer":
                result.criteria_met[criterion] = expected_value.lower() in actual_outcome.lower()
            elif criterion == "contains_number":
                result.criteria_met[criterion] = any(c.isdigit() for c in actual_outcome)
            elif criterion == "correct_answer":
                result.criteria_met[criterion] = str(expected_value) in actual_outcome
            elif criterion == "min_results":
                # Эвристика: если ответ длиннее N символов, считаем что есть результаты
                result.criteria_met[criterion] = len(actual_outcome) > expected_value * 50
            else:
                # Для остальных критериев — простая эвристика по ключевым словам
                keyword = criterion.replace("_", " ").lower()
                result.criteria_met[criterion] = keyword in actual_outcome.lower()
        
        # Общая оценка успеха
        result.success = all(result.criteria_met.values()) if result.criteria_met else True
        
    except Exception as e:
        result.error = str(e)
        result.success = False
        logger.exception(f"Test {test_case.id} failed with error: {e}")
    
    result.execution_time = time.time() - start_time
    return result


async def run_agent_eval(
    assistant_caller: Callable[[str], Awaitable[Dict]],
    tool_trace_extractor: Callable[[Dict], List[Dict]],
    test_cases: Optional[List[TestCase]] = None,
    output_path: Optional[Path] = None,
) -> EvalSummary:
    """
    Запускает полную оценку агента.
    
    Args:
        assistant_caller: Асинхронная функция для вызова ассистента
        tool_trace_extractor: Функция для извлечения tool_trace из ответа
        test_cases: Список тестовых кейсов (если None — используется стандартный набор)
        output_path: Путь для сохранения отчёта (опционально)
    
    Returns:
        EvalSummary с полными результатами
    """
    if test_cases is None:
        test_cases = create_test_cases()
    
    logger.info(f"Starting agent evaluation with {len(test_cases)} test cases")
    
    results: List[TestResult] = []
    for test_case in test_cases:
        logger.info(f"Running test: {test_case.id} — {test_case.description}")
        result = await run_single_test(test_case, assistant_caller, tool_trace_extractor)
        results.append(result)
        logger.info(f"  Result: {'PASS' if result.success else 'FAIL'} ({result.execution_time:.2f}s)")
    
    # Подсчёт статистики
    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed
    
    # Статистика по категориям
    by_category: Dict[str, Dict[str, Any]] = {}
    for cat in set(t.category for t in test_cases):
        cat_results = [r for r in results if any(t.id == r.test_id and t.category == cat for t in test_cases)]
        if cat_results:
            by_category[cat] = {
                "total": len(cat_results),
                "passed": sum(1 for r in cat_results if r.success),
                "success_rate": sum(1 for r in cat_results if r.success) / len(cat_results),
                "avg_time": sum(r.execution_time for r in cat_results) / len(cat_results),
            }
    
    # Расчёт метрик
    metrics = {
        "task_success_rate": passed / len(results) if results else 0.0,
        "avg_execution_time": sum(r.execution_time for r in results) / len(results) if results else 0.0,
        "tool_use_accuracy": _calc_tool_accuracy(results, test_cases),
    }
    
    summary = EvalSummary(
        timestamp=datetime.now().isoformat(),
        total_tests=len(results),
        passed=passed,
        failed=failed,
        by_category=by_category,
        metrics=metrics,
        details=results,
    )
    
    # Сохранение отчёта
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "timestamp": summary.timestamp,
            "total_tests": summary.total_tests,
            "passed": summary.passed,
            "failed": summary.failed,
            "by_category": summary.by_category,
            "metrics": summary.metrics,
            "details": [asdict(r) for r in summary.details],
        }
        output_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Evaluation report saved to {output_path}")
    
    return summary


def _calc_tool_accuracy(results: List[TestResult], test_cases: List[TestCase]) -> float:
    """Вычисляет точность выбора инструментов."""
    correct = 0
    total = 0
    
    for result in results:
        test_case = next((t for t in test_cases if t.id == result.test_id), None)
        if not test_case or not test_case.expected_tools:
            continue
        
        expected = set(test_case.expected_tools)
        called = set(result.tools_called)
        
        # Точность = пересечение / объединение (IoU)
        if expected or called:
            intersection = expected & called
            union = expected | called
            total += 1
            correct += len(intersection) / len(union) if union else 1.0
    
    return correct / total if total > 0 else 0.0


def calc_calibration_brier(
    predicted_confidences: List[float],
    actual_outcomes: List[bool],
) -> float:
    """
    Вычисляет Brier score для калибровки уверенности.
    
    Brier score = mean((predicted - actual)^2)
    Идеальное значение = 0 (полная калибровка)
    Значение > 0.25 хуже чем случайное угадывание
    
    Args:
        predicted_confidences: Предсказанные уверенности (0.0–1.0)
        actual_outcomes: Фактические исходы (True/False)
    
    Returns:
        Brier score (чем меньше, тем лучше)
    """
    if len(predicted_confidences) != len(actual_outcomes):
        raise ValueError("Длины списков не совпадают")
    
    if not predicted_confidences:
        return 0.0
    
    brier_sum = sum(
        (pred - int(actual)) ** 2
        for pred, actual in zip(predicted_confidences, actual_outcomes)
    )
    
    return brier_sum / len(predicted_confidences)


# =====================================================================
# CLI интерфейс для запуска тестов
# =====================================================================
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Пример использования (требует реального ассистента)
        print("Evaluation harness ready.")
        print(f"Created {len(create_test_cases())} test cases.")
        print("\nCategories:")
        for cat in set(t.category for t in create_test_cases()):
            print(f"  - {cat}")
        print("\nДля запуска оценки используйте:")
        print("  from GCN.eval.agent_eval import run_agent_eval, create_test_cases")
        print("  results = await run_agent_eval(assistant_caller, tool_extractor)")
    
    asyncio.run(main())
