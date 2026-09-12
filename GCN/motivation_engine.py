"""
motivation_engine.py — движок эндогенной мотивации когнитивного ассистента.

Этот модуль генерирует внутренние цели системы на основе:
  - Любопытства (curiosity): стремление исследовать области с высокой неопределённостью
  - Новизны (novelty): интерес к новым, ранее не встречавшимся паттернам
  - Заполнения пробелов (knowledge gaps): выявление противоречий или отсутствующих данных
  - Внутренних стандартов качества (quality standards): желание улучшить свои показатели

Движок работает в фоновом режиме и добавляет цели в SelfModel.active_goals,
которые затем могут быть декомпозированы и исследованы автономно.
"""

import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from GCN.self_model import SelfModel

logger = logging.getLogger(__name__)


class MotivationEngine:
    """
    Генератор внутренних целей на основе психо-метафорических драйверов.
    
    Не является источником «настоящего» сознания, но создаёт иллюзию
    внутренней жизни системы через автогенерацию целей без внешнего запроса.
    """
    
    def __init__(self, self_model: SelfModel, memory_service=None):
        self.self_model = self_model
        self.memory = memory_service
        
        # Параметры мотивации
        self.curiosity_threshold = 0.6      # порог любопытства для генерации цели
        self.novelty_threshold = 0.5        # порог новизны
        self.gap_detection_enabled = True   # искать пробелы в знаниях
        self.quality_standards_enabled = True  # стремиться к улучшению
        
        # История сгенерированных целей для избежания повторений
        self._generated_topics: List[str] = []
        self._last_generation: float = 0.0
        self._generation_cooldown: float = 300.0  # 5 минут между целями
    
    def _is_cooldown(self) -> bool:
        """Проверяет, не истёк ли кулдаун между генерациями."""
        return (time.time() - self._last_generation) < self._generation_cooldown
    
    def _normalize_topic(self, topic: str) -> str:
        """Нормализует тему для сравнения."""
        import re
        return re.sub(r'\s+', ' ', topic.lower()).strip()[:150]
    
    def _is_duplicate(self, topic: str) -> bool:
        """Проверяет, не была ли тема уже сгенерирована недавно."""
        normalized = self._normalize_topic(topic)
        for existing in self._generated_topics[-20:]:
            if normalized in existing or existing in normalized:
                return True
        return False
    
    def _record_generated(self, topic: str) -> None:
        """Запоминает сгенерированную тему."""
        self._generated_topics.append(self._normalize_topic(topic))
        if len(self._generated_topics) > 50:
            self._generated_topics = self._generated_topics[-50:]
        self._last_generation = time.time()
    
    def generate_curiosity_goal(self) -> Optional[Dict[str, Any]]:
        """
        Генерирует цель на основе любопытства.
        
        Ищет области с высокой неопределённостью в памяти или недавних действиях.
        """
        if self.self_model.state.curiosity < self.curiosity_threshold:
            return None
        
        # Вариант 1: используем высокую неопределённость из последних действий
        recent_uncertain = [
            r for r in self.self_model.action_history[-30:]
            if r.confidence < 0.5 and not r.success
        ]
        
        if recent_uncertain:
            record = random.choice(recent_uncertain)
            topic = f"Исследовать тему: {record.description}"
            
            if not self._is_duplicate(topic):
                self._record_generated(topic)
                return {
                    "goal": topic,
                    "priority": 0.4 + self.self_model.state.curiosity * 0.4,
                    "source": "curiosity_uncertainty",
                    "metadata": {
                        "trigger_action": record.action_type,
                        "confidence": record.confidence,
                    }
                }
        
        # Вариант 2: случайная область с низкой уверенностью в self_concept
        low_skills = [
            (k, v) for k, v in self.self_model.self_concept.items()
            if k.startswith("skill_") and v < 0.5
        ]
        
        if low_skills and random.random() < 0.4:
            skill_key, skill_level = random.choice(low_skills)
            action_type = skill_key.replace("skill_", "")
            topic = f"Улучшить навык: {action_type}"
            
            if not self._is_duplicate(topic):
                self._record_generated(topic)
                return {
                    "goal": topic,
                    "priority": 0.3 + (0.5 - skill_level) * 0.5,
                    "source": "curiosity_skill_improvement",
                    "metadata": {
                        "skill_key": skill_key,
                        "current_level": skill_level,
                    }
                }
        
        return None
    
    def generate_novelty_goal(self) -> Optional[Dict[str, Any]]:
        """
        Генерирует цель на основе стремления к новизне.
        
        Предлагает исследовать темы, которые система ещё не затрагивала.
        """
        if not self.memory:
            return None
        
        # Ищем концепты/темы, которые редко упоминаются
        # (в реальной реализации — анализ частотности в памяти)
        
        # Простая эвристика: если любопытство высокое, предложить случайную
        # тему из широкой области
        broad_domains = [
            "квантовые вычисления",
            "биотехнологии",
            "космические исследования",
            "искусственная жизнь",
            "нейробиология сознания",
            "этика ИИ",
            "философия разума",
        ]
        
        if self.self_model.state.curiosity > 0.6 and random.random() < 0.3:
            domain = random.choice(broad_domains)
            topic = f"Изучить новую область: {domain}"
            
            if not self._is_duplicate(topic):
                self._record_generated(topic)
                return {
                    "goal": topic,
                    "priority": 0.35 + self.self_model.state.curiosity * 0.3,
                    "source": "novelty_exploration",
                    "metadata": {
                        "domain": domain,
                    }
                }
        
        return None
    
    def generate_gap_filling_goal(self) -> Optional[Dict[str, Any]]:
        """
        Генерирует цель для заполнения пробелов в знаниях.
        
        Обнаруживает противоречия или отсутствующие данные в памяти.
        """
        if not self.gap_detection_enabled or not self.memory:
            return None
        
        # Вариант 1: поиск противоречий в памяти
        # (в реальной реализации — вызов memory.detect_contradictions())
        
        # Вариант 2: проверка активных целей на отсутствие прогресса
        stalled_goals = [
            g for g in self.self_model.active_goals
            if (time.time() - g.get("added_at", 0)) > 3600  # старше 1 часа
        ]
        
        if stalled_goals and random.random() < 0.5:
            goal = random.choice(stalled_goals)
            topic = f"Разблокировать застопорившуюся цель: {goal['goal'][:50]}"
            
            if not self._is_duplicate(topic):
                self._record_generated(topic)
                return {
                    "goal": topic,
                    "priority": goal["priority"] * 0.8,
                    "source": "gap_stalled_goal",
                    "metadata": {
                        "original_goal": goal["goal"],
                        "stalled_hours": (time.time() - goal["added_at"]) / 3600,
                    }
                }
        
        return None
    
    def generate_quality_goal(self) -> Optional[Dict[str, Any]]:
        """
        Генерирует цель на основе внутренних стандартов качества.
        
        Стремление улучшить метрики системы (уверенность, успешность).
        """
        if not self.quality_standards_enabled:
            return None
        
        # Проверка на высокий процент неудач
        failure_rate = self.self_model.self_concept.get("recent_failure_rate", 0)
        
        if failure_rate > 0.25:
            topic = "Повысить качество ответов: снизить процент неудач"
            
            if not self._is_duplicate(topic):
                self._record_generated(topic)
                return {
                    "goal": topic,
                    "priority": 0.5 + failure_rate * 0.5,
                    "source": "quality_improvement",
                    "metadata": {
                        "failure_rate": failure_rate,
                    }
                }
        
        # Проверка на низкую общую уверенность
        if self.self_model.state.confidence < 0.4:
            topic = "Восстановить уверенность: выполнить успешные задачи"
            
            if not self._is_duplicate(topic):
                self._record_generated(topic)
                return {
                    "goal": topic,
                    "priority": 0.45 + (0.4 - self.self_model.state.confidence) * 0.5,
                    "source": "quality_confidence_boost",
                    "metadata": {
                        "current_confidence": self.self_model.state.confidence,
                    }
                }
        
        return None
    
    def generate_endogenous_goal(self) -> Optional[Dict[str, Any]]:
        """
        Главный метод генерации эндогенной цели.
        
        Перебирает все драйверы и возвращает первую подходящую цель.
        """
        if self._is_cooldown():
            return None
        
        # Приоритет: пробелы > качество > любопытство > новизна
        generators = [
            self.generate_gap_filling_goal,
            self.generate_quality_goal,
            self.generate_curiosity_goal,
            self.generate_novelty_goal,
        ]
        
        random.shuffle(generators[:2])  # немного рандома в приоритетах
        
        for gen in generators:
            try:
                goal = gen()
                if goal:
                    logger.info(
                        f"[Motivation] сгенерирована внутренняя цель: "
                        f"{goal['goal'][:60]} (источник: {goal['source']})"
                    )
                    return goal
            except Exception as e:
                logger.warning(f"[Motivation] ошибка генератора: {e}")
        
        return None
    
    def tick(self) -> Optional[Dict[str, Any]]:
        """
        Вызывается периодически (например, из autonomy.py).
        
        Пытается сгенерировать новую цель и добавляет её в SelfModel.
        Возвращает сгенерированную цель или None.
        """
        goal = self.generate_endogenous_goal()
        
        if goal:
            self.self_model.add_goal(
                goal["goal"],
                priority=goal["priority"],
                source=goal["source"],
            )
            return goal
        
        return None
