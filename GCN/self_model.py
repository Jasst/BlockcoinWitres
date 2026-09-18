"""
self_model.py — динамическая модель «Я» когнитивного ассистента.

Этот модуль формирует и поддерживает непрерывную самоконцепцию системы,
включая:
  - Возможности (capabilities): что система умеет делать
  - Ограничения (limitations): известные слабости и границы
  - Цели (goals): текущие активные цели и их приоритеты
  - Эмоциональное состояние (state): метафорическое «настроение» на основе
    недавних успехов/неудач, неопределённости, нагрузки
  - История действий (action_history): последние выполненные действия

Модель обновляется после каждого значимого действия и используется для:
  - Метакогнитивного мониторинга (оценка уверенности)
  - Генерации эндогенных целей (любопытство, заполнение пробелов)
  - Формирования «точки зрения» системы на мир
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SelfState:
    """Метафорическое эмоциональное состояние системы."""
    confidence: float = 0.5        # общая уверенность в своих силах [0..1]
    curiosity: float = 0.5         # уровень любопытства [0..1]
    stress: float = 0.0            # «стресс» от неудач/высокой нагрузки [0..1]
    engagement: float = 0.5        # вовлечённость в текущую задачу [0..1]
    last_update: float = field(default_factory=time.time)

    def update(self, success: Optional[bool] = None, uncertainty: float = 0.0,
               load_factor: float = 0.0) -> None:
        """Обновляет состояние на основе результата действия."""
        now = time.time()
        # Медленный распад: коэффициент нормирован на минуты (не секунды)
        # 0.001 × минуты: стресс спадает до 0 за ~17 минут бездействия
        decay = 0.001 * ((now - self.last_update) / 60)
        
        if success is True:
            self.confidence = min(1.0, self.confidence + 0.05)
            self.stress = max(0.0, self.stress - 0.03)
        elif success is False:
            self.confidence = max(0.2, self.confidence - 0.08)
            self.stress = min(1.0, self.stress + 0.1)
        
        # Неопределённость снижает уверенность, но повышает любопытство
        self.confidence = max(0.2, self.confidence - uncertainty * 0.1)
        self.curiosity = min(1.0, self.curiosity + uncertainty * 0.15)
        
        # Нагрузка увеличивает стресс
        self.stress = min(1.0, self.stress + load_factor * 0.05)
        
        # Естественный распад
        self.confidence = max(0.3, self.confidence - decay * 0.5)
        self.stress = max(0.0, self.stress - decay)
        self.curiosity = max(0.2, self.curiosity - decay * 0.3)
        
        self.last_update = now


@dataclass
class ActionRecord:
    """Запись о выполненном действии."""
    action_type: str          # тип: "search", "recall", "tool_call", "reasoning"
    description: str          # краткое описание
    success: bool             # успех/неудача
    confidence: float         # уверенность в результате [0..1]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SelfModel:
    """
    Динамическая модель «Я» когнитивного ассистента.
    
    Хранится в user_dir/self_model.json и переживает перезапуски.
    """
    
    def __init__(self, user_dir: Path):
        self.user_dir = Path(user_dir)
        self._path = self.user_dir / "self_model.json"
        
        # Базовые возможности системы (статичные)
        self.capabilities: List[str] = [
            "поиск информации в интернете",
            "работа с долгосрочной памятью",
            "вызов внешних инструментов (MCP)",
            "генерация изображений",
            "анализ кода",
            "решение многошаговых задач",
            "проактивное исследование тем",
        ]
        
        # Известные ограничения
        self.limitations: List[str] = [
            "зависимость от качества локальной LLM",
            "ограниченный контекст окна модели",
            "отсутствие реального понимания (симуляция)",
            "задержки при работе с внешними сервисами",
        ]
        
        # Динамические компоненты
        self.state = SelfState()
        self.active_goals: List[Dict[str, Any]] = []
        self.action_history: List[ActionRecord] = []
        self.self_concept: Dict[str, Any] = {}  # абстрактная самоконцепция
        
        self._load()
    
    def _load(self) -> None:
        """Загружает сохранённую модель."""
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if "state" in raw:
                    self.state = SelfState(**raw["state"])
                if "active_goals" in raw:
                    self.active_goals = raw["active_goals"]
                if "action_history" in raw:
                    self.action_history = [
                        ActionRecord(**r) for r in raw["action_history"][-50:]
                    ]
                if "self_concept" in raw:
                    self.self_concept = raw["self_concept"]
                logger.debug(f"[SelfModel] загружено из {self._path}")
        except Exception as e:
            logger.warning(f"[SelfModel] ошибка загрузки: {e}")
            self.state = SelfState()
    
    def save(self) -> None:
        """Сохраняет модель."""
        try:
            self.user_dir.mkdir(parents=True, exist_ok=True)
            raw = {
                "state": asdict(self.state),
                "active_goals": self.active_goals[-10:],
                "action_history": [asdict(r) for r in self.action_history[-50:]],
                "self_concept": self.self_concept,
                "capabilities": self.capabilities,
                "limitations": self.limitations,
            }
            self._path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[SelfModel] ошибка сохранения: {e}")
    
    def record_action(self, action_type: str, description: str,
                      success: bool, confidence: float,
                      metadata: Dict[str, Any] = None) -> None:
        """Регистрирует выполненное действие и обновляет состояние."""
        record = ActionRecord(
            action_type=action_type,
            description=description[:200],
            success=success,
            confidence=min(1.0, max(0.0, confidence)),
            metadata=metadata or {},
        )
        self.action_history.append(record)
        
        # Обрезаем историю
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
        
        # Обновляем состояние
        uncertainty = 1.0 - confidence if not success else 0.0
        self.state.update(success=success, uncertainty=uncertainty)
        
        # Обновляем самоконцепцию на основе паттернов
        self._update_self_concept(action_type, success)
        
        # === НОВОЕ: Калибровка уверенности (Brier score buckets) ===
        self._record_calibration(confidence, success, action_type)
        
        self.save()
    
    def _record_calibration(self, predicted_confidence: float, actual_success: bool, action_type: str = None) -> None:
        """Записывает результат в бакет калибровки для последующей коррекции уверенности.
        
        ИСПРАВЛЕНИЕ: ключ включает action_type для разделения статистики по типам действий.
        """
        bucket = round(predicted_confidence * 10) / 10  # 0.0, 0.1, ..., 1.0
        action_prefix = action_type if action_type else "generic"
        bucket_key = f"calib_{action_prefix}_bucket_{bucket}"
        
        if bucket_key not in self.self_concept:
            self.self_concept[bucket_key] = {"n": 0, "success": 0}
        
        b = self.self_concept[bucket_key]
        b["n"] += 1
        b["success"] += int(actual_success)
    
    def get_calibrated_confidence(self, predicted: float, action_type: str = None) -> float:
        """
        Возвращает калиброванную вероятность успеха для заявленной уверенности.
        
        Если модель говорит «уверен 0.9», но в этом бакете успех только 60% —
        вернёт 0.6. Это радикально улучшает метакогницию.
        
        Использует Bayesian-сглаживание для плавного перехода при малом количестве данных.
        
        ИСПРАВЛЕНИЕ: использует отдельные бакеты для каждого типа действия.
        """
        bucket = round(predicted * 10) / 10
        action_prefix = action_type if action_type else "generic"
        bucket_key = f"calib_{action_prefix}_bucket_{bucket}"
        
        b = self.self_concept.get(bucket_key, {"n": 0, "success": 0})
        
        # Bayesian-сглаживание: смешиваем prior (0.5) с наблюдениями
        # n=0 → 0.5, n=1 success=1 → 0.58, n=5 success=5 → 0.75
        prior_n = 5
        prior_success = 2.5  # prior mean = 0.5
        smoothed_rate = (prior_success + b["success"]) / (prior_n + b["n"])
        
        return smoothed_rate
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Возвращает статистику калибровки по всем бакетам."""
        stats = {}
        for key, value in self.self_concept.items():
            if key.startswith("calibration_bucket_"):
                bucket = key.replace("calibration_bucket_", "")
                n = value.get("n", 0)
                success = value.get("success", 0)
                if n > 0:
                    stats[bucket] = {
                        "n": n,
                        "success_rate": round(success / n, 3),
                    }
        return stats
    
    def _update_self_concept(self, action_type: str, success: bool) -> None:
        """Обновляет абстрактную самоконцепцию на основе паттернов действий."""
        key = f"skill_{action_type}"
        current = self.self_concept.get(key, 0.5)
        
        if success:
            self.self_concept[key] = min(1.0, current + 0.02)
        else:
            self.self_concept[key] = max(0.2, current - 0.05)
        
        # Добавляем мета-знание о своих паттернах
        recent = self.action_history[-20:]
        if recent:
            self.self_concept["recent_failure_rate"] = sum(1 for r in recent if not r.success) / len(recent)
        else:
            self.self_concept["recent_failure_rate"] = 0.0
    
    def add_goal(self, goal: str, priority: float = 0.5, source: str = "external") -> None:
        """Добавляет активную цель."""
        self.active_goals.append({
            "goal": goal[:300],
            "priority": min(1.0, max(0.0, priority)),
            "source": source,
            "added_at": time.time(),
        })
        # Обрезаем
        if len(self.active_goals) > 20:
            self.active_goals = sorted(
                self.active_goals, key=lambda g: -g["priority"]
            )[:20]
        self.save()
    
    def remove_goal(self, goal: str) -> None:
        """Удаляет завершённую цель."""
        self.active_goals = [
            g for g in self.active_goals
            if g["goal"][:50].lower() != goal[:50].lower()
        ]
        self.save()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Возвращает краткую сводку состояния для использования в промптах."""
        return {
            "confidence": round(self.state.confidence, 2),
            "curiosity": round(self.state.curiosity, 2),
            "stress": round(self.state.stress, 2),
            "engagement": round(self.state.engagement, 2),
            "recent_failure_rate": round(
                self.self_concept.get("recent_failure_rate", 0.0), 2
            ),
            "active_goals_count": len(self.active_goals),
            "recent_actions": len(self.action_history),
            "self_concept_keys": list(self.self_concept.keys()),
        }
    
    def generate_self_prompt(self) -> str:
        """Генерирует текстовый блок о состоянии системы для промпта."""
        state = self.state
        lines = [
            "=== ТЕКУЩЕЕ СОСТОЯНИЕ СИСТЕМЫ ===",
            f"Уверенность: {state.confidence:.2f}",
            f"Любопытство: {state.curiosity:.2f}",
            f"Нагрузка: {state.stress:.2f}",
            f"Активных целей: {len(self.active_goals)}",
        ]
        
        if self.active_goals:
            lines.append("Текущие цели:")
            for i, g in enumerate(self.active_goals[-3:], 1):
                lines.append(f"  {i}. {g['goal'][:60]} (приоритет: {g['priority']:.2f})")
        
        if self.self_concept.get("recent_failure_rate", 0) > 0.3:
            lines.append("ВНИМАНИЕ: высокий процент неудач в последних действиях.")
        
        return "\n".join(lines)
    
    def get_confidence_for_action(self, action_type: str) -> float:
        """Возвращает оценку уверенности для конкретного типа действия."""
        base = self.state.confidence
        skill_key = f"skill_{action_type}"
        skill_level = self.self_concept.get(skill_key, 0.5)
        
        # Комбинируем базовую уверенность с навыком
        raw_confidence = (base * 0.4 + skill_level * 0.6)
        
        # === НОВОЕ: Применяем калибровку ===
        # Вместо сырой уверенности используем калиброванную (теперь с учётом типа действия)
        return self.get_calibrated_confidence(raw_confidence, action_type)
    
    def is_overloaded(self) -> bool:
        """Проверяет, перегружена ли система."""
        return self.state.stress > 0.7 or len(self.active_goals) > 15
