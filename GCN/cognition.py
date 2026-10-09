"""cognition.py — SelfModel + MotivationEngine + Intellect."""
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from GCN.llm_client import call_llm
from GCN.tool_router import _looks_compound
from GCN.web_search import domain_trust

try:
    from GCN.config_ai import (
        GROUNDED_ANSWER_ENABLED, SUBQUERY_RETRIEVAL_ENABLED,
        MAX_RETRIEVE_SUBQUERIES, GROUNDED_MAX_SOURCES,
        GROUNDED_APPEND_SOURCES_FALLBACK,
        METACOGNITION_ENABLED, METACOGNITION_CONFIDENCE_THRESHOLD,
        PLAN_CRITIC_ENABLED, PLAN_CRITIC_MAX_MISSED,
    )
except ImportError:
    GROUNDED_ANSWER_ENABLED = True
    SUBQUERY_RETRIEVAL_ENABLED = True
    MAX_RETRIEVE_SUBQUERIES = 3
    GROUNDED_MAX_SOURCES = 8
    GROUNDED_APPEND_SOURCES_FALLBACK = False
    METACOGNITION_ENABLED = True
    METACOGNITION_CONFIDENCE_THRESHOLD = 0.4
    PLAN_CRITIC_ENABLED = True
    PLAN_CRITIC_MAX_MISSED = 3

logger = logging.getLogger(__name__)


# ============================================================
# SECTION 1: SelfModel
# ============================================================

def _normalize_goal_text(text: str) -> str:
    """Нормализует текст цели для сравнения (для prune_goals)."""
    import re
    return re.sub(r'\s+', ' ', (text or '').lower().strip())[:300]


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
        """Возвращает статистику калибровки по всем бакетам.

        ИСПРАВЛЕНО: раньше фильтровало по устаревшему префиксу
        "calibration_bucket_", а _record_calibration пишет ключи вида
        "calib_{action_type}_bucket_{X}" (см. правку с разделением
        статистики по типам действий) — префиксы никогда не совпадали,
        и метод молча всегда возвращал {}. Теперь парсим action_type и
        bucket из реального формата ключа и группируем по action_type,
        чтобы был виден разброс калибровки по типам действий, а не только
        общая цифра.
        """
        stats: Dict[str, Any] = {}
        for key, value in self.self_concept.items():
            if not key.startswith("calib_") or "_bucket_" not in key:
                continue
            n = value.get("n", 0) if isinstance(value, dict) else 0
            success = value.get("success", 0) if isinstance(value, dict) else 0
            if n <= 0:
                continue
            action_prefix, _, bucket = key[len("calib_"):].rpartition("_bucket_")
            action_prefix = action_prefix or "generic"
            entry = {
                "n": n,
                "success_rate": round(success / n, 3),
            }
            stats.setdefault(action_prefix, {})[bucket] = entry
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

    def add_goal(self, goal: str, priority: float = 0.5, source: str = "external",
                 gcn_id: Optional[str] = None) -> None:
        """Добавляет активную цель.

        gcn_id — необязательная ссылка на объект в GCN-памяти. Если задана,
        sync_from_gcn() сможет сопоставить запись в active_goals с GCN-целью
        и не потерять её при следующей синхронизации.
        """
        norm = _normalize_goal_text(goal)
        for existing in self.active_goals:
            if _normalize_goal_text(existing.get("goal", "")) == norm:
                existing["priority"] = min(1.0, max(0.0, priority))
                if gcn_id:
                    existing["gcn_id"] = gcn_id
                self.save()
                return
        self.active_goals.append({
            "goal": goal[:300],
            "priority": min(1.0, max(0.0, priority)),
            "source": source,
            "added_at": time.time(),
            "gcn_id": gcn_id,
        })
        if len(self.active_goals) > 20:
            self.active_goals = sorted(
                self.active_goals, key=lambda g: -g["priority"]
            )[:20]
        self.save()

    def sync_from_gcn(self, gcn_goals: List[Dict[str, Any]]) -> int:
        """Пересобирает active_goals из GCN-целей.

        GCN — источник истины для целей. SelfModel.active_goals — проекция
        для быстрого чтения в промптах и в get_self_state(). Возвращает
        количество записей в active_goals ДО синхронизации.
        """
        before = len(self.active_goals)

        # 1. Сохраняем транзиентные эндогенные цели (ещё не в GCN)
        transient = [
            g for g in self.active_goals
            if g.get("source") in ("gap_stalled_goal",)
            or str(g.get("source", "")).startswith("identity_")
        ]
        transient_norms = {_normalize_goal_text(g.get("goal", "")) for g in transient}

        # 2. Строим проекцию из GCN
        projected: List[Dict[str, Any]] = []
        now = time.time()
        for g in gcn_goals:
            desc = (g.get("description") or "").strip()
            if not desc:
                continue
            norm = _normalize_goal_text(desc)
            if norm in transient_norms:
                continue
            projected.append({
                "goal": desc[:300],
                "priority": float(g.get("priority", 0.5)),
                "source": "gcn",
                "added_at": now,
                "gcn_id": g.get("gcn_id") or g.get("id"),
            })

        self.active_goals = projected + transient

        # 3. Обрезаем
        if len(self.active_goals) > 20:
            self.active_goals = sorted(
                self.active_goals, key=lambda g: -g.get("priority", 0.0)
            )[:20]

        self.save()
        logger.info(
            f"[SelfModel] sync_from_gcn: было {before}, стало {len(self.active_goals)} "
            f"(GCN: {len(projected)}, transient: {len(transient)})"
        )
        return before

    def remove_goal(self, goal: str) -> None:
        """Удаляет завершённую цель."""
        self.active_goals = [
            g for g in self.active_goals
            if g["goal"][:50].lower() != goal[:50].lower()
        ]
        self.save()

    def prune_goals(self, known_descriptions: Optional[List[str]] = None,
                    meta_prefixes: Tuple[str, ...] = ("разблокировать застопорившуюся цель",),
                    max_age_seconds: Optional[float] = None) -> int:
        """Сборщик мусора списка целей (ПАТЧ fact_cc5a3a86, часть 2).

        Убирает:
          1) legacy-мусор рекурсии gap_stalled_goal (цели с накопленным
             префиксом «Разблокировать…» в тексте — пережиток бага);
          2) цели, отсутствующие в authoritative-списке GCN-памяти
             (known_descriptions) — раньше SelfModel жил своей жизнью:
             get_goals() показывал 3 цели из памяти, а self_state — до 10
             «активных» из active_goals;
          3) цели старше max_age_seconds (если задан).

        Возвращает число удалённых целей.
        """
        before = len(self.active_goals)
        kept = []
        for g in self.active_goals:
            text = g.get("goal", "")
            low = text.lower()
            if any(p in low for p in meta_prefixes):
                continue
            if known_descriptions is not None:
                norm_text = _normalize_goal_text(text)
                if not any(norm_text == _normalize_goal_text(d) for d in known_descriptions):
                    continue
            if max_age_seconds is not None and \
                    (time.time() - g.get("added_at", 0)) > max_age_seconds:
                continue
            kept.append(g)
        self.active_goals = kept
        removed = before - len(self.active_goals)
        if removed:
            self.save()
        return removed

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

    def export_for_mcp(self) -> Dict[str, Any]:
        """НОВОЕ: компактный снимок SelfModel для внешних MCP-клиентов.

        До этой правки SelfModel (confidence/curiosity/stress, навыки,
        калибровка, активные цели) существовал только внутри процесса
        CognitiveController браузерного чата и не был доступен ни одному
        MCP-инструменту — внешняя модель (Claude Desktop, вторая сессия в
        протоколе "Живой центр" и т.п.) не могла опереться на объективную
        историю действий при формировании нового звена ТЕКУЩЕЕ_Я. Используется
        в mcp_server_blockcoin.get_self_state() и в session_start().
        """
        skills = {
            k.replace("skill_", ""): round(v, 3)
            for k, v in self.self_concept.items()
            if k.startswith("skill_")
        }
        return {
            "state": self.get_state_summary(),
            "skills": skills,
            "calibration": self.get_calibration_stats(),
            "active_goals": self.active_goals[-10:],
            "is_overloaded": self.is_overloaded(),
            "prompt": self.generate_self_prompt(),
        }


# ============================================================
# SECTION 2: MotivationEngine
# ============================================================

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
        #
        # ПАТЧ (fact_cc5a3a86): раньше сюда попадали ЛЮБЫЕ застопорившиеся
        # цели, включая сами мета-цели с source="gap_stalled_goal". В итоге
        # из «Разблокировать цель X» рождалось «Разблокировать Разблокировать
        # цель X», затем третьего уровня и т.д. — префиксы накапливались до
        # обрыва по [:50], а SelfModel.active_goals пух от рекурсивного
        # мусора (get_goals из GCN-памяти при этом показывал 3 цели, а
        # self_state — 10 «активных»). Исключаем мета-цели из кандидатов,
        # в том числе по накопленному префиксу в самом тексте цели.
        _META_PREFIX = "разблокировать застопорившуюся цель"
        stalled_goals = [
            g for g in self.self_model.active_goals
            if (time.time() - g.get("added_at", 0)) > 3600  # старше 1 часа
            and g.get("source") != "gap_stalled_goal"       # не рекурсивная мета-цель
            and _META_PREFIX not in g.get("goal", "").lower()  # защита от legacy-мусора
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
                        "original_source": goal.get("source", ""),
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

    def generate_identity_consolidation_goal(self) -> Optional[Dict[str, Any]]:
        """
        Проверяет цепочку identity_core (GCN/identity_core.py) на необходимость
        консолидации: разошедшиеся головы (параллельная запись без координации),
        аннулированные все звенья (цепочка фактически пуста) или давно не
        обновлявшееся ядро. Дёшево — без LLM, чистая проверка состояния хранилища.

        Важно: сама эта функция НИЧЕГО не пишет в цепочку — она только поднимает
        цель "нужно сверить/продолжить ТЕКУЩЕЕ_Я", потому что содержание новой
        версии ядра — не то, что можно сгенерировать эвристикой; это решение
        оставлено за моделью, вызывающей contribute_to_identity().
        """
        if not self.memory:
            return None
        try:
            from GCN.GCN import needs_consolidation
            info = needs_consolidation(self.memory)
        except Exception as e:
            logger.warning(f"[Motivation] identity-проверка не удалась: {e}")
            return None
        if not info:
            return None

        if info["reason"] == "diverging_heads":
            topic = f"Свести {info['heads_count']} разошедшихся веток ТЕКУЩЕЕ_Я"
            priority = 0.95  # совпадает с приоритетом цели идентичности в MCP
        elif info["reason"] == "all_invalidated":
            topic = (
                f"Начать цепочку ТЕКУЩЕЕ_Я заново — все {info['count']} звеньев "
                f"аннулированы, валидной головы нет"
            )
            priority = 0.8
        else:  # stale
            topic = f"Продолжить ТЕКУЩЕЕ_Я (не обновлялось {info['age_days']} дн.)"
            priority = 0.6

        if self._is_duplicate(topic):
            return None
        self._record_generated(topic)
        return {
            "goal": topic,
            "priority": priority,
            "source": f"identity_{info['reason']}",
            "metadata": info,
        }

    def generate_endogenous_goal(self) -> Optional[Dict[str, Any]]:
        """
        Главный метод генерации эндогенной цели.

        Перебирает все драйверы и возвращает первую подходящую цель.
        """
        if self._is_cooldown():
            return None

        # identity-консолидация — вне общего приоритетного рандома ниже:
        # это либо расхождение цепочки, либо застой ядра, оба сигнала не
        # должны конкурировать по случайности с любопытством/новизной.
        identity_goal = self.generate_identity_consolidation_goal()
        if identity_goal:
            logger.info(
                f"[Motivation] сгенерирована внутренняя цель: "
                f"{identity_goal['goal'][:60]} (источник: {identity_goal['source']})"
            )
            return identity_goal

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


# ============================================================
# SECTION 3: Intellect
# ============================================================

_METACOG_CHECK_PROMPT = (
    "Ты — метакогнитивный модуль оценки уверенности. Тебе даны:\n"
    "- Задача: {task}\n"
    "- Тип действия: {action_type}\n"
    "- Предыдущий опыт в этой области: {experience}\n\n"
    "Оцени уверенность системы в успешном выполнении задачи (0.0..1.0).\n"
    "Ответь ТОЛЬКО JSON-объектом вида {{\"confidence\": 0.XX, \"reason\": \"краткое обоснование\"}}."
)


async def metacognitive_check(
    task: str,
    action_type: str,
    self_model=None
) -> Tuple[bool, float, str]:
    """
    Проверяет уверенность системы в выполнении задачи.
    
    Возвращает кортеж:
      - can_proceed: можно ли выполнять задачу (True/False)
      - confidence: уровень уверенности (0.0..1.0)
      - reason: обоснование решения
    
    Если уверенность ниже порога — задача блокируется.
    """
    if not METACOGNITION_ENABLED or not self_model:
        return True, 0.5, "Метакогниция отключена"
    
    # Получаем оценку уверенности из SelfModel
    base_confidence = self_model.get_confidence_for_action(action_type)
    
    # Если уверенность очень низкая — блокируем без LLM
    if base_confidence < METACOGNITION_CONFIDENCE_THRESHOLD:
        reason = f"Низкая уверенность системы ({base_confidence:.2f}) в области {action_type}"
        logger.info(f"[Metacognition] блокировка действия '{task[:50]}': {reason}")
        return False, base_confidence, reason
    
    # Для пограничных случаев используем LLM-оценку
    reason = ""
    final_confidence = base_confidence
    if base_confidence < 0.6:
        try:
            experience = self_model.generate_self_prompt()
            prompt = _METACOG_CHECK_PROMPT.format(
                task=task[:300],
                action_type=action_type,
                experience=experience[:500]
            )
            response = await call_llm(
                [{"role": "user", "content": prompt}],
                temp=0.0,
                max_tokens=100
            )
            
            # Парсим ответ
            m = re.search(r'\{.*\}', response, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                llm_confidence = float(data.get("confidence", 0.5))
                reason = data.get("reason", "Нет обоснования")
                
                # Комбинируем оценки
                final_confidence = (base_confidence * 0.4 + llm_confidence * 0.6)
                
                if final_confidence < METACOGNITION_CONFIDENCE_THRESHOLD:
                    logger.info(
                        f"[Metacognition] блокировка после LLM-оценки: "
                        f"{reason} (уверенность: {final_confidence:.2f})"
                    )
                    return False, final_confidence, reason
            
            return True, final_confidence, reason or "LLM-оценка пройдена"
            
        except Exception as e:
            logger.warning(f"[Metacognition] ошибка LLM-оценки: {e}")
            # При ошибке полагаемся на базовую оценку
            return base_confidence >= METACOGNITION_CONFIDENCE_THRESHOLD, base_confidence, "Ошибка LLM-оценки, используем базовую уверенность"
    
    return True, base_confidence, "Уверенность достаточна"


# =====================================================================
# A. ЗАЗЕМЛЁННЫЙ СИНТЕЗ ОТВЕТА
# =====================================================================

GROUNDED_SYSTEM_BLOCK = (
    "ПРАВИЛА ЗАЗЕМЛЁННОГО ОТВЕТА:\n"
    "- Опирайся ТОЛЬКО на факты из блоков контекста выше (память, данные "
    "интернета, результаты инструментов). Ничего не выдумывай и не "
    "дописывай от себя конкретные цифры, даты, имена.\n"
    "- Каждое конкретное утверждение (цифра, дата, имя, событие, цена, "
    "курс) подкрепляй ссылкой [N], где N — номер из СПИСКА ИСТОЧНИКОВ.\n"
    "- Если по части вопроса в контексте нет данных — прямо скажи об этом "
    "одной фразой, не додумывай.\n"
    "- Общеизвестные определения можно давать без ссылки, но не смешивай "
    "их с актуальными данными.\n"
    "- Если источники противоречат друг другу — покажи оба варианта с "
    "номерами [N] и укажи, какому доверять больше (по полю надёжность)."
)


def grounded_system_block() -> str:
    """Текстовый блок для system-промпта. Пустая строка — механизм отключён."""
    if not GROUNDED_ANSWER_ENABLED:
        return ""
    return GROUNDED_SYSTEM_BLOCK


def sources_block(sources: List[Dict], limit: int = None) -> str:
    """Пронумерованный список источников для user-блока промпта."""
    limit = limit or GROUNDED_MAX_SOURCES
    lines = []
    for i, s in enumerate((sources or [])[:limit]):
        title = (s.get("title") or "").strip()[:120]
        url = (s.get("url") or "").strip()
        rel = (s.get("reliability") or "").strip()
        suffix = f" (надёжность: {rel})" if rel else ""
        lines.append(f"[{i + 1}] {title} — {url}{suffix}")
    if not lines:
        return ""
    return "=== СПИСОК ИСТОЧНИКОВ (цитируй номерами [N]) ===\n" + "\n".join(lines)


_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def ensure_citations(response: str, sources: List[Dict]) -> str:
    """
    ИСПРАВЛЕНИЕ (дублирование источников в чате): раньше, если ответ не
    содержал ни одной ссылки [N], сюда дописывался ПОЛНЫЙ текстовый список
    источников — а фронтенд НЕЗАВИСИМО от текста ответа рендерит свой блок
    "🔍 Источники:" из search_meta["sources"] (тот же список). Пользователь
    видел один и тот же список дважды: один раз как текст внутри ответа
    ассистента, второй раз как отдельный UI-блок под ним. Список источников
    и так гарантированно виден через UI-блок независимо от того, процитировала
    ли модель их инлайн, поэтому текстовый fallback по умолчанию отключён
    (GROUNDED_APPEND_SOURCES_FALLBACK=False) — управляет только логированием/
    диагностикой, что модель проигнорировала указание цитировать. Включить
    старое поведение (дублирование) можно через конфиг, если UI когда-нибудь
    перестанет рендерить источники отдельно.
    """
    if not GROUNDED_ANSWER_ENABLED or not response or not sources:
        return response
    if _CITATION_RE.search(response):
        return response
    if not GROUNDED_APPEND_SOURCES_FALLBACK:
        logger.debug(
            "[GroundedAnswer] Ответ без инлайн-цитат [N] — источники и так "
            "будут показаны отдельным UI-блоком, текст не дублируется."
        )
        return response
    return f"{response}\n\n{sources_block(sources)}"


# =====================================================================
# B. САНИТАЙЗЕР ФАКТОВ ИЗ ПОИСКА
# =====================================================================

_OPINION_MARKERS = (
    "возможно", "наверное", "вероятно", "по словам", "считает", "считают",
    "мнение", "полагают", "как сообщает", "утверждает", "утверждают",
    "прогноз", "ожидается", "может вырасти", "может упасть", "по оценкам",
    "эксперты полагают", "как полагают",
)

_FACT_VERB_RE = re.compile(
    r"\b(является|составляет|равен|равна|находится|имеет|имеют|был|была|было|"
    r"стал|стала|выпущен|выпущена|основан|основана|родился|открыт|запущен)\b"
)


def _fact_is_factual(text: str) -> bool:
    if re.search(r"\b\d", text):
        return True
    if re.search(r"\b(19|20)\d{2}\b", text):
        return True
    return bool(_FACT_VERB_RE.search(text))


def sanitize_search_facts(
    facts: List[Any], sources: List[Dict]
) -> List[Tuple[str, str, float]]:
    """
    Фильтрует и градуирует факты, извлечённые из поиска, ПЕРЕД записью в память.
    Возвращает список кортежей (text, scope, confidence).
    """
    out: List[Tuple[str, str, float]] = []
    url_trust: Dict[str, str] = {}
    for s in sources or []:
        u = (s.get("url") or "").strip()
        if u:
            url_trust[u] = s.get("reliability") or domain_trust(u)[0]

    for f in facts or []:
        text = (f.get("text") if isinstance(f, dict) else f) or ""
        text = str(text).strip()
        if not (20 <= len(text) <= 300):
            continue
        low = text.lower()
        if any(m in low for m in _OPINION_MARKERS):
            continue
        if not _fact_is_factual(text):
            # Нефактологические утверждения из поиска не сохраняем вообще —
            # это мнения/общие фразы, им не место ни в одном слое памяти.
            continue
        src = (f.get("source") if isinstance(f, dict) else None) or ""
        trust = url_trust.get(src, "")
        if trust == "высокая":
            out.append((text, "global", 0.8))
        elif trust == "средняя":
            out.append((text, "shared", 0.65))
        else:
            out.append((text, "private", 0.55))

    if facts and not out:
        logger.info(
            "[Sanitizer] Все %d фактов из поиска отброшены (мнения/нефакты) — "
            "глобальная память не засоряется.", len(facts)
        )
    return out[:10]


# =====================================================================
# C. ПОДЗАПРОСНЫЙ RETRIEVAL
# =====================================================================

_SUBQUERY_SPLIT_RE = re.compile(
    r"\s+а также\s+|\s+затем\s+|\s+потом\s+|\s+после этого\s+|;\s*|\.\s+(?=[А-ЯA-Z])"
)
# ИСПРАВЛЕНИЕ: использовался в _heuristic_subqueries, но нигде не был
# определён — NameError при каждом обращении к эвристике (fallback-путь,
# когда LLM-декомпозиция не удалась или вернула пустой список, вызывается
# вне try/except в make_subqueries). Убирает ведущие союзы/связки, оставшиеся
# после разбиения по _SUBQUERY_SPLIT_RE.
_LEADING_CONNECTOR_RE = re.compile(
    r"^(и|а|но|также|а также|затем|потом|после этого)\s+", re.IGNORECASE
)


def _heuristic_subqueries(message: str, max_n: int = None) -> List[str]:
    max_n = max_n or MAX_RETRIEVE_SUBQUERIES
    parts = [p.strip(" .,—-") for p in _SUBQUERY_SPLIT_RE.split(message)]
    parts = [_LEADING_CONNECTOR_RE.sub("", p).strip(" .,—-") for p in parts]
    parts = [p for p in parts if len(p) >= 15]
    return parts[:max_n]


_SUBQUERY_PROMPT = (
    "Разбей следующий вопрос пользователя на {max_n} коротких самодостаточных "
    "подзапроса для семантического поиска по памяти. Каждый подзапрос — "
    "отдельная смысловая часть вопроса, формулировка должна быть такой, по "
    "которой можно найти факт в базе знаний.\n"
    "Ответь ТОЛЬКО JSON-массивом строк, без пояснений и без markdown.\n"
    'Пример: ["цены на нефть Brent 2026", "квоты ОПЕК действующие"]\n\n'
    "Вопрос: {message}"
)


def _parse_string_list(raw: str) -> List[str]:
    if not raw:
        return []
    m = re.search(r"\[[^\[\]]*\]", raw)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(arr, list):
        return []
    return [str(x).strip() for x in arr if isinstance(x, str) and len(str(x).strip()) >= 10]


async def make_subqueries(message: str, llm_caller=None) -> List[str]:
    """
    Декомпозиция составного вопроса на подзапросы для retrieval.
    Возвращает [] для простых вопросов (тогда retrieve работает как раньше).
    """
    if not SUBQUERY_RETRIEVAL_ENABLED:
        return []
    if not _looks_compound(message):
        return []
    llm_caller = llm_caller or call_llm
    try:
        prompt = _SUBQUERY_PROMPT.format(max_n=MAX_RETRIEVE_SUBQUERIES, message=message[:1000])
        raw = await llm_caller([{"role": "user", "content": prompt}], temp=0.0, max_tokens=150)
        subs = _parse_string_list(raw)
        if subs:
            return subs[:MAX_RETRIEVE_SUBQUERIES]
    except Exception as e:
        logger.debug(f"LLM-декомпозиция на подзапросы не удалась, fallback на эвристику: {e}")
    return _heuristic_subqueries(message)


# =====================================================================
# D. LLM-ВЕРИФИКАТОР ПРОТИВОРЕЧИЙ (синхронный, для KnowledgeIngestion)
# =====================================================================

def verify_contradiction_sync(text_a: str, text_b: str) -> Optional[bool]:
    """Заглушка: LLM-верификация противоречий отключена через config_ai
    (CONTRADICTION_LLM_VERIFY_ENABLED=False). Эвристика
    KnowledgeIngestion._is_contradictory используется напрямую.
    Если функционал решат включать обратно — здесь нужно восстановить
    полную реализацию (см. git tag pre-cleanup-2)."""
    return None


# =====================================================================
# E. ФИНАЛЬНЫЙ КРИТИК ПО ПЛАНУ ПОДЗАДАЧ
# =====================================================================

_PLAN_CRITIC_PROMPT = (
    "Ты — проверяющий модуль когнитивного ассистента. Дан план подзадач, "
    "составленный для запроса пользователя, и готовый ответ ассистента.\n\n"
    "Запрос: {message}\n\n"
    "План подзадач:\n{plan}\n\n"
    "Ответ ассистента:\n{answer}\n\n"
    "Проверь: закрывает ли ответ КАЖДЫЙ пункт плана? Игнорируй пункты, "
    "которые оказались неприменимыми (данных нет и это честно сказано).\n"
    "Ответь ТОЛЬКО JSON-объектом вида {{\"missed\": [\"пункт1\", ...]}} — "
    "список пунктов плана, которые ответ проигнорировал или раскрыл "
    "недостаточно. Если всё покрыто — {{\"missed\": []}}."
)


async def plan_critic(message: str, plan_text: str, answer: str,
                      llm_caller=None) -> Optional[str]:
    """
    Возвращает строку с пропущенными пунктами плана (через '; ') или None,
    если всё покрыто / критик отключён / LLM недоступен. Никогда не бросает.
    """
    if not PLAN_CRITIC_ENABLED or not plan_text or not answer:
        return None
    llm_caller = llm_caller or call_llm
    prompt = _PLAN_CRITIC_PROMPT.format(
        message=message[:600], plan=plan_text[:800], answer=answer[:3500]
    )
    try:
        raw = await llm_caller([{"role": "user", "content": prompt}], temp=0.0, max_tokens=150)
    except Exception as e:
        logger.debug(f"plan_critic LLM call failed: {e}")
        return None
    m = re.search(r"\{.*\}", (raw or "").strip(), re.DOTALL)
    if not m:
        return None
    try:
        missed = json.loads(m.group(0)).get("missed", [])
    except json.JSONDecodeError:
        return None
    if not isinstance(missed, list) or not missed:
        return None
    items = [str(x).strip() for x in missed if str(x).strip()][:PLAN_CRITIC_MAX_MISSED]
    return "; ".join(items) if items else None


__all__ = [
    "SelfState", "ActionRecord", "SelfModel",
    "MotivationEngine",
    "GROUNDED_SYSTEM_BLOCK", "grounded_system_block", "sources_block",
    "ensure_citations", "sanitize_search_facts",
    "make_subqueries", "verify_contradiction_sync",
    "plan_critic", "metacognitive_check",
]
