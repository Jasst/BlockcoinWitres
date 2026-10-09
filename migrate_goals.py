"""Одноразовая чистка self_model.json от мусора автогенерированных целей."""
import json
import shutil
from datetime import datetime
from pathlib import Path

USER_ID = "2a0fe15382ab71ba6a0602049052dd58f0110d73e1d6947d207a816b3365ab8a"
SM_PATH = Path("ai_memory_v3") / USER_ID / "self_model.json"

# Паттерны мусора. Всё, что совпадает хотя бы с одним — удаляется.
GARBAGE_SOURCES = {
    "gap_stalled_goal",
    "quality_confidence_boost",
    "quality_improvement",
    "curiosity_uncertainty",
    "curiosity_skill_improvement",
    "novelty_exploration",
}
GARBAGE_PREFIXES = (
    "разблокировать застопорившуюся цель",
    "восстановить уверенность",
)


def is_garbage(g: dict) -> bool:
    src = str(g.get("source", "")).lower()
    text = str(g.get("goal", "")).lower()
    if src in GARBAGE_SOURCES:
        return True
    if src.startswith("identity_"):
        return True  # identity-цели тоже пересоздаются автоматически
    if any(text.startswith(p) for p in GARBAGE_PREFIXES):
        return True
    if text.count("разблокировать") >= 1:
        return True
    return False


def main():
    if not SM_PATH.exists():
        print(f"Не найден: {SM_PATH}")
        return

    # 1. Бэкап с timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = SM_PATH.with_suffix(f".json.bak_{ts}")
    shutil.copy2(SM_PATH, backup_path)
    print(f"Бэкап: {backup_path}")

    # 2. Читаем
    data = json.loads(SM_PATH.read_text(encoding="utf-8"))
    goals = data.get("active_goals", [])
    print(f"Было записей: {len(goals)}")

    # 3. Фильтруем
    kept, removed = [], []
    for g in goals:
        if is_garbage(g):
            removed.append(g)
        else:
            kept.append(g)

    print(f"\nУдалено ({len(removed)}):")
    for g in removed:
        print(f"  [{g.get('source')}] {g.get('goal', '')[:90]}")
    print(f"\nОставлено ({len(kept)}):")
    for g in kept:
        print(f"  [{g.get('source')}] {g.get('goal', '')[:90]}")

    # 4. Записываем
    data["active_goals"] = kept
    SM_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nСохранено: {SM_PATH}")


if __name__ == "__main__":
    main()