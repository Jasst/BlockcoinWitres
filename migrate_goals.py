# migrate_goals.py — запустить ОДИН РАЗ до применения правок кода, из корня проекта.
#
# Переносит активные цели из self_model.json в GCN, чтобы первый же вызов
# sync_from_gcn() не «схлопнул» active_goals (GCN становится источником истины).
#
# ВАЖНО (проверено по коду): get_memory_service() и MemoryService живут в
# модуле GCN.memory_service (в GCN/__init__.py они не реэкспортируются),
# поэтому импорт — `from GCN.memory_service import get_memory_service`.
#
# Запуск:  python migrate_goals.py [USER_ID]
# USER_ID можно передать аргументом; иначе используется константа ниже.

import asyncio
import json
import sys
from pathlib import Path

from GCN.config_ai import MEMORY_BASE_DIR
from GCN.memory_service import get_memory_service

# Целевой пользователь (можно переопределить аргументом командной строки).
USER_ID = "2a0fe15382ab71ba6a0602049052dd58f0110d73e1d6947d207a816b3365ab8a"


def _self_model_path(user_id: str) -> Path:
    # Layout как в SelfModel: <MEMORY_BASE_DIR>/<user_id>/self_model.json
    return Path(MEMORY_BASE_DIR) / user_id / "self_model.json"


async def main():
    user_id = sys.argv[1] if len(sys.argv) > 1 else USER_ID

    sm_path = _self_model_path(user_id)
    if not sm_path.exists():
        print(f"Файл не найден: {sm_path} — миграция не требуется")
        return

    data = json.loads(sm_path.read_text(encoding="utf-8"))
    goals = data.get("active_goals", [])
    if not goals:
        print("Нет целей для миграции")
        return

    svc = await get_memory_service(user_id)
    svc.refresh()
    existing_gcn = {
        g["description"].strip().lower()
        for g in await svc.get_goals()
        if g.get("description")
    }

    migrated = 0
    for g in goals:
        text = (g.get("goal") or "").strip()
        if not text:
            continue
        if text.lower() in existing_gcn:
            continue
        if "разблокировать застопорившуюся цель" in text.lower():
            continue  # legacy-мусор не мигрируем
        try:
            await svc.add_goal(text, priority=float(g.get("priority", 0.5)))
            migrated += 1
            print(f"  + {text[:80]}")
        except Exception as e:
            print(f"  ! {text[:60]}: {e}")

    print(f"\nМигрировано в GCN: {migrated}")


if __name__ == "__main__":
    asyncio.run(main())
