import json
from pathlib import Path

path = Path("ai_memory_v3/global/cognitive_memory/gcn_state.json")

with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)  # читает и превращает \u в символы

with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)  # сохраняет с русскими буквами

print("Готово!")