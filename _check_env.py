"""_check_env.py — вспомогательный модуль для проверок импорта в среде разработки.

В Docker-образе приложения установлены sentence_transformers и faiss, а также
заданы переменные окружения (SECRET_KEY и др.). В этой песочнице их нет, поэтому
для прогонов вида `python -c "import routes.ai_assistant; print('OK')"` мы
подключаем лёгкие заглушки/дефолты. Файл НЕ влияет на рантайм приложения:
он импортируется только вручную из команд проверки (sitecustomize не трогаем).

Использование:
    python -c "import _check_env; import routes.ai_assistant; print('OK')"
"""
import os
import sys
import types

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-import-checks-only")


class _Any:
    """Универсальная заглушка: вызывается, атрибутится, инстанцируется."""

    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return _Any()

    def __call__(self, *a, **k):
        return _Any()


def _install(name, attrs=None):
    if name in sys.modules:
        return
    try:
        import importlib.util
        if importlib.util.find_spec(name) is not None:
            return  # настоящая библиотека установлена — не подменяем
    except Exception:
        pass
    mod = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    mod.__getattr__ = lambda attr: _Any  # любые прочие имена
    sys.modules[name] = mod


_install("sentence_transformers", {"SentenceTransformer": _Any})
_install("faiss", {
    "IndexIDMap": _Any,
    "IndexFlatIP": _Any,
    "IndexIVFFlat": _Any,
    "normalize_L2": lambda x: x,
})
