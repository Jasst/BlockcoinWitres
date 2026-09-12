"""
code_analyzer.py — инструменты для самоанализа кода ИИ-системой.

Позволяет ИИ:
1. Читать файлы своего кода (с проверкой безопасности)
2. Анализировать структуру проекта
3. Искать по коду
4. Предлагать исправления на основе ошибок
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from GCN.config_ai import (
    ENABLE_CODE_SELF_REFLECTION,
    CODE_ANALYSIS_MAX_FILES,
    CODE_CONTEXT_MAX_LINES,
    AUTO_FIX_SUGGESTIONS,
    CODE_ACCESS_ROOT,
    ALLOWED_CODE_EXTENSIONS,
)

logger = logging.getLogger(__name__)


@dataclass
class CodeSearchResult:
    """Результат поиска в коде."""
    file_path: str
    line_number: int
    line_content: str
    context: str  # несколько строк вокруг


class CodeAnalyzer:
    """Анализатор кода для самообучения ИИ."""
    
    def __init__(self, root_path: Optional[Path] = None):
        self.root = root_path or CODE_ACCESS_ROOT
        self._file_cache: Dict[str, str] = {}  # кэш содержимого файлов
        self._structure_cache: Optional[Dict] = None  # кэш структуры проекта
        
    def _is_safe_path(self, file_path: str) -> bool:
        """Проверяет, что файл находится в разрешённой директории и имеет безопасное расширение."""
        try:
            # Нормализуем путь
            abs_path = Path(file_path).resolve()
            
            # Проверяем, что путь внутри root
            if not str(abs_path).startswith(str(self.root.resolve())):
                logger.warning(f"Попытка доступа к файлу вне root: {file_path}")
                return False
            
            # Проверяем расширение
            if abs_path.suffix.lower() not in ALLOWED_CODE_EXTENSIONS:
                logger.warning(f"Недопустимое расширение файла: {abs_path.suffix}")
                return False
            
            # Проверяем, что файл существует
            if not abs_path.is_file():
                logger.warning(f"Файл не существует: {file_path}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Ошибка проверки пути {file_path}: {e}")
            return False
    
    def _get_context_lines(self, lines: List[str], line_idx: int, context_size: int = 5) -> str:
        """Возвращает контекст вокруг указанной строки."""
        start = max(0, line_idx - context_size)
        end = min(len(lines), line_idx + context_size + 1)
        
        context_lines = []
        for i in range(start, end):
            prefix = ">>> " if i == line_idx else "    "
            context_lines.append(f"{prefix}{i+1:4d}: {lines[i]}")
        
        return "\n".join(context_lines)
    
    async def read_file(self, file_path: str, max_lines: int = 500) -> str:
        """
        Читает файл с проверкой безопасности.
        
        Args:
            file_path: Относительный путь от root (например, "GCN/config_ai.py")
            max_lines: Максимум строк для возврата
        
        Returns:
            Содержимое файла или сообщение об ошибке
        """
        if not ENABLE_CODE_SELF_REFLECTION:
            return "Самоанализ кода отключён в конфигурации."
        
        # Собираем полный путь
        full_path = self.root / file_path
        
        if not self._is_safe_path(str(full_path)):
            return f"Ошибка: доступ к файлу '{file_path}' запрещён по соображениям безопасности."
        
        try:
            # Проверяем кэш
            cache_key = str(full_path)
            if cache_key in self._file_cache:
                content = self._file_cache[cache_key]
            else:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self._file_cache[cache_key] = content
            
            # Ограничиваем размер
            lines = content.splitlines()
            if len(lines) > max_lines:
                truncated = "\n".join(lines[:max_lines])
                return f"{truncated}\n\n... (файл обрезан, показано {max_lines} из {len(lines)} строк)"
            
            return content
            
        except FileNotFoundError:
            return f"Ошибка: файл '{file_path}' не найден."
        except UnicodeDecodeError:
            return f"Ошибка: не удалось прочитать файл '{file_path}' (некорректная кодировка)."
        except Exception as e:
            logger.error(f"Ошибка чтения файла {file_path}: {e}")
            return f"Ошибка при чтении файла: {str(e)}"
    
    async def search_in_code(self, pattern: str, max_results: int = 20) -> str:
        """
        Ищет паттерн (строку или regex) в коде проекта.
        
        Args:
            pattern: Строка или regex для поиска
            max_results: Максимум результатов
        
        Returns:
            Форматированный список результатов
        """
        if not ENABLE_CODE_SELF_REFLECTION:
            return "Самоанализ кода отключён в конфигурации."
        
        try:
            # Компилируем как regex, если не получится — ищем как подстроку
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                use_regex = True
            except re.error:
                pattern_lower = pattern.lower()
                use_regex = False
            
            results: List[CodeSearchResult] = []
            files_checked = 0
            
            # Рекурсивно обходим директорию
            for root, dirs, files in os.walk(self.root):
                # Пропускаем скрытые директории и __pycache__
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                
                for filename in files:
                    if Path(filename).suffix.lower() not in ALLOWED_CODE_EXTENSIONS:
                        continue
                    
                    file_path = Path(root) / filename
                    if not self._is_safe_path(str(file_path)):
                        continue
                    
                    files_checked += 1
                    if files_checked > CODE_ANALYSIS_MAX_FILES * 10:  # жёсткий лимит
                        break
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        for line_idx, line in enumerate(lines):
                            found = False
                            if use_regex:
                                found = bool(regex.search(line))
                            else:
                                found = pattern_lower in line.lower()
                            
                            if found:
                                rel_path = str(file_path.relative_to(self.root))
                                context = self._get_context_lines(lines, line_idx, CODE_CONTEXT_MAX_LINES // 10)
                                results.append(CodeSearchResult(
                                    file_path=rel_path,
                                    line_number=line_idx + 1,
                                    line_content=line.strip(),
                                    context=context
                                ))
                                
                                if len(results) >= max_results:
                                    break
                        
                        if len(results) >= max_results:
                            break
                    except Exception as e:
                        logger.debug(f"Пропуск файла {file_path}: {e}")
                
                if len(results) >= max_results:
                    break
            
            if not results:
                return f"Ничего не найдено по запросу '{pattern}'."
            
            # Форматируем результат
            output_lines = [f"Найдено {len(results)} совпадений по запросу '{pattern}':\n"]
            for i, res in enumerate(results, 1):
                output_lines.append(f"\n{i}. Файл: {res.file_path}, строка {res.line_number}")
                output_lines.append(f"   {res.line_content[:200]}")
                output_lines.append("\n   Контекст:")
                for ctx_line in res.context.split('\n'):
                    output_lines.append(f"   {ctx_line}")
            
            return "\n".join(output_lines)
            
        except Exception as e:
            logger.error(f"Ошибка поиска в коде: {e}", exc_info=True)
            return f"Ошибка при поиске: {str(e)}"
    
    async def get_project_structure(self, max_depth: int = 3) -> str:
        """
        Возвращает структуру проекта в виде дерева.
        
        Args:
            max_depth: Максимальная глубина обхода
        
        Returns:
            Текстовое представление структуры
        """
        if not ENABLE_CODE_SELF_REFLECTION:
            return "Самоанализ кода отключён в конфигурации."
        
        def build_tree(path: Path, prefix: str = "", depth: int = 0) -> List[str]:
            if depth > max_depth:
                return []
            
            result = []
            try:
                items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
                items = [item for item in items if not item.name.startswith('.')]
                
                for i, item in enumerate(items):
                    is_last = (i == len(items) - 1)
                    connector = "└── " if is_last else "├── "
                    
                    if item.is_file():
                        if item.suffix.lower() in ALLOWED_CODE_EXTENSIONS:
                            result.append(f"{prefix}{connector}{item.name}")
                    else:
                        result.append(f"{prefix}{connector}{item.name}/")
                        extension = "    " if is_last else "│   "
                        result.extend(build_tree(item, prefix + extension, depth + 1))
            except PermissionError:
                pass
            
            return result
        
        tree_lines = build_tree(self.root)
        return f"Структура проекта (до глубины {max_depth}):\n\n" + "\n".join(tree_lines)
    
    async def analyze_error_location(self, error_message: str, traceback_str: str) -> str:
        """
        Анализирует ошибку и пытается найти проблемное место в коде.
        
        Args:
            error_message: Текст ошибки
            traceback_str: Трассировка стека
        
        Returns:
            Анализ ошибки с предложениями по исправлению
        """
        if not ENABLE_CODE_SELF_REFLECTION:
            return "Самоанализ кода отключён в конфигурации."
        
        # Извлекаем имена файлов и строки из traceback
        file_pattern = r'File "([^"]+)", line (\d+)'
        matches = re.findall(file_pattern, traceback_str)
        
        if not matches:
            return f"Не удалось извлечь информацию о местоположении ошибки из traceback.\n\nОшибка: {error_message}"
        
        analysis_parts = [f"Анализ ошибки: {error_message}\n"]
        analysis_parts.append("=" * 60)
        
        for file_path, line_num in matches[:3]:  # Берём первые 3 локации
            # Пытаемся найти относительный путь
            try:
                abs_path = Path(file_path)
                if str(abs_path).startswith(str(self.root)):
                    rel_path = str(abs_path.relative_to(self.root))
                    
                    # Читаем файл с контекстом
                    content = await self.read_file(rel_path, max_lines=int(line_num) + 10)
                    
                    analysis_parts.append(f"\n📁 Файл: {rel_path}, строка {line_num}")
                    analysis_parts.append("-" * 40)
                    
                    # Выделяем проблемную строку
                    lines = content.split('\n')
                    if int(line_num) <= len(lines):
                        target_line = lines[int(line_num) - 1]
                        analysis_parts.append(f"Код:\n{target_line}")
                    
                    if AUTO_FIX_SUGGESTIONS:
                        analysis_parts.append("\n💡 Возможные причины:")
                        
                        # Простые эвристики для типичных ошибок
                        if "IndexError" in error_message:
                            analysis_parts.append("  - Выход за пределы списка/массива")
                            analysis_parts.append("  - Проверь границы индекса перед доступом")
                        elif "KeyError" in error_message:
                            analysis_parts.append("  - Доступ к несуществующему ключу в словаре")
                            analysis_parts.append("  - Используй .get() или проверку in перед доступом")
                        elif "AttributeError" in error_message:
                            analysis_parts.append("  - Доступ к несуществующему атрибуту объекта")
                            analysis_parts.append("  - Проверь тип объекта и доступные методы")
                        elif "TypeError" in error_message:
                            analysis_parts.append("  - Несоответствие типов аргументов")
                            analysis_parts.append("  - Проверь сигнатуры функций и типы данных")
                        elif "NameError" in error_message:
                            analysis_parts.append("  - Использование необъявленной переменной")
                            analysis_parts.append("  - Проверь импорты и область видимости")
                        elif "ImportError" in error_message or "ModuleNotFoundError" in error_message:
                            analysis_parts.append("  - Модуль не найден")
                            analysis_parts.append("  - Проверь установку зависимостей (pip install)")
                        else:
                            analysis_parts.append("  - Требуется ручной анализ кода")
                
            except Exception as e:
                logger.debug(f"Не удалось проанализировать файл {file_path}: {e}")
        
        if AUTO_FIX_SUGGESTIONS:
            analysis_parts.append("\n" + "=" * 60)
            analysis_parts.append("\n🔧 Общие рекомендации:")
            analysis_parts.append("  1. Проверь типы данных всех переменных")
            analysis_parts.append("  2. Добавь обработку исключений в критичных местах")
            analysis_parts.append("  3. Убедись, что все зависимости установлены")
            analysis_parts.append("  4. Проверь соответствие версий библиотек")
        
        return "\n".join(analysis_parts)
    
    def clear_cache(self):
        """Очищает кэш файлов."""
        self._file_cache.clear()
        self._structure_cache = None


# Глобальный экземпляр (ленивая инициализация)
_analyzer: Optional[CodeAnalyzer] = None


def get_analyzer() -> CodeAnalyzer:
    """Возвращает глобальный экземпляр анализатора."""
    global _analyzer
    if _analyzer is None:
        _analyzer = CodeAnalyzer()
    return _analyzer
