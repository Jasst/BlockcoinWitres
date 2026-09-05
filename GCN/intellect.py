"""
intellect.py — пакет улучшений "интеллекта" когнитивного ассистента BlockcoinWitres.

Пять независимых механизмов (все отключаемые через config_ai.py):

  A. Заземлённый синтез ответа (grounded answering).
     Модель обязана опираться ТОЛЬКО на переданный контекст (память/поиск/
     инструменты) и цитировать источники номерами [N] из явного списка.
     ensure_citations() гарантирует, что источники видны даже если модель
     проигнорировала инструкцию.

  B. Санитайзер фактов из поиска.
     Раньше ВСЁ извлечённое из поискового снаппета улетало в глобальную
     память (remember(scope="global"), confidence=0.9) — включая мнения,
     прогнозы и факты с ненадёжных доменов, что отравляло глобальный граф.
     Теперь факт проверяется: фактологичность (число/дата/глагол), маркеры
     мнений, доверие домена источника. Скоуп и confidence подбираются
     автоматически: высокое доверие → global/0.8, среднее → shared/0.65,
     остальное → private/0.55, немфактологическое — не сохраняется.

  C. Подзапросный retrieval.
     Составной вопрос разбивается на 2-3 самодостаточных подзапроса (LLM +
     heuristic fallback), каждый ищется отдельно через обычный pipeline
     retrieve(), результаты сливаются с бустом мультихитов (факт, найденный
     по нескольким подзапросам, важнее), затем один общий LLM-реранк.

  D. LLM-верификатор противоречий.
     Эвристика _is_contradictory (наличие/отсутствие отрицания) даёт много
     ложных срабатываний и пропускает реальные (разные числа по одному
     предмету). Теперь при similarity > 0.7 и эвристическом срабатывании
     противоречие подтверждается коротким LLM-проходом. Синхронный
     (urllib в ThreadPoolExecutor), т.к. KnowledgeIngestion.submit_candidate
     — синхронный. При сбое/таймауте (None) — прежнее поведение.

  E. Финальный критик по плану подзадач.
     ToolRouter теперь отдаёт plan наружу (run()["plan"] и _last_plan).
     После генерации ответа LLM сверяет его с планом; пропущенные пункты
     добираются одним дополнительным проходом генерации (не ReAct-циклом,
     просто "дополни ответ").
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

try:
    from GCN.config_ai import (
        LM_STUDIO_URL,
        LM_STUDIO_API_KEY,
        GROUNDED_ANSWER_ENABLED,
        PLAN_CRITIC_ENABLED,
        CONTRADICTION_LLM_VERIFY_ENABLED,
        SUBQUERY_RETRIEVAL_ENABLED,
        MAX_RETRIEVE_SUBQUERIES,
        PLAN_CRITIC_MAX_MISSED,
        GROUNDED_MAX_SOURCES,
    )
except ImportError:
    LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
    LM_STUDIO_API_KEY = "lm-studio"
    GROUNDED_ANSWER_ENABLED = True
    PLAN_CRITIC_ENABLED = True
    CONTRADICTION_LLM_VERIFY_ENABLED = True
    SUBQUERY_RETRIEVAL_ENABLED = True
    MAX_RETRIEVE_SUBQUERIES = 3
    PLAN_CRITIC_MAX_MISSED = 3
    GROUNDED_MAX_SOURCES = 8

from GCN.llm_client import call_llm
from GCN.web_search import domain_trust

logger = logging.getLogger(__name__)


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
    Если ответ не содержит ни одной ссылки [N], а источники были — дописывает
    в конец ответа пронумерованный список источников. Иначе возвращает как есть.
    """
    if not GROUNDED_ANSWER_ENABLED or not response or not sources:
        return response
    if _CITATION_RE.search(response):
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
_COMPOUND_MARKERS_C = (" и ", " а также ", " затем ", " потом ", ";", " или ")
_LEADING_CONNECTOR_RE = re.compile(
    r"^(?:затем|потом|а также|также|и|а|после этого|далее|еще|ещё)\s+",
    re.IGNORECASE,
)


def _looks_compound(message: str) -> bool:
    if len(message) >= 120:
        return True
    if message.count("?") >= 2:
        return True
    lowered = f" {message.lower()} "
    return any(m in lowered for m in _COMPOUND_MARKERS_C)


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

_CONTRADICTION_PROMPT = (
    "Даны два утверждения из памяти AI-ассистента, помеченные как возможно "
    "противоречащие друг другу.\n"
    "Утверждение A: {a}\n"
    "Утверждение B: {b}\n\n"
    "Ответь ТОЛЬКО JSON-объектом вида {{\"verdict\": true}} или {{\"verdict\": false}}, "
    "без пояснений.\n"
    "verdict=true — утверждения действительно противоречат (одно отрицает "
    "другое, или дают несовместимые значения одного и того же параметра: "
    "разные числа/даты/статус для одного объекта).\n"
    "verdict=false — не противоречат (разные объекты, разное время, разный "
    "контекст, или совместимые утверждения)."
)

_llm_sync_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="intellect-llm")


def _post_json_sync(payload: Dict, timeout: float) -> Dict:
    import urllib.request

    req = urllib.request.Request(
        LM_STUDIO_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LM_STUDIO_API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def verify_contradiction_sync(text_a: str, text_b: str) -> Optional[bool]:
    """
    True — реально противоречат; False — не противоречат; None — проверить
    не удалось (вызывающий код откатывается на эвристику). НИКОГДА не бросает
    исключений наружу.
    """
    if not CONTRADICTION_LLM_VERIFY_ENABLED:
        return None
    prompt = _CONTRADICTION_PROMPT.format(a=text_a[:500], b=text_b[:500])
    payload = {
        "model": "local-model",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 30,
    }
    try:
        fut = _llm_sync_pool.submit(_post_json_sync, payload, 12.0)
        data = fut.result(timeout=16.0)
        content = ((data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or "").strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return None
        verdict = json.loads(m.group(0)).get("verdict")
        if isinstance(verdict, bool):
            return verdict
        if isinstance(verdict, str):
            return verdict.strip().lower() in ("true", "yes", "да", "1")
    except Exception as e:
        logger.debug(f"LLM-верификация противоречия недоступна: {e}")
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