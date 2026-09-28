import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional
from GCN.config_ai import LM_STUDIO_URL, LM_STUDIO_API_KEY, LM_STUDIO_TIMEOUT, LM_STUDIO_STREAM_TIMEOUT, DEFAULT_MAX_TOKENS
import json

logger = logging.getLogger(__name__)

# Переиспользуемая сессия с пулом соединений (keep-alive) к локальному LM
# Studio вместо нового TCP-хендшейка на каждый вызов — это самый горячий
# путь в проекте (каждое сообщение чата, каждый tool-call реранкинг,
# verify_response). Закрывается через close_session() при остановке процесса.
_session: Optional[aiohttp.ClientSession] = None
_session_lock = asyncio.Lock()

# Статусы, при которых имеет смысл повторить попытку (временная перегрузка/
# рейт-лимит бэкенда), а не только 5xx.
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        async with _session_lock:
            if _session is None or _session.closed:
                _session = aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(limit=20, keepalive_timeout=30)
                )
    return _session


async def close_session() -> None:
    """Вызывать при штатном завершении процесса, чтобы не оставлять открытый пул соединений."""
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None


async def call_llm_raw(
    messages: List[Dict[str, str]],
    temp: float = 0.7,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    tools: Optional[List[Dict]] = None,
    retries: int = 3
) -> Dict:
    """
    Возвращает сырой объект message от LLM целиком (не только content), чтобы
    вызывающий код (GCN.tool_router) мог прочитать tool_calls при нативном
    function calling — тем же механизмом, что использует внешний MCP-клиент.
    При исчерпании retry возвращает {"_error": "..."} вместо {}.
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LM_STUDIO_API_KEY}"}
    payload = {
        "model": "local-model",
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    last_error = None
    for attempt in range(retries):
        try:
            session = await _get_session()
            async with session.post(LM_STUDIO_URL, json=payload, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=LM_STUDIO_TIMEOUT)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("choices", [{}])[0].get("message", {}) or {}

                error_text = await resp.text()
                logger.error(f"LLM error {resp.status}: {error_text[:200]}")
                # tools может быть не поддержан бэкендом (400) — пробуем без tools один раз
                if tools and resp.status == 400 and attempt == 0:
                    payload.pop("tools", None)
                    payload.pop("tool_choice", None)
                    continue
                if resp.status in _RETRYABLE_STATUSES and attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                last_error = f"LLM returned status {resp.status}: {error_text[:100]}"
                break
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            last_error = str(e)
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            break
    # Все retry исчерпаны — возвращаем ошибку явно
    return {"_error": last_error or "LLM call failed after all retries"}


async def call_llm(
    messages: List[Dict[str, str]],
    temp: float = 0.7,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = 3,
    include_reasoning: bool = False,
) -> str:
    """Универсальный вызов локальной LLM (LM Studio) — только текст ответа.

    include_reasoning=True — если модель прислала reasoning_content (нативный
    thinking Qwen3/DeepSeek-R1 и т.п.), обернуть его в <thought>...</thought>
    и склеить с content. Используется ТОЛЬКО в главном ответе чата; для
    служебных вызовов (планирование, верификация, JSON-парсинг) — False,
    чтобы reasoning не путал парсеры.
    """
    msg = await call_llm_raw(messages, temp=temp, max_tokens=max_tokens,
                             tools=None, retries=retries)
    content = msg.get("content", "") or ""
    if not include_reasoning:
        return content

    reasoning = (
        msg.get("reasoning_content")
        or msg.get("reasoning")
        or ""
    ).strip()

    if not reasoning:
        return content
    if not content:
        # Модель думала, но не уложилась в лимит токенов.
        return f"<thought>{reasoning}</thought>"
    return f"<thought>{reasoning}</thought>\n\n{content}"


async def call_llm_stream(
    messages: List[Dict[str, str]],
    temp: float = 0.7,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stop: Optional[List[str]] = None,
    include_reasoning: bool = True,
):
    """Потоковый вызов LLM (LM Studio).

    Yield-ит dict-события, чтобы разделить reasoning и content в потоке:
      {"kind": "reasoning", "text": "..."} — reasoning-токен (нативный thinking)
      {"kind": "content",   "text": "..."} — токен финального ответа
      {"kind": "truncated", "reason": "no_done"|"length"} — поток закончился
                                              без [DONE] или по лимиту токенов.

    include_reasoning=False — reasoning-события не эмитятся (модель может
    тратить токены на thinking, но UI его не увидит).

    Обратная несовместимость: раньше функция yield-ила строки, теперь dict.
    Единственный потребитель — _stream_response_worker в ai_assistant.py.
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LM_STUDIO_API_KEY}"}
    payload = {
        "model": "local-model",
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if stop:
        payload["stop"] = stop
    timeout = aiohttp.ClientTimeout(total=LM_STUDIO_STREAM_TIMEOUT)
    stream_did_complete = False
    finish_reason = None
    try:
        session = await _get_session()
        async with session.post(LM_STUDIO_URL, json=payload,
                                headers=headers, timeout=timeout) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(f"Stream error {resp.status}: {error_text[:200]}")
                yield {"kind": "content", "text": "[Ошибка LLM]"}
                return
            async for line in resp.content:
                if not line:
                    break
                line = line.decode('utf-8').strip()
                if not line or not line.startswith('data: '):
                    continue
                data = line[6:]
                if data == '[DONE]':
                    stream_did_complete = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = chunk.get('choices', [{}])[0]
                delta = choice.get('delta', {})

                if 'finish_reason' in choice:
                    finish_reason = choice['finish_reason']

                # Reasoning-токен — эмитим сразу, без накопления.
                if include_reasoning:
                    r_chunk = (
                        delta.get('reasoning_content')
                        or delta.get('reasoning')
                        or ''
                    )
                    if r_chunk:
                        yield {"kind": "reasoning", "text": r_chunk}

                # Content-токен — тоже сразу.
                content = delta.get('content', '')
                if content:
                    yield {"kind": "content", "text": content}

            if not stream_did_complete:
                logger.warning(
                    f"LLM stream ended without [DONE] marker "
                    f"(finish_reason={finish_reason})"
                )
                yield {"kind": "truncated", "reason": "no_done"}
                return
            if finish_reason == "length":
                logger.warning("LLM stream truncated by token limit (finish_reason=length)")
                yield {"kind": "truncated", "reason": "length"}
                return
            logger.debug(f"LLM stream completed with finish_reason: {finish_reason}")
    except asyncio.CancelledError:
        logger.debug("Stream cancelled")
        raise
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield {"kind": "content", "text": f"[Ошибка: {e}]"}