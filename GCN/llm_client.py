import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional
from GCN.config_ai import LM_STUDIO_URL, LM_STUDIO_API_KEY, LM_STUDIO_TIMEOUT, LM_STUDIO_STREAM_TIMEOUT
import json
logger = logging.getLogger(__name__)

async def call_llm(
    messages: List[Dict[str, str]],
    temp: float = 0.7,
    max_tokens: int = 2048,
    retries: int = 3
) -> str:
    """
    Универсальная функция вызова локальной LLM (LM Studio).
    Используется и в ai_assistant, и в MCP-сервере.
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LM_STUDIO_API_KEY}"}
    payload = {
        "model": "local-model",
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens
    }
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(LM_STUDIO_URL, json=payload, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=LM_STUDIO_TIMEOUT)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    else:
                        error_text = await resp.text()
                        logger.error(f"LLM error {resp.status}: {error_text[:200]}")
                        if resp.status >= 500 and attempt < retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return ""
    return ""

async def call_llm_stream(
    messages: List[Dict[str, str]],
    temp: float = 0.7,
    max_tokens: int = 2048
):
    """Потоковый вызов LLM (LM Studio) с теми же параметрами, что и call_llm."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LM_STUDIO_API_KEY}"}
    payload = {
        "model": "local-model",
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": True,
    }
    timeout = aiohttp.ClientTimeout(total=LM_STUDIO_STREAM_TIMEOUT)  # убедитесь, что константа определена в config_ai
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(LM_STUDIO_URL, json=payload, headers=headers, timeout=timeout) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Stream error {resp.status}: {error_text[:200]}")
                    yield "[Ошибка LLM]"
                    return
                async for line in resp.content:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get('choices', [{}])[0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
    except asyncio.CancelledError:
        logger.debug("Stream cancelled")
        raise
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"[Ошибка: {e}]"