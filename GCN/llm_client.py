import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional
from GCN.config_ai import LM_STUDIO_URL, LM_STUDIO_API_KEY, LM_STUDIO_TIMEOUT

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