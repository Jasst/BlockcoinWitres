import asyncio
import os
import aiohttp
import logging
from typing import Optional

from GCN.config_ai import (
    EASYDIFFUSION_ENABLED,
    EASYDIFFUSION_URL,
    EASYDIFFUSION_ENDPOINT,
    EASYDIFFUSION_TIMEOUT,
    EASYDIFFUSION_DEFAULT_STEPS,
    EASYDIFFUSION_DEFAULT_WIDTH,
    EASYDIFFUSION_DEFAULT_HEIGHT,
    EASYDIFFUSION_MODEL,
    EASYDIFFUSION_DEFAULT_LORA,
    EASYDIFFUSION_DEFAULT_LORA_WEIGHT,
    EASYDIFFUSION_DEFAULT_LORA_USE,
)
from GCN.llm_client import call_llm

logger = logging.getLogger(__name__)

# Кеш текущей модели, чтобы не слать /options при каждом вызове.
_current_model = None

# Переиспользуемая сессия к локальному EasyDiffusion-хосту (как в llm_client.py) —
# закрывается через close_session() при остановке процесса.
_session: Optional[aiohttp.ClientSession] = None
_session_lock = asyncio.Lock()

# EasyDiffusion/A1111-совместимые бэкенды требуют width/height, кратные 8,
# иначе — артефакты или ошибка API. Верхние границы — защита от того, что
# mcp_server_blockcoin.generate_image объявляет width/height/steps как
# открытые Field-параметры без ge/le.
_MAX_IMAGE_DIMENSION = 1536
_MAX_IMAGE_STEPS = 60

# LoRA задаётся A1111-стилем прямо в prompt: "<lora:имя_файла:вес>".
_LORA_TAG = None
if EASYDIFFUSION_DEFAULT_LORA_USE and EASYDIFFUSION_DEFAULT_LORA:
    _lora_name = os.path.splitext(os.path.basename(EASYDIFFUSION_DEFAULT_LORA))[0]
    _LORA_TAG = f"<lora:{_lora_name}:{EASYDIFFUSION_DEFAULT_LORA_WEIGHT}>"


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        async with _session_lock:
            if _session is None or _session.closed:
                _session = aiohttp.ClientSession()
    return _session


async def close_session() -> None:
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None


def _round_to_multiple(value: int, multiple: int = 8) -> int:
    if value <= 0:
        return multiple
    return max(multiple, round(value / multiple) * multiple)


async def set_easy_diffusion_model(model_name: str) -> bool:
    """Устанавливает модель через /v1/sdapi/v1/options. Возвращает True при успехе."""
    global _current_model
    if _current_model == model_name:
        return True

    url = f"{EASYDIFFUSION_URL}/v1/sdapi/v1/options"
    try:
        session = await _get_session()
        async with session.post(
            url, json={"sd_model_checkpoint": model_name},
            timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            if resp.status == 200:
                _current_model = model_name
                logger.info(f"Model set to: {model_name}")
                return True
            # Некоторые бэкенды принимают только поле "model"
            async with session.post(
                url, json={"model": model_name},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp2:
                if resp2.status == 200:
                    _current_model = model_name
                    logger.info(f"Model set via 'model' field: {model_name}")
                    return True
                logger.error(f"Failed to set model: {resp2.status}")
                return False
    except Exception as e:
        logger.error(f"Error setting model: {e}")
        return False


async def generate_image(
    prompt: str,
    steps: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    seed: Optional[int] = None,
    sampler_name: Optional[str] = None,
) -> Optional[str]:
    """Генерирует изображение через EasyDiffusion API: сначала /options, затем txt2img."""
    if not EASYDIFFUSION_ENABLED:
        logger.warning("EasyDiffusion disabled")
        return None

    if not await set_easy_diffusion_model(EASYDIFFUSION_MODEL):
        logger.warning("Could not set model, continuing with default")

    if len(prompt) > 500:
        prompt = prompt[:500] + "..."
    if _LORA_TAG and _LORA_TAG not in prompt:
        prompt = f"{prompt} {_LORA_TAG}"

    safe_width = min(_MAX_IMAGE_DIMENSION, _round_to_multiple(
        width if width is not None else EASYDIFFUSION_DEFAULT_WIDTH))
    safe_height = min(_MAX_IMAGE_DIMENSION, _round_to_multiple(
        height if height is not None else EASYDIFFUSION_DEFAULT_HEIGHT))
    safe_steps = max(1, min(_MAX_IMAGE_STEPS, steps if steps is not None else EASYDIFFUSION_DEFAULT_STEPS))

    payload = {
        "prompt": prompt,
        "steps": safe_steps,
        "width": safe_width,
        "height": safe_height,
    }
    if cfg_scale is not None:
        payload["cfg_scale"] = cfg_scale
    if seed is not None and seed >= 0:
        payload["seed"] = seed
    if sampler_name:
        payload["sampler_name"] = sampler_name

    logger.info(f"Sending generation payload: {payload}")

    url = f"{EASYDIFFUSION_URL}{EASYDIFFUSION_ENDPOINT}"
    try:
        session = await _get_session()
        async with session.post(
            url, json=payload,
            timeout=aiohttp.ClientTimeout(total=EASYDIFFUSION_TIMEOUT)
        ) as resp:
            if resp.status != 200:
                logger.error(f"EasyDiffusion error: {resp.status}")
                return None
            data = await resp.json()
            images = data.get("images")
            if not images:
                logger.error("No images in response")
                return None
            return images[0]
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return None


async def enhance_prompt(prompt: str) -> str:
    """Улучшает промпт через LLM."""
    system_msg = (
        "Ты — эксперт по улучшению промптов для Stable Diffusion. "
        "Верни ТОЛЬКО улучшенный промпт, без пояснений, без кавычек, без маркдауна. "
        "Улучши описание, добавь детали стиля, освещения, качества. "
        "Ответ должен быть одним предложением, не более 20 слов."
    )
    try:
        enhanced = await call_llm(
            [{"role": "system", "content": system_msg},
             {"role": "user", "content": prompt}],
            temp=0.5,
            max_tokens=150
        )
        enhanced = enhanced.strip().strip('"').strip("'")
        if '\n' in enhanced:
            enhanced = enhanced.split('\n')[0].strip()
        if len(enhanced) > 200:
            enhanced = enhanced[:200]
        return enhanced if enhanced else prompt
    except Exception:
        return prompt