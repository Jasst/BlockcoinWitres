import asyncio
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
    EASYDIFFUSION_DEFAULT_LORA_USE,      # <-- добавить
)
from GCN.llm_client import call_llm

logger = logging.getLogger(__name__)

# ---- Кеш текущей модели (чтобы не отправлять /options при каждом вызове) ----
_current_model = None

# УЛУЧШЕНИЕ: как и в llm_client.py — переиспользуемая сессия вместо новой
# на каждый вызов к тому же локальному EasyDiffusion-хосту.
_session: Optional[aiohttp.ClientSession] = None
_session_lock = asyncio.Lock()


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
    """
    УЛУЧШЕНИЕ: Stable Diffusion (и большинство совместимых с A1111 бэкендов,
    включая EasyDiffusion) требует width/height, кратные 8 (обычно 64 для
    некоторых моделей/VAE) — иначе тихо получаем артефакты или прямую ошибку
    API. Раньше произвольные width/height от вызывающего кода (в т.ч. из
    mcp_server_blockcoin.generate_image, где это открытые параметры Field)
    шли в payload без всякой проверки.
    """
    if value <= 0:
        return multiple
    return max(multiple, round(value / multiple) * multiple)

async def set_easy_diffusion_model(model_name: str) -> bool:
    """
    Устанавливает модель через /v1/sdapi/v1/options.
    Возвращает True, если успешно.
    """
    global _current_model
    if _current_model == model_name:
        return True  # уже установлена

    url = f"{EASYDIFFUSION_URL}/v1/sdapi/v1/options"
    payload = {"sd_model_checkpoint": model_name}  # стандартное поле A1111
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    _current_model = model_name
                    logger.info(f"Model set to: {model_name}")
                    return True
                else:
                    # Если sd_model_checkpoint не работает, пробуем model
                    payload2 = {"model": model_name}
                    async with session.post(url, json=payload2, timeout=10) as resp2:
                        if resp2.status == 200:
                            _current_model = model_name
                            logger.info(f"Model set via 'model' field: {model_name}")
                            return True
                        else:
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
    """
    Генерирует изображение через EasyDiffusion API.
    Сначала устанавливает модель через /options, затем отправляет txt2img.
    """
    if not EASYDIFFUSION_ENABLED:
        logger.warning("EasyDiffusion disabled")
        return None

    # 1. Установка модели
    if not await set_easy_diffusion_model(EASYDIFFUSION_MODEL):
        logger.warning("Could not set model, continuing with default")

    if len(prompt) > 500:
        prompt = prompt[:500] + "..."

    # 2. Формируем payload для генерации (БЕЗ override_settings)
    payload = {
        "prompt": prompt,
        "steps": steps if steps is not None else EASYDIFFUSION_DEFAULT_STEPS,
        "width": width if width is not None else EASYDIFFUSION_DEFAULT_WIDTH,
        "height": height if height is not None else EASYDIFFUSION_DEFAULT_HEIGHT,
        # Не добавляем override_settings – модель уже установлена через options
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
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=EASYDIFFUSION_TIMEOUT)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    images = data.get("images")
                    if images and len(images) > 0:
                        return images[0]
                    else:
                        logger.error("No images in response")
                        return None
                else:
                    logger.error(f"EasyDiffusion error: {resp.status}")
                    return None
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return None

async def enhance_prompt(prompt: str) -> str:
    """Улучшает промпт через LLM (без изменений)."""
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