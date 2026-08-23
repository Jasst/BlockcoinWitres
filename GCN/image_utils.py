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
)
from GCN.llm_client import call_llm

logger = logging.getLogger(__name__)

async def enhance_prompt(prompt: str) -> str:
    """
    Улучшает промпт для генерации изображений с помощью LLM.
    Возвращает только улучшенный промпт (без пояснений), обрезанный до 200 символов.
    """
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

async def generate_image(
    prompt: str,
    steps: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> Optional[str]:
    """
    Генерирует изображение через EasyDiffusion API.
    Возвращает base64-строку изображения или None в случае ошибки.
    """
    if not EASYDIFFUSION_ENABLED:
        logger.warning("EasyDiffusion disabled")
        return None

    if len(prompt) > 500:
        prompt = prompt[:500] + "..."

    payload = {
        "prompt": prompt,
        "steps": steps if steps is not None else EASYDIFFUSION_DEFAULT_STEPS,
        "width": width if width is not None else EASYDIFFUSION_DEFAULT_WIDTH,
        "height": height if height is not None else EASYDIFFUSION_DEFAULT_HEIGHT,
        "model": "realisticVisionV60B1_v51HyperVAE",  # или другая модель
    }
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