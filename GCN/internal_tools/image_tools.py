"""
image_tools.py — регистрация внутреннего инструмента генерации изображений.

Вынесено из CognitiveController.__init__ (routes/ai_assistant.py).

Инструмент:
  - generate_image — генерация изображения через EasyDiffusion

Использует controller.enhance_prompt() и controller.generate_image() —
оба публичные методы CognitiveController.
"""
import base64
import logging
import os
from datetime import datetime
from typing import Dict

from GCN.config_ai import GENERATED_IMAGES_DIR
from GCN.tool_router import ToolRegistry

logger = logging.getLogger(__name__)


def register(registry: ToolRegistry, controller) -> None:
    """
    Регистрирует инструмент generate_image.

    Args:
        registry:   экземпляр ToolRegistry
        controller: CognitiveController — источник enhance_prompt и generate_image
    """

    async def _internal_generate_image(args: Dict) -> Dict:
        prompt = args.get("prompt", "")
        enhance = args.get("enhance_prompt", True)
        steps = args.get("steps", 20)
        width = args.get("width", 512)
        height = args.get("height", 512)
        cfg_scale = args.get("cfg_scale", 7.0)
        seed = args.get("seed", -1)
        sampler = args.get("sampler", "dpmpp_2m")
        if enhance:
            prompt = await controller.enhance_prompt(prompt)
        image_b64 = await controller.generate_image(
            prompt, steps=steps, width=width, height=height,
            cfg_scale=cfg_scale, seed=seed, sampler_name=sampler
        )
        if image_b64:
            output_dir = GENERATED_IMAGES_DIR
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = output_dir / f"image_{timestamp}.png"
            with open(filename, "wb") as f:
                f.write(base64.b64decode(image_b64))
            BASE_URL = os.getenv("SERVER_BASE_URL", "http://localhost:8000")
            image_url = f"{BASE_URL}/generated_images/{filename.name}"
            return {"status": "ok", "image_url": image_url, "prompt": prompt}
        return {"status": "error", "message": "Не удалось сгенерировать изображение."}

    registry.register(
        name="generate_image",
        description="Генерирует изображение по текстовому описанию. Аргументы: prompt (str), enhance_prompt (bool, опционально), steps, width, height, cfg_scale, seed, sampler",
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "enhance_prompt": {"type": "boolean", "default": True},
                "steps": {"type": "integer", "default": 20},
                "width": {"type": "integer", "default": 512},
                "height": {"type": "integer", "default": 512},
                "cfg_scale": {"type": "number", "default": 7.0},
                "seed": {"type": "integer", "default": -1},
                "sampler": {"type": "string", "default": "dpmpp_2m"}
            },
            "required": ["prompt"]
        },
        handler=_internal_generate_image,
        server="internal",
        timeout_seconds=300
    )
    logger.debug("[internal_tools.image] зарегистрирован: generate_image")
