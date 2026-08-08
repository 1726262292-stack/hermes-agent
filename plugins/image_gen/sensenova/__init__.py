"""SenseNova (商汤) image generation backend.

Exposes SenseNova's ``sensenova-u1-fast`` model (infographic generation) as
an :class:`ImageGenProvider` implementation. Uses the OpenAI-compatible
``/v1/images/generations`` endpoint at ``token.sensenova.cn``.

Note: SenseNova's image API is a *separate* generation endpoint, not Chat
Completions. It does NOT accept image input (no image-to-image editing) —
text-to-image only. Returned image URLs are temporary (~1 hour expiry).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agent.secret_scope import get_secret
from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)

API_MODEL = "sensenova-u1-fast"
DEFAULT_MODEL = "sensenova-u1-fast"
BASE_URL = "https://token.sensenova.cn/v1"
ENV_KEY = "SENSENOVA_API_KEY"

# SenseNova accepts specific sizes; map aspect ratios to them.
# Valid sizes (from API error): 1664x2496, 2496x1664, 1760x2368, 2368x1760,
# 1824x2272, 2272x1824, 2048x2048, 2752x1536, 1536x2752, 3072x1376,
# 1344x3136, 2560x720, 3072x864
_SIZES = {
    "landscape": "2752x1536",
    "square": "2048x2048",
    "portrait": "1536x2752",
}


def _load_sensenova_config() -> Dict[str, Any]:
    """Read ``image_gen`` section from config.yaml (returns {} on failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


class SenseNovaImageGenProvider(ImageGenProvider):
    """SenseNova ``images.generate`` backend — sensenova-u1-fast."""

    @property
    def name(self) -> str:
        return "sensenova"

    @property
    def display_name(self) -> str:
        return "SenseNova (商汤)"

    def is_available(self) -> bool:
        if not get_secret(ENV_KEY):
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "sensenova-u1-fast",
                "display_name": "SenseNova U1 Fast (信息图生成)",
                "speed": "~5-15s",
                "strengths": "信息图/图示生成,支持中文排版",
            },
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "title": "SenseNova (商汤) API Key",
            "description": (
                "在 https://platform.sensenova.cn 获取 API Key。"
                "图像生成接口为 https://token.sensenova.cn/v1/images/generations"
            ),
            "fields": [
                {
                    "key": ENV_KEY,
                    "label": "SenseNova API Key",
                    "secret": True,
                }
            ],
        }

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def capabilities(self) -> Dict[str, Any]:
        return {"text_to_image": True, "image_to_image": False}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="sensenova",
                aspect_ratio=aspect,
            )

        # SenseNova image API does not accept image input.
        sources = []
        if isinstance(image_url, str) and image_url.strip():
            sources.append(image_url.strip())
        if sources:
            return error_response(
                error=(
                    "SenseNova image API does not support image-to-image "
                    "editing (text-to-image only)."
                ),
                error_type="unsupported_modality",
                provider="sensenova",
                aspect_ratio=aspect,
            )

        api_key = get_secret(ENV_KEY)
        if not api_key:
            return error_response(
                error=(
                    "SENSENOVA_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → SenseNova to configure."
                ),
                error_type="auth_required",
                provider="sensenova",
                aspect_ratio=aspect,
            )

        # Resolve model: image_gen.sensenova.model → image_gen.model → default
        cfg = _load_sensenova_config()
        sensenova_cfg = cfg.get("sensenova") if isinstance(cfg.get("sensenova"), dict) else {}
        model = (
            str(sensenova_cfg.get("model") or cfg.get("model") or DEFAULT_MODEL)
        ).strip() or DEFAULT_MODEL

        size = _SIZES.get(aspect, _SIZES["square"])

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="sensenova",
                aspect_ratio=aspect,
            )

        try:
            client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)
            result = client.images.generate(
                model=model,
                prompt=prompt,
                n=1,
                size=size,
            )
        except Exception as exc:
            return error_response(
                error=f"SenseNova image generation failed: {exc}",
                error_type="provider_exception",
                provider="sensenova",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            url = result.data[0].url if result.data else None
        except Exception:
            url = None

        if url:
            try:
                image_ref = str(save_url_image(url))
            except Exception:
                # Temporary URL (1h expiry) — keep it usable even if download fails
                image_ref = url
        else:
            return error_response(
                error="SenseNova response contained no image URL",
                error_type="empty_response",
                provider="sensenova",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=image_ref,
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="sensenova",
            modality="text",
            extra={"size": size},
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire SenseNovaImageGenProvider into the registry."""
    ctx.register_image_gen_provider(SenseNovaImageGenProvider())
