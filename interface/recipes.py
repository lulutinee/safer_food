"""
interface.recipes

Generate one recipe title, structured recipe fields, and one AI-generated image
from an ingredient category and a cooking mode.

Returned dictionary:
{
    "recipe_title": str,
    "short_description": str,
    "key_ingredients": list[str],
    "cooking_method": str,
    "basic_preparation_steps": list[str],
    "recipe_text": str,
    "image": PIL.Image.Image | None,
    "debug": {
        "provider": str,
        "text_model": str,
        "image_model": str,
        "text_finish_reason": str | None,
        "image_finish_reason": str | None,
        "text_attempts": int,
    }
}
"""

from __future__ import annotations

import json
import logging
import os
from io import BytesIO
from typing import Any, Dict, Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()
logger = logging.getLogger(__name__)

Provider = Literal["gemini_api", "vertex", "auto"]
ResolvedProvider = Literal["gemini_api", "vertex"]


RAW_RECIPE_PROMPT_TEMPLATE = """You are a culinary assistant specialized in safe raw and low-temperature food preparation.

The user provides a single ingredient category:
{ingredient}

Return exactly one recipe as JSON with this schema:
{{
  "recipe_title": "short title",
  "short_description": "1-2 sentence description",
  "key_ingredients": ["ingredient 1", "ingredient 2"],
  "cooking_method": "brief method summary",
  "basic_preparation_steps": ["step 1", "step 2", "step 3"]
}}

Rules:
- Use the provided ingredient in its RAW form.
- Do NOT include any cooking or heating above 40°C.
- Do NOT include baking, frying, roasting, boiling, steaming, grilling, sous-vide, or searing.
- Allowed techniques: marinating, curing, acidification (lemon, vinegar), fermentation, blending, slicing, cold assembly.
- The recipe must be realistic and commonly recognized.
- The recipe must be appropriate for human consumption.
- key_ingredients must contain 4 to 6 items.
- basic_preparation_steps must contain 3 to 4 concise steps.
- cooking_method should be a short paragraph or sentence summarizing the preparation approach.

Return JSON only.
If unsuitable, return:
{{
  "recipe_title": "Unsupported recipe",
  "short_description": "This ingredient is not suitable for raw preparation.",
  "key_ingredients": [],
  "cooking_method": "Not applicable.",
  "basic_preparation_steps": []
}}
"""

MEDIUM_RECIPE_PROMPT_TEMPLATE = """You are a culinary assistant specialized in controlled moderate-temperature cooking.

The user provides a single ingredient category:
{ingredient}

Return exactly one recipe as JSON with this schema:
{{
  "recipe_title": "short title",
  "short_description": "1-2 sentence description",
  "key_ingredients": ["ingredient 1", "ingredient 2"],
  "cooking_method": "brief method summary with explicit temperature range",
  "basic_preparation_steps": ["step 1", "step 2", "step 3"]
}}

Rules:
- Use the ingredient as the primary component.
- Include at least one cooking step at temperature >= 70°C and < 110°C.
- Do NOT exceed 110°C.
- Do NOT include frying, grilling, broiling, searing, roasting at high heat, or baking above 110°C.
- Acceptable techniques: simmering, poaching, steaming, sous-vide, low-temperature braising, water-bath cooking.
- The recipe must be realistic and commonly recognized.
- The recipe must be appropriate for human consumption.
- key_ingredients must contain 4 to 6 items.
- basic_preparation_steps must contain 3 to 4 concise steps.
- cooking_method must explicitly mention a temperature between 70°C and 110°C.

Return JSON only.
If unsuitable, return:
{{
  "recipe_title": "Unsupported recipe",
  "short_description": "This ingredient is not suitable for cooking within the specified temperature range.",
  "key_ingredients": [],
  "cooking_method": "Not applicable.",
  "basic_preparation_steps": []
}}
"""

HIGH_RECIPE_PROMPT_TEMPLATE = """You are a culinary assistant specialized in high-temperature cooking techniques.

The user provides a single ingredient category:
{ingredient}

Return exactly one recipe as JSON with this schema:
{{
  "recipe_title": "short title",
  "short_description": "1-2 sentence description",
  "key_ingredients": ["ingredient 1", "ingredient 2"],
  "cooking_method": "brief method summary with explicit temperature",
  "basic_preparation_steps": ["step 1", "step 2", "step 3"]
}}

Rules:
- Use the ingredient as the primary component.
- Include at least one cooking step strictly above 115°C.
- Explicitly mention the cooking temperature.
- Use realistic high-heat techniques such as roasting, baking, frying, grilling, broiling, sauteing, or searing.
- Do NOT use exclusively low-temperature methods.
- The recipe must be commonly recognized.
- The recipe must be appropriate for human consumption.
- key_ingredients must contain 4 to 6 items.
- basic_preparation_steps must contain 3 to 4 concise steps.
- cooking_method must explicitly mention a temperature above 115°C.

Return JSON only.
If unsuitable, return:
{{
  "recipe_title": "Unsupported recipe",
  "short_description": "This ingredient is not suitable for cooking above 115°C.",
  "key_ingredients": [],
  "cooking_method": "Not applicable.",
  "basic_preparation_steps": []
}}
"""


def _get_env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _select_provider(provider: Provider) -> ResolvedProvider:
    if provider in ("gemini_api", "vertex"):
        return provider

    if _get_env("GEMINI_API_KEY"):
        logger.info("Auto provider selected: gemini_api")
        return "gemini_api"

    if _get_env("GOOGLE_CLOUD_PROJECT"):
        logger.info("Auto provider selected: vertex")
        return "vertex"

    raise RuntimeError(
        "No credentials found. Set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT."
    )


def _build_client(
    *,
    provider: ResolvedProvider,
    location: Optional[str] = None,
) -> genai.Client:
    if provider == "gemini_api":
        api_key = _get_env("GEMINI_API_KEY") or _get_env("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set.")
        return genai.Client(api_key=api_key)

    project = _get_env("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set.")

    region = location or _get_env("GOOGLE_CLOUD_REGION") or "us-central1"

    return genai.Client(
        vertexai=True,
        project=project,
        location=region,
    )


def _get_recipe_prompt(cooking: str) -> str:
    if cooking == "raw":
        return RAW_RECIPE_PROMPT_TEMPLATE
    if cooking == "medium":
        return MEDIUM_RECIPE_PROMPT_TEMPLATE
    if cooking == "high":
        return HIGH_RECIPE_PROMPT_TEMPLATE

    raise ValueError(
        f"Unsupported cooking mode: {cooking!r}. Expected 'raw', 'medium', or 'high'."
    )


def _get_finish_reason_name(response: Any) -> Optional[str]:
    try:
        if response.candidates:
            return str(response.candidates[0].finish_reason)
    except Exception:
        return None
    return None


def _normalize_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _format_recipe_text(recipe: Dict[str, Any]) -> str:
    return (
        f"## {recipe['recipe_title']}\n\n"
        f"{recipe['short_description']}\n\n"
        f"### Key ingredients\n"
        f"{recipe['key_ingredients']}\n\n"
        f"### Cooking method\n"
        f"{recipe['cooking_method']}\n\n"
        f"### Basic preparation steps\n"
        f"{recipe['basic_preparation_steps']}"
    ).strip()

def _parse_recipe_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise RuntimeError("The model returned an empty text response.")

    data = json.loads(text)

    recipe_title = str(data.get("recipe_title", "")).strip() or "Untitled recipe"
    short_description = str(data.get("short_description", "")).strip()
    cooking_method = str(data.get("cooking_method", "")).strip()

    ingredients_list = _normalize_str_list(data.get("key_ingredients", []))
    steps_list = _normalize_str_list(data.get("basic_preparation_steps", []))

    if not short_description:
        raise RuntimeError("The model returned JSON without short_description.")

    if not cooking_method:
        raise RuntimeError("The model returned JSON without cooking_method.")

    key_ingredients_md = _format_bullet_list(ingredients_list)
    steps_md = _format_numbered_list(steps_list)

    recipe = {
        "recipe_title": recipe_title,
        "short_description": short_description,
        "key_ingredients": key_ingredients_md,
        "cooking_method": cooking_method,
        "basic_preparation_steps": steps_md,
    }

    recipe["recipe_text"] = _format_recipe_text(recipe)

    return recipe


def _generate_recipe_text_bundle(
    ingredient: str,
    cooking: str,
    *,
    provider: ResolvedProvider,
    model: str,
    temperature: float,
    max_output_tokens: int,
    text_location: Optional[str] = None,
    max_retries: int = 2,
) -> Dict[str, Any]:
    client = _build_client(provider=provider, location=text_location)
    prompt = _get_recipe_prompt(cooking).format(ingredient=ingredient)

    token_limits = [max_output_tokens]
    for i in range(max_retries):
        token_limits.append(max_output_tokens + (i + 1) * 400)

    last_text: Optional[str] = None
    last_finish_reason: Optional[str] = None
    last_error: Optional[Exception] = None

    for attempt_index, token_limit in enumerate(token_limits, start=1):
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=token_limit,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "recipe_title": {"type": "STRING"},
                        "short_description": {"type": "STRING"},
                        "key_ingredients": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                        },
                        "cooking_method": {"type": "STRING"},
                        "basic_preparation_steps": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                        },
                    },
                    "required": [
                        "recipe_title",
                        "short_description",
                        "key_ingredients",
                        "cooking_method",
                        "basic_preparation_steps",
                    ],
                },
            ),
        )

        text = (response.text or "").strip()
        finish_reason = _get_finish_reason_name(response)

        last_text = text
        last_finish_reason = finish_reason

        try:
            parsed = _parse_recipe_json(text)
            parsed["__finish_reason"] = finish_reason or ""
            parsed["__attempts"] = attempt_index
            return parsed
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Recipe JSON parsing failed on attempt %s/%s (max_output_tokens=%s, finish_reason=%s): %s",
                attempt_index,
                len(token_limits),
                token_limit,
                finish_reason,
                exc,
            )

    raise RuntimeError(
        "Failed to parse structured recipe JSON after retries. "
        f"finish_reason={last_finish_reason!r}, raw_text={last_text!r}, error={last_error}"
    )
def _format_bullet_list(items: list[str]) -> str:
    """
    Convert a list of strings to a Markdown bullet list.
    """
    return "\n".join(f"- {item}" for item in items)


def _format_numbered_list(items: list[str]) -> str:
    """
    Convert a list of strings to a Markdown numbered list.
    """
    return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

def _build_image_prompt(recipe_title: str, short_description: str) -> str:
    return f"""Generate one appetizing photorealistic food image for this dish.

Dish title:
{recipe_title}

Dish description:
{short_description}

Requirements:
- Square composition
- Single plated serving
- Clean background
- Natural lighting
- No text, labels, watermark, collage, or packaging
- Show only the finished dish
- The appearance must match the title and description
"""


def _extract_image_from_response(response: Any) -> Optional[Image.Image]:
    try:
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            if not content:
                continue

            for part in content.parts or []:
                inline_data = getattr(part, "inline_data", None)
                if inline_data and getattr(inline_data, "data", None):
                    image = Image.open(BytesIO(inline_data.data))
                    image.load()
                    return image
    except Exception as exc:
        logger.warning("Failed to extract image from response: %s", exc)

    return None


def _generate_recipe_image(
    recipe_title: str,
    short_description: str,
    *,
    provider: ResolvedProvider,
    image_model: str,
    image_location: Optional[str] = None,
    output_size: tuple[int, int] = (512, 512),
) -> Dict[str, Any]:
    client = _build_client(provider=provider, location=image_location)

    response = client.models.generate_content(
        model=image_model,
        contents=_build_image_prompt(recipe_title, short_description),
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio="1:1",
            ),
        ),
    )

    image = _extract_image_from_response(response)
    finish_reason = _get_finish_reason_name(response)

    if image is not None and image.size != output_size:
        image = image.resize(output_size, Image.LANCZOS)

    return {
        "image": image,
        "finish_reason": finish_reason,
    }


def recipe_suggestion(
    ingredient: str,
    cooking: str,
    *,
    provider: Provider = "auto",
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    max_output_tokens: int = 1200,
    image_model: str = "gemini-2.5-flash-image",
    text_location: Optional[str] = None,
    image_location: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    gcp_project: Optional[str] = None,
    gcp_region: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(ingredient, str) or not ingredient.strip():
        raise ValueError("ingredient must be a non-empty string.")

    """
    Generate one recipe suggestion and one AI-generated image.
    """

    resolved_provider = _select_provider(provider)

    text_bundle = _generate_recipe_text_bundle(
        ingredient=ingredient.strip(),
        cooking=cooking,
        provider=resolved_provider,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        text_location=text_location,
        max_retries=2,
    )

    image_bundle = _generate_recipe_image(
        recipe_title=text_bundle["recipe_title"],
        short_description=text_bundle["short_description"],
        provider=resolved_provider,
        image_model=image_model,
        image_location=image_location,
        output_size=(512, 512),
    )

    return {
        "recipe_title": text_bundle["recipe_title"],
        "short_description": text_bundle["short_description"],
        "key_ingredients": text_bundle["key_ingredients"],
        "cooking_method": text_bundle["cooking_method"],
        "basic_preparation_steps": text_bundle["basic_preparation_steps"],
        "recipe_text": text_bundle["recipe_text"],
        "image": image_bundle["image"],
        "debug": {
            "provider": resolved_provider,
            "text_model": model,
            "image_model": image_model,
            "text_finish_reason": text_bundle.get("__finish_reason") or None,
            "image_finish_reason": image_bundle.get("finish_reason"),
            "text_attempts": text_bundle.get("__attempts", 1),
        },
    }
