"""
Docstring for interface.recipes
Gives recipe suggestions depending on the predictions made on the user's food.
Returns one recipe and one AI-generated image suitable for Streamlit.
"""

from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Any, Dict, Literal, Optional

from dotenv import load_dotenv
from PIL import Image
from google import genai
from google.genai.types import GenerateContentConfig, Modality
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
logger = logging.getLogger(__name__)

Provider = Literal["gemini_api", "vertex", "auto"]

RAW_RECIPE_PROMPT_TEMPLATE = """You are a culinary assistant specialized in safe raw and low-temperature food preparation.

The user provides a single ingredient category:
{ingredient}

Your task:

Return exactly 1 recipe suggestion that:
- Uses the provided ingredient in its RAW form.
- Does NOT include any cooking or heating above 40°C.
- Does NOT include baking, frying, roasting, boiling, steaming, grilling, sous-vide, or searing.
- May include marinating, curing, acidification (lemon, vinegar), fermentation, blending, slicing, or cold assembly.
- Is realistic and commonly recognized.
- Is appropriate for human consumption.

Provide:

1. Recipe name
2. Photo of the recipe
3. Short description (2–3 sentences)
4. Key ingredients (bullet list)
5. Basic preparation steps (concise, 4–6 steps)

Formatting rules:
- Use clear section headers.
- Keep the recipe concise.
- Do not add disclaimers unless necessary.
- Do not explain safety risks unless explicitly asked.
- Return only the recipe.

If the ingredient is not suitable for raw preparation, return exactly:
"This ingredient is not suitable for raw preparation."
"""

MEDIUM_RECIPE_PROMPT_TEMPLATE = """You are a culinary assistant specialized in controlled moderate-temperature cooking.

The user provides a single ingredient category:
{ingredient}

Your task:

Return exactly 1 recipe suggestion that:
- Uses the provided ingredient as the primary component.
- Includes at least one cooking step performed at a temperature >= 70°C and < 110°C.
- Does NOT include any cooking method exceeding 110°C.
- Does NOT include frying, grilling, broiling, searing, roasting at high heat, or baking above 110°C.
- Acceptable techniques include simmering (70–100°C), poaching (70–95°C), steaming (<100°C), sous-vide (70–95°C), controlled low-temperature braising (<110°C), or water-bath cooking.
- Is realistic and commonly recognized.
- Is appropriate and safe for human consumption.

Provide:

1. Recipe name
2. Short description (2–3 sentences)
3. Key ingredients (bullet list)
4. Cooking method with explicit temperature range
5. Basic preparation steps (4–6 concise steps)

Formatting rules:
- Use clear section headers.
- Explicitly state the cooking temperature range.
- Keep the recipe concise.
- Do not add safety disclaimers unless explicitly requested.
- Return only the recipe.

If the ingredient is incompatible with cooking within the specified temperature range, return exactly:
"This ingredient is not suitable for cooking within the specified temperature range."
"""

HIGH_RECIPE_PROMPT_TEMPLATE = """You are a culinary assistant specialized in high-temperature cooking techniques.

The user provides a single ingredient category:
{ingredient}

Your task:

Return exactly 1 recipe suggestion that:
- Uses the provided ingredient as the primary component.
- Includes at least one cooking step performed at a temperature strictly above 115°C.
- Explicitly mentions the cooking temperature.
- Uses realistic high-heat culinary techniques such as roasting, baking, frying, grilling, broiling, sautéing, searing, or pressure cooking (>115°C).
- Does NOT include exclusively low-temperature methods.
- Is commonly recognized.
- Is appropriate and safe for human consumption.

Provide:

1. Recipe name
2. Short description (2–3 sentences)
3. Key ingredients (bullet list)
4. Cooking method with explicit temperature indication
5. Basic preparation steps (4–6 concise steps)

Formatting rules:
- Use clear section headers.
- Explicitly state the cooking temperature.
- Keep the recipe concise.
- Do not add safety disclaimers unless explicitly requested.
- Return only the recipe.

If the ingredient is incompatible with cooking above 115°C, return exactly:
"This ingredient is not suitable for cooking above 115°C."
"""


def _get_env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _select_provider(provider: Provider) -> Literal["gemini_api", "vertex"]:
    if provider in ("gemini_api", "vertex"):
        return provider

    if _get_env("GEMINI_API_KEY"):
        logger.info("Auto provider: gemini_api")
        return "gemini_api"

    if _get_env("GOOGLE_CLOUD_PROJECT"):
        logger.info("Auto provider: vertex")
        return "vertex"

    raise RuntimeError(
        "No credentials found. Set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT."
    )


def _build_text_llm(
    *,
    provider: Literal["gemini_api", "vertex"],
    model: str,
    temperature: float,
    max_output_tokens: int,
    gemini_api_key: Optional[str] = None,
    gcp_project: Optional[str] = None,
    gcp_region: Optional[str] = None,
) -> ChatGoogleGenerativeAI:
    if provider == "gemini_api":
        api_key = (gemini_api_key or _get_env("GEMINI_API_KEY"))
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            google_api_key=api_key,
        )

    # Vertex AI mode
    project = (gcp_project or _get_env("GOOGLE_CLOUD_PROJECT"))
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set.")

    location = (gcp_region or _get_env("GOOGLE_CLOUD_REGION") or "us-central1")

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        project=project,
        location=location,
    )


def _build_image_client(
    *,
    provider: Literal["gemini_api", "vertex"],
    image_location: Optional[str] = None,
) -> genai.Client:
    if provider == "gemini_api":
        api_key = _get_env("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        return genai.Client(api_key=api_key)

    project = _get_env("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set.")

    # For image generation on Vertex AI, "global" is often the safest default.
    location = image_location or _get_env("GOOGLE_CLOUD_REGION") or "global"

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )


def _get_prompt(cooking: str) -> ChatPromptTemplate:
    if cooking == "raw":
        return ChatPromptTemplate.from_template(RAW_RECIPE_PROMPT_TEMPLATE)
    if cooking == "medium":
        return ChatPromptTemplate.from_template(MEDIUM_RECIPE_PROMPT_TEMPLATE)
    if cooking == "high":
        return ChatPromptTemplate.from_template(HIGH_RECIPE_PROMPT_TEMPLATE)

    return ChatPromptTemplate.from_template(
        "Explain in simple terms that the selected food type or cooking requirements does not belong to a recognized category."
    )


def _build_image_prompt(recipe_text: str) -> str:
    return f"""Generate one high-quality appetizing food photograph of the dish described below.

Requirements:
- Photorealistic food photography
- Single plated serving
- Clean background
- Natural lighting
- No text, labels, watermark, or collage
- Show the finished dish only
- Make the appearance consistent with the recipe

Recipe:
{recipe_text}
"""


def _generate_recipe_image(
    recipe_text: str,
    *,
    provider: Literal["gemini_api", "vertex"],
    image_model: str = "gemini-2.5-flash-image",
    image_location: Optional[str] = None,
) -> Optional[Image.Image]:
    client = _build_image_client(provider=provider, image_location=image_location)

    response = client.models.generate_content(
        model=image_model,
        contents=_build_image_prompt(recipe_text),
        config=GenerateContentConfig(
            response_modalities=[Modality.TEXT, Modality.IMAGE],
        ),
    )

    for candidate in response.candidates or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue

        for part in content.parts or []:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                image = Image.open(BytesIO(inline_data.data))
                image.load()  # fully load into memory
                return image

    logger.warning("No image returned by the image generation model.")
    return None


def recipe_suggestion(
    ingredient: str,
    cooking: str,
    *,
    provider: Provider = "auto",
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    max_output_tokens: int = 800,
    image_model: str = "gemini-2.5-flash-image",
    image_location: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    gcp_project: Optional[str] = None,
    gcp_region: Optional[str] = None,
    ) -> Dict[str, Any]:
    """
    Generate one recipe suggestion and one AI-generated image.

    Returns:
        dict with:
        - "recipe_text": str
        - "image": PIL.Image.Image | None

    Notes:
        - The image is kept in memory only.
        - The returned PIL image can be used directly with Streamlit:
            st.image(result["image"])
    """

    selected_provider = _select_provider(provider)

    llm = _build_text_llm(
        provider=selected_provider,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        gemini_api_key=gemini_api_key,
        gcp_project=gcp_project,
        gcp_region=gcp_region,
    )

    prompt = _get_prompt(cooking)
    chain = prompt | llm | StrOutputParser()

    answer: str = chain.invoke({"ingredient": ingredient})
    recipe_text = answer.strip()

    image = _generate_recipe_image(
        recipe_text,
        provider=selected_provider,
        image_model=image_model,
        image_location=image_location,
    )

    return {
        "recipe_text": recipe_text,
        "image": image,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    result = recipe_suggestion(
        ingredient="beef",
        cooking="raw",
        provider="auto",
    )

    print(result["recipe_text"])
    print(type(result["image"]))
