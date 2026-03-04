"""
Docstring for interface.recipes
Gives recipes depending on the predictiosn made on the user's food"""

from __future__ import annotations

#notes : utiliser LangChain pour transformer nos sources en embeddings, puis
#faire la génération de texte. Voir challenges et leçons Gen IA, RAG 13 janvier


"""
Copié de explanations.py

"""



import logging
import os
from typing import List, Literal, Optional, Sequence

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
logger = logging.getLogger(__name__)

Provider = Literal["gemini_api", "vertex", "auto"]

RAW_RECIPES_PROMPT_TEMPLATE = """You are a culinary assistant specialized in safe raw and low-temperature food preparation.

The user provides a single ingredient category:
{ingredient}

Your task:

Return exactly 5 recipe suggestions that:
- Use the provided ingredient in its RAW form.
- Do NOT include any cooking or heating above 40°C.
- Do NOT include baking, frying, roasting, boiling, steaming, grilling, sous-vide, or searing.
- May include marinating, curing, acidification (lemon, vinegar), fermentation, blending, slicing, or cold assembly.
- Are realistic and commonly recognized culinary preparations.
- Are appropriate for human consumption (do not suggest unsafe or illegal practices).

For each recipe, provide:

1. Recipe name
2. Short description (2–3 sentences)
3. Key ingredients (bullet list)
4. Basic preparation steps (concise, 4–6 steps)

Formatting rules:
- Use clear section headers.
- Keep each recipe concise.
- Do not add disclaimers unless necessary.
- Do not explain safety risks unless explicitly asked.
- Return only the 5 recipes.

If the ingredient is not suitable for raw preparation, return:
"This ingredient is not suitable for raw preparation."
"""

MEDIUM_RECIPES_PROMPT_TEMPLATE = """You are a culinary assistant specialized in controlled moderate-temperature cooking.

The user provides a single ingredient category:
{ingredient}

Your task:

Return exactly 5 recipe suggestions that:

- Use the provided ingredient as the primary component.
- Include at least one cooking step performed at a temperature ≥ 70°C and < 110°C.
- Do NOT include any cooking method exceeding 110°C.
- Do NOT include frying, grilling, broiling, searing, roasting at high heat, or baking above 110°C.
- Acceptable techniques include simmering (70–100°C), poaching (70–95°C), steaming (<100°C), sous-vide (70–95°C), controlled low-temperature braising (<110°C), or water-bath cooking.
- Are realistic, commonly recognized culinary preparations.
- Are appropriate and safe for human consumption.

For each recipe, provide:

1. Recipe name
2. Short description (2–3 sentences)
3. Key ingredients (bullet list)
4. Cooking method with explicit temperature range (must clearly state a temperature between 70°C and 110°C)
5. Basic preparation steps (4–6 concise steps)

Formatting rules:
- Use clear section headers.
- Explicitly state the cooking temperature range in each recipe.
- Keep each recipe concise.
- Do not add safety disclaimers unless explicitly requested.
- Return only the 5 recipes.

If the ingredient is incompatible with cooking within the specified temperature range (70–110°C), return:
"This ingredient is not suitable for cooking within the specified temperature range."
"""

HIGH_RECIPES_PROMPT_TEMPLATE = """You are a culinary assistant specialized in high-temperature cooking techniques.

The user provides a single ingredient category:
{ingredient}

Your task:

Return exactly 5 recipe suggestions that:

- Use the provided ingredient as the primary component.
- Include at least one cooking step performed at a temperature strictly above 115°C.
- Explicitly mention the cooking temperature (e.g., 180°C oven, 200°C roasting, 170°C frying oil, etc.).
- Use realistic high-heat culinary techniques such as roasting, baking, frying, grilling, broiling, sautéing, searing, or pressure cooking (>115°C).
- Do NOT include exclusively low-temperature methods (no sous-vide, poaching, steaming, simmering below 110°C).
- Are commonly recognized culinary preparations.
- Are appropriate and safe for human consumption.

For each recipe, provide:

1. Recipe name
2. Short description (2–3 sentences)
3. Key ingredients (bullet list)
4. Cooking method with explicit temperature indication (must clearly state a temperature >115°C)
5. Basic preparation steps (4–6 concise steps)

Formatting rules:
- Use clear section headers.
- Explicitly state the cooking temperature in each recipe.
- Keep each recipe concise.
- Do not add safety disclaimers unless explicitly requested.
- Return only the 5 recipes.

If the ingredient is incompatible with cooking above 115°C, return:
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

    # auto mode
    if _get_env("GEMINI_API_KEY"):
        logger.info("Auto provider: gemini_api")
        return "gemini_api"

    if _get_env("GOOGLE_CLOUD_PROJECT"):
        logger.info("Auto provider: vertex")
        return "vertex"

    raise RuntimeError(
        "No credentials found. Set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT."
    )


def _build_llm(
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


def recipe_suggestion(
    ingredient: Optional[Sequence[str]],
    cooking: str,
    *,
    provider: Provider = "auto",
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    max_output_tokens: int = 800,
    gemini_api_key: Optional[str] = None,
    gcp_project: Optional[str] = None,
    gcp_region: Optional[str] = None,
) -> str:

    selected_provider = _select_provider(provider)

    llm = _build_llm(
        provider=selected_provider,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        gemini_api_key=gemini_api_key,
        gcp_project=gcp_project,
        gcp_region=gcp_region,
    )

    if cooking == 'raw':
        prompt = ChatPromptTemplate.from_template(RAW_RECIPES_PROMPT_TEMPLATE)
    elif cooking == 'medium' :
        prompt = ChatPromptTemplate.from_template(MEDIUM_RECIPES_PROMPT_TEMPLATE)
    elif cooking == 'high':
        prompt = ChatPromptTemplate.from_template(HIGH_RECIPES_PROMPT_TEMPLATE)
    else:
        prompt = ChatPromptTemplate.from_template('Explain in simple terms that the selected food type or cooking requirements does not belong to a recognized category ')


    chain = prompt | llm | StrOutputParser()

    answer: str = chain.invoke({"ingredient": ingredient})
    return answer.strip()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(recipe_suggestion(ingredient="beef", cooking='raw', provider="auto"))
