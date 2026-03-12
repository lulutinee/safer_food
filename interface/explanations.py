"""
risk_answer.py

Production-safe Gemini integration (API key or Vertex AI).
Uses ONLY ChatGoogleGenerativeAI (no deprecated ChatVertexAI).

Dependencies:
- langchain-core
- langchain-google-genai
- python-dotenv

Use :
risk_explanation(
    microorganisms: Optional[Sequence[str]],   : name(s) of the microorganism(s) above infectious dose (Listeria monocytogenes, Salmonella enterica, Escherichia coli)
    *,
    provider: Provider = "auto",                : llm model provider ["gemini_api", "vertex", "auto"]
    model: str = "gemini-2.5-flash",            : model name
    temperature: float = 0.2,                   : model temperature
    max_output_tokens: int = 800,
) -> str:


"""

from __future__ import annotations

import logging
import os
from typing import List, Literal, Optional, Sequence

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Optional for local dev only; Streamlit Cloud won't use .env
load_dotenv(override=False)
logger = logging.getLogger(__name__)

Provider = Literal["gemini_api", "vertex", "auto"]

RISK_PROMPT_TEMPLATE = """You are a food microbiology risk communicator.
Write in clear, concise language for a general audience.

Given the following list of microorganisms detected above infectious dose in a food:
{microorganisms}

For EACH microorganism, write exactly:
1) One short paragraph providing a clear and concise explanation of the risks if I eat this food.
2) If some categories of population are more sensitive or at higher risk, add a couple of lines on sensitive populations and added risk.
3) One paragraph detailing the symptoms I should look for if I have food poisoning caused by this microorganism.

Formatting rules:
- Use the microorganism name as a header line.
- Keep each microorganism section compact.
- Within each microorganism section use markdown-formatted bullet points
- If the list is empty or None, return:
  "No microorganisms detected above infectious dose based on the provided list."
"""
 


def _normalize_microorganisms(microorganisms: Optional[Sequence[str]]) -> List[str]:
    if not microorganisms:
        return []
    cleaned: List[str] = []
    seen = set()
    for m in microorganisms:
        if not m:
            continue
        s = str(m).strip()
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
    return cleaned


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
        api_key = _get_env("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            google_api_key=api_key,
        )

    # Vertex AI mode
    project = _get_env("GOOGLE_CLOUD_PROJECT")
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


def risk_explanation(
    microorganisms: Optional[Sequence[str]],
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

    micro_list = _normalize_microorganisms(microorganisms)
    micro_text = "None" if not micro_list else ", ".join(micro_list)

    llm = _build_llm(
        provider=selected_provider,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        gemini_api_key=gemini_api_key,
        gcp_project=gcp_project,
        gcp_region=gcp_region,
    )

    prompt = ChatPromptTemplate.from_template(RISK_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    answer: str = chain.invoke({"microorganisms": micro_text})
    return answer.strip()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(risk_explanation(["Salmonella enterica", "Escherichia coli", "Listeria monocytogenes", "Salmonella enterica"], provider="auto"))
