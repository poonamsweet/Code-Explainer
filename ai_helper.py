from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


class CodeExplanationError(RuntimeError):
    """Raised when the OpenAI call fails in a user-facing way."""


@dataclass(frozen=True)
class ExplainResult:
    text: str
    model: str


def get_openai_client(*, api_key: Optional[str] = None) -> OpenAI:
    """
    Create an OpenAI client.

    Configuration:
      - OPENAI_API_KEY: required (unless api_key passed)
    """
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise CodeExplanationError(
            "Missing OPENAI_API_KEY. Set it as an environment variable before running."
        )
    return OpenAI(api_key=resolved_key)


def explain_code(
    code: str,
    *,
    filename: Optional[str] = None,
    model: Optional[str] = None,
    client: Optional[OpenAI] = None,
    api_key: Optional[str] = None,
    timeout_s: float = 60.0,
) -> str:
    """
    Explain the given source code using the OpenAI API.

    Configuration:
      - OPENAI_API_KEY: required
      - OPENAI_MODEL: optional (default: gpt-4.1-mini)
    """
    result = explain_code_detailed(
        code,
        filename=filename,
        model=model,
        client=client,
        api_key=api_key,
        timeout_s=timeout_s,
    )
    return result.text


def explain_code_detailed(
    code: str,
    *,
    filename: Optional[str] = None,
    model: Optional[str] = None,
    client: Optional[OpenAI] = None,
    api_key: Optional[str] = None,
    timeout_s: float = 60.0,
) -> ExplainResult:
    """
    Reusable explanation function that returns metadata along with the text.
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a non-empty string")

    resolved_model = (model or DEFAULT_MODEL).strip()
    if not resolved_model:
        resolved_model = DEFAULT_MODEL

    resolved_client = client or get_openai_client(api_key=api_key)

    name_line = f"Filename: {filename}\n\n" if filename else ""
    prompt = (
        "You are an expert software engineer. Explain the code clearly and concisely.\n"
        "Include:\n"
        "- High-level purpose\n"
        "- Key functions/classes and data flow\n"
        "- Any notable edge cases, risks, or improvements\n"
        "- If relevant, a short usage example\n\n"
        f"{name_line}"
        "Code:\n"
        f"{code}"
    )

    try:
        resp = resolved_client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": "You explain code to developers."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            timeout=timeout_s,
        )
    except AuthenticationError as e:
        raise CodeExplanationError(
            "Authentication failed. Check that OPENAI_API_KEY is set correctly."
        ) from e
    except RateLimitError as e:
        raise CodeExplanationError(
            "Rate limited by the API. Please wait a bit and try again."
        ) from e
    except (APITimeoutError, APIConnectionError) as e:
        raise CodeExplanationError(
            "Network/timeout error calling the API. Check your connection and try again."
        ) from e
    except BadRequestError as e:
        raise CodeExplanationError(
            "Bad request to the API (often an invalid model name or input). "
            "Try a different model or smaller file."
        ) from e
    except APIError as e:
        raise CodeExplanationError(
            "The API returned an error. Please try again shortly."
        ) from e
    except Exception as e:
        raise CodeExplanationError("Unexpected error while generating explanation.") from e

    text = (resp.choices[0].message.content or "").strip()
    return ExplainResult(text=text, model=resolved_model)


def answer_question_about_code(
    *,
    question: str,
    context_chunks: list[str],
    filename: Optional[str] = None,
    model: Optional[str] = None,
    client: Optional[OpenAI] = None,
    api_key: Optional[str] = None,
    timeout_s: float = 60.0,
) -> ExplainResult:
    """
    Answer a question about code using retrieved context chunks.

    The model must answer based only on provided context chunks.
    If the answer is not supported by the context, it must reply exactly: "Not in code".
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(context_chunks, list) or not context_chunks:
        raise ValueError("context_chunks must be a non-empty list of strings")

    resolved_model = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    resolved_client = client or get_openai_client(api_key=api_key)

    context = "\n\n---\n\n".join(context_chunks)
    name_line = f"File: {filename}\n\n" if filename else ""

    try:
        resp = resolved_client.chat.completions.create(
            model=resolved_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior software engineer. "
                        "Answer the user's question clearly and directly.\n\n"
                        "Rules:\n"
                        "- Use ONLY the provided Context. Do not guess.\n"
                        "- If the answer is not explicitly present in the Context, reply with exactly: Not in code\n"
                        "- Keep answers concise, with bullet points when helpful.\n"
                        "- If referencing code, quote short snippets from the Context."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{name_line}"
                        "Context (retrieved code snippets):\n"
                        f"{context}\n\n"
                        "Question:\n"
                        f"{question}"
                    ),
                },
            ],
            temperature=0.2,
            timeout=timeout_s,
        )
    except AuthenticationError as e:
        raise CodeExplanationError(
            "Authentication failed. Check that OPENAI_API_KEY is set correctly."
        ) from e
    except RateLimitError as e:
        raise CodeExplanationError(
            "Rate limited by the API. Please wait a bit and try again."
        ) from e
    except (APITimeoutError, APIConnectionError) as e:
        raise CodeExplanationError(
            "Network/timeout error calling the API. Check your connection and try again."
        ) from e
    except BadRequestError as e:
        raise CodeExplanationError(
            "Bad request to the API (often an invalid model name or input). "
            "Try a different model or smaller context."
        ) from e
    except APIError as e:
        raise CodeExplanationError(
            "The API returned an error. Please try again shortly."
        ) from e
    except Exception as e:
        raise CodeExplanationError("Unexpected error while generating an answer.") from e

    text = (resp.choices[0].message.content or "").strip()
    return ExplainResult(text=text, model=resolved_model)

