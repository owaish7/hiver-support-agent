"""One wrapper over both providers, so switching is a config line and not a rewrite.

Two providers on purpose:

  generator -> Gemini 2.5 Flash
  judge     -> Llama 3.3 70B on Groq

They are different model families deliberately. LLM judges show self-preference bias:
they score text from their own family more generously. Gemini writing the reply and
Gemini grading it would inflate the headline number by an amount we cannot measure.
Using a different family for the judge removes that specific confound. It does not
make the judge correct -- that is what eval/agreement.py is for.

Everything returns a validated Pydantic object. Nothing in this project parses prose
out of a model response.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

import config

T = TypeVar("T", bound=BaseModel)

MAX_RETRIES = 4


@dataclass
class LLMResult:
    """Parsed output plus what it cost. Token counts feed the cost column in the
    results table -- an agent that is 2% better for 40x the tokens is a finding."""
    parsed: BaseModel
    input_tokens: int
    output_tokens: int
    raw: str


class LLMError(RuntimeError):
    pass


# ------------------------------------------------------------------ clients
# Built lazily so that every offline path (--report, the escalation oracle, the
# TF-IDF baseline, the tests) runs with no API key present at all.

_gemini = None
_groq = None


def _gemini_client():
    global _gemini
    if _gemini is None:
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY not set (see .env.example)")
        _gemini = genai.Client(api_key=key)
    return _gemini


def _groq_client():
    global _groq
    if _groq is None:
        from groq import Groq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise LLMError("GROQ_API_KEY not set (see .env.example)")
        _groq = Groq(api_key=key)
    return _groq


# ------------------------------------------------------------------ retry

def _with_retry(fn):
    """Free tiers rate-limit. A 200-item eval run that dies at item 180 because of a
    429 has wasted 180 real API calls, so back off and keep going."""
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
            last = exc
            msg = str(exc).lower()
            transient = any(s in msg for s in
                            ("429", "rate", "quota", "503", "500", "overload",
                             "timeout", "unavailable", "deadline"))
            if not transient or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt * 2)  # 2s, 4s, 8s
    raise LLMError(str(last))


# ------------------------------------------------------------------ public API

def complete(system: str, user: str, schema: type[T], *,
             provider: str = "gemini", model: str | None = None,
             temperature: float = 0.0) -> LLMResult:
    """Ask a model for structured output. Returns a validated `schema` instance.

    temperature=0 everywhere by default: a re-run of the eval must produce the same
    numbers, otherwise "the system improved" and "the sampler got lucky" are the same
    observation.
    """
    if provider == "gemini":
        return _complete_gemini(system, user, schema, model or config.GEN_MODEL, temperature)
    if provider == "groq":
        return _complete_groq(system, user, schema, model or config.JUDGE_MODEL, temperature)
    raise LLMError(f"unknown provider {provider!r}")


def _complete_gemini(system: str, user: str, schema: type[T], model: str,
                     temperature: float) -> LLMResult:
    from google.genai import types

    client = _gemini_client()

    def call():
        return client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )

    resp = _with_retry(call)
    raw = resp.text or ""
    parsed = resp.parsed
    if parsed is None:
        try:
            parsed = schema.model_validate_json(raw)
        except ValidationError as exc:
            raise LLMError(f"gemini returned unparseable output: {raw[:200]}") from exc

    usage = getattr(resp, "usage_metadata", None)
    return LLMResult(
        parsed=parsed,
        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        raw=raw,
    )


def _complete_groq(system: str, user: str, schema: type[T], model: str,
                   temperature: float) -> LLMResult:
    """Groq has no native Pydantic schema binding, so the JSON schema goes into the
    system prompt and we validate on the way out. Same contract as the Gemini path."""
    client = _groq_client()
    schema_text = json.dumps(schema.model_json_schema(), indent=2)
    sys_full = (
        f"{system}\n\n"
        f"Reply with a single JSON object matching this schema exactly. "
        f"No markdown fences, no commentary.\n\n{schema_text}"
    )

    def call():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_full},
                      {"role": "user", "content": user}],
            temperature=temperature,
            response_format={"type": "json_object"},
        )

    resp = _with_retry(call)
    raw = resp.choices[0].message.content or ""
    try:
        parsed = schema.model_validate_json(raw)
    except ValidationError as exc:
        raise LLMError(f"groq returned unparseable output: {raw[:200]}") from exc

    return LLMResult(
        parsed=parsed,
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
        raw=raw,
    )
