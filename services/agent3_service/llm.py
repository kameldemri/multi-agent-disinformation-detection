import json
import os
from dataclasses import dataclass

import httpx

from services.agent3_service.schemas import DecisionResponse


SYSTEM_PROMPT = """You are the explanation writer for Agent 3 in a multi-agent disinformation detection system.

You receive evidence from two independent agents and a final deterministic decision. Agent 1 reports fake_news_probability for article content. It measures the likelihood that content has fake-news characteristics. Agent 2 reports bot_probability for account behavior. It measures the likelihood of bot-like temporal behavior, not misinformation or malicious intent. final_risk_probability is the maximum of those two probabilities.

Your only job is to write a concise reason that explains the already-decided outcome. Do not change the decision. Do not change final_risk_probability. Do not invent evidence. Do not add enforcement actions other than BAN or MODERATION. Use the policy_reason as the factual basis. Return only the reason as plain text, with no labels or extra commentary."""


class LLMError(Exception):
    pass


@dataclass(frozen=True)
class LLMSettings:
    mode: str
    api_key: str | None
    base_url: str
    model: str
    timeout: float
    max_tokens: int
    allow_no_key: bool = False


def load_settings() -> LLMSettings:
    mode = os.getenv("LLM_MODE", "offline").strip().lower()
    if mode not in {"offline", "api"}:
        raise RuntimeError("LLM_MODE must be 'offline' or 'api'")

    try:
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "10"))
    except ValueError as exc:
        raise RuntimeError("LLM_TIMEOUT_SECONDS must be a float") from exc
    if timeout <= 0:
        raise RuntimeError("LLM_TIMEOUT_SECONDS must be positive")

    try:
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "160"))
    except ValueError as exc:
        raise RuntimeError("LLM_MAX_TOKENS must be an integer") from exc
    if max_tokens <= 0:
        raise RuntimeError("LLM_MAX_TOKENS must be positive")

    api_key = os.getenv("LLM_API_KEY", "").strip() or None
    allow_no_key = os.getenv("LLM_ALLOW_NO_KEY", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    return LLMSettings(
        mode=mode,
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip(),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        timeout=timeout,
        max_tokens=max_tokens,
        allow_no_key=allow_no_key,
    )


def build_payload(decision: DecisionResponse) -> dict:
    return {
        "user_id": decision.user_id,
        "decision": decision.decision,
        "final_risk_probability": decision.final_risk_probability,
        "evidence": decision.evidence.model_dump(),
        "policy_reason": decision.reason,
    }


def parse_reason(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        if text.lower().startswith("reason:"):
            reason = text[len("reason:") :].strip()
        elif text.endswith((".", "!", "?")) and len(text.split()) >= 3:
            reason = text
        else:
            raise LLMError("LLM response did not contain a usable reason")
    else:
        if not isinstance(data, dict):
            raise LLMError("LLM response must be a JSON object")
        reason = data.get("reason")
        if not isinstance(reason, str):
            raise LLMError("LLM response must contain a non-empty reason")
        reason = reason.strip()

    if not reason:
        raise LLMError("LLM response must contain a non-empty reason")

    return reason


def _post_completion(client, url: str, headers: dict, body: dict) -> str:
    response = client.post(url, headers=headers, json=body)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return parse_reason(content)


def request_reason(
    settings: LLMSettings,
    payload: dict,
    client=None,
) -> str:
    if settings.mode != "api":
        raise LLMError("LLM mode is not api")
    if not settings.api_key and not settings.allow_no_key:
        raise LLMError("LLM API key is missing")

    url = f"{settings.base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    body = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ],
        "temperature": 0,
        "max_tokens": settings.max_tokens,
    }

    if client is not None:
        return _post_completion(client, url, headers, body)

    with httpx.Client(timeout=settings.timeout) as http_client:
        return _post_completion(http_client, url, headers, body)


def explain_result(
    decision: DecisionResponse,
    settings: LLMSettings,
    client=None,
) -> tuple[str, str]:
    if settings.mode == "offline":
        return decision.reason, "disabled"

    payload = build_payload(decision)
    try:
        reason = request_reason(settings, payload, client=client)
    except (
        LLMError,
        httpx.HTTPError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ):
        return decision.reason, "fallback"

    return reason, "used"
