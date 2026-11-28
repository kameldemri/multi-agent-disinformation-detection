import os
from dataclasses import dataclass

import httpx
from fastapi import HTTPException


@dataclass(frozen=True)
class ServiceSettings:
    agent1_url: str
    agent2_url: str
    agent3_url: str
    timeout: float


def load_settings() -> ServiceSettings:
    try:
        timeout = float(os.getenv("COORDINATOR_TIMEOUT_SECONDS", "30"))
    except ValueError as exc:
        raise RuntimeError("COORDINATOR_TIMEOUT_SECONDS must be a float") from exc
    if timeout <= 0:
        raise RuntimeError("COORDINATOR_TIMEOUT_SECONDS must be positive")

    return ServiceSettings(
        agent1_url=os.getenv("AGENT1_URL", "http://agent1:8001").rstrip("/"),
        agent2_url=os.getenv("AGENT2_URL", "http://agent2:8002").rstrip("/"),
        agent3_url=os.getenv("AGENT3_URL", "http://agent3:8003").rstrip("/"),
        timeout=timeout,
    )


def _extract_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return "agent service returned an invalid error response"
    detail = data.get("detail", "agent service error") if isinstance(data, dict) else "agent service error"
    return str(detail)


def post_json(
    url: str,
    payload: dict,
    settings: ServiceSettings,
    client=None,
) -> dict:
    try:
        if client is not None:
            response = client.post(url, json=payload)
        else:
            with httpx.Client(timeout=settings.timeout) as http_client:
                response = http_client.post(url, json=payload)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="agent service timed out") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="agent service unavailable") from exc

    if response.status_code == 422:
        raise HTTPException(status_code=422, detail=_extract_detail(response))
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=_extract_detail(response))

    try:
        data = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="agent service returned invalid JSON") from exc

    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="agent service returned invalid JSON")

    return data
