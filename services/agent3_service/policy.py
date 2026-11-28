import os

from services.agent3_service.schemas import DecisionRequest, DecisionResponse


def _load_threshold(name: str) -> float:
    raw_value = os.getenv(name, "0.5")
    try:
        threshold = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float") from exc
    if not 0 <= threshold <= 1:
        raise RuntimeError(f"{name} must be between 0 and 1")
    return threshold


BOT_THRESHOLD = _load_threshold("BOT_THRESHOLD")
FAKE_NEWS_THRESHOLD = _load_threshold("FAKE_NEWS_THRESHOLD")


def build_reason(
    bot_probability: float,
    fake_news_probability: float,
    bot_threshold: float,
    fake_news_threshold: float,
) -> str:
    bot_high = bot_probability >= bot_threshold
    fake_high = fake_news_probability >= fake_news_threshold

    if bot_high and fake_high:
        return "Both behavioral and content evidence exceed their configured thresholds."
    if bot_high:
        return "The account shows a high probability of bot-like behavior."
    if fake_high:
        return "The content shows a high probability of fake-news characteristics."
    return "Both content and behavioral risks are below their configured thresholds."


def evaluate_decision(
    request: DecisionRequest,
    bot_threshold: float = BOT_THRESHOLD,
    fake_news_threshold: float = FAKE_NEWS_THRESHOLD,
) -> DecisionResponse:
    bot_probability = request.agent_2.bot_probability
    fake_news_probability = request.agent_1.fake_news_probability

    if bot_probability >= bot_threshold:
        decision = "BAN"
    elif fake_news_probability >= fake_news_threshold:
        decision = "BAN"
    else:
        decision = "MODERATION"

    return DecisionResponse(
        user_id=request.user_id,
        final_risk_probability=max(bot_probability, fake_news_probability),
        decision=decision,
        reason=build_reason(
            bot_probability,
            fake_news_probability,
            bot_threshold,
            fake_news_threshold,
        ),
        evidence={
            "fake_news_probability": fake_news_probability,
            "bot_probability": bot_probability,
        },
        llm_status="disabled",
    )
