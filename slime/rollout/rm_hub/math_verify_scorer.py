import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_math_verify_reward(response: str | None, label: Any) -> float:
    if not response or label is None:
        return 0.0
    try:
        from math_verify import parse, verify

        pred = parse(str(response))
        gold = parse(str(label))
        return float(verify(pred, gold))
    except Exception as e:
        logger.debug(f"math_verify scoring evaluation failed: {e}")
        return 0.0
