import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _extract_gsm8k_gold(label_str: str) -> str | None:
    if not label_str:
        return None
    if "####" in label_str:
        ans = label_str.split("####")[-1].strip()
        return re.sub(r"[,$%\s]", "", ans)
    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", label_str.replace(",", ""))
    return numbers[-1] if numbers else None


def _extract_gsm8k_pred(response_str: str) -> str | None:
    if not response_str:
        return None
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", response_str)
    if boxed_matches:
        return re.sub(r"[,$%\s]", "", boxed_matches[-1].strip())
    if "####" in response_str:
        ans = response_str.split("####")[-1].strip()
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", ans.replace(",", ""))
        if numbers:
            return numbers[0]
    m = re.findall(
        r"(?:answer is|is equal to|equals|total of|result is|needs)\s*:?\s*([-$+]?\d+(?:\.\d+)?)",
        response_str,
        re.IGNORECASE,
    )
    if m:
        return m[-1]
    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", response_str.replace(",", ""))
    return numbers[-1] if numbers else None


def compute_math_verify_reward(response: str | None, label: Any) -> float:
    if not response or label is None:
        return 0.0
    try:
        from math_verify import parse, verify  # pylint: disable=import-error,import-outside-toplevel

        pred = parse(str(response))
        gold = parse(str(label))
        return float(verify(pred, gold))
    except Exception:
        # Fallback to robust GSM8K numerical regex extraction
        gold = _extract_gsm8k_gold(str(label))
        pred = _extract_gsm8k_pred(str(response))
        if gold is None or pred is None:
            return 0.0
        try:
            return 1.0 if abs(float(pred) - float(gold)) < 1e-4 else 0.0
        except ValueError:
            return 1.0 if str(pred).strip().lower() == str(gold).strip().lower() else 0.0
