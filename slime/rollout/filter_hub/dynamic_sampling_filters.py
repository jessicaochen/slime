from __future__ import annotations

import logging
import math
import re
import statistics
from typing import TYPE_CHECKING, Any

from slime.rollout.filter_hub.base_types import DynamicFilterOutput

if TYPE_CHECKING:
    from slime.utils.types import Sample

logger = logging.getLogger(__name__)

__all__ = ["check_reward_nonzero_std", "check_gsm8k_nonzero_std"]

# State tracker per rollout step to guarantee bounded candidate pool and backfill
_FILTER_STATE: dict[str, Any] = {
    "current_rollout_id": None,
    "dropped_count": 0,
    "kept_count": 0,
}


def _calc_std(values: list[float]) -> float:
    """Calculate sample standard deviation with pure Python without torch overhead."""
    if len(values) < 2:
        return 0.0
    try:
        return float(statistics.stdev(values))
    except Exception:
        mean_val = sum(values) / len(values)
        return math.sqrt(sum((x - mean_val) ** 2 for x in values) / (len(values) - 1))


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


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = _calc_std(rewards) > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def check_gsm8k_nonzero_std(args, samples: list[Sample], **kwargs):
    """Evaluate GSM8K numerical accuracy across candidate rollouts with bounded backfill.

    Prioritizes non-zero variance prompt groups (std > 0) while strictly capping total
    evaluations to over_sampling_batch_size to maintain predictable Sampling:Training ratios.
    """
    if not samples:
        return DynamicFilterOutput(keep=True)

    gold = _extract_gsm8k_gold(str(samples[0].label))
    if gold is None:
        return DynamicFilterOutput(keep=True)

    # Track rollout step reset
    rollout_id = getattr(samples[0], "rollout_id", None) or kwargs.get("rollout_id")
    if _FILTER_STATE["current_rollout_id"] != rollout_id:
        _FILTER_STATE["current_rollout_id"] = rollout_id
        _FILTER_STATE["dropped_count"] = 0
        _FILTER_STATE["kept_count"] = 0

    rewards = []
    for s in samples:
        pred = _extract_gsm8k_pred(str(s.response))
        try:
            is_correct = (pred is not None) and (abs(float(pred) - float(gold)) < 1e-4)
        except ValueError:
            is_correct = (pred is not None) and (str(pred).strip().lower() == str(gold).strip().lower())
        rewards.append(1.0 if is_correct else 0.0)

    std_val = _calc_std(rewards)
    has_variance = std_val > 1e-6

    # Calculate max drops allowed before backfill kicks in to prevent fetching extra batches
    over_sampling_batch = getattr(args, "over_sampling_batch_size", 512) or 512
    rollout_batch = getattr(args, "rollout_batch_size", 128) or 128
    max_allowed_drops = max(0, over_sampling_batch - rollout_batch)

    if has_variance:
        _FILTER_STATE["kept_count"] += 1
        return DynamicFilterOutput(keep=True)

    # If we reached the drop ceiling, activate backfill to keep the sample and terminate on budget
    if _FILTER_STATE["dropped_count"] >= max_allowed_drops:
        _FILTER_STATE["kept_count"] += 1
        return DynamicFilterOutput(keep=True, reason="backfill_budget_reached")

    # Otherwise drop zero-variance sample
    _FILTER_STATE["dropped_count"] += 1
    return DynamicFilterOutput(
        keep=False,
        reason=f"zero_std_{round(rewards[0], 1)}",
    )
