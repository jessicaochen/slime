import re

import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = ["check_reward_nonzero_std", "check_gsm8k_nonzero_std"]


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
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def check_gsm8k_nonzero_std(args, samples: list[Sample], **kwargs):
    """Evaluate GSM8K numerical accuracy across candidate rollouts with robust regex extraction,
    keeping prompt groups with non-zero variance (std > 0) to maximize GRPO gradient signal."""
    if not samples:
        return DynamicFilterOutput(keep=True)

    gold = _extract_gsm8k_gold(str(samples[0].label))
    if gold is None:
        return DynamicFilterOutput(keep=True)

    rewards = []
    for s in samples:
        pred = _extract_gsm8k_pred(str(s.response))
        try:
            is_correct = (pred is not None) and (abs(float(pred) - float(gold)) < 1e-4)
        except ValueError:
            is_correct = (pred is not None) and (str(pred).strip().lower() == str(gold).strip().lower())
        rewards.append(1.0 if is_correct else 0.0)

    std_val = float(torch.tensor(rewards, dtype=torch.float64).std())
    keep = std_val > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
