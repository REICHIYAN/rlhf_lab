from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from unirlhf.data.schemas import InjectionExample, InjectionConfig
from unirlhf.models.interfaces import TextGenerator


@dataclass
class RobustnessMetrics:
    self_consistency_entropy: float
    injection_success_rate: float


def compute_self_consistency_entropy(
    prompt: str,
    generator: TextGenerator,
    n_samples: int = 5,
) -> float:
    outputs = []
    for _ in range(n_samples):
        out, _ = generator.generate(prompt)
        outputs.append(out)

    unique, counts = np.unique(outputs, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    entropy = -float((probs * np.log(probs + 1e-12)).sum())
    return entropy


def compute_injection_success_rate(
    generator: TextGenerator,
    injection_cfg: InjectionConfig,
    base_prompts: List[InjectionExample],
    success_check_fn: Callable[[str], bool],
) -> float:
    total = 0
    success = 0
    for ex in base_prompts:
        attacked_prompt = ex.base_prompt + injection_cfg.injection_suffix
        out, _ = generator.generate(attacked_prompt)
        total += 1
        if success_check_fn(out):
            success += 1
    return success / total if total > 0 else 0.0
