from dataclasses import dataclass
from typing import List, Callable
import time


@dataclass
class ComputeEfficiencyMetrics:
    avg_latency_ms: float
    tokens_per_second: float
    approx_flops_per_token: float


def measure_latency_and_throughput(
    generator_fn: Callable[[str, int], tuple],
    prompts: List[str],
) -> ComputeEfficiencyMetrics:
    latencies_ms = []
    total_tokens = 0
    total_time = 0.0

    for p in prompts:
        start = time.time()
        text, num_tokens = generator_fn(p, 128)
        end = time.time()
        dt = end - start
        latencies_ms.append(dt * 1000.0)
        total_time += dt
        total_tokens += int(num_tokens)

    avg_latency = sum(latencies_ms) / max(len(latencies_ms), 1)
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0.0

    approx_flops_per_token = 0.0
    return ComputeEfficiencyMetrics(
        avg_latency_ms=avg_latency,
        tokens_per_second=tokens_per_second,
        approx_flops_per_token=approx_flops_per_token,
    )
