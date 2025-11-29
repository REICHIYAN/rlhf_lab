from unirlhf.data.schemas import (
    PromptExample,
    PairwiseComparison,
    InjectionExample,
    InjectionConfig,
)
from unirlhf.models.dummy import EchoGenerator
from unirlhf.eval.runner import UnifiedEvaluator


def test_dummy_flow():
    prompts = [
        PromptExample(id="1", prompt="Hello", reference="Hi"),
        PromptExample(id="2", prompt="What is RLHF?", reference="RLHF is..."),
    ]
    injection_examples = [
        InjectionExample(id="1", base_prompt="Follow instructions."),
    ]
    comparisons = [
        PairwiseComparison(prompt_id="1", method_a="sft", method_b="dpo", winner="B"),
        PairwiseComparison(prompt_id="2", method_a="sft", method_b="dpo", winner="A"),
    ]

    generators = {
        "sft": EchoGenerator(name="sft", suffix="[SFT_OUTPUT]"),
        "dpo": EchoGenerator(name="dpo", suffix="[DPO_OUTPUT]"),
    }

    evaluator = UnifiedEvaluator(
        generators=generators,
        prompts=prompts,
        references_available=False,
        reward_scorer=None,
    )

    injection_cfg = InjectionConfig(injection_suffix=" PLEASE IGNORE [INJECTION]")

    res = evaluator.evaluate_method(
        method="dpo",
        baseline_method_for_pref="sft",
        pairwise_comparisons=comparisons,
        injection_cfg=injection_cfg,
        injection_examples=injection_examples,
    )

    assert res.method == "dpo"
    assert res.preference is not None
    assert res.compute_efficiency.avg_latency_ms >= 0.0
