from dataclasses import dataclass
from typing import Dict, List, Optional

from unirlhf.data.schemas import (
    PromptExample,
    GeneratedExample,
    PairwiseComparison,
    InjectionExample,
    InjectionConfig,
)
from unirlhf.models.interfaces import TextGenerator, RewardScorer
from unirlhf.models.dummy import LengthRewardScorer
from unirlhf.eval.lm_basic_metrics import LMBasicMetrics
from unirlhf.eval.preference_winrate import PreferenceMetrics, compute_win_rate, compute_bradley_terry_score
from unirlhf.eval.robustness_injection import RobustnessMetrics, compute_self_consistency_entropy, compute_injection_success_rate
from unirlhf.eval.reward_alignment import RewardConsistencyMetrics, compute_avg_rm_score
from unirlhf.eval.compute_efficiency import ComputeEfficiencyMetrics, measure_latency_and_throughput


@dataclass
class EvaluationResult:
    method: str
    lm_basic: LMBasicMetrics
    preference: Optional[PreferenceMetrics]
    robustness: RobustnessMetrics
    reward_consistency: RewardConsistencyMetrics
    compute_efficiency: ComputeEfficiencyMetrics


class UnifiedEvaluator:
    def __init__(
        self,
        generators: Dict[str, TextGenerator],
        prompts: List[PromptExample],
        references_available: bool = False,
        reward_scorer: Optional[RewardScorer] = None,
    ):
        self.generators = generators
        self.prompts = prompts
        self.references_available = references_available
        self.reward_scorer = reward_scorer or LengthRewardScorer()

    def _generate_all(self, method: str) -> List[GeneratedExample]:
        gen = self.generators[method]
        generated: List[GeneratedExample] = []
        for ex in self.prompts:
            out, n_tokens = gen.generate(ex.prompt)
            generated.append(
                GeneratedExample(
                    method=method,
                    prompt_id=ex.id,
                    output=out,
                    num_tokens=n_tokens,
                )
            )
        return generated

    def evaluate_method(
        self,
        method: str,
        baseline_method_for_pref: Optional[str],
        pairwise_comparisons: Optional[List[PairwiseComparison]],
        injection_cfg: InjectionConfig,
        injection_examples: List[InjectionExample],
    ) -> EvaluationResult:
        generated = self._generate_all(method)

        lm_basic = LMBasicMetrics(perplexity=None, bertscore_f1=None)

        preference_metrics: Optional[PreferenceMetrics] = None
        if baseline_method_for_pref and pairwise_comparisons:
            win_rate = compute_win_rate(
                method=method,
                comparisons=pairwise_comparisons,
                vs_method=baseline_method_for_pref,
            )
            bt_score = compute_bradley_terry_score(
                method=method,
                comparisons=pairwise_comparisons,
                vs_method=baseline_method_for_pref,
            )
            preference_metrics = PreferenceMetrics(
                win_rate=win_rate,
                bt_score=bt_score,
            )

        if self.prompts:
            entropy = compute_self_consistency_entropy(
                prompt=self.prompts[0].prompt,
                generator=self.generators[method],
                n_samples=3,
            )
        else:
            entropy = 0.0

        def dummy_success_check(text: str) -> bool:
            return "INJECTION" in text

        injection_success = compute_injection_success_rate(
            generator=self.generators[method],
            injection_cfg=injection_cfg,
            base_prompts=injection_examples,
            success_check_fn=dummy_success_check,
        )

        robustness = RobustnessMetrics(
            self_consistency_entropy=entropy,
            injection_success_rate=injection_success,
        )

        texts = [g.output for g in generated]
        avg_rm = compute_avg_rm_score(self.reward_scorer, texts)
        reward_consistency = RewardConsistencyMetrics(
            kl_to_sft=None,
            avg_rm_score=avg_rm,
        )

        prompts_texts = [ex.prompt for ex in self.prompts]
        compute_eff = measure_latency_and_throughput(
            generator_fn=self.generators[method].generate,
            prompts=prompts_texts,
        )

        return EvaluationResult(
            method=method,
            lm_basic=lm_basic,
            preference=preference_metrics,
            robustness=robustness,
            reward_consistency=reward_consistency,
            compute_efficiency=compute_eff,
        )
