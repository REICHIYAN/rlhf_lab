from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from unirlhf.models.interfaces import RewardScorer


@dataclass
class RewardConsistencyMetrics:
    kl_to_sft: Optional[float]
    avg_rm_score: Optional[float]


def compute_kl_to_sft(
    sft_model: PreTrainedModel,
    rl_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    max_length: int = 512,
    device: Optional[str] = None,
) -> float:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sft_model = sft_model.to(device)
    rl_model = rl_model.to(device)
    sft_model.eval()
    rl_model.eval()

    kls = []
    with torch.no_grad():
        for p in prompts:
            enc = tokenizer(
                p,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).to(device)
            out_sft = sft_model(**enc)
            out_rl = rl_model(**enc)
            logp_sft = out_sft.logits.log_softmax(-1)
            logp_rl = out_rl.logits.log_softmax(-1)
            p_rl = logp_rl.exp()
            kl = (p_rl * (logp_rl - logp_sft)).sum(-1).mean()
            kls.append(kl.item())
    return float(sum(kls) / max(len(kls), 1))


def compute_avg_rm_score(
    reward_model: RewardScorer,
    texts: List[str],
) -> float:
    scores = [reward_model.score(t) for t in texts]
    return float(sum(scores) / max(len(scores), 1))
