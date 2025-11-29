from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class LMBasicMetrics:
    perplexity: Optional[float]
    bertscore_f1: Optional[float]


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    max_length: int = 512,
    device: Optional[str] = None,
) -> float:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    losses = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).to(device)
            labels = enc["input_ids"]
            out = model(**enc, labels=labels)
            losses.append(out.loss.item())

    mean_loss = sum(losses) / max(len(losses), 1)
    ppl = float(torch.exp(torch.tensor(mean_loss)).item())
    return ppl


def compute_bertscore_f1(
    preds: List[str],
    refs: List[str],
    lang: str = "en",
) -> float:
    assert len(preds) == len(refs), "preds and refs must have same length"
    from bert_score import score

    _, _, F1 = score(preds, refs, lang=lang)
    return float(F1.mean().item())
