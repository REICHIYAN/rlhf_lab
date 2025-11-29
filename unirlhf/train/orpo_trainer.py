from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .datasets import PreferenceDataset


@dataclass
class ORPOConfig:
    lr: float = 1e-5
    batch_size: int = 2
    num_epochs: int = 1
    max_length: int = 128
    kl_coeff: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ORPOTrainer:
    """Simplified ORPO-like trainer.

    Combines cross-entropy on chosen responses with a KL penalty towards a fixed SFT model.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: PreferenceDataset,
        config: ORPOConfig,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config

    def train(self) -> PreTrainedModel:
        self.model.to(self.config.device)
        self.ref_model.to(self.config.device)
        self.model.train()
        self.ref_model.eval()

        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

        for _ in range(self.config.num_epochs):
            for batch in loader:
                chosen = [p + "\n" + c for p, c in zip(batch["prompt"], batch["chosen"])]

                enc = self.tokenizer(
                    chosen,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                ).to(self.config.device)

                out = self.model(**enc, labels=enc["input_ids"])
                loss_ce = out.loss

                with torch.no_grad():
                    out_ref = self.ref_model(**enc)
                    logp_ref = out_ref.logits.log_softmax(-1)
                logp = out.logits.log_softmax(-1)
                p = logp.exp()
                kl = (p * (logp - logp_ref)).sum(-1).mean()

                loss = loss_ce + self.config.kl_coeff * kl

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

        return self.model
