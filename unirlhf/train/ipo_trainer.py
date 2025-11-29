from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .datasets import PreferenceDataset


@dataclass
class IPOConfig:
    lr: float = 1e-5
    batch_size: int = 2
    num_epochs: int = 1
    max_length: int = 128
    margin: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class IPOTrainer:
    """Simplified IPO-like trainer.

    We enforce a margin between chosen and rejected log-prob via a hinge loss.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: PreferenceDataset,
        config: IPOConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config

    def train(self) -> PreTrainedModel:
        self.model.to(self.config.device)
        self.model.train()

        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

        for _ in range(self.config.num_epochs):
            for batch in loader:
                chosen = [p + "\n" + c for p, c in zip(batch["prompt"], batch["chosen"])]
                rejected = [p + "\n" + r for p, r in zip(batch["prompt"], batch["rejected"])]

                enc_c = self.tokenizer(
                    chosen,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                ).to(self.config.device)
                enc_r = self.tokenizer(
                    rejected,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                ).to(self.config.device)

                logits_c = self.model(**enc_c).logits
                logits_r = self.model(**enc_r).logits

                logp_c = torch.log_softmax(logits_c, dim=-1)
                logp_r = torch.log_softmax(logits_r, dim=-1)

                last_ids_c = enc_c["input_ids"][:, -1].unsqueeze(-1)
                last_ids_r = enc_r["input_ids"][:, -1].unsqueeze(-1)
                logp_c_last = logp_c.gather(-1, last_ids_c).squeeze(-1)
                logp_r_last = logp_r.gather(-1, last_ids_r).squeeze(-1)

                margin_violation = self.config.margin - (logp_c_last - logp_r_last)
                loss = torch.mean(torch.clamp(margin_violation, min=0.0))

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

        return self.model
