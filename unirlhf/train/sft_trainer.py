from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from .datasets import SFTDataset


@dataclass
class SFTConfig:
    lr: float = 1e-4
    batch_size: int = 2
    num_epochs: int = 1
    max_length: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SFTTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: SFTDataset,
        config: SFTConfig,
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
        total_steps = len(loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            opt, num_warmup_steps=0, num_training_steps=max(total_steps, 1)
        )

        for _ in range(self.config.num_epochs):
            for batch in loader:
                texts = [p + "\n" + r for p, r in zip(batch["prompt"], batch["response"])]
                enc = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                ).to(self.config.device)
                labels = enc["input_ids"].clone()
                out = self.model(**enc, labels=labels)
                loss = out.loss

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                scheduler.step()

        return self.model
