from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .datasets import PreferenceDataset


@dataclass
class ActivePLConfig:
    lr: float = 1e-5
    batch_size: int = 2
    num_epochs: int = 1
    max_length: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ActivePreferenceLearningTrainer:
    """Simplified Active Preference Learning trainer.

    Here we simulate an "active" step by:
    - Computing the current model's logit gap for each pair
    - Selecting the most uncertain pairs (gap near 0) for an extra epoch of DPO-like training.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: PreferenceDataset,
        config: ActivePLConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config

    def _logit_gap(self, batch) -> torch.Tensor:
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

        with torch.no_grad():
            logits_c = self.model(**enc_c).logits
            logits_r = self.model(**enc_r).logits
            logp_c = torch.log_softmax(logits_c, dim=-1)
            logp_r = torch.log_softmax(logits_r, dim=-1)

            last_ids_c = enc_c["input_ids"][:, -1].unsqueeze(-1)
            last_ids_r = enc_r["input_ids"][:, -1].unsqueeze(-1)
            logp_c_last = logp_c.gather(-1, last_ids_c).squeeze(-1)
            logp_r_last = logp_r.gather(-1, last_ids_r).squeeze(-1)

            gap = logp_c_last - logp_r_last
        return gap.cpu()

    def train(self) -> PreTrainedModel:
        self.model.to(self.config.device)
        self.model.train()

        # Step 1: compute uncertainty (gap magnitude)
        loader_full = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=False)
        gaps = []
        batches = []
        for batch in loader_full:
            gap = self._logit_gap(batch)
            gaps.extend(gap.abs().tolist())
            batches.append(batch)

        if not gaps:
            return self.model

        threshold = sorted(gaps)[max(len(gaps) // 2 - 1, 0)]
        uncertain_indices = [i for i, g in enumerate(gaps) if g <= threshold]

        # Step 2: train DPO-like on uncertain pairs only
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        for idx in uncertain_indices:
            batch = batches[idx]
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

            diff = logp_c_last - logp_r_last
            loss = -torch.mean(torch.log(torch.sigmoid(diff)))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()

        return self.model
