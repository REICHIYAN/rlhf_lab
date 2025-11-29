from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class PPORecord:
    prompt: str
    response: str
    old_logprob: float
    reward: float


class PPODataset(Dataset):
    def __init__(self, records: List[PPORecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]
        return {
            "prompt": r.prompt,
            "response": r.response,
            "old_logprob": torch.tensor(r.old_logprob, dtype=torch.float32),
            "reward": torch.tensor(r.reward, dtype=torch.float32),
        }


@dataclass
class PPOConfig:
    lr: float = 1e-5
    batch_size: int = 2
    num_epochs: int = 1
    max_length: int = 128
    clip_range: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PPOTrainer:
    """Very small, didactic PPO-style trainer for offline RLHF.

    This is *not* a full PPO pipeline (no value function, no GAE).
    It uses an advantage = reward baseline-subtracted and a clipped
    log-prob ratio objective.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: PPODataset,
        config: PPOConfig,
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

        # simple baseline: mean reward
        rewards = torch.stack([b["reward"] for b in self.dataset])  # type: ignore
        baseline = rewards.mean().item()

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

                out = self.model(**enc)
                logits = out.logits
                logprobs = torch.log_softmax(logits, dim=-1)

                # Take the logprob of the last token as a simple surrogate.
                last_token_ids = enc["input_ids"][:, -1].unsqueeze(-1)
                last_logprobs = logprobs.gather(-1, last_token_ids).squeeze(-1)

                old_logprob = batch["old_logprob"].to(self.config.device)
                reward = batch["reward"].to(self.config.device)
                advantage = reward - baseline

                ratio = torch.exp(last_logprobs - old_logprob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * advantage
                loss = -torch.mean(torch.min(surr1, surr2))

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

        return self.model
