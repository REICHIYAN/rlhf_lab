from typing import Protocol, Tuple


class TextGenerator(Protocol):
    def generate(self, prompt: str, max_new_tokens: int = 128) -> Tuple[str, int]:
        ...


class RewardScorer(Protocol):
    def score(self, text: str) -> float:
        ...
