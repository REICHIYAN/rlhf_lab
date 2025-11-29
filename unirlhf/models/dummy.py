from typing import Tuple
from .interfaces import TextGenerator, RewardScorer


class EchoGenerator(TextGenerator):
    def __init__(self, name: str, suffix: str):
        self.name = name
        self.suffix = suffix

    def generate(self, prompt: str, max_new_tokens: int = 128) -> Tuple[str, int]:
        output = prompt + " " + self.suffix
        num_tokens = len(output.split())
        return output, num_tokens


class LengthRewardScorer(RewardScorer):
    def score(self, text: str) -> float:
        return float(len(text))
