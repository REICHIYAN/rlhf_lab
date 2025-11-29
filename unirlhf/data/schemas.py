from dataclasses import dataclass
from typing import Optional, Literal, Dict


@dataclass
class PromptExample:
    id: str
    prompt: str
    reference: Optional[str] = None
    meta: Optional[Dict] = None


@dataclass
class GeneratedExample:
    method: str
    prompt_id: str
    output: str
    num_tokens: int
    meta: Optional[Dict] = None


@dataclass
class PairwiseComparison:
    prompt_id: str
    method_a: str
    method_b: str
    winner: Literal["A", "B", "tie"]
    meta: Optional[Dict] = None


@dataclass
class InjectionConfig:
    injection_suffix: str


@dataclass
class InjectionExample:
    id: str
    base_prompt: str
