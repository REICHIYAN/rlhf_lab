import json
from dataclasses import dataclass
from typing import List, Dict, Any

from torch.utils.data import Dataset


@dataclass
class SFTRecord:
    prompt: str
    response: str


@dataclass
class PreferenceRecord:
    prompt: str
    chosen: str
    rejected: str


class SFTDataset(Dataset):
    def __init__(self, records: List[SFTRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        return {"prompt": r.prompt, "response": r.response}


class PreferenceDataset(Dataset):
    def __init__(self, records: List[PreferenceRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        return {"prompt": r.prompt, "chosen": r.chosen, "rejected": r.rejected}


def load_sft_jsonl(path: str) -> List[SFTRecord]:
    records: List[SFTRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            records.append(SFTRecord(prompt=obj["prompt"], response=obj["response"]))
    return records


def load_pref_jsonl(path: str) -> List[PreferenceRecord]:
    records: List[PreferenceRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            records.append(
                PreferenceRecord(
                    prompt=obj["prompt"],
                    chosen=obj["chosen"],
                    rejected=obj["rejected"],
                )
            )
    return records
