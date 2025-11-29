import os
from copy import deepcopy

from transformers import AutoModelForCausalLM, AutoTokenizer

from unirlhf.train.datasets import load_sft_jsonl, load_pref_jsonl, SFTDataset, PreferenceDataset
from unirlhf.train.sft_trainer import SFTTrainer, SFTConfig
from unirlhf.train.dpo_trainer import DPOTrainer, DPOConfig


def test_sft_and_dpo_smoke():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    td = os.path.join(base_dir, "test_data")

    sft_records = load_sft_jsonl(os.path.join(td, "sft_train.jsonl"))
    pref_records = load_pref_jsonl(os.path.join(td, "pref_train.jsonl"))

    sft_ds = SFTDataset(sft_records)
    pref_ds = PreferenceDataset(pref_records)

    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model_sft = deepcopy(model)
    trainer_sft = SFTTrainer(
        model=model_sft,
        tokenizer=tokenizer,
        dataset=sft_ds,
        config=SFTConfig(num_epochs=1, batch_size=1),
    )
    model_sft = trainer_sft.train()

    model_dpo = deepcopy(model_sft)
    trainer_dpo = DPOTrainer(
        model=model_dpo,
        tokenizer=tokenizer,
        dataset=pref_ds,
        config=DPOConfig(num_epochs=1, batch_size=1),
    )
    model_dpo = trainer_dpo.train()
