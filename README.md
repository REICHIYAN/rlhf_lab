# RLHF-Lab

RLHF-Lab is a **unified laboratory** for experimenting with RLHF-style methods under a
*single, consistent pipeline*.

It provides **toy but faithful** implementations (for small models and tiny datasets) of:

- **SFT** (Supervised Fine-Tuning)
- **PPO-style RLHF** (InstructGPT-like, simplified)
- **DPO** (Direct Preference Optimization, simplified no-ref variant)
- **IPO** (margin-based preference optimization, simplified)
- **ORPO** (SFT + KL penalty to reference model)
- **RLAIF** (AI feedback; implemented as DPO with AI-labeled preferences)
- **Active Preference Learning** (uncertainty-based pair selection, simplified)

On top of these, RLHF-Lab provides a **shared evaluation toolkit**:

1. **LM Basic Quality** — Perplexity (PPL), BERTScore
2. **Preference Fit** — Win Rate, Bradley–Terry score
3. **Robustness** — Self-consistency entropy, injection success rate
4. **Reward Consistency** — KL to SFT, average reward-model score
5. **Compute Efficiency** — Latency, tokens/sec, approx FLOPs/token (placeholder)

> ⚠️ Scope & Intended Use
>
> - Designed for **research prototypes, teaching, and benchmarking on tiny models**
> - Not intended as a production RLHF library or large-scale training framework
> - Uses only free/open-source dependencies (PyTorch, HuggingFace, BERTScore, NumPy, etc.)

---

## 1. Installation (local or Colab, free)

### From source (recommended for research)

```bash
git clone https://github.com/REICHIYAN/rlhf_lab.git
cd rlhf_lab
pip install -e .
```

or, inside the project root:

```bash
pip install .
```

### Dependencies

Core dependencies are:

- `torch`
- `transformers`
- `bert-score`
- `numpy`
- `pandas`

These are declared in `pyproject.toml` and `requirements.txt`.

---

## 2. Project structure

```text
rlhf_lab/
  pyproject.toml
  README.md
  requirements.txt
  LICENSE

  unirlhf/
    __init__.py

    data/
      __init__.py
      schemas.py

    models/
      __init__.py
      interfaces.py
      dummy.py

    eval/
      __init__.py
      lm_basic.py
      preference.py
      robustness.py
      reward_consistency.py
      compute_efficiency.py
      runner.py

    train/
      __init__.py
      datasets.py
      sft_trainer.py
      ppo_trainer.py
      dpo_trainer.py
      ipo_trainer.py
      orpo_trainer.py
      rlaif_trainer.py
      active_pl_trainer.py

  examples/
    run_dummy_evaluation.py
    run_all_methods_tiny_gpt2.py

  test_data/
    prompts.jsonl
    injection_base_prompts.jsonl
    comparisons.jsonl
    sft_train.jsonl
    pref_train.jsonl

  tests/
    unit/
      test_basic_flow.py
    integration/
      test_train_smoke.py
```

---

## 3. How to run everything in Colab

1. Upload the zip (`rlhf_lab.zip`) and unzip:

```bash
!unzip rlhf_lab.zip -d .
%cd rlhf_lab
!pip install -e .
```

2. Run **all methods training + evaluation** (tiny GPT-2):

```bash
python -m examples.run_all_methods_tiny_gpt2
```

This will:

- Download a tiny causal LM (`sshleifer/tiny-gpt2`)
- Train small models for:
  - SFT
  - PPO-style RLHF
  - DPO / IPO / ORPO
  - RLAIF
  - Active Preference Learning
- Run the unified evaluator (`UnifiedEvaluator`) on these models
- Print a **comparison table** over all 5 metric groups

3. To test the evaluation pipeline alone (no HF / internet needed):

```bash
python -m examples.run_dummy_evaluation
```

---

## 4. Running tests

Unit tests (no external downloads):

```bash
pytest tests/unit
```

Integration tests (downloads tiny HF model):

```bash
pytest tests/integration
```

---

## 5. License

This project is distributed under the **MIT License**. See `LICENSE` for details.

---

## 6. Citation (example)

If you use RLHF-Lab in academic work, you might cite it informally as:

> RLHF-Lab: A Unified Laboratory for RLHF-style Methods  
> R. Taguchi, 2025.  
> https://github.com/REICHIYAN/rlhf_lab

Adjust the author / URL as appropriate.
