# Verum LLM

A domain-specific language model trained on the complete TCC (Trajectory Completion Conjecture) framework corpus. This includes 40+ papers, validation code, Rust and Python source code, and experimental results from the Verum project.

## Purpose

The Verum LLM serves as the **Zangalewa AI interceptor layer** — a specialized model that deeply understands the mathematical foundations, categorical structures, and validation methodologies of the TCC framework. Unlike general-purpose LLMs that approximate domain knowledge, the Verum LLM has internalized the precise definitions, theorems, proofs, and code that constitute the framework.

This follows the architecture of [Purpose](https://github.com/fullscreen-triangle/purpose), which trains domain-specific LLMs by fine-tuning base models on curated domain corpora.

## Training Pipeline

### 1. Corpus Preparation

```bash
python prepare_corpus.py
```

Extracts and processes all knowledge sources from the Verum project:
- **LaTeX papers** — stripped of formatting, theorems and equations preserved in readable form
- **Python code** — validation scripts, analysis tools, corpus preparation
- **Rust code** — core framework implementations
- **Markdown documentation** — READMEs, design documents, specifications
- **JSON validation results** — experimental outputs and metrics

Output: `corpus.jsonl` (one document per line, JSON format)

### 2. Training

```bash
python train.py --base-model "microsoft/phi-2" --epochs 3 --lr 2e-4
```

Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning. This modifies only a small subset of model weights while preserving the base model's general language capabilities.

**Requirements:**
```
pip install transformers peft datasets accelerate bitsandbytes torch
```

**Options:**
- `--base-model` — HuggingFace model ID (default: `microsoft/phi-2`)
- `--epochs` — training epochs (default: 3)
- `--lr` — learning rate (default: 2e-4)
- `--lora-r` — LoRA rank (default: 16)
- `--lora-alpha` — LoRA alpha (default: 32)
- `--batch-size` — batch size (default: 4)
- `--max-length` — max sequence length (default: 2048)
- `--dry-run` — validate corpus without training

### 3. Inference

```bash
python inference.py --model verum-model --prompt "What is the universal transport formula?"
```

## Architecture

The training uses LoRA adapters targeting the attention projection layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`). With rank 16 and alpha 32, this adds approximately 0.5% trainable parameters while achieving strong domain adaptation.

The corpus is structured as a causal language modeling task — the model learns to predict the next token given the preceding context from Verum documents. This naturally encodes the relationships between mathematical concepts, code implementations, and validation results.
