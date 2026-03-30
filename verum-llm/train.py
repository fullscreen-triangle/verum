"""
Train the Verum-specialized LLM.

Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
on the Verum framework corpus.

Usage:
    python train.py --base-model "microsoft/phi-2" --epochs 3 --lr 2e-4
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Train Verum LLM")
    parser.add_argument("--base-model", default="microsoft/phi-2", help="Base model to fine-tune")
    parser.add_argument("--corpus", default="corpus.jsonl", help="Training corpus path")
    parser.add_argument("--output", default="verum-model", help="Output model directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--dry-run", action="store_true", help="Just validate corpus, don't train")
    args = parser.parse_args()

    # Load corpus
    corpus_path = os.path.join(os.path.dirname(__file__), args.corpus)
    if not os.path.exists(corpus_path):
        print(f"Corpus not found at {corpus_path}. Run prepare_corpus.py first.")
        return

    print(f"Loading corpus from {corpus_path}...")
    documents = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))

    total_chars = sum(len(d["text"]) for d in documents)
    print(f"Loaded {len(documents)} documents ({total_chars:,} chars, ~{total_chars // 4:,} tokens)")

    if args.dry_run:
        print("\nDry run — corpus validated. Ready for training.")
        print(f"\nTo train, run:")
        print(f"  python train.py --base-model {args.base_model}")
        print(f"\nRequirements:")
        print(f"  pip install transformers peft datasets accelerate bitsandbytes")
        return

    # Training requires GPU + transformers. Check availability.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install transformers peft datasets accelerate bitsandbytes torch")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: Training on CPU will be very slow. GPU recommended.")

    # Prepare dataset
    print("Preparing dataset...")
    texts = [d["text"] for d in documents if len(d["text"]) > 100]
    dataset = Dataset.from_dict({"text": texts})

    # Load tokenizer and model
    print(f"Loading base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # Configure LoRA
    print(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Training
    print(f"Training for {args.epochs} epochs...")
    training_args = TrainingArguments(
        output_dir=os.path.join(os.path.dirname(__file__), args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        fp16=(device == "cuda"),
        report_to="none",
    )

    from transformers import Trainer, DataCollatorForLanguageModeling
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    # Save
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    main()
