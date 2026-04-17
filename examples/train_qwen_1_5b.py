import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
import numpy as np

from tpo_torch.trainer import TPOTrainer


def create_synthetic_rlhf_dataset(tokenizer, num_samples=100):
    print(f"[*] Generating {num_samples} synthetic RLHF preference samples...")
    data = []
    for i in range(num_samples):
        prompt_text = f"What is Target Policy Optimization? Sample {i}"
        response_text = "TPO is a stable cross-entropy method for RLHF."
        full_text = prompt_text + " " + response_text

        tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=128,
            padding="max_length",
        )

        advantage_score = float(np.random.uniform(0.1, 1.5))

        data.append({
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": tokens["input_ids"],
            "advantages": advantage_score,
        })

    return Dataset.from_list(data)


def main():
    print("[*] TPO REAL-WORLD TRAINING PROOF")
    print("=" * 50)

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"[*] Loading Tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Loading Policy Model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print(f"[*] Loading Reference Model (Frozen): {model_name}")
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)

    print("[*] Creating synthetic RLHF dataset...")
    train_dataset = create_synthetic_rlhf_dataset(tokenizer, num_samples=20)

    print("[*] Configuring TrainingArguments...")
    training_args = TrainingArguments(
        output_dir="./tpo_qwen_results",
        num_train_epochs=1,
        max_steps=5,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_steps=5,
        learning_rate=2e-5,
        remove_unused_columns=False,
        report_to=["none"],
    )

    print("[*] Initializing TPOTrainer with TPO loss...")
    trainer = TPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=0.1,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("[*] Running TPO training steps...")
    train_metrics = trainer.train()

    print()
    print("[*] TRAINING COMPLETE")
    print(f"[*] Final train_loss: {train_metrics.training_loss:.6f}")
    print("[*] Output artifacts saved to ./tpo_qwen_results/")
    print()
    print("TPO is working correctly if:")
    print("  - Loss values are non-NaN")
    print("  - Gradients flow to policy model")
    print("  - Reference model remains frozen")


if __name__ == "__main__":
    main()
