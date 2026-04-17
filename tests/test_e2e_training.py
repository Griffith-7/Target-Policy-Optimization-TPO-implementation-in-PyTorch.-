import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestTPOEndToEnd:
    def test_trainer_uses_correct_loss_function(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from datasets import Dataset
        from tpo_torch.trainer import TPOTrainer

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name)
        ref_model = AutoModelForCausalLM.from_pretrained(model_name)

        data = []
        for i in range(4):
            tokens = tokenizer(f"Sample {i}", truncation=True, max_length=32, padding="max_length")
            data.append({
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "labels": tokens["input_ids"],
                "advantages": float(i % 2) + 0.1,
            })

        train_dataset = Dataset.from_list(data)

        args = TrainingArguments(
            output_dir="./test_output",
            max_steps=2,
            per_device_train_batch_size=2,
            logging_steps=1,
            save_steps=999,
            learning_rate=2e-5,
            remove_unused_columns=False,
            report_to=["none"],
        )

        trainer = TPOTrainer(
            model=model,
            ref_model=ref_model,
            beta=0.1,
            args=args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

        losses = []

        original_compute = trainer.compute_loss

        def capturing_compute_loss(mdl, inp, ret=False, num_items_in_batch=None):
            loss, _ = original_compute(mdl, inp, return_outputs=True, num_items_in_batch=num_items_in_batch)
            losses.append(loss.item())
            return loss

        trainer.compute_loss = capturing_compute_loss

        trainer.train()

        assert len(losses) >= 1, "Trainer should have computed at least 1 loss"
        assert all(not (l != l) for l in losses), "No NaN losses"
        assert all(l >= -1e-6 for l in losses), f"All losses should be >= ~0 (floating noise OK), got {losses}"
        print(f"\nEnd-to-end losses: {losses}")

    def test_no_ref_model_uses_policy_as_ref(self):
        from tpo_torch.trainer import TPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from datasets import Dataset

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name)

        data = []
        for i in range(2):
            tokens = tokenizer(f"Test {i}", truncation=True, max_length=16, padding="max_length")
            data.append({
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "labels": tokens["input_ids"],
                "advantages": 0.5,
            })

        train_dataset = Dataset.from_list(data)

        args = TrainingArguments(
            output_dir="./test_output",
            max_steps=1,
            per_device_train_batch_size=2,
            save_steps=999,
            remove_unused_columns=False,
            report_to=["none"],
        )

        trainer = TPOTrainer(
            model=model,
            ref_model=None,
            beta=0.1,
            args=args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

        assert trainer.ref_model is None

        losses = []
        original_compute = trainer.compute_loss

        def capture_loss(mdl, inp, ret=False, num_items_in_batch=None):
            loss, _ = original_compute(mdl, inp, return_outputs=True, num_items_in_batch=num_items_in_batch)
            losses.append(loss.item())
            return loss

        trainer.compute_loss = capture_loss
        trainer.train()

        assert len(losses) == 1
        assert not (losses[0] != losses[0]), "Loss should not be NaN"
        assert losses[0] >= -1e-6, f"Loss should be >= ~0, got {losses[0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
