import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tpo_torch.loss import tpo_loss, tpo_loss_from_logits


class TestTPOLossCore:
    def test_no_nan_on_extreme_advantages(self):
        batch, seq, vocab = 4, 10, 32
        policy_logits = torch.randn(batch, seq, vocab, requires_grad=True)
        ref_logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        # Test very high and very low advantages
        advantages = torch.tensor([10.0, -10.0, 100.0, -100.0])
        
        loss = tpo_loss_from_logits(policy_logits, ref_logits, labels, advantages, beta=0.1)
        assert not torch.isnan(loss), "TPO Loss exploded into NaN!"
        loss.backward()
        assert not torch.isnan(policy_logits.grad).any(), "Gradients resulted in NaN!"

    def test_gradient_proportional_scaling(self):
        """
        ULTIMATE PROOF: Ensure that higher advantages produce larger gradients
        toward the target token.
        """
        torch.manual_seed(42)
        batch, seq, vocab = 1, 1, 10
        # Initialize uniform logits
        policy_logits = torch.zeros(batch, seq + 1, vocab, requires_grad=True)
        ref_logits = torch.zeros(batch, seq + 1, vocab)
        labels = torch.tensor([[5, 5]])
        
        # 1. Zero Advantage
        loss_zero = tpo_loss_from_logits(policy_logits, ref_logits, labels, torch.tensor([0.0]), beta=1.0)
        loss_zero.backward()
        grad_zero = policy_logits.grad.clone()
        policy_logits.grad.zero_()
        
        # 2. High Advantage
        loss_high = tpo_loss_from_logits(policy_logits, ref_logits, labels, torch.tensor([5.0]), beta=1.0)
        loss_high.backward()
        grad_high = policy_logits.grad.clone()
        policy_logits.grad.zero_()
        
        # 3. Negative Advantage
        loss_neg = tpo_loss_from_logits(policy_logits, ref_logits, labels, torch.tensor([-5.0]), beta=1.0)
        loss_neg.backward()
        grad_neg = policy_logits.grad.clone()
        
        # Assertions
        # Magnitudes: High > Zero > Negative (for reinforcing the token)
        sum_high = grad_high.abs().sum().item()
        sum_zero = grad_zero.abs().sum().item()
        sum_neg = grad_neg.abs().sum().item()
        
        assert sum_high > sum_zero, f"High reward ({sum_high}) should pull harder than zero reward ({sum_zero})"
        assert sum_zero > sum_neg, f"Zero reward ({sum_zero}) should pull harder than negative reward ({sum_neg})"
        
        # Direction: Gradient at index 5 should be negative (decreasing loss via increasing logprob)
        assert grad_high[0, 0, 5] < 0
        assert grad_high[0, 0, 5] < grad_zero[0, 0, 5], "High reward should have a more negative gradient (stronger pull)"


class TestTPOMasking:
    def test_padding_masked_out_of_loss(self):
        batch, seq, vocab = 2, 8, 16
        policy_logits = torch.randn(batch, seq, vocab, requires_grad=True)
        ref_logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        advantages = torch.tensor([1.0, 2.0])
        mask = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.float
        )

        loss_with_mask = tpo_loss_from_logits(policy_logits, ref_logits, labels, advantages, beta=0.5, attention_mask=mask)
        loss_with_mask.backward()
        assert torch.all(policy_logits.grad[0, 4:] == 0), "Masked tokens must have ZERO gradient"


class TestTPODataCollator:
    def test_collator_preserves_advantages(self):
        from tpo_torch.trainer import TPODataCollator
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3], "advantages": 0.5},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [4, 5], "advantages": 1.0},
        ]

        collator = TPODataCollator(tokenizer=tokenizer)
        batch = collator(features)
        assert "advantages" in batch
        assert batch["advantages"].shape[0] == 2
        assert batch["advantages"][0].item() == pytest.approx(0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
