# TPO-Torch 🎯

Target Policy Optimization (TPO) - An experimental RLHF implementation using cross-entropy with advantage-weighted target distributions.

## ⚠️ Important Caveats

- **NOT peer-reviewed** - No academic paper or published benchmarks
- **Unverified claims** - Stability/performance vs PPO has NOT been independently validated
- **Experimental** - Use at your own risk for research purposes

## What is TPO?

TPO is a training method that computes target probabilities for tokens based on reference policy + advantage signals, then uses cross-entropy loss to fit the policy toward those targets.

```python
target_prob = sigmoid(log_odds(P_ref) + advantage/beta)
loss = -target_prob * log P_policy(token)
```

## Installation

```bash
pip install tpo-torch
```

## Quick Start

```python
from tpo_torch import TPOTrainer
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B")
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B")

trainer = TPOTrainer(
    model=model,
    ref_model=ref_model,
    beta=0.1,
    train_dataset=my_rewarded_dataset,
)

trainer.train()
```

**Requirements:**
- `advantages` column in your dataset (float: higher = better response)
- Reference model (frozen) for computing baseline distributions

## Core Features

- PEFT/LoRA support via `peft` package
- Integrates with HuggingFace Trainer
- Sequence-level and token-level advantages
- Custom data collator that preserves advantages

## Verification Status

| Component | Status |
|-----------|--------|
| Loss function computes | ✅ Verified |
| Gradients flow | ✅ Verified |
| Trainer integration | ✅ Verified |
| Beats PPO stability | ❌ Unverified |
| Benchmarks | ❌ Not provided |

## Roadmap

- [ ] Independent benchmarks vs PPO/GRPO
- [ ] Peer review or academic citation
- [ ] Multi-GPU support

## License

MIT

## Disclaimer

This is a research implementation. No guarantees about performance compared to PPO, GRPO, or other established RLHF methods. The "stability" claims are theoretical/hypothetical, not empirically validated.