import torch.nn.functional as F

def tpo_loss(logits, target_probs):
    return -torch.sum(target_probs * F.log_softmax(logits, dim=-1), dim=-1)
