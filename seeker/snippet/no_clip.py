#date: 2025-06-09T16:47:18Z
#url: https://api.github.com/gists/82231ebec3bb38f3fe96bf79b290b0fe
#owner: https://api.github.com/users/radi-cho

def compute_grpo_no_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    b, sequence_length = policy_log_probs.shape
    if advantages.dim() == 2:
        advantages = advantages.squeeze(1)

    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    adv_broad = advantages.unsqueeze(1).expand(-1, sequence_length)
    loss = -ratio * adv_broad

    metadata = {
        "ratio": ratio,
        "mean_ratio": ratio.mean().item(),
        "mean_advantage": advantages.mean().item(),
        "mean_loss": loss.mean().item(),
    }

    return loss, metadata