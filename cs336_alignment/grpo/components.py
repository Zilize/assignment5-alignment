import torch
from typing import Callable, Literal


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    device = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    total_reward, total_format_reward, total_answer_reward = 0, 0, 0
    rewards = []
    for rollout_response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward = reward_fn(rollout_response, ground_truth)
        rewards.append(reward["reward"])
        total_reward += reward["reward"]
        total_format_reward += reward["format_reward"]
        total_answer_reward += reward["answer_reward"]

    return torch.tensor(rewards, dtype=torch.float, device=device), {
        "mean_reward": total_reward / len(rollout_responses),
        "mean_format_reward": total_format_reward / len(rollout_responses),
        "mean_answer_reward": total_answer_reward / len(rollout_responses),
    }


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    rollout_batch_size = raw_rewards.shape[0]
    reshaped_raw_rewards = raw_rewards.reshape(rollout_batch_size // group_size, group_size)
    if baseline == "mean":
        mean_rewards = reshaped_raw_rewards.mean(dim=-1, keepdim=True)
        shifted_rewards = reshaped_raw_rewards - mean_rewards
    elif baseline == "none":
        shifted_rewards = reshaped_raw_rewards
    else:
        raise NotImplementedError

    if advantage_normalizer == "std":
        std_rewards = reshaped_raw_rewards.std(dim=-1, keepdim=True) + advantage_eps
        rewards = shifted_rewards / std_rewards
    elif advantage_normalizer == "none":
        rewards = shifted_rewards
    elif advantage_normalizer == "mean":
        mean_rewards = reshaped_raw_rewards.mean(dim=-1, keepdim=True) + advantage_eps
        rewards = shifted_rewards / mean_rewards
    else:
        raise NotImplementedError
    return rewards.reshape(rollout_batch_size), {
        "mean_rewards": rewards.mean().item(),
        "std_rewards": rewards.std().item(),
        "max_rewards": rewards.max().item(),
        "min_rewards": rewards.min().item()
    }


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_token_loss = -raw_rewards_or_advantages * policy_log_probs
    if importance_reweighting_method == "none":
        pass
    else:
        raise NotImplementedError
    return per_token_loss, {}


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    response_length = mask.sum(dim=-1)
    response_loss = per_token_policy_gradient_loss.masked_fill(~mask, 0).sum(dim=-1)
    if loss_normalization == "sequence":
        response_loss = (response_loss / response_length).mean()  # .mean() -> */mBG, mB: microBatch
    elif loss_normalization == "constant":
        response_loss = response_loss.sum() / normalization_constant  # assert const == BGL
    else:
        raise NotImplementedError

    return response_loss
