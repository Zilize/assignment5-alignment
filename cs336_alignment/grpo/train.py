from typing import Callable, Literal

import torch
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase


from cs336_alignment.grpo.tokenizer import tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.grpo.components import (compute_rollout_rewards, compute_group_normalized_rewards,
                                             compute_policy_gradient_loss, aggregate_loss_across_microbatch)


def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    tokenization_result = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = tokenization_result["input_ids"]
    labels = tokenization_result["labels"]
    response_mask = tokenization_result["response_mask"]

    rollout_rewards, _ = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    rollout_rewards, _ = compute_group_normalized_rewards(rollout_rewards, group_size, baseline, advantage_eps,
                                                          advantage_normalizer)

    micro_batch_size = len(repeated_prompts) // gradient_accumulation_steps
    total_loss = torch.zeros((1,), device=model.device)
    for i in range(0, len(repeated_prompts), micro_batch_size):
        inputs_micro_batch = input_ids[i: i + micro_batch_size]
        labels_micro_batch = labels[i: i + micro_batch_size]
        mask_micro_batch = response_mask[i: i + micro_batch_size]
        rewards_micro_batch = rollout_rewards[i: i + micro_batch_size].unsqueeze(1)

        get_result = get_response_log_probs(model, inputs_micro_batch, labels_micro_batch)
        log_probs = get_result["log_probs"]  # (micro_batch_size, seq_len)

        per_token_loss, _ = compute_policy_gradient_loss(rewards_micro_batch, log_probs, importance_reweighting_method,
                                                         old_log_probs, cliprange, mask_micro_batch)
        loss = aggregate_loss_across_microbatch(per_token_loss, mask_micro_batch, loss_normalization,
                                                normalization_constant)

        loss = loss * len(inputs_micro_batch) / len(repeated_prompts)
        loss.backward()

        total_loss += loss.detach()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    return total_loss, {}
