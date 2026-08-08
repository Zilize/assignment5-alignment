import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel


def get_model(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        dtype=torch.bfloat16,
    )
    return model


def get_tokenizer(model_id_or_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return tokenizer


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase
) -> dict[str, torch.Tensor]:
    prompt_tokens = tokenizer(prompt_strs)['input_ids']
    output_tokens = tokenizer(output_strs)['input_ids']

    merged_tokens = []
    for prompt_token, output_token in zip(prompt_tokens, output_tokens):
        merged_tokens.append(prompt_token + output_token)
    max_prompt_and_output_len = max(len(merged_token) for merged_token in merged_tokens)

    # build mask for labels
    response_mask = []
    for prompt_token, output_token in zip(prompt_tokens, output_tokens):
        prefix = [0] * (len(prompt_token) - 1)
        middle = [1] * len(output_token)
        suffix = [0] * (max_prompt_and_output_len - 1 - len(prefix) - len(middle))
        response_mask.append(prefix + middle + suffix)
    response_mask = torch.tensor(response_mask, dtype=torch.long)

    padded_merged_tokens = [merged_token + [0] * (max_prompt_and_output_len - len(merged_token))
                            for merged_token in merged_tokens]
    padded_merged_tokens = torch.tensor(padded_merged_tokens, dtype=torch.long)

    truncated_merged_tokens = padded_merged_tokens[:, :-1]
    shifted_merged_tokens = padded_merged_tokens[:, 1:]
    return {
        "input_ids": truncated_merged_tokens,
        "labels": shifted_merged_tokens,
        "response_mask": response_mask
    }


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    batch_size, seq_len, vocab_size = log_probs.shape

    batch_indices = torch.tensor(range(batch_size), dtype=torch.long).repeat_interleave(seq_len).reshape(batch_size, seq_len)
    seq_indices = torch.tensor(range(seq_len), dtype=torch.long).repeat(batch_size).reshape(batch_size, seq_len)

    result = {"log_probs": log_probs[batch_indices, seq_indices, labels]}
    if return_token_entropy:
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        result["token_entropy"] = entropy
    return result
