import json
import wandb
import torch
import random
import argparse

from tqdm import tqdm

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.grpo.components import compute_rollout_rewards
from cs336_alignment.grpo.step import grpo_train_step
from cs336_alignment.grpo.tokenizer import get_response_log_probs, tokenize_prompt_and_output
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.checkpoint import get_model_and_tokenizer


def fetch_prompt_and_reward_fn(prompt_type):
    assert prompt_type in ["question_only", "zero_shot", "three_shot"]
    if prompt_type == "question_only":
        prompt = open('cs336_alignment/prompts/question_only.prompt', 'r').read().strip()
        reward_fn = question_only_reward_fn
    elif prompt_type == "zero_shot":
        prompt = open('cs336_alignment/prompts/r1_zero.prompt', 'r').read().strip()
        reward_fn = r1_zero_reward_fn
    else:
        prompt = open('cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt', 'r').read().strip()
        reward_fn = r1_zero_reward_fn
    return prompt, reward_fn


def fetch_data(data_file, num_examples):
    with open(data_file, 'r') as file:
        data_items = []
        for json_text in file.readlines():
            data_item = json.loads(json_text)
            data_items.append(data_item)
            if len(data_items) == num_examples:
                break
        return data_items


def fetch_method_config(method):
    if method == "grpo":
        return {
            "baseline": "mean",
            "advantage_eps": 1e-6,
            "advantage_normalizer": "std",
            "importance_reweighting_method": "none",
            "cliprange": None,
            "loss_normalization": "sequence",
        }
    elif method == "grpo_constant":
        return {
            "baseline": "mean",
            "advantage_eps": 1e-6,
            "advantage_normalizer": "std",
            "importance_reweighting_method": "none",
            "cliprange": None,
            "loss_normalization": "constant",
        }
    elif method == "dr_grpo":
        return {
            "baseline": "mean",
            "advantage_eps": 1e-6,
            "advantage_normalizer": "none",
            "importance_reweighting_method": "none",
            "cliprange": None,
            "loss_normalization": "constant",
        }
    elif method == "rft":
        return {
            "baseline": "none",
            "advantage_eps": 1e-6,
            "advantage_normalizer": "none",
            "importance_reweighting_method": "none",
            "cliprange": None,
            "loss_normalization": "constant",
        }
    elif method == "maxrl":
        return {
            "baseline": "mean",
            "advantage_eps": 1e-6,
            "advantage_normalizer": "mean",
            "importance_reweighting_method": "none",
            "cliprange": None,
            "loss_normalization": "constant",
        }
    elif method == "off_policy_naive":
        return {
            "baseline": "mean",
            "advantage_eps": 1e-6,
            "advantage_normalizer": "std",
            "importance_reweighting_method": "none",
            "cliprange": None,
            "loss_normalization": "sequence",
        }
    elif method == "off_policy_noclip":
        return {
            "baseline": "mean",
            "advantage_eps": 1e-6,
            "advantage_normalizer": "std",
            "importance_reweighting_method": "noclip",
            "cliprange": None,
            "loss_normalization": "sequence",
        }
    elif method == "off_policy_grpo":
        return {
            "baseline": "mean",
            "advantage_eps": 1e-6,
            "advantage_normalizer": "std",
            "importance_reweighting_method": "grpo",
            "cliprange": 0.2,
            "loss_normalization": "sequence",
        }
    elif method == "off_policy_gspo":
        return {
            "baseline": "mean",
            "advantage_eps": 1e-6,
            "advantage_normalizer": "std",
            "importance_reweighting_method": "gspo",
            "cliprange": 3e-4,
            "loss_normalization": "sequence",
        }
    else:
        raise NotImplementedError


def data_sampler(data, prompt, batch_size):
    while True:
        random.shuffle(data)
        questions, ground_truths = [], []
        for data_item in data:
            questions.append(prompt.format(question=data_item["question"]))
            ground_truths.append(data_item["answer"].split("####")[1].strip())
            if len(questions) == batch_size:
                yield questions, ground_truths
                questions, ground_truths = [], []


def validate_model(server, sampling_params, reward_fn, valid_sampler, valid_step, num_logged_samples=0):
    count, total_reward, total_format_reward, total_answer_reward = 0, 0.0, 0.0, 0.0
    total_rollout_length = 0
    candidates = []
    for _ in range(valid_step):
        questions, ground_truths = next(valid_sampler)

        completions = server.generate_completions(questions, sampling_params)
        rollout_responses = [completion.text for completion in completions]
        rollout_lengths = [len(completion.token_ids) for completion in completions]
        total_rollout_length += sum(rollout_lengths)

        _, stats = compute_rollout_rewards(reward_fn, rollout_responses, ground_truths)

        count += len(rollout_responses)
        total_reward += stats["mean_reward"] * len(rollout_responses)
        total_format_reward += stats["mean_format_reward"] * len(rollout_responses)
        total_answer_reward += stats["mean_answer_reward"] * len(rollout_responses)

        if num_logged_samples > 0:
            candidates.extend(zip(questions, ground_truths, rollout_responses, rollout_lengths))

    samples = []
    for question, ground_truth, response, length in random.sample(candidates, min(num_logged_samples,
                                                                                  len(candidates))):
        reward = reward_fn(response, ground_truth)
        samples.append({
            "question": question,
            "ground_truth": ground_truth,
            "response": response,
            "length": length,
            "reward": reward["reward"],
            "format_reward": reward["format_reward"],
            "answer_reward": reward["answer_reward"],
        })

    return {
        "mean_reward": total_reward / count,
        "mean_format_reward": total_format_reward / count,
        "mean_answer_reward": total_answer_reward / count,
        "mean_rollout_length": total_rollout_length / count
    }, samples


def train(args):
    run = wandb.init(
        entity='zilize',
        project='grpo',
        name=args.exp_name,
        config=dict(vars(args))
    )

    prompt, reward_fn = fetch_prompt_and_reward_fn(args.prompt)
    train_data = fetch_data(args.train_data, args.n_train_examples)
    valid_data = fetch_data(args.valid_data, args.n_val_examples)

    assert args.rollout_batch_size % args.group_size == 0
    prompt_batch_size = args.rollout_batch_size // args.group_size
    train_sampler = data_sampler(train_data, prompt, prompt_batch_size)
    valid_sampler = data_sampler(valid_data, prompt, args.valid_batch_size)

    model, tokenizer = get_model_and_tokenizer(args.model, f'cuda:{args.policy_device}')
    model.eval()  # 确保log-prob和vllm推理时计算方式一致，避免dropout影响
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2),
                                  weight_decay=args.weight_decay)

    server = VLLMServer(model_id=args.model, port=args.vllm_port, gpu=args.rollout_device, seed=args.sampling_seed,
                        gpu_memory_utilization=0.9)
    server.start()
    server.init_weight_sync(policy_device=args.policy_device)

    model_device = next(model.parameters()).device
    train_sampling_params = {
        "temperature": args.sampling_temperature,
        "max_tokens": args.sampling_max_tokens,
        "min_tokens": args.sampling_min_tokens,
        "n": args.group_size,
        "seed": args.sampling_seed,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True
    }
    valid_sampling_params = {
        "temperature": args.sampling_temperature,
        "max_tokens": args.sampling_max_tokens,
        "min_tokens": args.sampling_min_tokens,
        "n": 1,
        "seed": args.sampling_seed,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True
    }
    method_config = fetch_method_config(args.method)
    off_policy = True if args.method in ["off_policy_naive", "off_policy_noclip", "off_policy_grpo",
                                         "off_policy_gspo"] else False

    sample_columns = ["step", "question", "ground_truth", "response", "length",
                      "reward", "format_reward", "answer_reward"]
    sample_rows = []

    # training loop
    if off_policy:
        total_steps = args.num_rollout_steps * (args.rollout_batch_size // args.train_batch_size) * args.gradient_accumulation_steps
    else:
        total_steps = args.num_rollout_steps * args.gradient_accumulation_steps
    pbar = tqdm(total=total_steps)

    for rollout_step in range(args.num_rollout_steps):
        server.sync_policy_weights(model)

        # validation
        if rollout_step % args.validation_intervals == 0:
            assert args.n_val_examples % args.valid_batch_size == 0

            valid_step = rollout_step if not off_policy else rollout_step * (args.rollout_batch_size // args.train_batch_size)
            valid_stats, valid_samples = validate_model(server, valid_sampling_params, reward_fn, valid_sampler,
                                                        args.n_val_examples // args.valid_batch_size,
                                                        args.n_logged_samples)
            valid_logs = {
                "valid/mean_reward": valid_stats["mean_reward"],
                "valid/mean_format_reward": valid_stats["mean_format_reward"],
                "valid/mean_answer_reward": valid_stats["mean_answer_reward"],
                "valid/mean_rollout_length": valid_stats["mean_rollout_length"],
            }
            if valid_samples:
                sample_rows.extend([[valid_step] + [sample[column] for column in sample_columns[1:]]
                                    for sample in valid_samples])
                valid_logs["valid/samples"] = wandb.Table(columns=sample_columns, data=sample_rows)
            run.log(valid_logs, step=valid_step)

        questions, ground_truths = next(train_sampler)
        repeated_prompts = [question for question in questions for _ in range(args.group_size)]
        repeated_ground_truths = [ground_truth for ground_truth in ground_truths for _ in range(args.group_size)]

        completions = server.generate_completions(questions, train_sampling_params)
        rollout_responses = [completion.text for completion in completions]

        if not off_policy:
            loss, train_stats = grpo_train_step(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                reward_fn=reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=args.group_size,
                baseline=method_config["baseline"],
                advantage_eps=method_config["advantage_eps"],
                advantage_normalizer=method_config["advantage_normalizer"],
                importance_reweighting_method=method_config["importance_reweighting_method"],
                old_log_probs=None,
                cliprange=method_config["cliprange"],
                loss_normalization=method_config["loss_normalization"],
                normalization_constant=args.rollout_batch_size * args.sampling_max_tokens,
                pbar=pbar,
                device=model_device,
            )
            run.log({
                "train/loss": loss.item(),
                "train/gradient_norm": train_stats["gradient_norm"],
                "train/mean_reward": train_stats["mean_reward"],
                "train/mean_format_reward": train_stats["mean_format_reward"],
                "train/mean_answer_reward": train_stats["mean_answer_reward"],
                "train/mean_token_entropy": train_stats["mean_token_entropy"],
            }, step=rollout_step)
        else:
            # off-policy train loop: (args.rollout_batch_size // args.train_batch_size) times
            assert args.rollout_batch_size % args.train_batch_size == 0
            assert args.train_batch_size % args.gradient_accumulation_steps == 0
            train_step_per_inference = args.rollout_batch_size // args.train_batch_size
            num_samples_per_train_step = args.train_batch_size

            # get old log probs
            # 按 train step 分块 tokenize，保证 padding 长度与 grpo_train_step 内部对同一块的 tokenize 结果一致
            micro_batch_size = num_samples_per_train_step // args.gradient_accumulation_steps
            old_log_probs_chunks = []
            with torch.no_grad():
                for train_step in range(train_step_per_inference):
                    index_start = train_step * num_samples_per_train_step
                    index_end = (train_step + 1) * num_samples_per_train_step
                    tokenization_result = tokenize_prompt_and_output(repeated_prompts[index_start: index_end],
                                                                     rollout_responses[index_start: index_end],
                                                                     tokenizer, device=model_device)
                    input_ids = tokenization_result["input_ids"]
                    labels = tokenization_result["labels"]
                    old_log_probs_chunks.append(torch.cat([
                        get_response_log_probs(model, input_ids[i: i + micro_batch_size],
                                               labels[i: i + micro_batch_size],
                                               device=model_device)["log_probs"]
                        for i in range(0, num_samples_per_train_step, micro_batch_size)
                    ], dim=0))  # (num_samples_per_train_step, max_seq_len of this chunk)

            for train_step in range(train_step_per_inference):
                index_start = train_step * num_samples_per_train_step
                index_end = (train_step + 1) * num_samples_per_train_step
                loss, train_stats = grpo_train_step(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    max_grad_norm=args.max_grad_norm,
                    reward_fn=reward_fn,
                    repeated_prompts=repeated_prompts[index_start: index_end],
                    rollout_responses=rollout_responses[index_start: index_end],
                    repeated_ground_truths=repeated_ground_truths[index_start: index_end],
                    group_size=args.group_size,
                    baseline=method_config["baseline"],
                    advantage_eps=method_config["advantage_eps"],
                    advantage_normalizer=method_config["advantage_normalizer"],
                    importance_reweighting_method=method_config["importance_reweighting_method"],
                    old_log_probs=old_log_probs_chunks[train_step],
                    cliprange=method_config["cliprange"],
                    loss_normalization=method_config["loss_normalization"],
                    normalization_constant=None,
                    pbar=pbar,
                    device=model_device,
                )
                run.log({
                    "train/loss": loss.item(),
                    "train/gradient_norm": train_stats["gradient_norm"],
                    "train/mean_reward": train_stats["mean_reward"],
                    "train/mean_format_reward": train_stats["mean_format_reward"],
                    "train/mean_answer_reward": train_stats["mean_answer_reward"],
                    "train/mean_token_entropy": train_stats["mean_token_entropy"],
                }, step=rollout_step * train_step_per_inference + train_step)

    server.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--policy_device', type=int, default=0)
    parser.add_argument('--rollout_device', type=int, default=1)
    parser.add_argument('--vllm_port', type=int, default=None)

    parser.add_argument('--model', type=str, default='../model')
    parser.add_argument('--prompt', type=str, default='zero_shot')
    parser.add_argument('--method', type=str, default='grpo')

    parser.add_argument('--train_data', type=str, default='data/gsm8k/train.jsonl')
    parser.add_argument('--valid_data', type=str, default='data/gsm8k/test.jsonl')
    parser.add_argument('--n_train_examples', type=int, default=6400)
    parser.add_argument('--n_val_examples', type=int, default=1024)
    parser.add_argument('--n_logged_samples', type=int, default=8)

    parser.add_argument('--sampling_temperature', type=float, default=1.0)
    parser.add_argument('--sampling_max_tokens', type=int, default=512)
    parser.add_argument('--sampling_min_tokens', type=int, default=1)
    parser.add_argument('--sampling_seed', type=int, default=42)

    parser.add_argument('--num_rollout_steps', type=int, default=200)
    parser.add_argument('--validation_intervals', type=int, default=10)
    parser.add_argument('--rollout_batch_size', type=int, default=256)
    parser.add_argument('--train_batch_size', type=int, default=8)  # only for off-policy
    parser.add_argument('--group_size', type=int, default=8)
    parser.add_argument('--valid_batch_size', type=int, default=1024)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)

    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.95)

    args = parser.parse_args()
    if args.vllm_port is None:
        # 端口按 rollout GPU 区分，否则多组实验会抢同一个 server 并互相 pkill
        args.vllm_port = 8000 + args.rollout_device
    train(args)
