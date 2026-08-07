import json
from typing import Any, Generator

from cs336_alignment.vllm_utils import VLLMServer, VLLMCompletion
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn


def data_loader(data_file: str, batch_size: int) -> Generator[tuple[list[Any], list[Any]], Any, None]:
    with open(data_file, 'r') as file:
        counter, questions, answers = 0, [], []
        for json_text in file.readlines():
            data_item = json.loads(json_text)
            questions.append(data_item['question'])
            answers.append(data_item['answer'])

            counter += 1
            if counter == batch_size:
                yield questions, answers
                counter, questions, answers = 0, [], []
        yield questions, answers


def evaluate(server, prompt, reward_fn, batch_size):
    counter, format_rewards, answer_rewards = 0, 0, 0
    for questions, answers in data_loader("../../data/gsm8k/test.jsonl", batch_size=batch_size):
        questions = [prompt.replace('{question}', question) for question in questions]
        ground_truths = [answer.split("####")[1].strip() for answer in answers]

        completions = server.generate_completions(questions, {
            "temperature": 1.0,
            "max_tokens": 512,
            "n": 1,
            "seed": 41,
            "top-p": 1.0,
            "stop": ["</answer>"],
            "include_stop_str_in_output": True
        })
        responses = [completion.text for completion in completions]

        for response, ground_truth in zip(responses, ground_truths):
            reward = reward_fn(response, ground_truth)
            format_rewards += reward["format_reward"]
            answer_rewards += reward["answer_reward"]

            # debug
            if reward["format_reward"] == 1 and reward["answer_reward"] == 0:
                pass
        counter += len(responses)
    return format_rewards / counter, answer_rewards / counter


def main():
    server = VLLMServer(model_id="allenai/OLMo-2-0425-1B", gpu=0)
    server.start()

    format_rate, answer_rate = evaluate(server, open('../prompts/question_only.prompt', 'r').read(),
                                        question_only_reward_fn, batch_size=256)
    print(format_rate, answer_rate)
    format_rate, answer_rate = evaluate(server, open('../prompts/r1_zero.prompt', 'r').read(),
                                        r1_zero_reward_fn, batch_size=256)
    print(format_rate, answer_rate)
    format_rate, answer_rate = evaluate(server, open('../prompts/r1_zero_three_shot_gsm8k.prompt', 'r').read(),
                                        r1_zero_reward_fn, batch_size=256)
    print(format_rate, answer_rate)

    server.stop()


if __name__ == '__main__':
    main()
