import argparse
import asyncio
import re
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DPO dataset from reasoning traces.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--dataset_id", type=str, default="lightonai/DPO_Native_Reasoning_FR_EN_chat_template")
    parser.add_argument("--output_repo", type=str, default="lightonai/OPSD-DPO-dataset-olmo")
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--max_model_len", type=int, default=20000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--concurrency", type=int, default=256, help="Max concurrent requests to vLLM")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    return parser.parse_args()


def clean_reasoning_tags(text: str) -> str:
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = cleaned.replace('<think>', '').replace('</think>', '')
    return cleaned.strip()


def prepare_tasks(dataset, tokenizer, max_input_tokens):
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}],
        tokenize=False, add_generation_prompt=True,
    )
    template_adds_think = dummy.rstrip().endswith("<think>")

    tasks = []
    for i in range(len(dataset["id"])):
        row_id = dataset["id"][i]
        lang = dataset["language"][i]
        prompt_msgs = dataset["prompt"][i]

        clean_answer = clean_reasoning_tags(dataset["chosen"][i][-1]["content"])

        # Build demonstration prompt with answer in context
        query = prompt_msgs[-1]["content"]
        demo_content = (
            f"{query}\n\n"
            f"This is an example for a response to the question:\n"
            f"{clean_answer}\n\n"
            f"Now answer with a response of your own, including the thinking process:\n"
        )
        demo_msgs = prompt_msgs[:-1] + [{"role": "user", "content": demo_content}]

        full_prompt = tokenizer.apply_chat_template(
            demo_msgs, tokenize=False, add_generation_prompt=True,
        )

        if template_adds_think:
            prompt_no_think = re.sub(r"<think>\n?$", "", full_prompt).rstrip() + "\n"
        else:
            prompt_no_think = full_prompt

        think_start = full_prompt if template_adds_think else full_prompt + "<think>\n"

        if lang != "fr":
            continue

        # Drop samples where the prompt exceeds input token budget
        token_count = len(tokenizer.encode(think_start + "D'accord,"))
        if token_count > max_input_tokens:
            continue

        # chosen: thinks in French (forced with D'accord,)
        tasks.append({
            "id": row_id, "prompt": demo_msgs, "lang": lang, "type": "chosen",
            "prompt_text": think_start + "D'accord,",
            "prefix": "<think>\nD'accord,",
        })
        # rejected: thinks freely (likely English)
        tasks.append({
            "id": row_id, "prompt": demo_msgs, "lang": lang, "type": "rejected",
            "prompt_text": full_prompt,
            "prefix": "<think>\n" if template_adds_think else "",
        })

    return tasks


async def generate_all(client, prompts, tasks, args):
    semaphore = asyncio.Semaphore(args.concurrency)
    outputs = [None] * len(prompts)
    done = 0

    async def generate_one(idx, prompt):
        nonlocal done
        async with semaphore:
            try:
                response = await client.completions.create(
                    model=args.model_id,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    stop=["<|im_end|>"],
                )
                text = response.choices[0].text
            except Exception as e:
                print(f"  [SKIP {idx}] {e}")
                text = ""
            outputs[idx] = text
            done += 1
            if done % 200 == 0:
                task = tasks[idx]
                content = task["prefix"] + text
                print(f"\n{'='*80}")
                print(f"  [{done}/{len(prompts)}] id={task['id']} type={task['type']}")
                print(f"{'='*80}")
                print(f"{prompt}{content}")
                print(f"{'='*80}\n")

    await tqdm.gather(*[generate_one(i, p) for i, p in enumerate(prompts)], desc="Generating")
    return outputs


async def amain():
    args = parse_args()

    ds = load_dataset(args.dataset_id, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    max_input_tokens = args.max_model_len - args.max_tokens
    tasks = prepare_tasks(ds[:], tokenizer, max_input_tokens)
    print(f"Kept {len(tasks) // 2}/{len(ds)} samples after filtering by token length (max input: {max_input_tokens})")
    prompts = [t["prompt_text"] for t in tasks]

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    print(f"Generating {len(prompts)} completions (concurrency={args.concurrency})...")
    outputs = await generate_all(client, prompts, tasks, args)

    # Reconstruct dataset
    processed = {}
    for task, generated in zip(tasks, outputs):
        rid = task["id"]
        if rid not in processed:
            processed[rid] = {
                "id": rid, "language": task["lang"],
                "prompt": task["prompt"], "chosen": None, "rejected": None,
            }
        content = task["prefix"] + generated
        conversation = task["prompt"] + [{"role": "assistant", "content": content}]
        processed[rid][task["type"]] = conversation

    new_dataset = Dataset.from_list(list(processed.values()))
    print(f"Pushing to Hub: {args.output_repo}")
    new_dataset.push_to_hub(args.output_repo)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(amain())
