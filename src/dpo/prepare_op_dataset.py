import argparse
import asyncio
import re
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from langdetect import detect

def parse_args():
    parser = argparse.ArgumentParser(description="Generate DPO dataset from reasoning traces.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--dataset_id", type=str, default="lightonai/DPO_Native_Reasoning_FR_EN_chat_template")
    parser.add_argument("--output_repo", type=str, default="lightonai/OPSD-DPO-dataset-olmo")
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--max_model_len", type=int, default=20000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--concurrency", type=int, default=512, help="Max concurrent requests to vLLM")
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
            f"Voici un exemple de réponse à la question :\n"
            f"{clean_answer}\n\n"
            f"Maintenant, réponds avec ta propre réponse, en incluant le processus de réflexion en français.\n"
        )
        demo_msgs = prompt_msgs[:-1] + [{"role": "user", "content": demo_content}]

        # Demo prompt (with example answer) for chosen generation
        demo_prompt = tokenizer.apply_chat_template(
            demo_msgs, tokenize=False, add_generation_prompt=True,
        )
        demo_think_start = demo_prompt if template_adds_think else demo_prompt + "<think>\n"

        # Raw prompt (no example) for rejected generation
        raw_prompt = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
        )

        if lang != "fr":
            continue

        # Drop samples where the longer prompt (demo) exceeds input token budget
        token_count = len(tokenizer.encode(demo_think_start + "D'accord,"))
        if token_count > max_input_tokens:
            continue

        # chosen: conditioned on prompt + example answer, forced French reasoning
        tasks.append({
            "id": row_id, "prompt": prompt_msgs, "lang": lang, "type": "chosen",
            "prompt_text": demo_think_start + "D'accord,",
            "prefix": "<think>\nD'accord,",
        })
        # rejected: just the prompt, model thinks freely (likely English)
        raw_think_start = raw_prompt if template_adds_think else raw_prompt + "<think>\n"
        tasks.append({
            "id": row_id, "prompt": prompt_msgs, "lang": lang, "type": "rejected",
            "prompt_text": raw_think_start,
            "prefix": "<think>\n",
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
            if done % 500 == 0:
                task = tasks[idx]
                print(f"\n{'='*80}")
                print(f"  [{done}/{len(prompts)}] id={task['id']} type={task['type']}")
                print(f"{'='*80}")
                print(f"{prompt}{text}")
                print(f"{'='*80}\n")

    await tqdm.gather(*[generate_one(i, p) for i, p in enumerate(prompts)], desc="Generating")
    return outputs


async def amain():
    args = parse_args()

    ds = load_dataset(args.dataset_id, split="train")
    total_before = len(ds)
    ds = ds.filter(lambda x: x["language"] != "en")
    print(f"Filtered out English samples: {total_before} -> {len(ds)}")
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
        processed[rid][task["type"]] = [{"role": "assistant", "content": content}]

    # Filter samples: keep only where chosen=French and rejected=English reasoning
    def extract_thinking(text):
        match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def detect_lang(text):
        try:
            return detect(text) if text else "unknown"
        except Exception:
            return "unknown"

    def has_valid_think_block(text):
        return bool(re.search(r'<think>\n.+?\n</think>', text, flags=re.DOTALL))

    filtered = []
    total = 0
    malformed = 0
    for row in processed.values():
        if row["chosen"] is None or row["rejected"] is None:
            continue
        total += 1
        chosen_content = row["chosen"][-1]["content"]
        rejected_content = row["rejected"][-1]["content"]
        if not has_valid_think_block(chosen_content) or not has_valid_think_block(rejected_content):
            malformed += 1
            continue
        chosen_thinking = extract_thinking(chosen_content)
        rejected_thinking = extract_thinking(rejected_content)
        chosen_lang = detect_lang(chosen_thinking)
        rejected_lang = detect_lang(rejected_thinking)
        if chosen_lang == "fr" and rejected_lang == "en":
            filtered.append(row)

    print(f"\n{'='*60}")
    print(f"Malformed <think> blocks: {malformed}/{total} samples discarded")
    print(f"Language filtering: kept {len(filtered)}/{total - malformed} valid samples")
    print(f"  (chosen=French AND rejected=English)")
    print(f"{'='*60}\n")

    new_dataset = Dataset.from_list(filtered)
    print(f"Pushing to Hub: {args.output_repo}")
    new_dataset.push_to_hub(args.output_repo, private=True)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(amain())
