"""
Generate answers from a dataset with (id, question, demonstration) fields using a vLLM API.
"""

import argparse
import asyncio
import itertools
import json
import os
from pathlib import Path

import httpx
import openai
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from tqdm.asyncio import tqdm


ASSISTANT_PREFILL = "<think>\nD'accord, voyons voir."


def build_messages(prompt: str, demonstration: str) -> list[dict]:
    """Build chat messages giving the model the prompt and demonstration as context."""
    return [
        {
            "role": "user",
            "content": (
                f"{prompt}\n\n"
                "This is an example for a response to the question:\n"
                f"{demonstration}\n\n"
                "Now answer with a response of your own, including the thinking process."
            ),
        },
        {
            "role": "assistant",
            "content": ASSISTANT_PREFILL,
        },
    ]


async def generate_one(
    sem: asyncio.Semaphore,
    client: openai.AsyncOpenAI,
    model_name: str,
    sample: dict,
    temperature: float,
    top_p: float,
) -> dict:
    """Generate an answer for a single sample."""
    messages = build_messages(sample["prompt"], sample["demonstration"])
    async with sem:
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            extra_body={
                "continue_final_message": True,
                "add_generation_prompt": False,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
    generation = ASSISTANT_PREFILL + (response.choices[0].message.content or "")
    return {
        "id": sample["id"],
        "messages": [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": generation},
        ],
        "finish_reason": response.choices[0].finish_reason,
        "completion_tokens": (
            response.usage.completion_tokens if response.usage else None
        ),
    }


def load_already_generated(output_path: str) -> set:
    """Load IDs already present in the output JSONL to allow resuming."""
    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return done_ids


async def run(args: argparse.Namespace) -> None:
    # Load dataset
    print(f"Loading dataset '{args.dataset_name}' (split=train)...")
    ds = load_dataset(args.dataset_name, split="train")
    print(f"Loaded {len(ds)} samples")

    # Check required columns
    for col in ("id", "prompt", "demonstration"):
        if col not in ds.column_names:
            raise ValueError(
                f"Dataset is missing required column '{col}'. Found: {ds.column_names}"
            )

    # Resume support: skip already generated IDs
    done_ids = load_already_generated(args.output_path)
    if done_ids:
        print(f"Found {len(done_ids)} already generated samples, resuming...")
    remaining = [s for s in ds if s["id"] not in done_ids]
    if args.max_samples:
        remaining = remaining[: args.max_samples]
    print(f"Generating for {len(remaining)} samples")

    if not remaining:
        print("Nothing to generate, all samples already done.")
        return

    # Setup async OpenAI clients (round-robin across base URLs)
    base_urls = [url.strip() for url in args.base_urls.split(",")]
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout=3600.0))
    clients = [
        openai.AsyncOpenAI(base_url=url, api_key=args.api_key, http_client=http_client)
        for url in base_urls
    ]
    client_cycle = itertools.cycle(clients)

    sem = asyncio.Semaphore(args.concurrency)

    # Create all tasks
    tasks = [
        generate_one(
            sem,
            next(client_cycle),
            args.model_name,
            sample,
            args.temperature,
            args.top_p,
        )
        for sample in remaining
    ]

    # Open output file in append mode for crash-safe incremental writes
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    generated_count = len(done_ids)
    error_count = 0

    with open(args.output_path, "a") as out_f:
        for coro in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Generating"
        ):
            try:
                result = await coro
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                generated_count += 1
            except Exception as e:
                error_count += 1
                print(f"\nError: {e}", flush=True)

    total = len(ds)
    print(f"\nGeneration complete:")
    print(f"  Total samples in dataset: {total}")
    print(f"  Successfully generated:   {generated_count}")
    print(f"  Errors:                   {error_count}")

    # Push to HuggingFace Hub
    if args.hf_repo_id:
        print(f"\nPushing to HuggingFace Hub: {args.hf_repo_id} ...")
        # Read back the full JSONL
        records = []
        with open(args.output_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        hf_ds = Dataset.from_list(records)
        hf_ds.push_to_hub(args.hf_repo_id, private=True)
        print(
            f"Pushed {len(records)} samples to https://huggingface.co/datasets/{args.hf_repo_id}"
        )

        # Also upload the raw JSONL as a file
        api = HfApi()
        api.upload_file(
            path_or_fileobj=args.output_path,
            path_in_repo="generations.jsonl",
            repo_id=args.hf_repo_id,
            repo_type="dataset",
        )
        print("Also uploaded raw JSONL file to the repo.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate answers from a (id, question, demonstration) dataset using a vLLM API."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lightonai/olmo3-fr-prompts",
        help="HuggingFace dataset name (e.g. 'user/my-dataset').",
    )
    parser.add_argument(
        "--base_urls",
        type=str,
        default="http://localhost:8000/v1",
        help="Comma-separated vLLM API base URLs.",
    )
    parser.add_argument(
        "--api_key", type=str, default="EMPTY", help="API key for the vLLM server."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name served by vLLM."
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of samples to generate. None = all.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Max number of concurrent async requests.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/generations.jsonl",
        help="Path to the output JSONL file (appended incrementally).",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default="lightonai/synthetic-data-french-reasoning",
        help="HuggingFace repo ID to push results to (e.g. 'user/my-generations'). "
        "Created as private. Omit to skip pushing.",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
