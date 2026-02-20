import argparse
import asyncio
import itertools
import json
import random
from collections import Counter

import httpx
import openai
from datasets import load_dataset
from tqdm.asyncio import tqdm


def detect_loop_ngram(text: str, ngram_size: int, ngram_threshold: int) -> bool:
    words = text.split()
    if len(words) < ngram_size:
        return False
    ngrams = [tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
    return Counter(ngrams).most_common(1)[0][1] > ngram_threshold


async def check_sample(sem, client, model_name, prompt, temperature, top_p, max_tokens, ngram_size, ngram_threshold):
    async with sem:
        response = await client.completions.create(
            model=model_name,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    completion = response.choices[0]
    hit_length = completion.finish_reason == "length"
    has_ngram_loop = detect_loop_ngram(completion.text, ngram_size, ngram_threshold)
    return {
        "prompt": prompt,
        "generated": completion.text,
        "finish_reason": completion.finish_reason,
        "no_eos": hit_length,
        "ngram_loop": has_ngram_loop,
        "looping": hit_length or has_ngram_loop,
    }


async def run(args):
    # Dataset
    print("Loading GSM8K train split...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    indices = random.Random(args.seed).sample(range(len(ds)), min(args.num_samples, len(ds)))
    samples = ds.select(sorted(indices))
    print(f"Selected {len(samples)} / {len(ds)} samples (seed={args.seed})")
    prompts = [f"Question: {s['question']}\n" for s in samples]

    # Clients
    base_urls = [url.strip() for url in args.base_urls.split(",")]
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout=3600.0))
    clients = [openai.AsyncOpenAI(base_url=url, api_key=args.api_key, http_client=http_client) for url in base_urls]
    client_cycle = itertools.cycle(clients)
    print(f"Using {len(clients)} vLLM instance(s): {base_urls}\n")

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        check_sample(
            sem, next(client_cycle), args.model_name, prompt,
            args.temperature, args.top_p, args.max_tokens,
            args.ngram_size, args.ngram_threshold,
        )
        for prompt in prompts
    ]

    results = []
    loop_no_eos = 0
    loop_ngram = 0
    loop_any = 0

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await coro
        results.append(result)
        if result["no_eos"]:
            loop_no_eos += 1
        if result["ngram_loop"]:
            loop_ngram += 1
        if result["looping"]:
            loop_any += 1

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} samples to {args.output}")

    n = len(prompts)
    print(f"\n{'=' * 50}")
    print(f"  Model      : {args.model_name}")
    print(f"  Samples    : {n}  |  temp: {args.temperature}  top_p: {args.top_p}")
    print(f"  ngram_size : {args.ngram_size}  (threshold: {args.ngram_threshold})")
    print(f"{'=' * 50}")
    print(f"  No EOS (filled context) : {loop_no_eos:3d} / {n}  ({100 * loop_no_eos / n:.1f}%)")
    print(f"  N-gram repetition       : {loop_ngram:3d} / {n}  ({100 * loop_ngram / n:.1f}%)")
    print(f"  Looping (any)           : {loop_any:3d} / {n}  ({100 * loop_any / n:.1f}%)")
    print(f"{'=' * 50}")


def main(args):
    asyncio.run(run(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--base_urls", type=str, default="http://localhost:8000/v1",
                        help="Comma-separated list of vLLM base URLs")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=64,
                        help="Max number of in-flight async requests")
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--ngram_size", type=int, default=10)
    parser.add_argument("--ngram_threshold", type=int, default=3)
    parser.add_argument("--output", type=str, default="loop_sanity_results.json")
    main(parser.parse_args())
