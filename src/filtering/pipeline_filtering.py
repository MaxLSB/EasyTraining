"""
Multi-step dataset filtering pipeline for French reasoning datasets.

Stages (in order):
  1. Load a HuggingFace dataset with a `messages` field.
  2. Remove samples with empty content in any turn.
  3. Verify that every assistant turn has correctly formatted <think></think> tags.
  4. Language filtering — keep only French samples (langdetect via ProcessPool).
  5. N-gram repetition filtering — remove degenerate looping (10-grams, threshold 5).
  6. LLM-as-a-judge quality filtering (async OpenAI calls).
  7. Push the filtered dataset to a private HuggingFace repository.
"""

import argparse
import asyncio
import itertools
import os
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

import httpx
import openai
from datasets import Dataset, load_dataset
from langdetect import detect
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm


# ── Stage 2: Empty content filtering ─────────────────────────────────────────


def filter_empty_content(dataset: Dataset) -> Dataset:
    """Remove samples where any turn has empty or whitespace-only content."""

    def has_content(sample: Dict[str, Any]) -> bool:
        for msg in sample["messages"]:
            if not msg.get("content") or not msg["content"].strip():
                return False
        return True

    return dataset.filter(has_content, desc="Empty content")


# ── Stage 3: Think-tag verification ──────────────────────────────────────────


def has_valid_think_tags(text: str) -> bool:
    """Return True if text contains exactly matched <think>...</think> pairs."""
    opens = [m.start() for m in re.finditer(r"<think>", text)]
    closes = [m.start() for m in re.finditer(r"</think>", text)]

    if len(opens) != len(closes) or len(opens) == 0:
        return False

    for o, c in zip(opens, closes):
        if o >= c:
            return False

    return True


def filter_think_tags(dataset: Dataset) -> Dataset:
    """Remove samples where any assistant turn has malformed <think></think> tags."""

    def is_valid(sample: Dict[str, Any]) -> bool:
        for msg in sample["messages"]:
            if msg["role"] == "assistant":
                if not has_valid_think_tags(msg["content"]):
                    return False
        return True

    return dataset.filter(is_valid, desc="Think tags")


# ── Stage 4: Language filtering (French only, parallel) ──────────────────────


def _detect_lang(messages: List[Dict[str, str]]) -> bool:
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    text = " ".join(msg["content"] for msg in messages)
    try:
        return detect(text) == "fr"
    except Exception:
        return False


def filter_language(dataset: Dataset, num_workers: int = None) -> Dataset:
    """Remove non-French samples using langdetect with ProcessPoolExecutor."""
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, 16)

    all_messages = dataset["messages"]
    keep = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(_detect_lang, all_messages, chunksize=64)
        keep = list(tqdm(results, total=len(dataset), desc="Language (fr)"))

    indices = [i for i, passed in enumerate(keep) if passed]
    return dataset.select(indices)


# ── Stage 5: N-gram repetition filtering ─────────────────────────────────────


def detect_ngram_loop(text: str, ngram_size: int = 10, threshold: int = 5) -> bool:
    """Return True if any n-gram appears more than `threshold` times."""
    words = text.split()
    if len(words) < ngram_size:
        return False
    counts: Dict[tuple, int] = {}
    for i in range(len(words) - ngram_size + 1):
        gram = tuple(words[i : i + ngram_size])
        count = counts.get(gram, 0) + 1
        if count > threshold:
            return True
        counts[gram] = count
    return False


def filter_ngram_repetition(
    dataset: Dataset, ngram_size: int = 10, threshold: int = 5
) -> Dataset:
    """Remove samples with degenerate n-gram repetition in assistant turns."""

    def is_clean(sample: Dict[str, Any]) -> bool:
        assistant_text = " ".join(
            msg["content"] for msg in sample["messages"] if msg["role"] == "assistant"
        )
        return not detect_ngram_loop(assistant_text, ngram_size, threshold)

    return dataset.filter(is_clean, desc="N-gram repetition")


# ── Stage 6: LLM-as-a-judge filtering (async) ────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are a French text quality judge. Given a text sample, return True if it passes all checks, False otherwise.

Return False if ANY condition is met:
- Language switching: non-French phrases appear in the prose (exceptions: math formulas, code, URLs, proper nouns, integrated loanwords like "email" or "weekend")
- Grammatical incoherence: sentences are structurally broken and incomprehensible
- Semantic incoherence: text is meaningless or word salad
- Poor quality: truncated fragments, repetitive/degenerate text, machine-translation artifacts, or no communicative value

Return True if the text is coherent, grammatically sound French of sufficient quality and completeness.

Respond with exactly one token: True or False. No explanation, no punctuation."""


async def _judge_one(
    sem: asyncio.Semaphore,
    client: openai.AsyncOpenAI,
    model_name: str,
    sample: Dict[str, Any],
    temperature: float,
    top_p: float,
) -> bool:
    """Judge a single sample. Returns True if it passes."""
    text = "\n\n".join(
        f"{m['role'].upper()}:\n{m['content']}" for m in sample["messages"]
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    async with sem:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=10,
                    top_p=top_p,
                )
                content = response.choices[0].message.content or ""
                if "true" in content.lower():
                    return True
                if "false" in content.lower():
                    return False
                raise ValueError(f"Unexpected judge response: {content}")
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Judge API failed after 3 attempts: {e}")
    return False


async def _run_llm_judge(
    dataset: Dataset,
    base_urls: List[str],
    api_key: str,
    model_name: str,
    concurrency: int,
    temperature: float,
    top_p: float,
) -> List[int]:
    """Run async LLM judge over the dataset, return accepted indices."""
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout=3600.0))
    clients = [
        openai.AsyncOpenAI(base_url=url, api_key=api_key, http_client=http_client)
        for url in base_urls
    ]
    client_cycle = itertools.cycle(clients)
    sem = asyncio.Semaphore(concurrency)

    async def _wrapped(idx: int, client, sample):
        try:
            passed = await _judge_one(
                sem, client, model_name, sample, temperature, top_p
            )
            return idx, passed
        except Exception as e:
            print(f"  Judge error on sample {idx}: {e}")
            return idx, None

    coros = [_wrapped(i, next(client_cycle), dataset[i]) for i in range(len(dataset))]

    accepted = []
    for coro in atqdm(asyncio.as_completed(coros), total=len(coros), desc="LLM judge"):
        idx, passed = await coro
        if passed:
            accepted.append(idx)

    await http_client.aclose()
    accepted.sort()
    return accepted


def filter_llm_judge(
    dataset: Dataset,
    base_urls: List[str],
    api_key: str,
    model_name: str,
    concurrency: int = 64,
    temperature: float = 0.15,
    top_p: float = 0.1,
) -> Dataset:
    """Remove samples that do not pass the LLM-as-a-judge quality check."""
    accepted = asyncio.run(
        _run_llm_judge(
            dataset,
            base_urls,
            api_key,
            model_name,
            concurrency,
            temperature,
            top_p,
        )
    )
    return dataset.select(accepted)


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_stage(name: str, fn, dataset: Dataset, initial: int, **kwargs) -> Dataset:
    """Run a filtering stage and print a one-line summary."""
    before = len(dataset)
    dataset = fn(dataset, **kwargs)
    removed = before - len(dataset)
    print(f"  {name:<25} {before:>7,} -> {len(dataset):>7,}  (-{removed:,})")
    return dataset


def main(args: argparse.Namespace) -> None:
    print(f"Loading '{args.dataset_name}' ...")
    dataset = load_dataset(args.dataset_name, split="train")
    initial = len(dataset)
    print(f"Loaded {initial:,} samples\n")

    print(f"  {'Stage':<25} {'Before':>7}    {'After':>7}   Removed")
    print(f"  {'-' * 58}")

    dataset = run_stage("Empty content", filter_empty_content, dataset, initial)
    dataset = run_stage("Think tags", filter_think_tags, dataset, initial)
    dataset = run_stage("Language (fr)", filter_language, dataset, initial)
    dataset = run_stage(
        "N-gram repetition",
        filter_ngram_repetition,
        dataset,
        initial,
        ngram_size=args.ngram_size,
        threshold=args.ngram_threshold,
    )

    # base_urls = [url.strip() for url in args.base_urls.split(",")]
    # print(f"\n  LLM judge ({args.model_name}, {len(base_urls)} endpoint(s))")
    # dataset = run_stage(
    #     "LLM judge",
    #     filter_llm_judge,
    #     dataset,
    #     initial,
    #     base_urls=base_urls,
    #     api_key=args.api_key,
    #     model_name=args.model_name,
    #     concurrency=args.concurrency,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    # )

    print(f"  {'-' * 58}")
    print(
        f"  {'TOTAL':<25} {initial:>7,} -> {len(dataset):>7,}  (-{initial - len(dataset):,})\n"
    )

    print(f"Pushing to {args.hf_repo_id} ...")
    dataset.push_to_hub(args.hf_repo_id, private=True)
    print(f"Done. https://huggingface.co/datasets/{args.hf_repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-step filtering pipeline for French reasoning datasets."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g. 'user/my-dataset').",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID to push the filtered dataset to.",
    )

    # API configuration (for LLM judge)
    parser.add_argument(
        "--base_urls",
        type=str,
        default="http://localhost:8000/v1",
        help="Comma-separated vLLM API base URLs.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key for the vLLM server.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model name for the LLM judge.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Max concurrent async requests for the LLM judge.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.15,
        help="Sampling temperature for the LLM judge.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.1,
        help="Top-p for the LLM judge.",
    )

    # N-gram filtering
    parser.add_argument("--ngram_size", type=int, default=10)
    parser.add_argument("--ngram_threshold", type=int, default=5)

    main(parser.parse_args())
