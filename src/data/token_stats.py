"""Compute average token length per sample in a HF dataset with a `messages` field."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

TOKENIZER_NAME = "Qwen/Qwen3-4B"
_tokenizer = None


def _init_tokenizer(name: str):
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(name)


def _count_tokens(messages):
    text = " ".join(m["content"] for m in messages)
    return len(_tokenizer.encode(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lightonai/olmo-think-160k-fr-v2-FILTERED-NO-JUDGE",
    )
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 16))
    args = parser.parse_args()

    print(f"Loading '{args.dataset_name}' ...")
    ds = load_dataset(args.dataset_name, split="train")
    print(f"Loaded {len(ds):,} samples")

    print(f"Tokenizer: {TOKENIZER_NAME}  |  Workers: {args.workers}")
    all_messages = ds["messages"]

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_tokenizer,
        initargs=(TOKENIZER_NAME,),
    ) as executor:
        lengths = list(
            tqdm(
                executor.map(_count_tokens, all_messages, chunksize=64),
                total=len(all_messages),
                desc="Tokenizing",
            )
        )

    avg = sum(lengths) / len(lengths)
    mn, mx = min(lengths), max(lengths)
    median = sorted(lengths)[len(lengths) // 2]

    print(f"\nSamples:  {len(lengths):,}")
    print(f"Avg:      {avg:,.1f} tokens")
    print(f"Median:   {median:,} tokens")
    print(f"Min:      {mn:,} tokens")
    print(f"Max:      {mx:,} tokens")


if __name__ == "__main__":
    main()
