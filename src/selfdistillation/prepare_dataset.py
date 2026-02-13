"""Reformat lightonai/olmo_think_fr_450k to {id, query, demonstration_think, demonstration_no_think} for SDFT."""

import argparse
import re
from datasets import load_dataset


def fix_think_tags(text):
    """Fix <think>/<​/think> to match Qwen3 chat template: <think>\n...\n</think>\n\n"""
    text = re.sub(r"<think>\n*", "<think>\n", text)
    text = re.sub(r"</think>\n*", "</think>\n\n", text)
    return text


def strip_think(text):
    """Remove <think>...</think> blocks, returning only the answer."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def reformat(example):
    msgs = example["messages"]
    query, raw_demo = "", ""
    for m in msgs:
        if m["role"] == "user" and not query:
            query = m["content"]
        elif m["role"] == "assistant" and not raw_demo:
            raw_demo = m["content"]
        if query and raw_demo:
            break
    return {
        "query": query,
        "demonstration_think": fix_think_tags(raw_demo),
        "demonstration_no_think": strip_think(raw_demo),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="lightonai/olmo_think_fr_450k", help="Source HF dataset")
    parser.add_argument("--repo", default="lightonai/olmo_think_fr_sdft", help="Output HF repo id")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    ds = load_dataset(args.input, split=args.split)
    ds = ds.shuffle(seed=42).select(range(min(args.max_samples, len(ds))))
    ds = ds.map(reformat, num_proc=8, remove_columns=["messages"])
    print(f"Columns: {ds.column_names}")
    ds.push_to_hub(args.repo, private=args.private)
    print(f"Pushed {len(ds)} samples to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
