"""
Analyze generations.jsonl: thinking language distribution and finish reasons.
"""

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from langdetect import detect

LANG_NAMES = {
    "fr": "French",
    "en": "English",
    "es": "Spanish",
    "de": "German",
    "zh-cn": "Chinese",
    "ja": "Japanese",
    "ar": "Arabic",
    "unknown": "Unknown",
    "no_think": "Unclosed reasoning (no </think>)",
}

BAR_WIDTH = 30


def get_think_content(text: str) -> str | None:
    """Return the content before </think> if present, else None (unclosed reasoning)."""
    idx = text.find("</think>")
    if idx == -1:
        return None
    start = text.find("<think>")
    content_start = start + len("<think>") if start != -1 else 0
    return text[content_start:idx]


def has_ngram_loop(text: str, n: int = 10, threshold: int = 3) -> bool:
    """Return True if any n-gram appears more than threshold times."""
    words = text.split()
    if len(words) < n:
        return False
    seen: dict[tuple, int] = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i : i + n])
        count = seen.get(gram, 0) + 1
        if count > threshold:
            return True
        seen[gram] = count
    return False


def process_sample(sample: dict) -> tuple[str, bool]:
    """Process a single sample; return (lang_key, is_looping)."""
    assistant_turns = [
        m["content"] for m in sample["messages"] if m["role"] == "assistant"
    ]
    is_looping = has_ngram_loop(" ".join(assistant_turns))

    think_contents = [c for t in assistant_turns if (c := get_think_content(t)) is not None]
    if not think_contents:
        return "no_think", is_looping

    combined = " ".join(think_contents)
    try:
        lang = detect(combined)
    except Exception:
        lang = "unknown"
    return lang, is_looping


def print_table(title: str, rows: list[tuple[str, int]], total: int):
    """Print a formatted table with label, count, percentage and bar."""
    label_width = max(len(r[0]) for r in rows) if rows else 10
    print(f"\n  {title}")
    print(f"  {'-' * (label_width + 40)}")
    for label, count in rows:
        pct = 100 * count / total if total else 0
        filled = int(BAR_WIDTH * count / total) if total else 0
        bar = f"[{'|' * filled}{'.' * (BAR_WIDTH - filled)}]"
        print(f"  {label:<{label_width}}  {count:>6}  {pct:>5.1f}%  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="outputs/Qwen__Qwen3-30B-A3B-Thinking-2507_generations.jsonl",
    )
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 16))
    args = parser.parse_args()

    with open(args.path) as f:
        lines = [json.loads(line) for line in f]

    total = len(lines)
    finish_reasons = Counter(sample["finish_reason"] for sample in lines)
    lang_stats: Counter = Counter()
    looping_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for lang, is_looping in tqdm(executor.map(process_sample, lines, chunksize=64), total=total, desc="Analyzing"):
            lang_stats[lang] += 1
            if is_looping:
                looping_count += 1

    token_lengths = [
        s["completion_tokens"] for s in lines if s.get("completion_tokens") is not None
    ]

    # Display
    w = 60
    print(f"\n{'=' * w}")
    print(f"  Generations Report  |  {total:,} samples")
    print(f"{'=' * w}")

    print_table(
        "Finish Reasons",
        [(r, c) for r, c in finish_reasons.most_common()],
        total,
    )

    print_table(
        "Thinking Language",
        [(LANG_NAMES.get(l, l), c) for l, c in lang_stats.most_common()],
        total,
    )

    print_table(
        "Looping Behavior (10-gram repeated >3x)",
        [("Looping", looping_count), ("Clean", total - looping_count)],
        total,
    )

    if token_lengths:
        print(f"\n  Completion Tokens")
        print(f"  {'-' * 40}")
        print(f"  {'Min':<10} {min(token_lengths):>8,}")
        print(f"  {'Max':<10} {max(token_lengths):>8,}")
        print(f"  {'Average':<10} {sum(token_lengths) / len(token_lengths):>8,.1f}")

    print(f"\n{'=' * w}\n")


if __name__ == "__main__":
    main()
