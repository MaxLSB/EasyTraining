"""
Script to quality-check a dataset of conversations for French LLM training using an LLM judge.
"""

import argparse
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import openai
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm


class DatasetBuilder:
    """Class to build a dataset using French prompts for quality filtering."""

    SYS_PROMPT = """
    You are a dataset quality assistant.

    Evaluate the conversation below between a user and an AI assistant.

    # Return **False** if **any** of the following apply:

    - The user or assistant messages are not in French
    - Significant grammar or spelling mistakes
    - Incoherent or nonsensical text
    - Reasoning traces contain text in languages other than French (e.g., Malagasy, Arabic, Chinese)
    - The assistant discusses or mentions translating the user's query from a non-French language
    - An assistant response doesn't answer the user's last question
    - Contradictions, invalid reasoning, or factual errors
    - The reasoning traces are illogical with respect to the user's question
    - The assistant fails to follow instructions in the prompt

    ### Example of a **False** case (third-language contamination):
    USER:
    Que faire si une femme est prise de suffocation ?
    ASSISTANT:
    <think>
    D'accord, l'utilisateur demande "Inona no atao raha tratra ny sovoka ny vehivavy iray ?"
    qui est en malgache. Permets-moi de traduire cela d'abord...
    </think>

    This should return **False** because the user query is in French, but the assistant
    responds in Malagasy, indicating contamination from another language.

    Otherwise, return **True**.

    # REQUIREMENTS:
    Respond only with: `True` or `False`
    """

    def __init__(
        self,
        clients: List[openai.Client],
        model_name: str,
        temperature: float,
        top_p: float,
    ):
        self.clients = clients
        self.client_cycle = itertools.cycle(clients)
        self._cycle_lock = threading.Lock()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    def _build_api_messages(self, question: str) -> List[Dict[str, Any]]:
        """Build the API message format."""
        return [
            {"role": "system", "content": self.SYS_PROMPT},
            {"role": "user", "content": question},
        ]

    def _make_api_call(self, messages: List[Dict[str, Any]]) -> str:
        """Make an API call with retry logic."""
        with self._cycle_lock:
            client = next(self.client_cycle)
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=10,
                    top_p=self.top_p,
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"API call failed after 3 attempts: {e}")
                continue
        raise RuntimeError("API call failed after 3 attempts")

    def generate(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate a single sample and return it if it passes quality checks."""
        # Judges the entire "messages" field: all roles (system/user/assistant)
        # concatenated into one string.
        text = "\n\n".join(
            f"{m['role'].upper()}:\n{m['content']}" for m in sample["messages"]
        )
        status = self._make_api_call(self._build_api_messages(text))

        if "true" in status.lower():
            return {"id": sample["id"], "messages": sample["messages"]}
        elif "false" in status.lower():
            return None
        else:
            raise ValueError(
                f"Unexpected response from model: {status}. Expected 'True' or 'False'."
            )


def get_clients(base_urls: List[str], api_key: str) -> List[openai.Client]:
    """Initialize OpenAI clients for each base URL."""
    return [openai.Client(base_url=url, api_key=api_key) for url in base_urls]


def filter_and_quality_pipeline(
    dataset: Dataset,
    clients: List[openai.Client],
    model_name: str,
    num_workers: int,
    temperature: float,
    top_p: float,
) -> List[Dict[str, Any]]:
    """Run the LLM quality check pipeline."""
    print("Applying quality filtering via LLM API...")
    ds_builder = DatasetBuilder(clients, model_name, temperature, top_p)
    generated = []
    rejected_count = 0
    processed_count = 0
    last_rejected_sample = None  # Track most recent rejected sample

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(ds_builder.generate, sample): sample for sample in dataset
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Quality filtering"
        ):
            try:
                original_sample = futures[future]
                result = future.result()
                if result is not None:
                    generated.append(result)
                else:
                    rejected_count += 1
                    last_rejected_sample = original_sample

                processed_count += 1
                if processed_count % 1000 == 0:
                    print(
                        f"\n[Progress] Processed: {processed_count:,} | "
                        f"Accepted: {len(generated):,} | "
                        f"Rejected: {rejected_count:,}",
                        flush=True,
                    )

                    if last_rejected_sample:
                        print("\n" + "─" * 80)
                        print("REJECTED SAMPLE EXAMPLE")
                        print("─" * 80)
                        for msg in last_rejected_sample["messages"]:
                            role = msg["role"].upper()
                            content = msg["content"][:500]
                            continuation = "..." if len(msg["content"]) > 500 else ""
                            print(f"\n{role}:\n{content}{continuation}")
                        print("─" * 80 + "\n", flush=True)

            except Exception as e:
                print(f"Error processing sample: {e}", flush=True)
                rejected_count += 1

    # Print summary
    print(f"\nFiltering complete:")
    print(f"  Accepted samples: {len(generated)}")
    print(f"  Rejected samples: {rejected_count}")
    total = len(generated) + rejected_count
    if total > 0:
        print(f"  Pass rate: {len(generated) / total * 100:.2f}%")

    return generated


def main(args: argparse.Namespace) -> None:
    """Main execution function."""
    # Load dataset
    print(f"Loading dataset from {args.input_dataset_path}...")
    dataset = load_from_disk(args.input_dataset_path)

    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            first_split = list(dataset.keys())[0]
            print(f"Using split: {first_split}")
            dataset = dataset[first_split]

    print(f"Loaded {len(dataset)} samples")

    # Setup API clients
    base_urls = [url.strip() for url in args.base_urls.split(",")]
    clients = get_clients(base_urls, args.api_key)
    print(f"Using {len(clients)} vLLM instance(s): {base_urls}")

    # Run pipeline
    generated = filter_and_quality_pipeline(
        dataset=dataset,
        clients=clients,
        model_name=args.model_name,
        num_workers=args.num_workers,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Save filtered dataset
    print(f"\nSaving filtered dataset to {args.output_dataset_path}...")
    if generated:
        hf_ds = Dataset.from_list(generated)
    else:
        hf_ds = Dataset.from_dict({"id": [], "messages": []})

    ds_dict = DatasetDict({"train": hf_ds})
    ds_dict.save_to_disk(args.output_dataset_path)
    print(f"Saved {len(generated)} samples to {args.output_dataset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter a dataset using an LLM judge for quality filtering."
    )

    # Input/Output paths
    parser.add_argument(
        "--input_dataset_path",
        type=str,
        default="/gpfs/projects/ehpc507/datasets/olmo_think_fr_200k/",
        help="Path to the input dataset directory.",
    )
    parser.add_argument(
        "--output_dataset_path",
        type=str,
        default="/gpfs/projects/ehpc507/datasets/olmo_think_fr_200k_filtered/",
        help="Path to save the filtered dataset directory.",
    )

    # API configuration
    parser.add_argument(
        "--base_urls",
        type=str,
        default="http://localhost:8000/v1",
        help="Comma-separated list of base URLs for the API instances. "
        "Example: http://ip1:8000/v1,http://ip2:8000/v1",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key for the model API.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-4b-it",
        help="Model name to use for quality filtering.",
    )

    # Processing configuration
    parser.add_argument(
        "--num_workers",
        type=int,
        default=64,
        help="Number of workers for parallel processing.",
    )

    # vLLM hyperparameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.15,
        help="Sampling temperature for vLLM API.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.1,
        help="Nucleus sampling probability (top_p) for vLLM API.",
    )

    args = parser.parse_args()
    main(args)
