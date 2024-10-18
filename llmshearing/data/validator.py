import random
from pathlib import Path
from typing import Dict

import numpy as np
from streaming import StreamingDataset
from tqdm import tqdm


def validate_mds_datasets(root_path: str):
    root = Path(root_path)
    datasets = [d for d in root.iterdir() if d.is_dir() and (d / "index.json").exists()]

    results = {}
    for dataset in datasets:
        print(f"Validating dataset: {dataset.name}")
        results[dataset.name] = validate_single_dataset(dataset)

    visualize_results(results)
    return results


def validate_single_dataset(
    dataset_path: Path, seq_length: int = 4096, sample_size: int = 100
) -> Dict:
    dataset = StreamingDataset(local=str(dataset_path))

    # Get metadata
    total_sequences = len(dataset)

    # Sparse sampling
    sampled_indices = random.sample(
        range(total_sequences), min(sample_size, total_sequences)
    )

    sequence_lengths = []
    token_frequencies = {}

    for idx in tqdm(sampled_indices, desc="Processing samples"):
        sample = dataset[idx]
        tokens = np.frombuffer(sample["tokens"], dtype=np.uint16)
        sequence_lengths.append(len(tokens))

        for token in tokens:
            token_frequencies[token] = token_frequencies.get(token, 0) + 1

    return {
        "total_sequences": total_sequences,
        "total_tokens": total_sequences * seq_length,
        "sampled_sequences": len(sampled_indices),
        "avg_sequence_length": np.mean(sequence_lengths).tolist(),
        "min_sequence_length": np.min(sequence_lengths).tolist(),
        "max_sequence_length": np.max(sequence_lengths).tolist(),
        "unique_tokens": len(token_frequencies),
    }


def visualize_results(results: Dict[str, Dict]):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # import markdown from jupyter
    from IPython.display import Markdown, display

    # show core data: total tokens per dataset and total overall
    total_tokens = sum([r["total_tokens"] for r in results.values()])
    display(Markdown("## Summary"))
    display(Markdown(f"**Total tokens:** {total_tokens / 1e9:.1f}B"))

    # Sequence count comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(results.keys()), y=[r["total_tokens"] for r in results.values()])
    plt.title("Total Tokens per Dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Average sequence length comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=list(results.keys()), y=[r["avg_sequence_length"] for r in results.values()]
    )
    plt.title("Average Sequence Length per Dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Unique tokens comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=list(results.keys()), y=[r["unique_tokens"] for r in results.values()]
    )
    plt.title("Unique Tokens per Dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Save detailed results as JSON
    import json

    with open("validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved to validation_results.json")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python validator.py <root_path>")
        sys.exit(1)

    root_path = sys.argv[1]
    results = validate_mds_datasets(root_path)
    print(
        "Validation complete. Results saved to validation_results.json and comparison plots."
    )
