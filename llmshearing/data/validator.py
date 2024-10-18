import argparse
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from streaming import StreamingDataset
from tqdm import tqdm


def validate_mds_datasets(root_paths: List[str], n_samples: int = 1000):
    results = {}

    for root_path in root_paths:
        results[root_path] = validate_mds_dataset_group(root_path, n_samples)

    visualize_results(results)
    return results


def validate_mds_dataset_group(root_path: str, n_samples: int = 1000) -> Dict:
    root = Path(root_path)
    datasets = [d for d in root.iterdir() if d.is_dir() and (d / "index.json").exists()]

    group_results = {}
    total_sequences_overall = 0
    for dataset in datasets:
        print(f"Validating dataset: {dataset.name} in {root_path}")
        dataset_results = validate_single_dataset(dataset, n_samples)
        group_results[dataset.name] = dataset_results
        total_sequences_overall += dataset_results["total_sequences"]

    # Calculate normalized ratios
    for dataset_results in group_results.values():
        dataset_results["normalized_total_sequences"] = (
            dataset_results["total_sequences"] / total_sequences_overall
        )

    return group_results


def validate_single_dataset(dataset_path: Path, sample_size: int = 1000) -> Dict:
    dataset = StreamingDataset(local=str(dataset_path))

    # Get metadata
    total_sequences = len(dataset)

    # Sparse sampling
    sampled_indices = random.sample(
        range(total_sequences), min(sample_size, total_sequences)
    )

    sequence_lengths = []
    token_frequencies: Dict[int, int] = {}

    for idx in tqdm(sampled_indices, desc="Processing samples"):
        sample = dataset[idx]
        tokens = np.frombuffer(sample["tokens"], dtype=np.uint16)
        sequence_lengths.append(len(tokens))

        for token in tokens:
            token_frequencies[token] = token_frequencies.get(token, 0) + 1

    return {
        "total_sequences": total_sequences,
        "sampled_sequences": len(sampled_indices),
        "avg_sequence_length": np.mean(sequence_lengths),
        "min_sequence_length": np.min(sequence_lengths),
        "max_sequence_length": np.max(sequence_lengths),
        "unique_tokens": len(token_frequencies),
        "top_10_tokens": sorted(
            token_frequencies.items(), key=lambda x: x[1], reverse=True
        )[:10],
    }


def visualize_results(results: Dict[str, Dict[str, Dict]]):
    metrics = [
        "total_sequences",
        "normalized_total_sequences",
        "avg_sequence_length",
        "unique_tokens",
    ]
    n_groups = len(results)
    n_datasets = max(len(group) for group in results.values())

    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 7 * len(metrics)))

    for i, metric in enumerate(metrics):
        data = []
        for group, group_results in results.items():
            for dataset, result in group_results.items():
                data.append((group, dataset, result[metric]))

        df = pd.DataFrame(data, columns=["Group", "Dataset", "Value"])
        sns.barplot(x="Dataset", y="Value", hue="Group", data=df, ax=axes[i])

        axes[i].set_title(f"{metric.replace('_', ' ').title()} per Dataset")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].legend(title="Group")

        if metric == "normalized_total_sequences":
            axes[i].set_ylabel("Ratio")
            axes[i].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    print("Detailed results saved to validation_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate MDS datasets")
    parser.add_argument(
        "root_paths", nargs="+", help="Root paths containing the datasets"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of samples to process per dataset",
    )
    args = parser.parse_args()

    results = validate_mds_datasets(args.root_paths, args.n_samples)
    print(
        "Validation complete. Results saved to validation_results.json and comparison plots displayed."
    )
