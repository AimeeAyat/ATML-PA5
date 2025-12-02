"""
Inspect the ultrafeedback_binarized dataset structure
"""

from datasets import load_dataset
import json

def inspect_dataset():
    """Load and inspect the dataset"""

    print("=" * 80)
    print("DATASET INSPECTION")
    print("=" * 80)

    # Load the dataset
    print("\nLoading dataset: trl-lib/ultrafeedback_binarized")
    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

    print(f"\nDataset Info:")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Column names: {dataset.column_names}")
    print(f"  - Features: {dataset.features}")

    # Show first few samples
    print("\n" + "=" * 80)
    print("SAMPLE DATA")
    print("=" * 80)

    for idx in range(min(3, len(dataset))):
        print(f"\nSample {idx + 1}:")
        print("-" * 80)
        sample = dataset[idx]

        # Pretty print the sample
        for key, value in sample.items():
            if isinstance(value, str):
                # Truncate long strings
                if len(value) > 200:
                    print(f"{key}: {value[:200]}...\n")
                else:
                    print(f"{key}: {value}\n")
            else:
                print(f"{key}: {value}\n")

    # Dataset statistics
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    print(f"\nColumn names and types:")
    for col_name, feature in dataset.features.items():
        print(f"  - {col_name}: {feature}")


if __name__ == "__main__":
    inspect_dataset()
