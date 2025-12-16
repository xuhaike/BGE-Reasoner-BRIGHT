#!/usr/bin/env python3
"""
Script to upload BGE-Reasoner embeddings to Hugging Face Hub
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path

def upload_embeddings(
    repo_id: str,
    embeddings_dir: str = "./embedding_data/BRIGHT",
    private: bool = False,
    token: str = None
):
    """
    Upload embeddings to Hugging Face Hub

    Args:
        repo_id: Repository ID (e.g., "username/bge_embedding_bright")
        embeddings_dir: Directory containing embeddings
        private: Whether to create a private repository
        token: Hugging Face token (if not set via huggingface-cli login)
    """

    api = HfApi(token=token)

    # Create repository if it doesn't exist
    print(f"Creating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token
        )
        print(f"✓ Repository created/verified: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Note: {e}")

    # Check embeddings directory
    embeddings_path = Path(embeddings_dir)
    if not embeddings_path.exists():
        raise ValueError(f"Embeddings directory not found: {embeddings_dir}")

    # List files to upload
    embedding_files = list(embeddings_path.glob("*.npy"))
    print(f"\nFound {len(embedding_files)} embedding files:")
    total_size = 0
    for f in sorted(embedding_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  - {f.name}: {size_mb:.1f} MB")
    print(f"Total size: {total_size/1024:.2f} GB")

    # Create README
    readme_content = f"""---
license: apache-2.0
task_categories:
- feature-extraction
- sentence-similarity
tags:
- embeddings
- bge-reasoner
- BRIGHT
size_categories:
- 10K<n<100K
---

# BGE-Reasoner Embeddings for BRIGHT Benchmark

This dataset contains pre-computed embeddings for the [BRIGHT benchmark](https://huggingface.co/datasets/xlangai/BRIGHT) using the BGE-Reasoner model.

## Dataset Structure

The dataset contains {len(embedding_files)} embedding files for various domains:

### Embedding Files
"""

    # Group files by dataset
    datasets_with_embeddings = {}
    for f in sorted(embedding_files):
        name = f.name
        if "passage_embeddings" in name:
            dataset = name.replace("_bge-reasoner_passage_embeddings.npy", "")
            if dataset not in datasets_with_embeddings:
                datasets_with_embeddings[dataset] = {}
            datasets_with_embeddings[dataset]["passage"] = f.stat().st_size / (1024 * 1024)
        elif "query_embeddings" in name:
            dataset = name.replace("_test_bge-reasoner_query_embeddings.npy", "")
            if dataset not in datasets_with_embeddings:
                datasets_with_embeddings[dataset] = {}
            datasets_with_embeddings[dataset]["query"] = f.stat().st_size / (1024 * 1024)

    for dataset, sizes in sorted(datasets_with_embeddings.items()):
        readme_content += f"\n**{dataset}**:\n"
        if "passage" in sizes:
            readme_content += f"- Passage embeddings: `{dataset}_bge-reasoner_passage_embeddings.npy` ({sizes['passage']:.1f} MB)\n"
        if "query" in sizes:
            readme_content += f"- Query embeddings: `{dataset}_test_bge-reasoner_query_embeddings.npy` ({sizes['query']:.1f} MB)\n"

    readme_content += f"""
## Usage

```python
import numpy as np
from huggingface_hub import hf_hub_download

# Download passage embeddings
passage_file = hf_hub_download(
    repo_id="{repo_id}",
    filename="biology_bge-reasoner_passage_embeddings.npy",
    repo_type="dataset"
)
passage_embeddings = np.load(passage_file)

# Download query embeddings
query_file = hf_hub_download(
    repo_id="{repo_id}",
    filename="biology_test_bge-reasoner_query_embeddings.npy",
    repo_type="dataset"
)
query_embeddings = np.load(query_file)

print(f"Passage embeddings shape: {{passage_embeddings.shape}}")
print(f"Query embeddings shape: {{query_embeddings.shape}}")
```

## Model Information

- **Model**: BGE-Reasoner
- **Embedding Dimension**: 1024
- **Normalization**: L2 normalized

## Citation

If you use these embeddings, please cite the BRIGHT benchmark:

```bibtex
@article{{bright2024,
  title={{BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval}},
  author={{...}},
  journal={{...}},
  year={{2024}}
}}
```

## License

Apache 2.0
"""

    # Write README to embeddings directory
    readme_path = embeddings_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"\n✓ Created README.md")

    # Upload folder
    print(f"\nUploading embeddings to Hugging Face...")
    print(f"This may take a while ({total_size/1024:.2f} GB)...\n")

    try:
        upload_folder(
            folder_path=str(embeddings_path),
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Upload BGE-Reasoner embeddings for BRIGHT benchmark"
        )
        print(f"\n✓ Successfully uploaded embeddings!")
        print(f"View at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check your token has write permissions")
        print("3. Verify the repository name is available")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload BGE-Reasoner embeddings to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID (e.g., 'username/bge_embedding_bright')"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="./embedding_data/BRIGHT",
        help="Directory containing embeddings (default: ./embedding_data/BRIGHT)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (optional if logged in via CLI)"
    )

    args = parser.parse_args()

    upload_embeddings(
        repo_id=args.repo_id,
        embeddings_dir=args.embeddings_dir,
        private=args.private,
        token=args.token
    )


if __name__ == "__main__":
    main()
