import argparse
import os
import sys
import time
import yaml
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

BrightShortInstructions = {
    # StackExchange
    "biology": "Given a Biology post, retrieve relevant passages that help answer the post.",
    "earth_science": "Given an Earth Science post, retrieve relevant passages that help answer the post.",
    "economics": "Given an Economics post, retrieve relevant passages that help answer the post.",
    "psychology": "Given a Psychology post, retrieve relevant passages that help answer the post.",
    "robotics": "Given a Robotics post, retrieve relevant passages that help answer the post.",
    "stackoverflow": "Given a Stack Overflow post, retrieve relevant passages that help answer the post.",
    "sustainable_living": "Given a Sustainable Living post, retrieve relevant passages that help answer the post.",
    # Coding
    "leetcode": "Given a Coding problem, retrieve relevant examples that help answer the problem.",
    "pony": "Given a Pony question, retrieve relevant passages that help answer the question.",
    # Theorem-based
    "aops": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "theoremqa_questions": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "theoremqa_theorems": "Given a Math problem, retrieve relevant theorems that help answer the problem.",
}


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_embeddings(dataset_name, model_name, config, force=False, generate_type="all"):
    print("="*70)
    print(f"Dataset: {dataset_name} | Model: {model_name} | Type: {generate_type}")
    print("="*70)

    if model_name not in config['embedding_models']:
        raise ValueError(f"Model '{model_name}' not found in config")

    model_config = config['embedding_models'][model_name]
    save_path = config['experiment']['embedding_path']
    os.makedirs(save_path, exist_ok=True)

    print(f"\nLoading BRIGHT {dataset_name} dataset...")
    try:
        data = load_dataset('xlangai/BRIGHT', 'examples', keep_in_memory=False)[dataset_name]
        documents = load_dataset('xlangai/BRIGHT', 'documents', keep_in_memory=False)[dataset_name]
        print(f"✓ Loaded {len(data)} queries, {len(documents)} documents")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

    queries = {q["id"]: q["query"] for q in data}
    corpus = {doc["id"]: {"title": "", "text": doc["content"]} for doc in documents}

    print(f"\nLoading model: {model_config['model_id']}")
    device = model_config.get('device', 'cpu')
    if device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = 'cpu'

    try:
        model = SentenceTransformer(model_config['model_id'], device=device)
        print(f"Model loaded on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    if generate_type in ["all", "doc"]:
        print(f"\n{'='*70}")
        print("PHASE 1: Encoding Documents")
        print(f"{'='*70}")

        passage_file = os.path.join(save_path, f"{dataset_name}_{model_name}_passage_embeddings.npy")

        if os.path.exists(passage_file) and not force:
            print(f"File exists: {passage_file}")
        else:
            passages = [v["title"] + " " + v["text"] for v in corpus.values()]
            print(f"Encoding {len(passages)} passages...")

            batch_size = model_config.get('batch_size', 512)
            embeddings_list = []

            for i in tqdm(range(0, len(passages), batch_size), desc="Encoding"):
                batch = passages[i:min(len(passages), i+batch_size)]
                emb = model.encode(
                    batch,
                    batch_size=32,
                    normalize_embeddings=model_config.get('normalize', True),
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings_list.append(emb)

            passage_embeddings = np.vstack(embeddings_list)
            np.save(passage_file, passage_embeddings)
            file_size = os.path.getsize(passage_file) / 1024 / 1024
            print(f"Saved: {passage_file} ({passage_embeddings.shape}, {file_size:.1f} MB)")
    else:
        print(f"\n{'='*70}")
        print("PHASE 1: Skipping Documents (generate_type={generate_type})")
        print(f"{'='*70}")

    if generate_type in ["all", "query"]:
        print(f"\n{'='*70}")
        print("PHASE 2: Encoding Queries")
        print(f"{'='*70}")

        query_file = os.path.join(save_path, f"{dataset_name}_test_{model_name}_query_embeddings.npy")

        if os.path.exists(query_file) and not force:
            print(f"File exists: {query_file}")
        else:
            query_list = list(queries.values())

            if 'bge-reasoner' in model_name and dataset_name in BrightShortInstructions:
                prompt = BrightShortInstructions[dataset_name]
                query_list = [f"Instruct: {prompt}\nQuery: {q}" for q in query_list]

            print(f"Encoding {len(query_list)} queries...")
            query_embeddings = model.encode(
                query_list,
                batch_size=16,
                normalize_embeddings=model_config.get('normalize', True),
                show_progress_bar=True,
                convert_to_numpy=True
            )

            np.save(query_file, query_embeddings)
            print(f"Saved: {query_file} ({query_embeddings.shape})")
    else:
        print(f"\n{'='*70}")
        print("PHASE 2: Skipping Queries (generate_type={generate_type})")
        print(f"{'='*70}")

    print(f"\n{'='*70}\nCompleted: {dataset_name}\n{'='*70}\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for BRIGHT benchmark datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset name or 'all' for all datasets"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bge-reasoner",
        help="Model name (bge-reasoner, bge-large, bge-base, bge-small)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if embeddings exist"
    )
    parser.add_argument(
        "--generate-type",
        type=str,
        default="all",
        choices=["all", "doc", "query"],
        help="Type of embeddings to generate: 'all' (both), 'doc' (documents only), or 'query' (queries only)"
    )

    args = parser.parse_args()

    config = load_config(args.config)
    datasets = config['bright_datasets'] if args.dataset == "all" else [args.dataset]

    start_time = time.time()
    successful, failed = 0, 0

    print(f"\nBGE-Reasoner Embedding Generation\nModel: {args.model}\nDatasets: {len(datasets)}\n")

    for i, dataset in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] Processing {dataset}...")

        try:
            success = generate_embeddings(dataset, args.model, config, args.force, args.generate_type)
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error: {e}")
            failed += 1

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*70}\nSummary: {successful} successful, {failed} failed\nTime: {elapsed:.1f} min\n{'='*70}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
