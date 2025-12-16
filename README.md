# BGE-Reasoner BRIGHT Reproduction

Reproduce BRIGHT benchmark results using [BGE-Reasoner](https://huggingface.co/BAAI/bge-reasoner-embed-qwen3-8b-0923) with domain-specific prompts and LLM-guided reranking.

## Updates (Dec 2025)

**New Features:**
- Domain-specific instruction prompts for all 12 BRIGHT datasets
- Selective embedding generation (`--generate-type` for doc/query/all)
- Enhanced LLM reranking with domain expertise
- Batch processing scripts for efficient embedding generation
- Hugging Face dataset upload script
- Increased parallelization (36 processes) for faster retrieval

**Recent Results (Psychology Dataset):**

| Algorithm | NDCG@10 | # Reranker Calls | # Tokens | Recall@500 |
|-----------|---------|------------------|----------|------------|
| Retrieve (Baseline) | 0.452 | 0 | 0 | 98.9% |
| Retrieve-and-Rerank | 0.560 | 300 | 277K | 93.1% |
| RetrieveGuidedSearch (RGS) | **0.561** | 290 | 362K | 92.6% |

RGS achieves **56.1% NDCG@10** on psychology dataset with ~290 reranker calls, outperforming baseline retrieval by **10.9 points** while maintaining high recall.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face
huggingface-cli login

# (Optional) Set up conda environment
conda create -n myenv python=3.11
conda activate myenv
```

### Generate Embeddings

```bash
# Single dataset (all embeddings)
python generate_embeddings.py --dataset psychology --model bge-reasoner

# Query embeddings only
python generate_embeddings.py --dataset psychology --model bge-reasoner --generate-type query

# Document embeddings only
python generate_embeddings.py --dataset psychology --model bge-reasoner --generate-type doc

# Batch processing for multiple datasets
bash generate_query_embeddings.sh
```

### Run Retrieval

```bash
# Retrieve-Guided Search (RGS)
python run_retrieval.py --algo_name RGS --dataset_name psychology --model_name bge-reasoner

# Retrieve-and-Rerank baseline
python run_retrieval.py --algo_name RR --dataset_name psychology --model_name bge-reasoner
```

## Dataset

**BRIGHT Benchmark**: 12 reasoning-intensive retrieval datasets across multiple domains

**Domains:**
- **StackExchange**: biology, earth_science, economics, psychology, robotics, stackoverflow, sustainable_living
- **Coding**: leetcode, pony
- **Math**: aops, theoremqa_questions, theoremqa_theorems

**Total**: 1,384 queries + 700K documents

## Models

| Model | Params | GPU Memory | Embedding Dim | Use Case |
|-------|--------|------------|---------------|----------|
| bge-reasoner | 8B | 24GB+ | 1024 | Target (best performance) |
| bge-large | 335M | 8GB+ | 1024 | Strong baseline |
| bge-base | 110M | 4GB+ | 768 | Fast baseline |
| bge-small | 33M | 2GB+ | 384 | Quick testing |

## Domain-Specific Instructions

The implementation uses task-specific prompts for better retrieval quality:

```python
BrightShortInstructions = {
    "biology": "Given a Biology post, retrieve relevant passages that help answer the post.",
    "psychology": "Given a Psychology post, retrieve relevant passages that help answer the post.",
    "leetcode": "Given a Coding problem, retrieve relevant examples that help answer the problem.",
    # ... (12 datasets total)
}
```

Query format: `Instruct: {task_instruction}\nQuery: {user_query}`

## GPU Requirements

### Recommended Hardware
- **BGE-Reasoner**: RTX A6000 (48GB) or A100 (40GB+)
  - Embedding generation: ~2-4h per dataset
  - Retrieval with reranking: ~1-2h per dataset
- **BGE-Large**: RTX 3090 (24GB) or better
  - Embedding generation: ~30min per dataset

### Cloud Options
- **Lambda Labs**: A100 40GB @ $1.10/hr → ~$40-60 total for full benchmark
- **RunPod**: RTX A6000 @ $0.79/hr
- **Colab Pro+**: A100 @ $50/month

## Structure

```
├── config.yaml                   # Model & dataset configuration
├── generate_embeddings.py        # Embedding generation with domain prompts
├── generate_query_embeddings.sh  # Batch script for query embeddings
├── run_retrieval.py              # Retrieval experiments (RGS, RR, SlideGAR)
├── upload_to_huggingface.py      # Upload embeddings to HF Hub
├── test_retrieval.sh             # Test script for single dataset
├── requirements.txt              # Python dependencies
└── results/                      # Experiment results (CSV format)
```

## Expected Results

**Psychology Dataset (This Reproduction):**
- RGS (300 reranks): **NDCG@10 = 56.1%**
- Baseline Retrieval: NDCG@10 = 45.2%
- Improvement: **+10.9 points**

**Original Paper (Average across BRIGHT):**
- BGE-Reasoner + RGS: NDCG@10 ≈ 37.1%
- BGE-Large: NDCG@10 ≈ 30%
- BM25: NDCG@10 ≈ 18%

*Note: Psychology dataset shows higher performance than benchmark average, suggesting domain-specific variation in retrieval difficulty.*

## API Configuration

The retrieval system supports multiple LLM APIs for reranking:

```bash
# Gemini (Recommended)
export GEMINI_API_KEY="your-key"

# OpenRouter
export OPENROUTER_API_KEY="your-key"

# OpenAI
export OPENAI_API_KEY="your-key"
```

Default models:
- Gemini: `gemini-2.0-flash-exp`
- OpenRouter: `google/gemini-2.5-flash`
- OpenAI: `gpt-4o-mini`

## Upload Embeddings

Share your embeddings on Hugging Face:

```bash
python upload_to_huggingface.py \
  --repo-id username/bge-reasoner-bright-embeddings \
  --embeddings-dir ./embedding_data/BRIGHT
```

## Key Implementation Details

1. **Increased Parallelization**: 36 parallel processes for faster retrieval
2. **Domain Expertise**: LLM system prompts include dataset-specific context
3. **Robust Error Handling**: 30s timeout per API call with exponential backoff
4. **Efficient Batching**: Configurable batch sizes for embedding generation
5. **DiskANN Indexing**: Fast approximate nearest neighbor search

## References

- [BGE-Reasoner Model](https://huggingface.co/BAAI/bge-reasoner-embed-qwen3-8b-0923)
- [BRIGHT Benchmark](https://brightbenchmark.github.io/)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [Original RGS Paper](https://github.com/xuhaike/Reranker-Guided-Search)


## License

MIT
