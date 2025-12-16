#!/bin/bash

# Script to generate query embeddings for all BRIGHT datasets with bge-reasoner

# Configuration
MODEL="bge-reasoner"
GENERATE_TYPE="query"  # Only generate query embeddings

# List of all BRIGHT datasets
DATASETS=(
    # "biology"
    # "earth_science"
    # "economics"
    # "psychology"
    "robotics"
    "stackoverflow"
    # "sustainable_living"
    # "leetcode"
    # "pony"
    # "aops"
    # "theoremqa_questions"
    # "theoremqa_theorems"
)

# Activate conda environment
source $HOME/miniconda3/bin/activate myenv

echo "========================================"
echo "BGE-Reasoner Query Embedding Generation"
echo "========================================"
echo "Model: $MODEL"
echo "Type: $GENERATE_TYPE"
echo "Total datasets: ${#DATASETS[@]}"
echo "========================================"
echo ""

# Loop through all datasets
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Processing dataset: $DATASET"
    echo "------------------------------------------------------------"

    # Generate query embeddings for this dataset
    python generate_embeddings.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --generate-type "$GENERATE_TYPE"

    if [ $? -eq 0 ]; then
        echo "✓ Successfully generated embeddings for: $DATASET"
    else
        echo "✗ Failed to generate embeddings for: $DATASET"
    fi
done

echo ""
echo "========================================"
echo "All query embedding generation completed!"
echo "========================================"
