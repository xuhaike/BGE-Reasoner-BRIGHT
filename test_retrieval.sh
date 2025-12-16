#!/bin/bash

# Test script for running retrieval on BRIGHT biology dataset with bge-reasoner

# Configuration
DATASET="psychology"
MODEL="bge-reasoner"
BENCHMARK="BRIGHT"
EMBEDDING_PATH="./embedding_data"
ALGO_NAME="RGS"  # Options: RGS (Retrieve-Guided-Search), RR (Retrieve-and-Rerank), SlideGAR
# ALGO_NAME="RR"  # Options: RGS (Retrieve-Guided-Search), RR (Retrieve-and-Rerank), SlideGAR
TIME_TAG=$(date +%Y%m%d_%H%M%S)
GRAPH_DEGREE=16

# Check if embeddings exist
QUERY_EMB="${EMBEDDING_PATH}/${BENCHMARK}/${DATASET}_test_${MODEL}_query_embeddings.npy"
PASSAGE_EMB="${EMBEDDING_PATH}/${BENCHMARK}/${DATASET}_${MODEL}_passage_embeddings.npy"


python run_retrieval.py \
    --benchmark_name "$BENCHMARK" \
    --dataset_name "$DATASET" \
    --model_name "$MODEL" \
    --embedding_path "$EMBEDDING_PATH" \
    --algo_name "$ALGO_NAME" \
    --graph_degree "$GRAPH_DEGREE" \

echo ""
echo "========================================"
echo "Test completed!"
echo "========================================"
echo "Check results in:"
echo "  - Logs: ./experiment_log/${BENCHMARK}_${MODEL}_None_log/"
echo "  - CSV: ./results/${BENCHMARK}_${MODEL}_None_csv/"
