#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.

# Ensure virtual environment is activated if running interactively
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        echo "Activating virtual environment..."
        source .venv/bin/activate
    else
        echo "Error: Virtual environment not found. Please create and activate it, or run this script from the project root." >&2
        exit 1
    fi
fi

# Download TinyStories if not present
if [ ! -f "TinyStoriesV2-GPT4-train.txt" ]; then
    echo "Downloading TinyStoriesV2-GPT4-train.txt..."
    wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt -O TinyStoriesV2-GPT4-train.txt
fi

if [ ! -f "TinyStoriesV2-GPT4-valid.txt" ]; then
    echo "Downloading TinyStoriesV2-GPT4-valid.txt..."
    wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt -O TinyStoriesV2-GPT4-valid.txt
fi

echo "Starting V18 Phase 1 (Predictive Pre-training)..."
python train_phase1.py

echo "Starting V18 Phase 2 (Reasoning Fine-tune)..."
python train_phase2.py

echo "V18 Training Complete."
