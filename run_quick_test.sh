#!/bin/bash

echo "=== Quick LLM Test Script ==="
echo "Testing model with single input sample"
echo

# Check if model files exist
if [ ! -d "resources/biomistral-7b" ]; then
    echo "❌ Error: resources/biomistral-7b not found!"
    echo "Please download BioMistral-7B model to resources/biomistral-7b/"
    exit 1
fi

if [ ! -d "saved/lora-0105/checkpoint-3000" ]; then
    echo "❌ Error: saved/lora-0105/checkpoint-3000 not found!"
    echo "Please train the model first using: bash experiment/llm_cls_t4_gpu.bash"
    exit 1
fi

if [ ! -f "data/mimic3/handled/voc_final.pkl" ]; then
    echo "❌ Error: data/mimic3/handled/voc_final.pkl not found!"
    echo "Please prepare the data first"
    exit 1
fi

echo "✓ All required files found"
echo

# Run the test
echo "Running quick test..."
python quick_test.py

echo
echo "=== Test completed ==="
echo "Check quick_test_result.json for results"
