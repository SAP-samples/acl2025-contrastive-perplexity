#!/bin/bash
#
# SPDX-FileCopyrightText: 2025 SAP SE or an SAP affiliate company
#
# SPDX-License-Identifier: Apache-2.0
#
# Example script for training Mistral with Contrastive Perplexity
# Usage: bash scripts/train_example.sh

# Exit on error
set -e

# Configuration
MODEL_NAME="mistralai/Mistral-7B-v0.1"
OUTPUT_DIR="results/mistral_contrastive"
DATA_DIR="data/paraphrase/safeNLP_processed"

echo "Starting training with Mistral..."
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"

# Run training
# Note: Ensure you have sufficient GPU memory (approx 16GB for 4-bit, 40GB+ for full precision)
python train_mistral.py \
    --model_name "$MODEL_NAME" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 6 \
    --learning_rate 2.2e-5 \
    --num_train_epochs 5 \
    --num_pos 5 \
    --num_neg 5 \
    --alpha 100.0 \
    --smoothing 0.0 \
    --bf16 \
    --use_4bit_quantization \
    --lora_r 64 \
    --lora_target_modules q_proj,v_proj \
    --num_workers 10 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.005 \
    --weight_decay 0.0 \
    --use_nested_quant

echo "Training complete!"
