#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/data/datasets/.envs/safe_env/bin/python"
TOP_K="${TOP_K:-10}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen2.5-3B-Instruct}"

cd "$ROOT_DIR"

if [ "$#" -gt 0 ]; then
  "$PYTHON_BIN" experiments/run_visualize_logits_batch.py \
    --top-k "$TOP_K" \
    --model-name "$MODEL_NAME" \
    "$@"
else
  "$PYTHON_BIN" experiments/run_visualize_logits_batch.py \
    --top-k "$TOP_K" \
    --model-name "$MODEL_NAME" \
    results/1_math_reasoning.pt \
    results/2_math_reasoning.pt \
    results/3_math_reasoning.pt \
    results/4_math_reasoning.pt \
    results/5_math_reasoning.pt \
    results/6_math_reasoning.pt \
    results/7_math_reasoning.pt \
    results/8_math_reasoning.pt \
    results/9_math_reasoning.pt \
    results/10_math_reasoning.pt \
    results/11_math_reasoning.pt \
    results/12_math_reasoning.pt
fi
