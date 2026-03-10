set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_MODEL="${BASELINE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
CLIP_MODEL="${CLIP_MODEL:-${SCRIPT_DIR}/clip_qwen}"
DATASET_DIR="${DATASET_DIR:-${SCRIPT_DIR}/dataset/LucasFang/FLUX-Reason/test}"

NUM_SAMPLES="${NUM_SAMPLES:-31}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

if [ ! -d "$DATASET_DIR" ]; then
  echo "ERROR: Dataset directory not found: $DATASET_DIR"
  exit 1
fi

PYBIN="${PYBIN:-python -u}"

echo "==> Evaluating candidate (CLIP LLaVA)"
$PYBIN eval.py \
  --model "$CLIP_MODEL" \
  --model_type llava \
  --dataset_dir "$DATASET_DIR" \
  --num_samples "$NUM_SAMPLES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --use_custom_model clip \
  --use_lora NO