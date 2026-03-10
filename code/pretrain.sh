
#!/usr/bin/env bash
export MASTER_PORT=29501
export TORCH_DISTRIBUTED_DEFAULT_PORT=29501

# export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
export CUDA_VISIBLE_DEVICES=0,1


export HF_ENDPOINT=https://hf-mirror.com

export HF_HOME="/huggingface_cache"

export TRANSFORMERS_CACHE="$HF_HOME"


export TRITON_CACHE_DIR=/tmp/$USER/triton_cache
mkdir -p $TRITON_CACHE_DIR

MODEL_NAME="dinov_llama3"

export TRITON_CACHE_DIR=/triton_cache   

deepspeed --master_port $MASTER_PORT --include localhost:0,1 pretrain_reason.py \
  --deepspeed zero2.json \
  --model_name_or_path /work/${MODEL_NAME} \
  --output_dir /work/${MODEL_NAME}/reason_trained \
  --data_path /work/dataset/FLUX-Reason/train \
  --train_type tune_mm_mlp_adapter \
  --bf16 true \
  --tf32 true \
  --dataloader_num_workers 10 \
  --dataloader_pin_memory true \
  --dataloader_persistent_workers true \
  --num_train_epochs 2 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 10000 \
  --save_total_limit 3 \
  --report_to "tensorboard" \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type "cosine" \
  --gradient_checkpointing true \
  --logging_steps 20 \
  --report_to none \

