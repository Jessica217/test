#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Qwen2.5-7B-Instruct
MODEL="/data/huggingface_models/Qwen2.5-7B-Instruct" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.

#DATA="/root/data/wjy/vip_vul_pro/Fine_tune/general_dataset/filtered_data/Fine_tuning_datasets/user_info_safe_data/user_info_and_safe_train_data.json"
Train_DATA="/root/data/wjy/vip_vul_pro/Fine_tune/general_dataset/300_data/300_data_train.json"
USE_LORA=True
EVAL_DATA="/root/data/wjy/vip_vul_pro/Fine_tune/general_dataset/300_data/300_data_val.json"
RESUME_FROM_CHECKPOINT="/root/data/wjy/vip_vul_pro/Fine_tune/translate/new_checkpoints/checkpoint-46"

export CUDA_VISIBLE_DEVICES=1,0
python finetune.py \
    --model_name_or_path $MODEL \
    --data_path $Train_DATA \
    --eval_data_path  $EVAL_DATA \
    --bf16 True \
    --output_dir /root/data/wjy/vip_vul_pro/Fine_tune/translate/new_checkpoints \
    --num_train_epochs 3 \
    --lora_r 256 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 500 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 8192 \
    --lazy_preprocess True \
    --use_lora ${USE_LORA} \
    --gradient_checkpointing \
    --load_best_model_at_end True \
    --ignore_data_skip True \
    --resume_from_checkpoint $RESUME_FROM_CHECKPOINT

    # > finetune2.log 2>&1 &
    #    --q_lora ${Q_LORA} \
#    --deepspeed ${DS_CONFIG_PATH}
