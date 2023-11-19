DATA_NAME_OR_PATH=kowndinya23/flan2022-mistral-512-450K-graphcut-logdet
MODEL_NAME_OR_PATH=mistralai/Mistral-7B-v0.1
OUTPUT_DIR=models/flan2022-512-mistral-graphcut-logdet-sub11-reproduce
HUB_TOKEN=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

export WANDB_API_KEY=ca1c97c6df99411f516b8750cce51b019683a2fa
wandb login --cloud --host https://api.wandb.ai --relogin $WANDB_API_KEY

accelerate launch --config_file config.yaml instruction_tuner.py \
    --dataset_name_or_path $DATA_NAME_OR_PATH \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --hf_access_token $HUB_TOKEN \
    --sliding_window 4096 \
    --torch_dtype bfloat16 \
    --max_seq_length 512 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --preprocessing_num_workers 12 \
    --seed 23 \
    --use_peft \
    --peft_lora_r 64 \
    --peft_lora_alpha 32 \
    --peft_lora_dropout 0.1 \
    --peft_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --weight_decay 0.0 \
    --lr_scheduler_type cosine \
    --lr_warmup_fraction 0.01 \
    --with_tracking \
    --report_to wandb \
    --output_dir $OUTPUT_DIR \
    --push_to_hub \
    --hub_token $HUB_TOKEN \