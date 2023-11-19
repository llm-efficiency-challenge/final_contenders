model_name_or_path='Qwen/Qwen-14B'
sft_checkpoint='../final_v3_test'
dataset='1025_dolly8k_cnn4kD_bbq8ks_mmlu19kRAW_sci6k_alpaca_plus'

CUDA_VISIBLE_DEVICES=0 python src/train_bash_limit.py \
    --stage sft \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --dataset $dataset \
    --template simple \
    --finetuning_type lora \
    --output_dir $sft_checkpoint \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --plot_loss \
    --max_grad_norm 1.0 \
    --fp16 \
    --lora_rank 3 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --save_steps 200000 \
    --lora_target "c_proj" \
    --ddp_find_unused_parameters False \
    --cutoff_len 4096 \
    --warmup_steps 100 \
    --group_by_length False \
    --preprocessing_num_workers 128


# upload
cd ..
python upload_model.py