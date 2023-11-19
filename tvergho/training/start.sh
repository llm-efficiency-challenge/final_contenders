#!/bin/bash

wget https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model-00001-of-00002.bin -P /app/model/Mistral-7B-v0.1
wget https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model-00002-of-00002.bin -P /app/model/Mistral-7B-v0.1

# Run datagen.py
python datagen.py

# Change directory to lit-gpt and run prepare_csv.py
cd lit-gpt
python scripts/prepare_csv.py --csv_path ../lima-platypus-final.csv --max_seq_length 2048

# Run convert_hf_checkpoint.py
python scripts/convert_hf_checkpoint.py --checkpoint_dir ../model/Mistral-7B-v0.1

# Run lora.py with specified arguments
python finetune/lora.py --quantize bnb.nf4-dq --precision bf16-true

python scripts/merge_lora.py --lora_path out/lora/lit_model_lora_finetuned.pth --checkpoint_dir ../model/Mistral-7B-v0.1 --out_dir out
python scripts/convert_lit_checkpoint.py --checkpoint_path out/lit_model.pth --output_path out/pytorch_model.bin --config_path ../model/Mistral-7B-v0.1/lit_config.json