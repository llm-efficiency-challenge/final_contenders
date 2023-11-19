from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import LoraConfig, PeftModel

import torch
import argparse

def input_path():
    parser = argparse.ArgumentParser(description='Path for LoRA')
    parser.add_argument('--lora_path', type=str)
    args = parser.parse_args()
    return args.lora_path

if __name__ == "__main__":
    trained_model_path = input_path()

    model_name = 'mistralai/Mistral-7B-v0.1'
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map='auto',
    )

    merged_model= PeftModel.from_pretrained(base_model, trained_model_path)
    merged_model= merged_model.merge_and_unload()

    # Save the merged model
    merged_model.save_pretrained("merged_model",safe_serialization=True)