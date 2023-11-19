from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, accelerate
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage='Download model')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.1')
    args = parser.parse_args()
    model_path: str = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )