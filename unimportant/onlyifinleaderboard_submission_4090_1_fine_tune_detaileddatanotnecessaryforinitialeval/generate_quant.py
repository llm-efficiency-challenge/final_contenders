from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from auto_gptq import AutoGPTQForCausalLM
import torch
import gc
import time

def load_model_and_tokenizer():
    start_time = time.time()
    model_name = "llama-2-7b-hf-gptq"
    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        use_safetensors=True, 
        inject_fused_attention=False, 
        device="cuda:0"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    end_time = time.time()

    loading_time = end_time - start_time
    print(f"Model loading time: {format(loading_time, '.2f')} seconds")
    return model, tokenizer

def generate_text(prompt, model, tokenizer):
    start_time = time.time()
    input_data = tokenizer(prompt, return_tensors="pt").to(model.device)
    result_tensor = model.generate(**input_data, max_length=2048)
    result = tokenizer.decode(result_tensor[0])
    end_time = time.time()

    tokens_generated = len(tokenizer.tokenize(result))
    time_taken = end_time - start_time
    tokens_per_second = tokens_generated / time_taken

    print(result)
    print(f"Tokens generated: {tokens_generated}")
    print(f"Time taken: {format(time_taken, '.2f')} seconds")
    print(f"Tokens per second: {format(tokens_per_second, '.2f')}")
    return result

def bytes_to_giga_bytes(bytes):
    return format(bytes / 1024 / 1024 / 1024, '.2f')

def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Main execution
model, tokenizer = load_model_and_tokenizer()

# Specify an appropriate prompt for text generation
prompt = "Creative ways of..."
generate_text(prompt, model, tokenizer)

# Print max memory utilization
print(f"Max memory utilization: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB")

# Flush memory
flush_memory()
