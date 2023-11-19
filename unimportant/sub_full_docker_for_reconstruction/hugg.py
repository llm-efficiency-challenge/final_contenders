import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from huggingface_hub import login

model = AutoPeftModelForCausalLM.from_pretrained(
    "mistral_guanaco_1ep",
    device_map="auto", 
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("mistral_guanaco_1ep")

# Merge LoRA and base model
merged_model = model.merge_and_unload(progressbar=True)

# Save the merged model
merged_model.save_pretrained("/submission/mistral_guanaco_1ep_merged")
tokenizer.save_pretrained("/submission/mistral_guanaco_1ep_merged")
