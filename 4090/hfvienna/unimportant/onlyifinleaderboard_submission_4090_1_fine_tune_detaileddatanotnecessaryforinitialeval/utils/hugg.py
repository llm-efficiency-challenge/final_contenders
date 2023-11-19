import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from huggingface_hub import login

model = AutoPeftModelForCausalLM.from_pretrained(
    "mistral_openbookqa",
    device_map="auto", 
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained("mistral_openbookqa")
# Merge LoRA and base model
merged_model = model.merge_and_unload(progressbar=True)

# Save the merged model
merged_model.save_pretrained("mistral_openbookqa_merged")
tokenizer.save_pretrained("mistral_openbookqa_merged")

login(token="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")

# push merged model to the hub
merged_model.push_to_hub("attkap/mistral-openbook_qa_2ep", token="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")
tokenizer.push_to_hub("attkap/mistral-openbook_qa_2ep")
