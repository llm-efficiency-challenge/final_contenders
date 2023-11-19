import torch
from peft import AutoPeftModelForCausalLM
from huggingface_hub import login


huggingface_write_token="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"
login(token=huggingface_write_token)
repo_name = "agoncharenko1992/repo_example"

peft_model_fp16 = AutoPeftModelForCausalLM.from_pretrained(
    '/train_data/llm_efficient_train/mistral_7b_v0.1_dolly_platty_mmlu_trqa_neftune_all_train/checkpoint-4200',
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    cache_dir='/train_data/.cache',
    use_auth_token=True,
    device_map='cpu',
)

peft_model_fp16.push_to_hub(repo_name)
