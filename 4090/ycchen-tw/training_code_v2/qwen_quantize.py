import os
import torch
import glob
import shutil
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login, HfApi

pretrained_model_dir = "Qwen/Qwen-14B"
quantized_model_dir = "Qwen-14B-8bit"

login(token=os.environ["HUGGINGFACE_TOKEN"])

def get_wikitext2(tokenizer):
    import numpy as np
    import torch
    import random
    wikidata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    wikilist = [' \n' if s == '' else s for s in wikidata['text'] ]

    text = ''.join(wikilist)
    trainenc = tokenizer(text, return_tensors='pt')

    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []

    num_example = 120
    seqlen = 4096

    for _ in range(num_example):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({'input_ids':inp,'attention_mask': attention_mask})
    return traindataset

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, trust_remote_code=True, use_fast=True)
examples = get_wikitext2(tokenizer)

quantize_config = BaseQuantizeConfig(
    bits=8,  # quantize model to 8-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    model_file_base_name='model',  # the name of the model file
)

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalLM.from_pretrained(
    pretrained_model_dir,
    quantize_config,
    # device_map="auto",
    trust_remote_code=True,
    # max_memory={0: "22GIB", 1: "22GIB"},
)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
print('Start quantize model')
model.quantize(examples)

# save quantized model
model.save_pretrained(quantized_model_dir, use_safetensors=True)

# copy model configs
for file in glob.glob('model_configs/*'):
    print(file)
    shutil.copy(file, quantized_model_dir)

# os.rename(quantized_model_dir + '/pytorch_model.bin.safetensors', quantized_model_dir + '/model.safetensors')

# upload quantized model to huggingface hub
api = HfApi()
api.create_repo(repo_id=os.environ["HUGGINGFACE_REPO"],
                exist_ok=True,
                )
api.upload_folder(
    folder_path=quantized_model_dir, 
    repo_id=os.environ["HUGGINGFACE_REPO"], 
    repo_type='model', 
)
