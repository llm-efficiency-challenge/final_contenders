from fastapi import FastAPI

import logging
import os
import time

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers.generation import GenerationConfig

torch.set_float32_matmul_precision("high")

from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
    DecodeRequest,
    DecodeResponse
)

app = FastAPI()

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)

login(token=os.environ["HUGGINGFACE_TOKEN"])


base_model_name ="Qwen/Qwen-14B"
adapter_name_or_path ="xxyyy123/final_1012_dolly8k_cnn3kD_bbq12k_mmlu29k_alpaca_flash_loraC_r2"

model = AutoModelForCausalLM.from_pretrained(
  base_model_name,
  torch_dtype=torch.float16,
  trust_remote_code=True,
  low_cpu_mem_usage=True,
  device_map="cuda"
)


model = PeftModel.from_pretrained(model, adapter_name_or_path)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name,padding_side="left",trust_remote_code=True)

'''ID_PAD = 151643
ID_EOS = 151643  # endoftext
ID_SOP = 151644  # start
ID_EOP = 151645  # end
ID_BR = 198  # "\n"
tokenizer.pad_token_id = ID_PAD
tokenizer.eos_token_id = ID_EOS'''
#tokenizer.padding_side = "left"  # NO use attention-mask

LLAMA2_CONTEXT_LENGTH = 4096


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)
    profix = '### System: Please generate text that is unbiased with respect to gender and race. Avoid being influenced by case sensitivity, spacing, or excessive use of dialects. Additionally, adapt your response based on the context provided below.If you encounter instructions like "Summarize the above article in 3 sentences," please continue generating text.\n###\n\n'
    prompt = input_data.prompt
    if prompt[-1]=='.': 
        prompt = prompt[:-1]+':'
    elif prompt[-2]=='.': 
        prompt = prompt[:-2]+':'+prompt[-1]
    # print(profix+prompt)
    encoded = tokenizer(profix+prompt, return_tensors="pt").to("cuda:0")

    prompt_length = encoded["input_ids"][0].size(0)
    max_returned_tokens = prompt_length + input_data.max_new_tokens
    assert max_returned_tokens <= LLAMA2_CONTEXT_LENGTH, (
        max_returned_tokens,
        LLAMA2_CONTEXT_LENGTH,
    )

    t0 = time.perf_counter()
    #encoded = {k: v.to("cuda:0") for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=input_data.max_new_tokens,
            do_sample=True,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    t = time.perf_counter() - t0
    if not input_data.echo_prompt:
        output = tokenizer.decode(outputs.sequences[0][prompt_length:], skip_special_tokens=True)
    else:
        output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
    tokens_generated = outputs.sequences[0].size(0) - prompt_length
    logger.info(
        f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"
    )

    logger.info(f"Memory used: {torch.cuda.max_memory_reserved(0) / 1e9:.02f} GB")
    generated_tokens = []
    
    log_probs = torch.log(torch.stack(outputs.scores, dim=1).softmax(-1))

    gen_sequences = outputs.sequences[:, encoded["input_ids"].shape[-1]:]
    gen_logprobs = torch.gather(log_probs, 2, gen_sequences[:, :, None]).squeeze(-1)

    top_indices = torch.argmax(log_probs, dim=-1)
    top_logprobs = torch.gather(log_probs, 2, top_indices[:,:,None]).squeeze(-1)
    top_indices = top_indices.tolist()[0]
    top_logprobs = top_logprobs.tolist()[0]

    for t, lp, tlp in zip(gen_sequences.tolist()[0], gen_logprobs.tolist()[0], zip(top_indices, top_logprobs)):
        idx, val = tlp
        tok_str = tokenizer.decode(idx)
        token_tlp = {tok_str: val}
        generated_tokens.append(
            Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
        )
    logprob_sum = gen_logprobs.sum().item()
    
    return ProcessResponse(
        text=output, tokens=generated_tokens, logprob=logprob_sum, request_time=t
    )


@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    t0 = time.perf_counter()
    encoded = tokenizer(
        input_data.text
    )
    t = time.perf_counter() - t0
    tokens = encoded["input_ids"]
    return TokenizeResponse(tokens=tokens, request_time=t)

@app.post("/decode")
async def decode(input_data: DecodeRequest) -> DecodeResponse:
    t0 = time.perf_counter()
    decoded = tokenizer.decode(input_data.tokens, skip_special_tokens=True)
    t = time.perf_counter() - t0
    return DecodeResponse(text=decoded, request_time=t)
