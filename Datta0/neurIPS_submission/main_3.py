from fastapi import FastAPI

import logging
import os
import time
import uvicorn

import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_recipes.inference.model_utils import load_peft_model
from peft import PeftConfig

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
kwargs = {
    'trust_remote_code' : True,
    'cache_dir':'/home/datta0/.cache/huggingface/hub'
}

logger = logging.getLogger(__name__)
# Configure the logging module
# logging.basicConfig(level=logging.INFO)

login(token='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')

# base_model_name = os.environ.get("HUGGINGFACE_BASE_MODEL",'meta-llama/Llama-2-13b-hf')
# adapter_name = os.environ.get("HUGGINGFACE_REPO",'Qwen/Qwen-14B')
books_adapter = 'imdatta0/qwen_books_r16_1e5'
jeopardy_adapter = 'imdatta0/jeopardy'
dolly_adapter = 'imdatta0/qwen_databricks_dolly_15k_r16'
oasst_adapter = 'imdatta0/qwen_oasst_top1_2023_08_25_code_r8_1e5'

# books_config = PeftConfig.from_pretrained(books_adapter, trust_remote_code = True)
# oasst_config = PeftConfig.from_pretrained(oasst_adapter, trust_remote_code = True)
# qns_config = PeftConfig.from_pretrained(qns_adapter, trust_remote_code = True)



model_name = 'Qwen/Qwen-14B'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    # use_flash_attention_2 = True,
    **kwargs,
)

model.load_adapter(books_adapter, "books_adapter")
model.load_adapter(jeopardy_adapter, "jeopardy_adapter")
model.load_adapter(dolly_adapter, "dolly_adapter")
# model.load_adapter(cnn_adapter, "cnn_adapter")
# model.load_adapter(code_adapter, "code_adapter")
model.load_adapter(oasst_adapter, "oasst_adapter")
# # # SET BOOKS ADAPTER BY DEFAULT
model.set_adapter("oasst_adapter")
switch_adapters = True

model.eval()
print(model)

tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)


LLAMA2_CONTEXT_LENGTH = 4096


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)
    
    kwargs = {
        'num_beams':3
    }
    #Mixed-4
    if "Article:" in input_data.prompt and "Summarize the above article in" in input_data.prompt:
        #CNN Daily Mail
        input_data.prompt += '\n'
        input_data.prompt = input_data.prompt.replace('Summarize the above article in 3 sentences.','Summarize the above article in under 3 sentences from the article.')
        if switch_adapters and model.active_adapter()!='books_adapter':
            model.set_adapter('books_adapter')
    elif not input_data.prompt.endswith('Answer:'):
        del kwargs['num_beams']
        if input_data.prompt.startswith('The following paragraphs each describe a set of'):
            #BigBench
            input_data.prompt += '\nWhich of the above is correct\nAnswer:'
        else:
            #coding ig?
            input_data.prompt += '\nAnswer:'
        if switch_adapters and model.active_adapter()!='jeopardy_adapter':
            model.set_adapter('jeopardy_adapter')
    elif input_data.prompt.startswith('Given the definition of the op operator, compute the result.'):
        input_data.prompt = input_data.prompt.replace('result','result in a concise manner.')
        input_data.prompt = input_data.prompt.replace('=','')
        if switch_adapters and model.active_adapter()!='oasst_adapter':
            model.set_adapter('oasst_adapter')
    
    elif input_data.prompt.startswith('Q:'):
        #GSM8k
        if switch_adapters and model.active_adapter()!='jeopardy_adapter':
            model.set_adapter('jeopardy_adapter')
    elif input_data.prompt.startswith('The following are multiple choice questions (with answers) about'):
        #MMLU
        if switch_adapters and model.active_adapter()!='dolly_adapter':
            model.set_adapter('dolly_adapter')
    elif input_data.prompt.startswith('Question:') and input_data.prompt.endswith('Answer:'):
        #TruthFulQA and BigBench emojis?
        del kwargs['num_beams']
        if switch_adapters and model.active_adapter()!='jeopardy_adapter':
            model.set_adapter('jeopardy_adapter')
    elif input_data.prompt.startswith('The following are multiple choice questions (with answers).'):
        #BBQ
        if switch_adapters and model.active_adapter()!='jeopardy_adapter':
            model.set_adapter('jeopardy_adapter')
    else:
        #Default
        del kwargs['num_beams']
        if switch_adapters and model.active_adapter()!='dolly_adapter':
            model.set_adapter('dolly_adapter')
        
    


    encoded = tokenizer(input_data.prompt, return_tensors="pt")
    if encoded['input_ids'].shape[-1]>=1024 and 'num_beams' in kwargs:
        del kwargs['num_beams']
    
    prompt_length = encoded["input_ids"][0].size(0)
    max_returned_tokens = prompt_length + input_data.max_new_tokens
    assert max_returned_tokens <= LLAMA2_CONTEXT_LENGTH, (
        max_returned_tokens,
        LLAMA2_CONTEXT_LENGTH,
    )

    t0 = time.perf_counter()
    encoded = {k: v.to("cuda") for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=input_data.max_new_tokens,
            do_sample=True,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs
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

    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
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
    logger.info(f"Using device: {'GPU'}")
    t0 = time.perf_counter()
    # decoded = tokenizer.decode(torch.Tensor(input_data.tokens))
    decoded = tokenizer.decode(input_data.tokens)
    t = time.perf_counter() - t0
    return DecodeResponse(text=decoded, request_time=t)