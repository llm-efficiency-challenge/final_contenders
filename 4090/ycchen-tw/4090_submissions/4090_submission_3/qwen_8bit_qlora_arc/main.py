from fastapi import FastAPI

import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time

import torch
from huggingface_hub import login
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria
from peft import PeftModel

torch.set_float32_matmul_precision("high")

from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
)

app = FastAPI()

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)

# login(token=os.environ["HUGGINGFACE_TOKEN"])

MODEL_PATH = "ycchen/yc-test1"
LORA_PATH = "ycchen/final-lora-r32-ep5-arc"

print('Model:', MODEL_PATH)
print('LoRA:', LORA_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, stop_sequences, tokenizer):
        super().__init__()
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_token = tokenizer._convert_id_to_token(input_ids[0, -1].item())
        if isinstance(last_token, bytes):
            last_token = last_token.decode()
        to_stop = any([ss in last_token for ss in self.stop_sequences])
        return to_stop

LLAMA2_CONTEXT_LENGTH = 4096


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)

    # print(input_data)
    
    encoded = tokenizer(input_data.prompt, return_tensors="pt")
    
    prompt_length = encoded["input_ids"][0].size(0)
    max_returned_tokens = prompt_length + input_data.max_new_tokens
    assert max_returned_tokens <= LLAMA2_CONTEXT_LENGTH, (
        max_returned_tokens,
        LLAMA2_CONTEXT_LENGTH,
    )

    stop_sequences = input_data.stop_sequences if input_data.stop_sequences is not None else []
    stopping_criteria = StopAtSpecificTokenCriteria(
        stop_sequences=stop_sequences,
        tokenizer=tokenizer,
    )

    t0 = time.perf_counter()
    encoded = {k: v.to("cuda") for k, v in encoded.items()}
    with torch.no_grad():
        eos_token_id = [151643]

        temperature = max(1e-3, input_data.temperature )
        outputs = model.generate(
            **encoded,
            max_new_tokens=input_data.max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=input_data.top_k,
            min_new_tokens=1,
            eos_token_id=eos_token_id,
            stopping_criteria=[stopping_criteria],
            return_dict_in_generate=True,
            output_scores=True,
        )
        # outputs.sequences = outputs.sequences[:, :-1]
        # outputs.scores = outputs.scores[:-1]
    
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
