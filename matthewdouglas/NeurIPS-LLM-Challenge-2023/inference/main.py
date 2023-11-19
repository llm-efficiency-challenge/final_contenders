import time
import torch
import logging
import os

from fastapi import FastAPI

from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
    DecodeRequest,
    DecodeResponse
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

torch.set_float32_matmul_precision("high")

app = FastAPI()

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mdouglas/Mistral-7B-sft-v1"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    use_flash_attention_2=True,
)

# Load a second adapter for multiple-choice question tasks
model.load_adapter("mdouglas/Mistral-7B-sft-v0", adapter_name="mcq")

def prompt_format(text: str) -> str:
    template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction\n{instruction}\n### Response\n"
    return template.format(instruction=text)

@app.post("/tokenize")
async def tokenize(input: TokenizeRequest) -> TokenizeResponse:
    t0 = time.perf_counter()
    tokens = tokenizer.encode(input.text)
    t = time.perf_counter() - t0
    return TokenizeResponse(tokens=tokens, request_time=t)

@app.post("/process")
async def process_request(input: ProcessRequest) -> ProcessResponse:
    if input.seed is not None:
        torch.manual_seed(input.seed)

    if input.max_new_tokens > 1:
        model.set_adapter("default")
        input.prompt = prompt_format(input.prompt)
    else:
        # Likely a few-shot multiple-choice question
        model.set_adapter("mcq")

    encoded = tokenizer([input.prompt], return_tensors="pt").to(model.device)
    prompt_length = encoded["input_ids"][0].size(0)    # type: ignore
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=input.max_new_tokens,
            do_sample=True,
            temperature=input.temperature,
            top_k=input.top_k,
            return_dict_in_generate=True,
            output_scores=True
        )

    t = time.perf_counter() - t0
    tokens_generated = outputs.sequences[0].size(0) - prompt_length

    if not input.echo_prompt:
        output = tokenizer.decode(outputs.sequences[0][prompt_length:], skip_special_tokens=True)
    else:
        output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    logger.info(f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec")
    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    return ProcessResponse(
        text=output,
        request_time=t,
        tokens=[],
        logprob=0
    )

@app.post("/decode")
async def decode(input: DecodeRequest) -> DecodeResponse:
    t0 = time.perf_counter()
    decoded = tokenizer.decode(input.tokens)
    t = time.perf_counter() - t0
    return DecodeResponse(text=decoded, request_time=t)