from api import ProcessRequest, Token, ProcessResponse, TokenizeRequest, TokenizeResponse, DecodeRequest, DecodeResponse

import time
import logging
import uvicorn
import torch
import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from fastapi import FastAPI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()
model_path: str = 'merged_model'
tokenizer_path: str = 'mistralai/Mistral-7B-v0.1'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)

@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    logger.info("Using device: {}".format(model.device))
    input_text = input_data.prompt

    t0 = time.perf_counter()

    inputs = tokenizer(input_text, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    model.eval()
    outputs = model.generate(**inputs, max_new_tokens=input_data.max_new_tokens, top_p=0.8, temperature=input_data.temperature, do_sample=True)
    output = outputs[0] # actually only 1 sequence is generated

    time_ellapsed = time.perf_counter() - t0

    prompt_length = inputs['input_ids'].shape[1]
    if input_data.echo_prompt is False:
        output_text = tokenizer.decode(output[prompt_length:])
    else:
        output_text = tokenizer.decode(output)

    tokens_generated = output.size(0) - prompt_length
    print(
        f"Time for inference: {time_ellapsed:.02f} sec total, {tokens_generated / time_ellapsed:.02f} tokens/sec"
    )

    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    generated_tokens_server = []
    for t in output:
        generated_tokens_server.append(
            Token(text=tokenizer.decode(t), logprob=0.7, top_logprob={'random': 0.7})
        )
    logprobs_sum = 0.7 * len(generated_tokens_server)
    # Process the input data here
    response = ProcessResponse(
        text=output_text, tokens=generated_tokens_server, logprob=logprobs_sum, request_time=time_ellapsed
    )
    return response

@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    t0 = time.perf_counter()
    encoded = tokenizer.encode(input_data.text)
    t = time.perf_counter() - t0
    return TokenizeResponse(tokens=encoded, request_time=t)

@app.post("/decode")
async def decode(input_data: DecodeRequest) -> DecodeResponse:
    t0 = time.perf_counter()
    decoded = tokenizer.decode(input_data.tokens, skip_special_tokens=True)
    t = time.perf_counter() - t0
    return DecodeResponse(text=decoded, request_time=t)