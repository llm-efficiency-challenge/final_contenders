from fastapi import FastAPI, Depends, HTTPException
import httpx
from pydantic import BaseModel
from typing import List, Dict, Optional
from transformers import MistralForCausalLM, AutoTokenizer, StoppingCriteria
from api import (
  ProcessRequest,
  ProcessResponse,
  TokenizeRequest,
  TokenizeResponse,
  Token,
  DecodeRequest,
  DecodeResponse
)
import torch
from dola import DoLa

app = FastAPI()
timeout = httpx.Timeout(20.0, read=50.0)

BASE_URL = "http://127.0.0.1:9000"
USE_INSTRUCTION_TEMPLATE = False
USE_LOCAL_MODEL = True

# Assuming you have your model and tokenizer saved in this directory
MODEL_DIR = "/app/model"

class MistralStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.eos_sequence = [13, 13]

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        last_2_ids = input_ids[:, -2:].tolist()[0]
        return self.eos_sequence == last_2_ids

# Loading model and tokenizer if using local model
if USE_LOCAL_MODEL:
  tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
  model = DoLa(MODEL_DIR, "cuda", 1)
  # model.set_stop_words(["\n\n"])
  model.stopping_criteria = [MistralStoppingCriteria(tokenizer)]
  # model = MistralForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)
  # model.to("cuda")
  print("Model loaded")

def apply_instruction_template(text: str) -> str:
    if USE_INSTRUCTION_TEMPLATE:
      return f"[INST] {text} [/INST]"
    return text

async def tokenize_with_transformers(text: str):
    encoded = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
    return {"tokens": encoded.tolist()[0]}

async def process_with_transformers(prompt: str, max_new_tokens, **kwargs):
    # Encode the prompt
    # encoded_prompt = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    # encoded_prompt = encoded_prompt.to(model.device)

    mature_layer = 32
    premature_layer = None
    candidate_premature_layers = list(range(1, 32))

    # Generate output using the custom stopping criteria
    output = model.generate(
        prompt,
        # encoded_prompt,
        # max_length=encoded_prompt.shape[1] + max_new_tokens,
        mode="dola",
        remove_stop_words=True,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.2,
        # stopping_criteria=[MistralStoppingCriteria(tokenizer)]
        mature_layer=mature_layer,
        premature_layer=premature_layer,
        candidate_premature_layers=candidate_premature_layers
    )

    # Only keep the new generated text, excluding the original prompt
    # decoded_output = tokenizer.decode(output[0][encoded_prompt.shape[1]:], skip_special_tokens=True)
    print(output)
    return {"content": output[0]}

@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize_endpoint(request: TokenizeRequest):
    text = apply_instruction_template(request.text)
    
    if USE_LOCAL_MODEL:
        response_data = await tokenize_with_transformers(text)
    else:
        data = {"content": text}
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{BASE_URL}/tokenize", json=data)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail="Tokenization failed")
        response_data = r.json()
        response_data["tokens"] = [1] + response_data["tokens"]
        
    response = {
        "tokens": response_data["tokens"],
        "request_time": 0.0  # Placeholder
    }
    
    return response

@app.post("/process", response_model=ProcessResponse)
async def process_endpoint(request: ProcessRequest):
  if USE_LOCAL_MODEL:
    text = apply_instruction_template(request.prompt)
    completion_response = await process_with_transformers(text, request.max_new_tokens)
    tokenized_response = await tokenize_with_transformers(text)
    tokens = tokenized_response["tokens"]
  else:
    text = request.prompt
    tokenize_data = TokenizeRequest(text=text)
    tokenized_response = await tokenize_endpoint(tokenize_data)
    # tokenized_response = await tokenize_with_transformers(text)
    tokens = tokenized_response["tokens"]

    # Generating completion
    completion_data = {
      "prompt": tokens,
      "n_predict": request.max_new_tokens,
      "top_k": request.top_k,
      "temperature": 0.0,
      "seed": request.seed if request.seed is not None else -1,
      "stop": ["</s>"],
      "top_p": 1.0,
      "repeat_penalty": 1.0,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
      r = await client.post(f"{BASE_URL}/completion", json=completion_data)

    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail="Completion failed")

    completion_response = r.json()

  # Generating a response based on the ProcessResponse class
  response = {
      "text": completion_response["content"],
      "tokens": [{"text": str(t), "logprob": 0.0, "top_logprob": {}} for t in tokens],  # logprob and top_logprob are placeholders
      "logprob": 0.0,  # Placeholder
      "request_time": 0.0
  }
  return response
