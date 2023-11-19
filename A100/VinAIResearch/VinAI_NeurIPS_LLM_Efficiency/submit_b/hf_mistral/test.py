import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import time
from typing import List, Dict, Optional

@dataclass
class ProcessRequest:
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.8
    echo_prompt: Optional[bool] = None

@dataclass
class Token:
    text: str
    logprob: float
    top_logprob: Dict[str, float]

@dataclass
class ProcessResponse:
    text: str
    tokens: List[Token]
    logprob: float
    request_time: float

if __name__ == "__main__":

    input_data = ProcessRequest(prompt='Greeting', max_new_tokens=50, temperature=0.8, echo_prompt=False)

    model_path: str = 'mistralai/Mistral-7B-v0.1'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print('Loading model ...')
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    input_text = input_data.prompt

    t0 = time.perf_counter()

    inputs = tokenizer(input_text, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

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
    breakpoint()