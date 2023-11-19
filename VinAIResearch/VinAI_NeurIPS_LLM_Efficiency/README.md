# VinAI NeurIPS LLM Efficiency

## Details

Base model: Mistral-7B-v0.1 from [huggingface](https://huggingface.co/mistralai/Mistral-7B-v0.1)

There are in total 3 submissions for the track of A100.

- `submit_a`
- `submit_b`
- `submit_c`

In each directory, there a source code for starting the FastAPI server (`hf_mistral`) and a `Dockerfile`.

The `Dockerfile` follows the `Dockerfile` in `lit-gpt` from the sample-submission of the challenge organizers, with server running on 80, export: `/process`, `/tokenizer` and `/decode` endpoint.

The dataset is available on [huggingface](https://huggingface.co/datasets/transZ/efficient_llm)