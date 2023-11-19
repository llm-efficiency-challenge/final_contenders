# This is the training code of our submission 1.

Our training code essentially only involves the quantization code of Qwen-14B.

It utilizes auto-gptq to quantize Qwen-14B and uploads it to the Hugging Face Hub. To start, replace "YOUR_TOKEN" and "YOUR_USERNAME/YOUR_REPO" in the Dockerfile with your personal token and repository details. Execute the following commands:

```bash
docker build -f ./Dockerfile -t qwen_quant .

docker run --gpus "device=0" --rm -ti qwen_quant
```

The program can complete in approximately 1~2 hours using a single RTX 3090 GPU.
