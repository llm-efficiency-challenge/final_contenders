# Important

The latest training code and Readme are in the "training_code_v2" directory.

# NeurIPS LLM Efficiency Challenge Solution

This repository presents the training and inference code developed by team ycchen.

We participated in the RTX4090 track.

We are graduate students at National Taiwan University.

## Method Introduction

Our method is described as follows:

- Base Model: 8-bit [GPTQ](https://github.com/wangitu/unpadded-AutoGPTQ) quantized [Qwen-14B](https://huggingface.co/Qwen/Qwen-14B), using 120 pieces of [Wikitext](https://huggingface.co/datasets/wikitext) as calibration data.

- Training Data: [Open Assistant](https://huggingface.co/datasets/OpenAssistant/oasst1), [LIMA](https://huggingface.co/datasets/GAIR/lima), [ARC](https://huggingface.co/datasets/ai2_arc), [C4](https://huggingface.co/datasets/c4) dataset, combined with [Humpback](https://arxiv.org/abs/2308.06259) self-alignment technology for data augmentation. 

- Training Method: [QLoRA](https://arxiv.org/abs/2305.14314)

Our three submissions are:

1. 4090_submissions_1.zip: Pure Qwen-14B quantized model, without fine-tuning.

2. 4090_submissions_2.zip: QLoRA instruction tuning using Open Assistant and LIMA.

3. 4090_submissions_3.zip: QLoRA instruction tuning using data from Open Assistant, LIMA, ARC and ~~C4 based Humpback dataset~~ (Upon checking, we found that our training code actually did not load the Humpback results 😂).

Results: 

1. 4090_submissions_1.zip: Score 0.6458

2. 4090_submissions_2.zip: Score 0.5845  

3. 4090_submissions_3.zip: Score 0.5954

Interestingly, our best-performing model was the one that was only quantized but not fine-tuned, Qwen-14B. This was an unexpected outcome 😢.

However, our respectable ranking was likely due to the use of GPTQ for higher quality quantization, and we precisely adjusted the generation settings (such as minimum, maximum token length, and stop criteria) to ensure the model produced outputs that met the challenge requirements.

## Training Code

Considering the above results, our training code essentially only involves the quantization code of Qwen-14B.

The training code is available in the folder named 'training_code'. It utilizes auto-gptq to quantize Qwen-14B and uploads it to the Hugging Face Hub. To start, replace "YOUR_TOKEN" and "YOUR_USERNAME/YOUR_REPO" in the Dockerfile with your personal token and repository details. Execute the following commands:

```bash
docker build -f ./Dockerfile -t qwen_quant .

docker run --gpus "device=0" --rm -ti qwen_quant
```

The program can complete in approximately 2 hours using a single RTX 3090 GPU.

Additionally, upon request, we also provide the training code for submission 2 and submission 3.
They are located in the folders 'training_code_submission_2' and 'training_code_submission_3' respectively.
The 'gen_dataset.ipynb' is for organizing the training dataset, and 'qlora.ipynb' is for training the model.

## Inference Code

After completing Qwen quantization, please replace MODEL_PATH in main.py of 4090_submissions_1.zip with "YOUR_USERNAME/YOUR_REPO" (originally "ycchen/yc-test1"). LORA_PATH can be ignored, because it does not actually participate in the subsequent program.

Then execute the Dockerfile in 4090_submissions_1.zip. 

## Data Format

The submissions for the 4090 challenge are contained within the '4090_submissions' folder, which includes the following files:

- 4090_submissions/

  - 4090_submissions_1.zip
  
  - 4090_submissions_2.zip
  
  - 4090_submissions_3.zip
  
# NeurIPS 1 LLM 1 GPU Challenge

---
