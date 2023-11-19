FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get update && apt-get install -y git python3-virtualenv wget 

RUN pip install -U transformers==4.32.0 accelerate tiktoken einops transformers_stream_generator==0.0.4 scipy
RUN pip install -U optimum peft
RUN pip install -U gekko
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX" 
RUN git clone https://github.com/wangitu/unpadded-AutoGPTQ.git && cd unpadded-AutoGPTQ && pip install -v -e .

WORKDIR /workspace

ENV HUGGINGFACE_TOKEN="YOUR_TOKEN"
ENV HUGGINGFACE_REPO="YOUR_USERNAME/YOUR_REPO"

COPY ./qwen_quantize.py qwen_quantize.py

CMD [ "python", "qwen_quantize.py"]
