FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

RUN apt-get update && apt-get install -y git python3-virtualenv wget 

RUN pip install -U transformers==4.32.0 accelerate tiktoken einops transformers_stream_generator==0.0.4 scipy
RUN pip install -U optimum==1.13.2 peft
RUN pip install -U gekko
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX" 

COPY ./unpadded-AutoGPTQ unpadded-AutoGPTQ
RUN cd unpadded-AutoGPTQ && pip install -v -e .

WORKDIR /workspace

ENV HUGGINGFACE_TOKEN="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"
ENV HUGGINGFACE_REPO="ycchen/sub_1"

COPY ./qwen_quantize.py qwen_quantize.py

# copy model_configs directory
COPY ./model_configs model_configs

CMD [ "python", "qwen_quantize.py"]
