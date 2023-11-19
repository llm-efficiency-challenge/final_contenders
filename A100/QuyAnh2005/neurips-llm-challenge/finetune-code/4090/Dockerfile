FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

RUN apt-get update  && apt-get install -y git python3-virtualenv wget
RUN pip install -q -U bitsandbytes
RUN pip install -q -U git+https://github.com/huggingface/transformers.git
RUN pip install -q -U git+https://github.com/huggingface/peft.git
RUN pip install -q -U git+https://github.com/huggingface/accelerate.git
RUN pip install datasets scipy protobuf py7zr peft bitsandbytes fire torch_tb_profiler ipywidgets

WORKDIR /workspace

COPY utils.py ./
COPY train.py ./

CMD [ "python", "train.py"]
