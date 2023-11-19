FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y git python3-pip

WORKDIR /workspace

RUN pip3 install packaging==23.2 ninja==1.11.1.1 --no-cache-dir
RUN pip3 install torch==2.0.1+cu118 --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip3 install flash-attn --no-build-isolation --no-cache-dir

RUN git clone https://github.com/OpenAccess-AI-Collective/axolotl.git \
    && cd axolotl \
    && git checkout 6c81c61b \
    && pip3 install -e . --no-cache-dir

COPY ./prepare.py /workspace/prepare.py
COPY ./sft-v0.yml /workspace/sft-v0.yml
COPY ./sft-v1.yml /workspace/sft-v1.yml

RUN python3 ./prepare.py

RUN huggingface-cli download mistralai/Mistral-7B-v0.1

CMD accelerate launch -m axolotl.cli.train sft-v1.yml