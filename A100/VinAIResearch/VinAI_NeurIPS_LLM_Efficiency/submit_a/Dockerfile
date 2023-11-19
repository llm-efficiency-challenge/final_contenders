# Use an official Python runtime as a parent image
# other options in https://github.com/orgs/pytorch/packages/container/pytorch-nightly/versions?filters%5Bversion_type%5D=tagged
# Lit-GPT requires current nightly (future 2.1) for the latest attention changes
#FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
# FROM ghcr.io/pytorch/pytorch-nightly:c69b6e5-cu11.8.0

# Set the working directory in the container to /submission
WORKDIR /submission

# Copy the specific file into the container at /submission
COPY /hf_mistral/ /submission/

RUN apt-get update && apt-get install -y git wget git-lfs
# Install any needed packages specified in requirements.txt that come from lit-gpt plus some optionals
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/TokisakiKurumi2001/transformers_mistral.git
RUN pip install accelerate peft

RUN python download.py --model mistralai/Mistral-7B-v0.1
RUN git lfs install
# RUN git clone https://huggingface.co/pminhtamnb/hf_mistral_v2
git clone https://huggingface.co/artidoro/model-vinaia
RUN python merge_and_save.py --lora_path model-vinaia
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
