# Use the Hugging Face transformers PyTorch GPU base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install Git and Git LFS
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install

# Set the working directory
WORKDIR /submission

# Copy the local files into the container
COPY . /submission


# Merge requirements from requirements.txt and other dependencies
RUN pip install pandas torch datasets peft transformers tqdm numpy~=1.24.3 matplotlib~=3.7.1 \
    scikit-learn~=1.2.2 joblib~=1.2.0 trl fastapi pydantic einops uvicorn \
    git+https://github.com/huggingface/transformers.git scipy ninja wheel

# Clone the bitsandbytes repository
RUN pip install bitsandbytes

RUN pip install accelerate==0.24.1

# Install flash-attn with specific MAX_JOBS
ENV MAX_JOBS=4
RUN pip install flash-attn --no-build-isolation

# Set the command to be run when starting the container
CMD python3 /submission/finetune.py && python3 /submission/hugg.py && python3 -m uvicorn main:app --host 0.0.0.0 --port 80
