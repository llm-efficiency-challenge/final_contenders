FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set the working directory in the container to /submission
WORKDIR /app
RUN apt-get update && apt-get install -y wget git && apt-get clean

# Setup server requirements
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install huggingface_hub tokenizers scipy accelerate transformers
RUN pip install flash-attn --no-build-isolation
RUN pip install pandas numpy tqdm

# Install lit-gpt
COPY ./lit-gpt /app/lit-gpt
WORKDIR /app/lit-gpt
RUN pip install -e .
WORKDIR /app

# Copy the rest of the submission
COPY . /app
RUN chmod +x /app/start.sh

# Download the model and tokenizer
# RUN wget https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model-00001-of-00002.bin -P /app/model/Mistral-7B-v0.1
# RUN wget https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model-00002-of-00002.bin -P /app/model/Mistral-7B-v0.1

RUN huggingface-cli login --token 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

RUN pip install bitsandbytes==0.41.0

# Use the start script as the entrypoint
ENTRYPOINT ["/app/start.sh"]
