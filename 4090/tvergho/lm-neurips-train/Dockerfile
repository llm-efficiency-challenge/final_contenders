FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set the working directory in the container to /submission
WORKDIR /app
RUN apt-get update && apt-get install -y wget git && apt-get clean

# Setup server requirements
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Install any needed packages specified in requirements.txt that come from lit-gpt plus some optionals
RUN pip install huggingface_hub tokenizers scipy accelerate
RUN pip install flash-attn --no-build-isolation

# Copy over scripts
COPY ./main.py /app/main.py
COPY ./api.py /app/api.py
COPY ./dola.py /app/dola.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

RUN pip install httpx
RUN pip install git+https://github.com/tvergho/transformers.git@dola

# Download the model and tokenizer
RUN mkdir /app/model
RUN wget https://huggingface.co/tvergho/neurips-training/resolve/main/mistral-platypus-lima-3ep.bin -P /app/model/
RUN wget https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/config.json -P /app/model/
RUN wget https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/generation_config.json -P /app/model/
RUN wget https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.json -P /app/model/
RUN wget https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.model -P /app/model/
RUN wget https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer_config.json -P /app/model/
RUN wget https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/special_tokens_map.json -P /app/model/

RUN mv /app/model/mistral-platypus-lima-3ep.bin /app/model/pytorch_model.bin

RUN pip install pandas numpy tqdm

# Use the start script as the entrypoint
ENTRYPOINT ["/app/start.sh"]
