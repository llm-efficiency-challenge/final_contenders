FROM ghcr.io/pytorch/pytorch:2.1.0-devel

RUN apt-get update  && apt-get install -y git python3-virtualenv wget 
RUN apt-get install build-essential -y

RUN pip install -U --no-cache-dir git+https://github.com/facebookresearch/llama-recipes.git@eafea7b366bde9dc3f0b66a4cb0a8788f560c793

WORKDIR /workspace
# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

ENV HUGGINGFACE_TOKEN="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"
# ENV HUGGINGFACE_REPO="imdatta0/qwen-oasst"
ENV WANDB_API_KEY='1663281c6220a7c530453cbf8d51869cd0e95580'
ENV WANDB_PROJECT='neurips_submission'
ENV TRAIN_MODEL=false

#Login to Huggingfacehub
RUN echo "Logging in to Huggingfacehub"
RUN huggingface-cli login --token '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'

# Copy over single file server
COPY ./main.py main.py
COPY ./api.py api.py
COPY ./train.py train.py


# Run the server
RUN echo "Inference server ready, hit with helm"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
