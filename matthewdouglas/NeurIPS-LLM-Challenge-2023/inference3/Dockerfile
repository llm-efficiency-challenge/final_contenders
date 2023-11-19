FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y git

WORKDIR /submission

# Install all dependencies.
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN MAX_JOBS=4 pip install flash-attn==2.3.3 --no-build-isolation 

COPY ./main.py /submission/main.py
COPY ./api.py /submission/api.py

# Startup a FastAPI inference server.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]