FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN apt-get update  && apt-get install -y git python3-virtualenv wget 

WORKDIR /workspace

RUN pip uninstall -y flash-attn
RUN pip uninstall -y transformer-engine
RUN pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy over single file server
COPY ./main.py main.py
COPY ./api.py api.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]