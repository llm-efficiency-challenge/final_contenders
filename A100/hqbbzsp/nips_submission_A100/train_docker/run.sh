
docker build -t trainer:flash -f Dockerfile.train_flash .
docker run --gpus "device=3" -it -p 127.0.0.1:8111:80 --name trainer_flash trainer:flash
