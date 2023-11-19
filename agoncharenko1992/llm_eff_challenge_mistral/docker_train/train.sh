docker build -f ./Dockerfile.train -t mistral_train . 

docker run --gpus "device=0" -ti -v /4tb_second/goncharenko_files/test_train_docker:/train_data -v /home/goncharenko/.cache/:/train_data/.cache/ mistral_train

