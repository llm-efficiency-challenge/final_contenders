docker build -f ./Dockerfile.push_local_results_to_hf_repo -t datapush . 

docker run --gpus "device=0" --rm -ti -v /4tb_second/goncharenko_files/test_train_docker:/train_data -v /home/goncharenko/.cache/:/train_data/.cache/ datapush

