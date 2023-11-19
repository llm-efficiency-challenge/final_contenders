# nips_submission_A100

## Team Introduction--We are a student team

- My school email: zspnjlgdx@njust.edu.cn

### Team Name

NJUST-A!dge

### Team Members

- Shupeng Zhong
- Yu Xia
- Shifeng Yi
- Qingguo Chen
- Yang Yang

## Project Structure

### Project Overview

Our team made three submissions at track 4090 and A100 respectively. A total of 6 submissions were submitted.

The submission of 4090 track is here: https://github.com/hqbbzsp/nips_submission

### Folder Structure

Project Root/

├── submission_of_A100/

│ ├── submission 1

│ ├── submission 2

│ └── submission 3  (THIS ONE IS WHAT WE CHOOSE)

├── final_dataset/

│ ├── datas

├── train_docker/

│ ├── LLaMA-Efficient-Tuning

└── README.md
## Data

Our dataset is open-sourced on Hugging Face, under the project name https://huggingface.co/datasets/zhongshupeng/dataset_A100, and the specific address is https://huggingface.co/datasets/zhongshupeng/dataset_A100/blob/main/1025_dolly8k_cnn4kD_bbq8ks_mmlu19kRAW_sci6k_alpaca_plus.json . The scripts used to construct our data are located in the 'final_dataset' folder. These scripts can be used to build the same dataset, with an MD5 value of 5a0a36d8c1d5429551cb7302e85df46d4a1b8f52.

## Train Docker
**Note**: When training our earlier submissions, we did not have the 40G version of the A100. We ran our scripts on the A100 80G version by **limiting the memory to 40G.** Unfortunately, we found that the A100 40G actually has only **39.35G** of memory available. This slight difference means that our early training scripts are not compatible with the A100 40G graphics card. However, we have found a solution: simply modify one parameter. This can be done by either **adding 'flash_attention'** or **setting the data's 'cut_len' to 2048**. Therefore, there may be some differences in the final trained model weights. **But the overall training process and dataset are completely same.**

Our training scripts are located in the 'train_docker' folder. You can create and run a Docker image by executing 'Dockerfile.train_flash'. If it runs successfully, the adaptor weights from the run will be uploaded to Hugging Face. The URL is https://huggingface.co/xxyyy123, and the specific weights will be similar to 'final_submit_v3_xxxxx'. The exact name depends on the time of the final upload.

Run steps:
```
# build image
docker build -t trainer:flash -f Dockerfile.train_flash .

# run
docker run -it -p 127.0.0.1:8111:80 --name trainer_flash trainer:flash /bin/bash

# Docker run.
# if CMD failed
cd LLaMA-Efficient-Tuning
bash nips_finetune_flash.sh


```
After completing these steps, the weights will be stored in 'final_v3_test' within the Docker environment. They will also be uploaded to https://huggingface.co/xxyyy123. The specific project will be named 'final_submit_v3_xxxxx'.

We also noticed that the installation process for the **'flash_attention'** library, which we used in our training, is quite complex. **We cannot guarantee successful installation.** Therefore, we provide an alternative Dockerfile that does not use 'flash_attention'.

If you encounter issues with the training scripts mentioned above, you can try the following scripts.

Run steps:

```
# build image
docker build -t trainer:context -f Dockerfile.train_context .

# run
docker run -it -p 127.0.0.1:8111:80 --name trainer_context trainer:context /bin/bash

# Docker run.
# if CMD failed
cd LLaMA-Efficient-Tuning
bash nips_finetune_context2048.sh

```
