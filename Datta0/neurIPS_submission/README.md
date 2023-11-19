# NeurIPS LLM Efficiency Challenge.

This repo contains my submission for [neurIPS LLM efficiency challenge](https://llm-efficiency-challenge.github.io/).
There are 3 submissions and each has its own dockerfile (Dockerfile, Dockerfile_2,Dockerfile_3) which run different combinations of adapters on the same model. Please make sure that no other process is consuming GPU while you run this.

## Info
- Base Model: [Qwen/Qwen-14B](https://huggingface.co/Qwen/Qwen-14B)
- Adapters: [HuggingFace](https://huggingface.co/imdatta0) [This submission trains multiple adapters and uses them depending on the task at hand.]
- dtype: bfloat16
- GPU/Track: A100
- Datasets: [nampdn-ai/tiny-textbooks](https://huggingface.co/datasets/nampdn-ai/tiny-textbooks)-80k, [OpenAssistant](OpenAssistant/oasst_top1_2023-08-25) ~13k, [jeopardy](https://huggingface.co/datasets/jeopardy) ~80k, [dolly](databricks/databricks-dolly-15k) -15k, [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail/viewer/3.0.0) (only train split and 100k randomly chosen) ~80k

- Training Samples: <200,000
- Eval Samples: 1000
- Approx Training time(if run): 20h

## How to Run
Note: If you want to run finetuning as well, follow:
To build the Image, run
```
docker build -f Dockerfile.train -t neurips_train .
```
To run the finetuning (tunes multiple adapters with diff configs on diff datasets, might take close to 20h)
```
docker run --gpus "device=0" -p 8080:80 --rm -ti neurips_train
```
Note: The submission that looks to have qualified is the 2nd one. That doesn't need to train books_adapter. The other two submissions (1 and 3) have better scores in many scenarios but unfortunately, have a few `NULL` for a couple of datasets. Those submissions do make use of books_adapter. If you want to train books_adapter, set `TRAIN_BOOKS` in Dockerfile.train to `true`. The reason behind those `NULL` *might*  be beam search.
For 2nd submission, the training runs under 24h (without `books_adapter`). 
For 1st and 3rd submission, training runs under 24h cuz they don't use `cnn_adapter`. Its basically a 1-1 swap. 

To run the inference with new artifacts, build the image using
```
docker build -f Dockerfile.final -t neurips_repro .
```
And run the same image using
```
docker run --gpus "device=0" -p 8080:80 --rm -ti neurips_repro
```

To build the Image, run
```
docker build -f Dockerfile -t neurips_inference .
```
For 2nd and 3rd submissions, please run
```
docker build -f Dockerfile_2 -t neurips_inference .
docker build -f Dockerfile_3 -t neurips_inference .
```

To start the server up and make it ready for inference, run
```
docker run -v --gpus "device=0" -p 8080:80 --rm -ti neurips_inference
```
This will start the server on port 8080.
Once the server is up, you can start sending requests via [HELM](https://github.com/stanford-crfm/helm/tree/main).
