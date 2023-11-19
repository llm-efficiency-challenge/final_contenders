## Build and run training
```
docker build -t mdouglas_training .
docker run --gpus all -ti mdouglas_training
```

## Outputs
The training process will output model artifacts in the container at `/workspace/out/Mistral-7B-sft-v0/` and `/workspace/out/Mistral-7B-sft-v1`. 
These artifacts can be copied from the container with `docker cp`.

Alternatively, mount a volume at /workspace/out/.
