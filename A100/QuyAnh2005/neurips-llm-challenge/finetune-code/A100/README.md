## Quick start to fine tune in A100

```
# Build
$ docker build -t neurips-a100-train:latest  .
$ docker run --runtime=nvidia -ti <container_id> bash

# Inside the container run
$ python train.py
```
