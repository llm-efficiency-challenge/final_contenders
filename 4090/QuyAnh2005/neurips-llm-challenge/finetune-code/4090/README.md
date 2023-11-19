## Quick start to fine tune in 4090

```
# Build
$ docker build -t neurips-4090-train:latest  .
$ docker run --runtime=nvidia -ti <container_id> bash

# Inside the container run
$ python train.py
```
