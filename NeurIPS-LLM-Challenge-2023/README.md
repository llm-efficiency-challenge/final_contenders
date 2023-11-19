# Challenge Submission
This is my submission to the NVIDIA RTX 4090 track of the [NeurIPS 2023 Large Language Model Efficiency Challenge: 1 LLM + 1 GPU + 1 Day](https://llm-efficiency-challenge.github.io/).

# Submission Details
There are three variants to be submitted. Each has a `Dockerfile` located in its directory.

* `inference`
* `inference2`
* `inference3`

Further information on the finetuning data and procedure is coming soon.

# Running Inference Server
1. Ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed.

2. Build container.
    ```bash
    cd inference
    docker build -t neurips_submission .
    ```

3. Run
    ```bash
    docker run --gpus all -p 8080:80 neurips_submission
    ```

4. Example API request.
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"prompt": "What is the meaning of life, the universe, and everything?","echo_prompt":0}' http://localhost:8080/proces
    ```
    > `{"text":"The answer is 42.","tokens":[],"logprob":0.0,"request_time":0.766957417014055}`

# Evaluation
Evaluation is performed with the [HELM](https://github.com/stanford-crfm/helm) project.
1. Install HELM.
    ```bash
    pip install git+https://github.com/stanford-crfm/helm.git
    ```
2. Run an evaluation with a `run_specs.conf` file.
    ```bash
    helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 10
    helm-summarize --suite v1
    ```
3. View the results.
    ```bash
    helm-server
    ```