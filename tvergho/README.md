# NeurIPS Large Language Model Efficiency Challenge Submission

This is my evaluation submission for the NeurIPS LLM efficiency challenge. This is a **student-only** team.

To start the server for the `neurips/local` model, build and run the Dockerfile on any device with a NVIDIA GPU. The model was trained and tested on an RTX 4090 with 24GB of memory. The server will start on port 80 by default.

## Training Reproduction

All training-related code is in the `training` directory. Building and running the Dockerfile should start the training loop. After that completes, the model file should be output in the `training/lit-gpt/out` directory.

The datasets used are [LIMA](https://huggingface.co/datasets/GAIR/lima) and specific components from [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus). The only sources from Open-Platypus that were used are ScienceQA, SciBench, ReClor, TheoremQA, ARB, and Guanaco, which are all human-generated and/or were clarified to fall within the scope of the competition rules.