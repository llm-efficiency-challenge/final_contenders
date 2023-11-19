# NeurIPS Large Language Model Efficiency Challenge

**Members:** Anmol Agarwal, Ajinkya Deshpande, Shashank Shet, Arun Iyer, Suresh Parthasarathy

**Affiliation:** Microsoft Research India


Our datasets were derived from CNN-DM, MMLU, BigBench, TruthfulQA, BBQ, **ARC (by AllenAI)**, GSM-8k and **MathQA (by Hendrycks)**. We not only include data from the tasks mentioned in the sample_conf file, but also include other diverse tasks from BigBench.

In order to add robustness and fairness to our models, we also include special queries in our dataset which have been perturbed to measure robustness and fairness in the same way as HELM does perturbations.
Our initial findings suggested that models like Mistral are very sensitive to the sequence in which various options are presented, so we also shuffled the options in the query to introduce **option permutation invariance** which led to slight gains in performance.

We also use an ensemble of models. For each query, we classify into which sort of task does it belong to (fact-based knowledge based task OR reasoning based-task) using Regex. The different models we use can be found below.

For any queries, please contact at either of the following emails: t-agarwalan@microsoft.com, anmolagarwal4453@gmail.com, ariy@microsoft.com


##### For Training reproduction
Please follow instructions in `README.md` in `training_dockers_for_models/training_dir_for_reproduction`.


#### FOR Inference RUNNING:
Go to the folder ```run_inference_<NUM>/lit-gpt``` and run the below:
```bash
anmol@GCRSANDBOX432:~/submission_nips/run_inference_1/lit-gpt$ docker build -t sample_submission .
anmol@GCRSANDBOX432:~/submission_nips/run_inference_1/lit-gpt$ docker run --gpus all -p 8080:80 sample_submission

```

#### Submission 1:
Please find eval docker in `run_inference_1`

#### Submission 2:
Please find eval docker in `run_inference_2`


#### Submission 3:
Please find eval docker in `run_inference_3`



## MODELS Description
| Model Name    | Huggingface Link              | Dataset Link                  |
|--------------|------------------------------|------------------------------|
| anmolagarwal999/32_8_mistral_model_59c7f323-74ae-4538-808b-fe054f5ccd84_general_WHOLE_best_model_yet_epoch_2_243      | [Huggingface Model A](https://huggingface.co/anmolagarwal999/32_8_mistral_model_59c7f323-74ae-4538-808b-fe054f5ccd84_general_WHOLE_best_model_yet_epoch_2_243)  | [pegasus_combined_general_train_dataset.json (9846 rows)](https://huggingface.co/datasets/ajdesh2000/pegasus_combined_general_train_dataset) 
| anmolagarwal999/32_8_mistral_model_59c7f323-74ae-4538-808b-fe054f5ccd84_general_WHOLE_best_model_yet_epoch_3_243      | [Huggingface Model B](https://huggingface.co/anmolagarwal999/32_8_mistral_model_59c7f323-74ae-4538-808b-fe054f5ccd84_general_WHOLE_best_model_yet_epoch_3_243)  | [pegasus_combined_general_train_dataset.json (9846 rows)](https://huggingface.co/datasets/ajdesh2000/pegasus_combined_general_train_dataset)  
| anmolagarwal999/mistral_base_finetuned_anmol_WHOLE_best_model_yet_epoch_2_1      | [Huggingface Model C](https://huggingface.co/anmolagarwal999/mistral_base_finetuned_anmol_WHOLE_best_model_yet_epoch_2_1)  | [backup_training_datasets_without_math/combined_train_dataset.json](https://huggingface.co/datasets/ajdesh2000/combined_train_dataset_v2) 
| anmolagarwal999/32_8_mistral_model_exodus_4d0d33a1-e716-41fb-99c1-b6ad3fb75080_WHOLE_best_model_yet_epoch_1_3      | [Huggingface Model D](https://huggingface.co/anmolagarwal999/32_8_mistral_model_exodus_4d0d33a1-e716-41fb-99c1-b6ad3fb75080_WHOLE_best_model_yet_epoch_1_3)  | [exodus_combined_general_train_dataset.json](https://huggingface.co/datasets/ajdesh2000/exodus_combined_general_train_dataset_v2)      
