# Data



| Dataset | Number of Entries | Source                                            | Dataset Part     |
|---------|-------------------|---------------------------------------------------|------------------|
| dolly   | 8000              | databricks/databricks-dolly-15k                   | train            |
| bbq     | 8000              | tasksource/bigbench: bbq_lite_json                | train            |
| cnn     | 3000              | "cnn_dailymail", "3.0.0"                          | train            |
| mmlu    | 19400             | cais/mmlu                                         | auxiliary_train  |
| sciqa   | 6508              | tasksource/ScienceQA_text_only                    | train            |
| gsm     | 2000              | gsm8k                                             | train            |
| bigbench| 1600              | tasksource/bigbench                               | train            |

This table provides a clear overview of the datasets with their respective numbers of entries, sources, and the part of the dataset they belong to.

Finally, we used the code in 'data_combine.ipynb' to combine all the data together to obtain the final dataset.
