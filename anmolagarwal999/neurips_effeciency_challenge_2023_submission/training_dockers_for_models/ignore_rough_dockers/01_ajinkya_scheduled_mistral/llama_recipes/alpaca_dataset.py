# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import torch
from torch.utils.data import Dataset


# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:"
#     ),
# }


class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=1000):
        print("INSIDE INIT FUNCTION")
        # self.ann = json.load(open(dataset_config.data_path))
        if partition == "train":
            # self.ann = self.ann
            # path = "training_datasets/bbq_train_dataset.json"
            # t1 = json.load(open(path))
            # path = "training_datasets/cnn_train_dataset.json"
            # t2 = json.load(open(path))
            # path = "training_datasets/mmlu_train_dataset.json"
            # t3 = json.load(open(path))
            # path = "training_datasets/tqa_train_dataset.json"
            # t4 = json.load(open(path))
            # self.ann = t1+t2+t3+t4  
            path = "combined_train_dataset.json"
            d_full = json.load(open(path))
            self.ann=d_full[:int(len(d_full)*0.8)]

        else:
            # path = "training_datasets/bbq_valid_dataset.json"
            # t1 = json.load(open(path))
            # path = "training_datasets/cnn_valid_dataset.json"
            # t2 = json.load(open(path))
            # path = "training_datasets/mmlu_valid_dataset.json"
            # t3 = json.load(open(path))
            # path = "training_datasets/tqa_valid_dataset.json"
            # t4 = json.load(open(path))
            # self.ann = t1+t2+t3+t4
            path = "combined_train_dataset.json"
            d_full = json.load(open(path))
            self.ann=d_full[int(len(d_full)*0.8):]
            tot_len = len(self.ann)
            # self.ann = self.ann[:tot_len//2]

        self.max_words = max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        # if ann.get("input", "") == "":
        #     prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        # else:
        #     prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        prompt = ann["instruction"]
        example = prompt + ann["output"]
        # print("Example is: ", example)
        # print("########")
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
            # print("Having to truncate")
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }
