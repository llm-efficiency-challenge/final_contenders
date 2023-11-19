import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import random_split
from tqdm import tqdm
import numpy as np
import random

random.seed(42)

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
logger = logging.getLogger(__name__)
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer

COLUMNS = np.array(["instruction", "input", "output"])


def prepare(
    csv_path: Path,
    destination_path: Path = Path("data/lima-platypus-final"),
    checkpoint_dir: Path = Path("../model/Mistral-7B-v0.1"),
    test_split_fraction: float = 0.05,
    seed: int = 42,
    mask_inputs: bool = False,
    ignore_index: int = -1,
    max_seq_length: Optional[int] = None,
) -> None:
    """Prepare a CSV dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    if max_seq_length is None:
        with open(checkpoint_dir / "lit_config.json", "r") as file:
            config = json.load(file)
            max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)
    logger.info("Loading data file ...")
    import pandas as pd

    df = pd.read_csv(csv_path, dtype=str, escapechar='\\').fillna("")
    if not np.array_equal(df.columns.values, COLUMNS):
        raise ValueError(f"CSV columns must be {COLUMNS}, found {df.columns.values}")
    data = json.loads(df.to_json(orient="records", indent=4))
    random.shuffle(data)

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    # Partition the dataset into train and test
    train_set, test_set = random_split(
        data, [1.0 - test_split_fraction, test_split_fraction], generator=torch.Generator().manual_seed(seed)
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"test has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set) if sample["instruction"] != "" and sample["output"] != ""
    ]
    train_set = [sample for sample in train_set if sample is not None]
    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(test_set) if sample["instruction"] != "" and sample["output"] != ""
    ]
    test_set = [sample for sample in test_set if sample is not None]
    torch.save(test_set, destination_path / "test.pt")


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool, ignore_index: int) -> dict:
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt_and_response_no_truncation = tokenizer.encode(full_prompt_and_response, eos=True)
    
    # Check if the untruncated sequence is too long
    if len(encoded_full_prompt_and_response_no_truncation) > max_length:
        return None

    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


def generate_prompt(example: dict) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    omit_tags = False

    if example["instruction"].strip()[-1] == ":" or (example["input"] and example["input"].strip()[-1] == ":"):
        omit_tags = random.random() < 0.3

    if omit_tags:
        if example["input"]:
            return f"{example['instruction']}\n\n{example['input']} "
        return f"{example['instruction']} "
    else:
        if example["input"]:
            return (
                f"[INST] {example['instruction']}\n\n{example['input']} [/INST]"
            )
        return (
            f"[INST] {example['instruction']} [/INST]"
        )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare, as_positional=False)
