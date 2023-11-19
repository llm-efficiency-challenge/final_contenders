from datasets import load_dataset, Dataset, concatenate_datasets
import json
import pandas as pd
from collections import Counter
import numpy as np
from tqdm import tqdm
import random
import os

seed = 42
np.random.seed(seed)
random.seed(seed)

def load_lima_dataset():
  lima_dataset = load_dataset("GAIR/lima")
  lima_dataset = lima_dataset['train']
  
  def process_example(example):
    instruction = ' '.join(example['conversations'][:-1])
    output = example['conversations'][-1]
    return {
      'instruction': instruction + "\n",
      'input': "",
      'output': output
    }
  
  def keep_example(example):
      return len(example['conversations']) > 1

  lima_dataset = lima_dataset.filter(keep_example).map(process_example, remove_columns=lima_dataset.column_names)  
  return lima_dataset

def load_platypus_dataset():
  platypus_dataset = load_dataset("garage-bAInd/Open-Platypus")
  allowed_platypus_sources = ['scienceqa', 'scibench', 'reclor', 'theoremqa', 'ARB', 'guanaco']
  platypus_dataset = platypus_dataset.filter(lambda example: example['data_source'] in allowed_platypus_sources)

  platypus_dataset = platypus_dataset['train']

  def process_example(example):
    instruction = example.get('instruction', "")
    input_text = example.get('input', "")
    return {
      'instruction': instruction,
      'input': input_text,
      'output': example['output']
    }
  
  platypus_dataset = platypus_dataset.map(process_example, remove_columns=platypus_dataset.column_names)  
  return platypus_dataset

if __name__ == "__main__":
  lima_dataset = load_lima_dataset()
  platypus_dataset = load_platypus_dataset()
  combined_dataset = concatenate_datasets([lima_dataset, platypus_dataset])
  print(combined_dataset)
  
  # Save the combined dataset as a CSV
  combined_dataset.to_csv("lima-platypus-final.csv", escapechar='\\', index=False)
