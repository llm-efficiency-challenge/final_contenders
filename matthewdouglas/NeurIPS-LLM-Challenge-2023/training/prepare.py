from datasets import load_dataset
dataset = load_dataset("mosaicml/instruct-v3", split="train")
dataset = dataset.filter(lambda x: x["source"] != "dolly_hhrlhf")
dataset.to_json("./instruct-data/train.jsonl")