from datetime import datetime
import pytz
import os
################
# os.environ['CUDA_VISIBLE_DEVICES']='2,3'
# os.environ['training_data_path']="/home/t-agarwalan/Desktop/nips_effeciency_challenge/EfficiencyChallenge/data/training_datasets/training_datasets/combined_train_dataset.json"
# os.environ['validation_data_path']="/home/t-agarwalan/Desktop/nips_effeciency_challenge/EfficiencyChallenge/data/training_datasets/training_datasets/combined_valid_dataset.json"
###############
import sys


from huggingface_hub import login, HfApi
from huggingface_hub import create_repo
from llama_recipes.finetuning import main as finetuning
from alpaca_dataset import InstructionDataset as get_anmol_dataset
import uuid
rand_str = uuid.uuid4()
rand_str = f"mistral_model_{rand_str}"
# rand_str = "debug_mistral"

################

# Get the timezone object for GMT+5.5
tz = pytz.timezone('Asia/Kolkata')

# Get the current time in the GMT+5.5 timezone
now = datetime.now(tz)

# Print the current time
print("Starting time is: ", now.strftime('%Y-%m-%d %H:%M:%S %Z%z'))
print("RANDOM STRING is: ", rand_str)
###############
os.environ["HUGGINGFACE_TOKEN"] = "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"
os.environ["HUGGINGFACE_REPO"] = f"anmolagarwal999/nips_challenge_{rand_str}"
os.environ['HUGGINGFACE_HUB_CACHE'] = "/home/anmol/huggingface_hub_cache_dir"
print("REPO DECIDED is: ", os.environ["HUGGINGFACE_REPO"])


def main():
    login(token=os.environ["HUGGINGFACE_TOKEN"])

    # TOT_BATCH_SIZE_WANTED = 128
    # EPOCH_BATCH_SIZE = 8

    TOT_BATCH_SIZE_WANTED = int(sys.argv[1])
    EPOCH_BATCH_SIZE = int(sys.argv[2])
    # NUM_EPOCHES = 10

    assert (TOT_BATCH_SIZE_WANTED % EPOCH_BATCH_SIZE == 0)
    TOT_GRAD_ACCUMULATION_STEPS = TOT_BATCH_SIZE_WANTED // EPOCH_BATCH_SIZE

    print("Total gradient accumulation steps are: ", TOT_GRAD_ACCUMULATION_STEPS)

    OUTPUT_DIR = "./"
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models_saved",
                              f"{TOT_BATCH_SIZE_WANTED}_{EPOCH_BATCH_SIZE}_{rand_str}")

    print("OUTPUT dir is: ", OUTPUT_DIR)

    custom_dataset_file_path = os.path.join(__file__)
    print("Custom dataset path is: ", custom_dataset_file_path)
    kwargs = {
        "model_name": "mistralai/Mistral-7B-v0.1",
        "use_peft": True,
        "peft_method": "lora",
        "quantization": True,
        "batch_size_training": EPOCH_BATCH_SIZE,
        "gradient_accumulation_steps": TOT_GRAD_ACCUMULATION_STEPS,
        "dataset": "custom_dataset",
        # "dataset": "alpaca_dataset",
        # "custom_dataset.file": "./custom_dataset.py",
        "custom_dataset.file": f"{custom_dataset_file_path}:get_anmol_dataset" ,  # "./train.py:get_anmol_dataset",
        # "output_dir": "./random_check",
        "output_dir": OUTPUT_DIR,
    }

    print("Going to begin finetuning")
    finetuning(**kwargs)

    print("Going to use the API to create HF repo")
    # api = HfApi()

    # create_repo(os.environ["HUGGINGFACE_REPO"], private=True, exist_ok=True)

    # api.upload_folder(
    #     # folder_path='./output_dir/',
    #     folder_path=OUTPUT_DIR,
    #     repo_id=os.environ["HUGGINGFACE_REPO"],
    #     repo_type='model',
    # )


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=2,3 python3 train.py 128 8 &> ./training_logs/logs_128_8.log
# CUDA_VISIBLE_DEVICES=0,1 python3 train.py 32 8 &> ./training_logs/logs_bbq_train_llama_2_7b_32_8.log
# CUDA_VISIBLE_DEVICES=0,1 training_data_path=/home/anmol/nips_challenge/efficiency_challenge_repo/data/training_datasets/cnn_train_dataset.json validation_data_path=/home/anmol/nips_challenge/efficiency_challenge_repo/data/training_datasets/cnn_valid_dataset.json python3 train.py 32 8 &> ./training_logs/logs_cnn_train_llama_2_7b_32_8.log

# CUDA_VISIBLE_DEVICES=2,3 training_data_path=/home/anmol/nips_challenge/efficiency_challenge_repo/data/training_datasets/mmlu_train_dataset.json validation_data_path=/home/anmol/nips_challenge/efficiency_challenge_repo/data/training_datasets/mmlu_valid_dataset.json python3 train.py 32 8 &> ./training_logs/logs_mmlu_train_llama_2_7b_32_8.log

# CUDA_VISIBLE_DEVICES=2,3 training_data_path=/home/anmol/nips_challenge/efficiency_challenge_repo/data/training_datasets/tqa_train_dataset.json validation_data_path=/home/anmol/nips_challenge/efficiency_challenge_repo/data/training_datasets/tqa_valid_dataset.json python3 train.py 32 8 &> ./training_logs/logs_tqa_train_llama_2_7b_32_8.log


# CUDA_VISIBLE_DEVICES=2,3 training_data_path=/home/t-agarwalan/Desktop/nips_effeciency_challenge/EfficiencyChallenge/data/training_datasets/training_datasets/combined_train_dataset.json validation_data_path=/home/t-agarwalan/Desktop/nips_effeciency_challenge/EfficiencyChallenge/data/training_datasets/training_datasets/combined_valid_dataset.json python3 train.py 32 8 &> ./training_logs/check_logs.log

# training_data_path=/home/t-agarwalan/Desktop/nips_effeciency_challenge/EfficiencyChallenge/data/new_training_datasets/pegasus_combined_general_train_dataset.json  python3 train.py 32 8 &> ./training_logs/logs_new_GENERAL_mistral_32_8.log

# Get the current time in the GMT+5.5 timezone
now = datetime.now(tz)

# Print the current time
print("Ending time is: ", now.strftime('%Y-%m-%d %H:%M:%S %Z%z'))
