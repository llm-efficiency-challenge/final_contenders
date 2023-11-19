import os, sys
from datetime import datetime


from huggingface_hub import login, HfApi 
from huggingface_hub import  create_repo
from llama_recipes.finetuning import main as finetuning
from alpaca_dataset import InstructionDataset as get_anmol_dataset
import uuid
rand_str = uuid.uuid4()

################
import pytz

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
os.environ['HUGGINGFACE_HUB_CACHE']= "/home/anmol/huggingface_hub_cache_dir"
print("REPO DECIDED is: ", os.environ["HUGGINGFACE_REPO"])

def main():
    login(token=os.environ["HUGGINGFACE_TOKEN"])
    
    # TOT_BATCH_SIZE_WANTED = 128
    # EPOCH_BATCH_SIZE = 8

    TOT_BATCH_SIZE_WANTED =int(sys.argv[1])
    EPOCH_BATCH_SIZE =int(sys.argv[2])
    # NUM_EPOCHES = 10
    
    assert(TOT_BATCH_SIZE_WANTED % EPOCH_BATCH_SIZE==0)
    TOT_GRAD_ACCUMULATION_STEPS = TOT_BATCH_SIZE_WANTED // EPOCH_BATCH_SIZE
    
    print("Total gradient accumulation steps are: ", TOT_GRAD_ACCUMULATION_STEPS)
    
    OUTPUT_DIR = "./"
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models_saved", f"{TOT_BATCH_SIZE_WANTED}_{TOT_BATCH_SIZE_WANTED}_{rand_str}")
    
    print("OUTPUT dir is: ", OUTPUT_DIR)
    
    kwargs = {
        "model_name": "Qwen/Qwen-14B",
        "use_peft": True,
        "peft_method": "lora",
        "quantization": True,
        "batch_size_training": EPOCH_BATCH_SIZE,
        "val_batch_size": 16,
        "gradient_accumulation_steps": TOT_GRAD_ACCUMULATION_STEPS, 
        "dataset": "custom_dataset",
        # "dataset": "alpaca_dataset",
        # "custom_dataset.file": "./custom_dataset.py",
        "custom_dataset.file": "./train.py:get_anmol_dataset",
        # "output_dir": "./random_check", 
        "output_dir": OUTPUT_DIR, 
        "target_modules" : ["c_attn","attn.c_proj"]
    }
    
    finetuning(**kwargs)

    api = HfApi() 
    
    create_repo(os.environ["HUGGINGFACE_REPO"], private=True, exist_ok=True)

    api.upload_folder( 
        # folder_path='./output_dir/', 
        folder_path=OUTPUT_DIR, 
        repo_id=os.environ["HUGGINGFACE_REPO"], 
        repo_type='model', 
    )

if __name__ == "__main__":
    main()
    
# CUDA_VISIBLE_DEVICES=2,3 python3 train.py 128 8 &> ./training_logs/logs_128_8.log
# CUDA_VISIBLE_DEVICES=0,1 python3 train.py 32 8 &> ./training_logs/logs_bbq_train_llama_2_7b_32_8.log

# Get the current time in the GMT+5.5 timezone
now = datetime.now(tz)

# Print the current time
print("Ending time is: ", now.strftime('%Y-%m-%d %H:%M:%S %Z%z'))