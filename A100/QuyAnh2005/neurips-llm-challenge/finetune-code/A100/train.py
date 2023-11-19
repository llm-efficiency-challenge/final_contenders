# %%writefile train.py

import torch
import transformers

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from huggingface_hub import login, HfApi

from utils import Concatenator, print_trainable_parameters


def main():
    login(token="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")

    # load dataset
    dataset = load_dataset("quyanh/helm-samsum-dolly-lima", split="train")
    print(dataset)

    # load model
    base_model_id = "Qwen/Qwen-14B"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, trust_remote_code=True)
    model.config.window = 2048

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        add_eos_token=True,
        trust_remote_code=True,
        eos_token="<|endoftext|>",
    )

    # prepare dataset
    def get_dataset(dataset):
        prompt = ("""{prompt}<|endoftext|>""")

        def apply_prompt_template(sample):
            return {
                "text": prompt.format(
                    prompt=sample["prompt"]
                )
            }

        dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

        dataset = dataset.map(
            lambda sample: tokenizer(sample["text"]),
            batched=True,
            remove_columns=list(dataset.features)
        )
        dataset = dataset.remove_columns(['token_type_ids'])
        dataset = dataset.map(Concatenator(), batched=True)
        return dataset

    tokenized_train_dataset = get_dataset(dataset)

    # set up lora
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    # training
    project = "qwen-finetune"
    base_model_name = "qwen"
    run_name = base_model_name + "-" + project
    output_dir = "./" + run_name
    
    tokenizer.pad_token = tokenizer.eos_token
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            max_steps=450,
            learning_rate=2e-5, # Want about 10x smaller than the Mistral learning rate
            logging_steps=20,
            bf16=True,
            optim="paged_adamw_8bit",
            logging_dir="./logs",        # Directory for storing logs
            save_strategy="steps",       # Save the model checkpoint every logging step
            save_steps=20,                # Save checkpoints every 50 steps
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    # save
    api = HfApi()
    api.upload_folder(
        folder_path="qwen-qwen-finetune/checkpoint-400",
        repo_id="quyanh/qwen-14b-neurips-a100",
        repo_type='model',
    )


if __name__ == "__main__":
    main()
