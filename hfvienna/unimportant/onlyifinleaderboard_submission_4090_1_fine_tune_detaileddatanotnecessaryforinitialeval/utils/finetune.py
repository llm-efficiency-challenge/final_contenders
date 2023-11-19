import torch 
from trl import SFTTrainer
from datasets import load_dataset 
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Load dataset and split it into training and testing subsets
dataset = load_dataset('lighteval/legal_summarization', 'MultiLexSum', split="train")

# Define a function to format prompts for summarization tasks
def formatting_prompts_func(examples):
    output_text = []
    for i in range(len(examples["article"])):
        article = examples["article"][i]
        summary = examples["summary"][i]

        text = f'''
{article}

Summarize the above article.        

{summary}
        '''
        output_text.append(text)

    return output_text

# Load pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", 
    device_map='auto', 
    load_in_4bit=True,
    use_flash_attention_2=True, 
    use_cache=False
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Configure PEFT (Projection for Effective Finetuning of Transformers) settings
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "v_proj", 
        "k_proj", 
        "o_proj",
        "gate_proj",
        #"up_proj",
        #"down_proj",
        #"lm_head"
        ],
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for k-bit training and apply PEFT
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# Define training arguments
args = TrainingArguments(
    output_dir="mistral_legal_summarization",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=6,
    gradient_checkpointing=True,
    warmup_ratio=0.03,
    logging_steps=1,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    lr_scheduler_type='constant',
)

# Initialize the SFTTrainer and set arguments
trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    formatting_func=formatting_prompts_func,
    args=args,
    train_dataset=dataset,
)

# Start finetuning
trainer.train()

# Save the finetuned model
trainer.save_model("mistral_legal_summarization")
