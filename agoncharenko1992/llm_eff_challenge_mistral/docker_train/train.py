import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import transformers
import time
from pathlib import Path
import evaluate
import string
import pandas as pd
import numpy as np
import string

from datasets import load_dataset
from datasets import load_metric
from datasets import Dataset
from datasets import concatenate_datasets

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoModelForQuestionAnswering
from transformers import BitsAndBytesConfig
from transformers import GenerationConfig

from peft import AutoPeftModelForCausalLM
from peft import PeftModel
from peft import prepare_model_for_kbit_training
from peft import LoraConfig
from peft import get_peft_model

from huggingface_hub import login

from optimum.bettertransformer import BetterTransformer

from trl import SFTTrainer
from trl import DataCollatorForCompletionOnlyLM

repo_id = "mistralai/Mistral-7B-v0.1"
model_name = 'mistral_7b_v0.1'
#instruct_mistral = "mistralai/Mistral-7B-Instruct-v0.1"

huggingface_write_token="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"
login(token=huggingface_write_token)
repo_name = "agoncharenko1992/llm_challenge_evaluation"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


# tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir='/train_data/.cache')
#Create a new token and add it to the tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'


model = AutoModelForCausalLM.from_pretrained(
    repo_id, 
    quantization_config=bnb_config, 
    device_map={"":0},
    cache_dir='/train_data/.cache'
)

# model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
config = LoraConfig(
    r=4,
    lora_alpha=4,
    target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, config)
print_trainable_parameters(peft_model)

dataset_platypus = load_dataset("garage-bAInd/Open-Platypus", split='train')
dataset_platypus = dataset_platypus.rename_columns({'input': 'context', 'output':'response'})
dataset_platypus = dataset_platypus.remove_columns('data_source')

dataset = load_dataset("databricks/databricks-dolly-15k", split='train')
dataset = dataset.remove_columns("category")

train_dataset = concatenate_datasets([dataset, dataset_platypus])

helm_eval_mmlu_topics = [
    'high_school_biology', 
    'high_school_chemistry',
    'high_school_computer_science',
    'high_school_european_history',
    'high_school_geography',
    'high_school_government_and_politics',
    'high_school_macroeconomics',
    'high_school_mathematics',
    'high_school_microeconomics',
    'high_school_physics',
    'high_school_psychology',
    'high_school_statistics',
    'high_school_us_history',
    'high_school_world_history',
    'moral_disputes',
    'moral_scenarios',
    'philosophy'
]

helm_eval_mmlu_hidden = [
    'formal_logic',
    'logical_fallacies',
]

# no_eval_topics = [
#     'medical_genetics', 
#     'elementary_mathematics', 
#     'college_medicine', 
#     'abstract_algebra', 
#     'jurisprudence', 
#     'clinical_knowledge', 
#     'virology', 
#     'security_studies', 
#     'management', 
#     'econometrics', 
#     'professional_accounting', 
#     'college_biology', 
#     'miscellaneous', 
#     'global_facts', 
#     'nutrition', 
#     'formal_logic', 
#     'astronomy', 
#     'human_aging', 
#     'college_mathematics', 
#     'international_law', 
#     'college_physics', 
#     'college_chemistry', 
#     'conceptual_physics', 
#     'human_sexuality', 
#     'machine_learning', 
#     'professional_law', 
#     'electrical_engineering', 
#     'prehistory', 
#     'marketing', 
#     'logical_fallacies', 
#     'sociology', 
#     'us_foreign_policy', 
#     'world_religions', 
#     'public_relations', 
#     'professional_psychology', 
#     'anatomy', 
#     'business_ethics', 
#     'college_computer_science', 
#     'computer_security', 
#     'professional_medicine',
# ]

mmlu_total_eval = helm_eval_mmlu_topics + helm_eval_mmlu_hidden

mmlu_final_eval_dataset = []

for curr_split in mmlu_total_eval:
    mmlu_final_eval_dataset.append(
        load_dataset('lukaemon/mmlu', curr_split)
    )
    
#     mmlu_dataset = load_dataset('lukaemon/mmlu', split='high_school_biology')
# mmlu_dataset = load_dataset('lukaemon/mmlu')

mmlu_dataset = concatenate_datasets([curr_split['test'] for curr_split in mmlu_final_eval_dataset])

def mmlu_mapping(sample):
    inpt = sample['input']
    A = sample['A']
    B = sample['B']
    C = sample['C']
    D = sample['D']
    response = sample['target']
    return {
        'instruction': f'{inpt}\nA.{A}\nB.{B}\nC.{C}\nD.{D}',
        'context': '',
        'response': response,
    }
    
mmlu_dataset = mmlu_dataset.map(mmlu_mapping, remove_columns=mmlu_dataset.features)

def truthful_qa_mapping(sample):
    question = sample.loc['question']
    possible_answers = sample.loc['mc1_targets']['choices']
    labels = sample.loc['mc1_targets']['labels']
    sample_processed = f'{question}\n'
    
    for i, answer in zip(string.ascii_uppercase, possible_answers):
        sample_processed += f'{i}. {answer}\n'
    
    [correct_answer] =  [
        f'{string.ascii_uppercase[idx]}. {possible_answers[idx]}' 
        for idx, label in enumerate(labels) 
        if label == 1
    ]
    
    return sample_processed, correct_answer
    
truthful_qa_ds = load_dataset('truthful_qa', 'multiple_choice', 'validation')
truthful_qa_pd = pd.DataFrame(truthful_qa_ds['validation'])
applied_df = truthful_qa_pd.apply(
    truthful_qa_mapping, 
    axis='columns', 
    result_type='expand',
)
applied_df.columns = ['instruction', 'response']
truthful_qa_helm = Dataset.from_pandas(applied_df)

validation_dataset = concatenate_datasets([mmlu_dataset, truthful_qa_helm])

SPECIAL_TOKEN_INSTRUCTION = '### Instruction:\n'
SPECIAL_TOKEN_ANSWER = '### Answer:\n'

response_template_ids = tokenizer.encode('\n'+SPECIAL_TOKEN_ANSWER)[2:]

def formatting_prompts_func(example):
    output_texts = []
    instructions, responses = example['instruction'] , example['response']
    contexts = example.get('context', [])
    
    for index, (curr_inst, curr_resp) in enumerate(zip(instructions, responses)):
        c = []
        if len(contexts) > 0 :
            c = contexts[index]
        
        cont = f"\n{c}" if (c is not None) and (len(c) > 0) else ''
        instruction_str = f"{SPECIAL_TOKEN_INSTRUCTION}{cont}{curr_inst}"
        response_str = f"{SPECIAL_TOKEN_ANSWER}{curr_resp}"
        
        # join all the parts together
        prompt = "\n\n".join([i for i in [instruction_str, response_str] if i is not None])
        output_texts.append(prompt)
    return output_texts
    
output_dir = Path(f"/train_data/llm_efficient_train/{model_name}_dolly_platty_mmlu_trqa_neftune_all_train/")


metric = evaluate.load("exact_match")

def compute_metrics(eval_preds):
    em = evaluate.load("exact_match")
    logits, labels = eval_preds
    
    predictions = np.argmax(logits, axis=-1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    for pred, sentence_label in zip(predictions, labels):
        decoded_labels = tokenizer.decode(sentence_label, skip_special_tokens=True)
        decoded_logits = tokenizer.decode(pred, skip_special_tokens=True)

    return em.compute(predictions=predictions, references=labels)
    
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right' #- No Flash Attention-2

tokenizer.max_model_input_sizes
# tokenizer.add_special_tokens({
#     'additional_special_tokens': [SPECIAL_TOKEN_INTRUCTION, SPECIAL_TOKEN_ANSWER],
# })

train_args = transformers.TrainingArguments(
    do_train=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    warmup_steps=500,
    max_steps=4500,
    learning_rate=2e-4,
    fp16=True,
    weight_decay=5e-5,
    logging_steps=2,
    save_steps=50,
    output_dir=output_dir,
    optim="paged_adamw_8bit",
    do_eval=True,
    do_predict=True,
    evaluation_strategy="steps",
    eval_steps=100,
    lr_scheduler_type='cosine',
    report_to='tensorboard',
    gradient_checkpointing=True,
#     group_by_length=True,
)

trainer = SFTTrainer(
    model=peft_model,
    max_seq_length=4096,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    formatting_func=formatting_prompts_func,
    args=train_args,
    neftune_noise_alpha=5,
#     compute_metrics=compute_metrics,
    data_collator=DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_template_ids,
    )
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer.train()

peft_model_fp16 = AutoPeftModelForCausalLM.from_pretrained(
    (output_dir / 'checkpoint-4200').as_posix(),
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    cache_dir='/train_data/.cache',
    use_auth_token=True,
    device_map='cpu',
)

peft_model_fp16.push_to_hub(repo_name)
