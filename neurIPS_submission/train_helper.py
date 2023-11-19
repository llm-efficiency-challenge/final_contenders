from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
import torch, os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
import os
from copy import deepcopy
from wandb import login as wandb_login

# Set torch config to probably avoid OOM maybe
print(f'Setting PYTORCH_CUDA_ALLOC_CONF to max_split_size_mb:256 to possibly avoid fragmentation')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

# Login to WANDB. Needs env key WANDB_API_KEY
wandb_login()
device = torch.device('cuda:0')
model_name = 'Qwen/Qwen-14B'


# train_size = 50000
test_size = 500
kwargs = {
    # 'cache_dir':'/home/datta0/.cache/huggingface/hub'
}


def create_cnn_prompt(sample):
    article = f'### \n Article:  {sample["article"]}'
    prompt = f'Summarize the above article in 3 sentences.'
    summary = f'{sample["highlights"]}'
    parts = [article, prompt, summary]
    formatted = "\n\n".join(parts)
    formatted = formatted.replace('\\n','\n')
    sample['text'] = formatted
    return sample

def create_prompt(rec):
    # intro = f'You are the most knowledgable AI assistant. You answer everything precise and concise. The following text explains what you need to help me with and what information is provided to you. Please answer appropriately'
    start = f" {rec['context']}"
    question = f"Question: {rec['input']}" if 'input' in rec else None
    response = f"Response: {rec['output']}" if 'output' in rec else None
    parts = [part for part in [start, question, response] if part]
    formatted_prompt = "\n\n".join(parts)
    formatted_prompt = formatted_prompt.replace('\\n', '\n')
    rec["text"] = formatted_prompt
    return rec

def create_jeopardy_prompt(rec):
    # intro = f'You are the most knowledgable AI assistant. You answer everything precise and concise. The following text explains what you need to help me with and what information is provided to you. Please answer appropriately'
    question = f'Question: {rec["question"]}' if "question" in rec else None
    response = f'Response: {rec["answer"]}' if "answer" in rec else None
    parts = [part for part in [question, response] if part]
    formatted_prompt = "\n\n".join(parts)
    formatted_prompt = formatted_prompt.replace('\\n', '\n')
    rec["text"] = formatted_prompt
    return rec

def create_dolly_prompt(rec):
    # intro = f'You are the most knowledgable AI assistant. You answer everything precise and concise. The following text explains what you need to help me with and what information is provided to you. Please answer appropriately'
    start = f" {rec['context']}" if 'context' in rec else None
    question = f"Question: {rec['instruction']}" if 'instruction' in rec else None
    response = f"Response: {rec['response']}" if 'response' in rec else None
    parts = [part for part in [start, question, response] if part]
    formatted_prompt = "\n\n".join(parts)
    formatted_prompt = formatted_prompt.replace('\\n', '\n')
    rec["text"] = formatted_prompt
    return rec



def get_model_and_tokenizer(lora_config):

    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code = True, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_name, config = model_config, use_fast = False,trust_remote_code = True, **kwargs)
    tokenizer.eos_token = '<|endoftext|>'
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config = model_config,
        torch_dtype = torch.bfloat16,
        device_map = 'auto',
        trust_remote_code = True,
        **kwargs
        # use_flash_attention_2 = True,
    )

    # Freeze Model Params  
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later

    model.gradient_checkpointing_enable()
    
    model = get_peft_model(model,lora_config)

    get_trainable_params(model)

    return tokenizer,model

def get_token_len(sample):
    sample['len_of_token'] = len(sample['input_ids'])
    return sample

def load_dataset_sorted(tokenizer,name):
    # Randomly shuffle and choose only 60k sample. We train on 50k and eval on 1k. Rest are buffer
    train_size = 100000
    if name=='nampdn-ai/tiny-textbooks':
        train_size = 80000
        dataset = load_dataset(name,split='train').shuffle().select(list(range(1,int(train_size*1.2))))
    elif name=='databricks/databricks-dolly-15k':
        dataset = load_dataset(name,split='train').map(create_dolly_prompt).filter(lambda x:len(x['text'].split(' '))>10)
    elif name=='cnn_dailymail':
        train_size = 80000
        dataset = load_dataset(name,'3.0.0',split='train').shuffle().select(list(range(1,int(train_size*1.2)))).map(create_cnn_prompt)
    elif name=='jeopardy':
        train_size = 80000
        dataset = load_dataset(name,split='train').shuffle().select(list(range(1,int(train_size*1.2)))).map(create_jeopardy_prompt)
    elif name=='OpenAssistant/oasst_top1_2023-08-25':
        dataset = load_dataset(name,split='train')
    cols = dataset.column_names
    dataset = dataset.map(lambda x:tokenizer(x['text']),batched = True).filter(lambda x:len(x['input_ids'])<=2048).map(get_token_len)
    dataset = dataset.train_test_split(shuffle = True,train_size=min(train_size,len(dataset)-500),test_size=test_size).sort('len_of_token').remove_columns(cols)
    train_data,eval_data = dataset['train'], dataset['test']
    
    return train_data,eval_data



def get_trainable_params(model):
    trainable_params = 0
    all_param = 0
    trainables = []
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainables.append((name,param.numel()))
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def train_model(dataset_name,):

    learning_rate = 1e-4
    lr_scheduler_type = "cosine"
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.1
    grad_acc_steps = 8
    train_batch_size = 2
    output_dir = '/home/supriyar/datta0/neurIPS_submission'+'/final_'+dataset_name.replace('/','_')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)
    run_name = output_dir.replace('/','_')
    num_epochs = 1
    

    if dataset_name=='cnn_dailymail':
        lora_r = 8
        lora_alpha = 16
        learning_rate =1e-5
        train_batch_size = 1    
        grad_acc_steps = 16
        target_modules = ['c_attn','c_proj','w1','w2']
        layers_to_transform = list(range(30,40))
    
    if dataset_name=='jeopardy':
        lora_r = 8
        lora_alpha = 16
        learning_rate =1e-4
        train_batch_size = 2
        grad_acc_steps = 8
        target_modules = ['c_attn','c_proj']
        layers_to_transform = list(range(30,40))
    
    if dataset_name=='databricks/databricks-dolly-15k':
        lora_r = 16
        lora_alpha = 32
        learning_rate = 3e-4
        train_batch_size = 2
        grad_acc_steps = 132
        target_modules = ['c_attn','c_proj']
        layers_to_transform = list(range(30,40))
    
    if dataset_name=='OpenAssistant/oasst_top1_2023-08-25':
        lora_r = 8
        lora_alpha = 16
        learning_rate =1e-4
        train_batch_size = 1
        grad_acc_steps = 16
        target_modules = ['c_proj']
        layers_to_transform = list(range(30,40))

    if dataset_name=='nampdn-ai/tiny-textbooks':
        lora_r = 16
        lora_alpha = 32
        learning_rate =1e-5
        train_batch_size = 1
        grad_acc_steps = 8
        target_modules = ['c_attn','c_proj','w1','w2']
        layers_to_transform = list(range(0,40))

    
    print(f'Training the model. Will save later to {output_dir}')
    
    lora_config = LoraConfig(
        r = lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        lora_dropout=0.1,
        bias = "none",
        task_type = "CAUSAL_LM"
    )

    tokenizer,model = get_model_and_tokenizer(lora_config)
    train_data, eval_data = load_dataset_sorted(tokenizer,dataset_name)

    trainer = Trainer(
        model = model,
        train_dataset = train_data,
        eval_dataset = eval_data,
        tokenizer = tokenizer,
        args = TrainingArguments(
            num_train_epochs=num_epochs,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=grad_acc_steps,
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            warmup_steps=0.01,
            # max_steps=50, #20,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            bf16=True,
            logging_steps=0.02,
            output_dir=output_dir,
            optim=f"paged_adamw_32bit",
            save_steps = 0.02,
            save_total_limit=3,
            disable_tqdm = False,
            resume_from_checkpoint=True,
            remove_unused_columns=True,
            evaluation_strategy="steps",
            eval_steps = 0.02,
            # eval_accumulation_steps=1,
            per_device_eval_batch_size=train_batch_size,
            # load_best_model_at_end=True,
            report_to="wandb",
            run_name=run_name,
            push_to_hub=True,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer,pad_to_multiple_of=8, return_tensors="pt",mlm=False)
    )
    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs
    print(model)

    with torch.cuda.amp.autocast(enabled=True, dtype = torch.bfloat16):
        trainer.train()

    trainer.push_to_hub(f'qwen-final-{dataset_name}')
    return True


