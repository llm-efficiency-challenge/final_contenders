from llama_recipes.utils.train_utils import evaluation
import torch
import numpy as np
from torch.utils.data import Subset,DataLoader, SequentialSampler
import sys

curriculum_learning_config={"starting_percent":1,"adaptive":False, "alpha":0.8, "flipped": False, "step_length": 50, "increase_factor":2}

def get_individual_losses(batch, outputs):
    # Shift so that tokens < n predict n
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = batch["labels"][..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    loss = loss.reshape(batch["labels"][..., 1:].shape)
    loss=loss.mean(-1)
    return loss

def get_scores(dataset, eval_dataloader_args ,model, train_config, local_rank, tokenizer):
    new_dataloader_args={}
    new_dataloader_args.update(eval_dataloader_args)
    new_dataloader_args["dataset"]=dataset
    new_dataloader_args["sampler"]=SequentialSampler(dataset)
    new_dataloader = DataLoader(**new_dataloader_args)
    scores=-1*evaluation(model, train_config, new_dataloader, local_rank, tokenizer, return_individual_losses=True)
    return scores

def get_new_percentage(global_step, previous_percentage):
    if previous_percentage==0 or global_step==0:
        return curriculum_learning_config["starting_percent"]
    elif previous_percentage==1:
        return 1
    elif global_step%curriculum_learning_config["step_length"]==0:    
        print("Updating curriculum")
        curr_percent = min(
            curriculum_learning_config["starting_percent"]
            * (
                curriculum_learning_config["increase_factor"]
                ** int(global_step / curriculum_learning_config["step_length"])
            ),
            1,
        )        
        return curr_percent
    else:
        return previous_percentage

def get_initial_dataloader(train_dataloader_args, eval_dataloader_args, teacher_model, train_config, local_rank, tokenizer, previous_percentage, global_step):
    if curriculum_learning_config["starting_percent"]==1:
        return DataLoader(**train_dataloader_args), 1, None
    else:
        teacher_scores=get_scores(train_dataloader_args["dataset"], eval_dataloader_args ,teacher_model, train_config, local_rank, tokenizer)
        new_percent=get_new_percentage(global_step, previous_percentage)
        sorted_indices=torch.argsort(teacher_scores).cpu().numpy().tolist()
        subset_indices=sorted_indices[:int(new_percent*len(train_dataloader_args["dataset"]))]
        selected_dataset=Subset(train_dataloader_args["dataset"], subset_indices)
        new_train_dataloader_args={}
        new_train_dataloader_args.update(train_dataloader_args)
        new_train_dataloader_args["dataset"]=selected_dataset
        return DataLoader(**new_train_dataloader_args), new_percent, teacher_scores

def get_dataloader(train_dataloader_args, eval_dataloader_args, model, train_config, local_rank, tokenizer, previous_percentage, global_step, latest_scores):
    if latest_scores==None:
       return None, previous_percentage, None
    new_percent=get_new_percentage(global_step, previous_percentage)   
    if new_percent!=previous_percentage:
        if curriculum_learning_config["adaptive"]:
            scores=get_scores(train_dataloader_args["dataset"], eval_dataloader_args ,model, train_config, local_rank, tokenizer)
            scores = (1 - curriculum_learning_config["alpha"]) * latest_scores+ (curriculum_learning_config["alpha"] * scores)
            if curriculum_learning_config["flipped"]:
                sorted_indices=torch.argsort(scores, descending=True)
            else:
                sorted_indices=torch.argsort(scores)
        else:
            scores=latest_scores
            sorted_indices=torch.argsort(scores)        
        subset_indices=sorted_indices[:int(new_percent*len(train_dataloader_args["dataset"]))]
        selected_dataset=Subset(train_dataloader_args["dataset"], subset_indices.cpu().numpy().tolist())
        new_train_dataloader_args={}
        new_train_dataloader_args.update(train_dataloader_args)
        new_train_dataloader_args["dataset"]=selected_dataset
        new_dataloader=DataLoader(**new_train_dataloader_args)
        list(new_dataloader)
        return new_dataloader, new_percent, scores
    else:
        return None, previous_percentage, latest_scores