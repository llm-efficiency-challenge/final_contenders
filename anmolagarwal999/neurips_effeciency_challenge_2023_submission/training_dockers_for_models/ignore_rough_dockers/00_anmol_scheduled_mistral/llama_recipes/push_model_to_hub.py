from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from huggingface_hub import login, HfApi
from huggingface_hub import create_repo
import os

os.environ["HUGGINGFACE_TOKEN"] = "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"

model_path = "/home/anmol/nips_challenge/efficiency_challenge_repo/code/00_starter_repo/neurips_llm_efficiency_challenge/sample-submissions/llama_recipes/models_saved/mistral_weights_by_aj/WHOLE_best_model_yet_epoch_1"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path)


repo_name = "anmolagarwal999/mistral_base_finetuned_WHOLE_best_model_yet_epoch_1"
# pt_model = DistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_tf=True)
# pt_model.save_pretrained("path/to/awesome-name-you-picked")
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)