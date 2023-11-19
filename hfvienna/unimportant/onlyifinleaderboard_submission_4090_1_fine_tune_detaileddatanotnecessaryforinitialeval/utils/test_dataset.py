from datasets import load_dataset

dataset = load_dataset('lighteval/legal_summarization', 'BillSum', split="train")


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

# Example usage
subset = {key: dataset[key][:3] for key in dataset.features}  # Get the first 3 examples
output_texts = formatting_prompts_func(subset)

# Print the formatted examples
for i, text in enumerate(output_texts):
    print(f"\nExample {i+1}:\n{text}")
