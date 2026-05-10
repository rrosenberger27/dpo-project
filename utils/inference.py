import torch
from peft import PeftModel
from transformers import PreTrainedTokenizer
from datasets import Dataset

def generate_and_print_example_outputs(model: PeftModel, tokenizer: PreTrainedTokenizer, dataset: Dataset) : 
    print("="*50)
    model.eval()
    for example in dataset : 
        if 'messages' in example:
            messages = [example['messages'][0]]
        elif 'chosen' in example:
            messages = [example['chosen'][0]]
        user_prompt = messages[0]['content']
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad() : 
            outputs = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.pad_token_id)
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = full_output.split("assistant\n")[-1] 
        print(f"PROMPT:\n{user_prompt}\n")
        print(f"MODEL OUTPUT:\n{reply}\n")
        print("="*50)
