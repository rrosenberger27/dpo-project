from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
# from .datasets import SFTDataset, DPODataset
from torch.utils.data import DataLoader


def load_sft_data(tokenizer : AutoTokenizer, batch_size=4) -> tuple[DataLoader, DataLoader, Dataset]:
    """
    Method to load the preference and sft data from HuggingFace
    """
    train_sft = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split='train_sft')
    test_sft = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split='test_sft')
    test_sample = test_sft.select([10, 100, 200, 500])

    def format_and_tokenize_sft(examples) : 
        full_texts = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in examples['messages']]
        prompt_texts = [tokenizer.apply_chat_template([msg[0]], tokenize=False, add_generation_prompt=True) for msg in examples['messages']]
        
        tokenized_full = tokenizer(full_texts, padding="max_length", truncation=True, max_length=256)
        
        tokenized_prompts = tokenizer(prompt_texts, truncation=True, max_length=256)
        
        batch_labels = []
        
        for i in range(len(full_texts)):
            seq_labels = tokenized_full["input_ids"][i].copy()
            prompt_len = len(tokenized_prompts["input_ids"][i])
            
            for j in range(prompt_len):
                seq_labels[j] = -100
                
            for j in range(len(seq_labels)):
                if tokenized_full["attention_mask"][i][j] == 0:
                    seq_labels[j] = -100
                    
            batch_labels.append(seq_labels)
            
        tokenized_full["labels"] = batch_labels
        
        return tokenized_full
    
    train_tokenized_sft_dataset = train_sft.map(
        format_and_tokenize_sft, 
        batched=True,
        num_proc=2
    )
    train_tokenized_sft_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_tokenized_sft_dataset = test_sft.map(
        format_and_tokenize_sft, 
        batched=True,
        num_proc=2
    )
    test_tokenized_sft_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    # train_sft, test_sft = SFTDataset(train_sft, tokenizer), SFTDataset(test_sft, tokenizer)
    train_sft_dataloader, test_sft_dataloader = DataLoader(train_tokenized_sft_dataset, batch_size=batch_size, shuffle=True, num_workers=2), DataLoader(test_tokenized_sft_dataset, batch_size=batch_size, shuffle=True)

    return train_sft_dataloader, test_sft_dataloader, test_sample

def load_dpo_data(tokenizer : AutoTokenizer, batch_size=2) -> tuple[DataLoader, DataLoader, Dataset]: 
    train_prefs = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split='train_prefs')
    test_prefs = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split='test_prefs')
    test_sample = test_prefs.select([10, 100, 200, 500]) # used later to see example outputs from dpo

    def format_and_tokenize_dpo(examples):
        prompt_texts = [tokenizer.apply_chat_template([msg[0]], tokenize=False, add_generation_prompt=True) for msg in examples['chosen']]
        chosen_texts = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in examples['chosen']]
        rejected_texts = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in examples['rejected']]
        
        tokenized_prompts = tokenizer(prompt_texts, truncation=True, max_length=256)
        tokenized_chosen = tokenizer(chosen_texts, padding="max_length", truncation=True, max_length=256)
        tokenized_rejected = tokenizer(rejected_texts, padding="max_length", truncation=True, max_length=256)
        
        chosen_labels_batch = []
        rejected_labels_batch = []
        
        for i in range(len(chosen_texts)):
            prompt_len = len(tokenized_prompts["input_ids"][i])
            
            c_labels = tokenized_chosen["input_ids"][i].copy()
            for j in range(prompt_len): 
                c_labels[j] = -100
            for j in range(len(c_labels)): 
                if tokenized_chosen["attention_mask"][i][j] == 0:
                    c_labels[j] = -100
            chosen_labels_batch.append(c_labels)
            
            r_labels = tokenized_rejected["input_ids"][i].copy()
            for j in range(prompt_len): 
                r_labels[j] = -100
            for j in range(len(r_labels)):
                if tokenized_rejected["attention_mask"][i][j] == 0:
                    r_labels[j] = -100
            rejected_labels_batch.append(r_labels)
            
        return {
            "chosen_ids": tokenized_chosen["input_ids"],
            "chosen_mask": tokenized_chosen["attention_mask"],
            "chosen_labels": chosen_labels_batch,
            "rejected_ids": tokenized_rejected["input_ids"],
            "rejected_mask": tokenized_rejected["attention_mask"],
            "rejected_labels": rejected_labels_batch,
        }
    
    train_tokenized_dpo_dataset = train_prefs.map(
        format_and_tokenize_dpo,
        batched=True,
        num_proc=2
    )
    train_tokenized_dpo_dataset.set_format(
        type='torch',
        columns=['chosen_ids', 'chosen_mask', 'chosen_labels',
                 'rejected_ids', 'rejected_mask', 'rejected_labels']
    )
    test_tokenized_dpo_dataset = test_prefs.map(
        format_and_tokenize_dpo,
        batched=True,
        num_proc=2
    )
    test_tokenized_dpo_dataset.set_format(
        type='torch',
        columns=['chosen_ids', 'chosen_mask', 'chosen_labels',
                 'rejected_ids', 'rejected_mask', 'rejected_labels']
    )


    train_prefs_dataloader, test_prefs_dataloader = DataLoader(train_tokenized_dpo_dataset, batch_size=batch_size, shuffle=True, num_workers=2), DataLoader(test_tokenized_dpo_dataset, batch_size=batch_size, shuffle=True)
    
    return train_prefs_dataloader, test_prefs_dataloader, test_sample

