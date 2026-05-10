import torch
from transformers import PreTrainedTokenizer

# Used in original pipeline but went with hugging face datasets and using .map for hugging face datasets since it runs faster

class SFTDataset(torch.utils.data.Dataset):
    """
    Pytorch Dataset to handle the supervised finetuning data
    """
    def __init__(self, dataset, tokenizer: PreTrainedTokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = 256

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        example = self.dataset[index]
        full_text = self.tokenizer.apply_chat_template(example['messages'], tokenize=False)
        prompt_text = self.tokenizer.apply_chat_template([example['messages'][0]], tokenize=False, add_generation_prompt=True) # add_generation_prompt ensures prompt length includes necessary starting headers

        encoded_prompt = self.tokenizer(prompt_text, return_tensors='pt')
        prompt_len = encoded_prompt['input_ids'].shape[1]

        encoded_text = self.tokenizer(full_text, return_tensors='pt', padding='max_length', max_length=self.max_len, truncation=True, return_attention_mask=True)
        labels = encoded_text['input_ids'].clone().squeeze(0)
        labels[:prompt_len] = -100
        labels[encoded_text['attention_mask'].squeeze(0) == 0] = -100
        

        return {
            'attention_mask' : encoded_text['attention_mask'].squeeze(0),
            'input_ids' : encoded_text['input_ids'].squeeze(0),
            'labels' : labels
        }
        
        

class DPODataset(torch.utils.data.Dataset):
    """
    Pytorch Dataset to handle the dpo preferences dataset
    """

    def __init__(self, dataset, tokenizer: PreTrainedTokenizer) :
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = 256
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        example = self.dataset[index]
        full_chosen_text = self.tokenizer.apply_chat_template(example['chosen'], tokenize=False)
        full_rejected_text = self.tokenizer.apply_chat_template(example['rejected'], tokenize=False)
        prompt_text = self.tokenizer.apply_chat_template([example['chosen'][0]], tokenize=False, add_generation_prompt=True)

        encoded_prompt = self.tokenizer(prompt_text, return_tensors='pt')
        prompt_len = encoded_prompt['input_ids'].shape[1]

        encoded_chosen_text = self.tokenizer(full_chosen_text, truncation=True, padding='max_length', max_length=self.max_len, return_attention_mask=True, return_tensors='pt')
        encoded_rejected_text = self.tokenizer(full_rejected_text, truncation=True, padding='max_length', max_length=self.max_len, return_attention_mask=True, return_tensors='pt')

        chosen_ids = encoded_chosen_text['input_ids'].squeeze(0)
        chosen_mask = encoded_chosen_text['attention_mask'].squeeze(0)
        chosen_labels = chosen_ids.clone()
        chosen_labels[:prompt_len] = -100
        chosen_labels[chosen_mask == 0] = -100

        rejected_ids = encoded_rejected_text['input_ids'].squeeze(0)
        rejected_mask = encoded_rejected_text['attention_mask'].squeeze(0)
        rejected_labels = rejected_ids.clone()
        rejected_labels[:prompt_len] = -100
        rejected_labels[rejected_mask == 0] = -100

        return {
            'chosen_ids': chosen_ids,
            'chosen_mask': chosen_mask,
            'chosen_labels': chosen_labels,
            'rejected_ids': rejected_ids,
            'rejected_mask': rejected_mask,
            'rejected_labels': rejected_labels
        }


        