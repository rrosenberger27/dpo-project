import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

def load_sft_model_and_tokenizer() -> tuple[PeftModel, AutoTokenizer]:
    """
    Method to load the model and tokenizer that we will finetune with sft

    Returns
    -------
        tuple[PeftModel, AutoTokenizer]
            The model we will finetune (integrated with LoRA adapters) and its corresponding tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", dtype=torch.float16, device_map='auto')
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05
    )
    peft_model = get_peft_model(base_model, lora_config)

    return peft_model, tokenizer

def load_dpo_model_and_tokenizer(sft_model_path="merged_sft_model") -> tuple[PeftModel, AutoTokenizer]:
    """
    Method to load the model and tokenizer that we will finetune with dpo

    Parameters
    ---------
        sft_model_path : string
            The path to the finetuned model to load

    Returns
    ------
        tuple[PeftModel, AutoTokenizer]
            The finetuned model that we will perform dpo on and the corresponding tokenizer
    """
    sft_model = AutoModelForCausalLM.from_pretrained(sft_model_path, dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05
    )

    dpo_peft_model = get_peft_model(sft_model, lora_config)

    return dpo_peft_model, tokenizer


def load_full_model(sft_model_path="merged_sft_model", dpo_adapters_path="dpo_adapter") :
    base_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path, 
        device_map="auto"
    )

    final_model = PeftModel.from_pretrained(
        base_model, 
        dpo_adapters_path
    )
    # Optionally merge into single unit
    # final_standalone_model = final_model.merge_and_unload()
    return final_model
    