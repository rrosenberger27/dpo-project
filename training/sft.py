import torch
from peft import PeftModel
from accelerate import Accelerator
from tqdm.auto import tqdm

def train_sft(model: PeftModel, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, accelerator: Accelerator, num_epochs: int = 2) -> None:
    """
    Perform supervised finetuning on the base model
    """
    model.train()
    step_losses = []
    for epoch in range(num_epochs) :
        epoch_loss = 0 
        num_iters = 0
        progress_bar = tqdm(dataloader, desc=f"DPO Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar: 
            with accelerator.accumulate(model) :
                input_ids = batch['input_ids'].to(model.device) 
                attention_mask = batch['attention_mask'].to(model.device) 
                labels = batch['labels'].to(model.device) 
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            num_iters += 1
            epoch_loss += loss.item()
            step_losses.append(loss.item())
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        
        print(f"Average epoch loss for epoch {epoch} : {epoch_loss / num_iters}")
    
    return step_losses
            

def test_sft(model: PeftModel, dataloader: torch.utils.data.DataLoader):
    model.eval()
    total_loss = 0
    total_correct_tokens = 0
    total_evaluated_tokens = 0
    with torch.no_grad() :
        progress_bar = tqdm(dataloader, desc="SFT Testing")
        for batch in progress_bar:
            batch = {k : v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            shift_predictions = predictions[:, :-1]
            labels = batch['labels']
            shift_labels = labels[:, 1:]
            mask = shift_labels != -100

            total_correct_tokens += ((shift_labels == shift_predictions) & mask).sum().item()
            total_evaluated_tokens += mask.sum().item()

    accuracy = total_correct_tokens / total_evaluated_tokens
    loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(loss)).item()

    return accuracy, loss, perplexity

    
