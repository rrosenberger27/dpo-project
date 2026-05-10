import torch
from peft import PeftModel
from accelerate import Accelerator
from tqdm.auto import tqdm
from torch.nn.functional import log_softmax, logsigmoid

def get_inputs_and_labels(prefix, device, batch):
    ids = batch[prefix + 'ids'].to(device)
    mask = batch[prefix + 'mask'].to(device)
    labels = batch[prefix + 'labels'].to(device)
    shift_labels = labels[:, 1:] #shift by 1 so that the predicted word lines up
    loss_mask = shift_labels != -100
    gather_labels = shift_labels.clone()
    gather_labels[~loss_mask] = 0
    return ids, mask, gather_labels, loss_mask

def get_logps(output, gather_labels, loss_mask, ) :
    logits = output.logits # batch size by seq_len by voc_size
    shift_logits = logits[:, :-1, :] #disclude the last word predicted since this was predicted on what was already the last token
    all_logps = log_softmax(shift_logits, dim=-1)
    seq_logps = torch.gather(all_logps, dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)
    seq_logps[~loss_mask] = 0
    logps = torch.sum(seq_logps, dim=-1)
    return logps

def train_dpo(model: PeftModel, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, accelerator: Accelerator, beta: float = 0.1, num_epochs: int = 2):
    model.train()
    device = model.device
    step_losses = []
    step_rewards = []
    for epoch in range(num_epochs) :
        epoch_loss = num_iters = 0
        progress_bar = tqdm(dataloader, desc=f"DPO Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            with accelerator.accumulate(model):
                # under policy :
                chosen_ids, chosen_mask, gather_chosen_labels, chosen_loss_mask = get_inputs_and_labels('chosen_', device, batch)
                rejected_ids, rejected_mask, gather_rej_labels, rej_loss_mask = get_inputs_and_labels('rejected_', device, batch)

                # compute chosen logps
                pi_chosen_output = model(input_ids=chosen_ids, attention_mask=chosen_mask)
                pi_logps_chosen = get_logps(pi_chosen_output, gather_chosen_labels, chosen_loss_mask)

                # compute log probabilities for rejected example
                pi_rejected_output = model(input_ids=rejected_ids, attention_mask=rejected_mask)
                pi_logps_rejected = get_logps(pi_rejected_output, gather_rej_labels, rej_loss_mask)

                # under reference: 
                with model.disable_adapter(), torch.no_grad():
                    ref_chosen_output = model(input_ids=chosen_ids, attention_mask=chosen_mask)
                    ref_logps_chosen = get_logps(ref_chosen_output, gather_chosen_labels, chosen_loss_mask) 

                    # compute log probabilities for rejected example
                    ref_rejected_output = model(input_ids=rejected_ids, attention_mask=rejected_mask)
                    ref_logps_rejected =  get_logps(ref_rejected_output, gather_rej_labels, rej_loss_mask)
                
                loss, rewards = dpo_loss(pi_logps_chosen, pi_logps_rejected, ref_logps_chosen, ref_logps_rejected, beta)                
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            step_losses.append(loss.item())
            step_rewards.append(rewards.mean().item())
            epoch_loss += loss.item()
            num_iters += 1
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'reward': f"{rewards.mean().item():.4f}"})
        
        print(f"Average epoch loss for epoch {epoch} : {epoch_loss / num_iters}")
    return step_losses, step_rewards

def test_dpo(model: PeftModel, dataloader:torch.utils.data.DataLoader) : 
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad() :
        progress_bar = tqdm(dataloader, desc="DPO Testing")
        for batch in progress_bar : 
            chosen_ids, chosen_mask, chosen_gather_labels, chosen_loss_mask = get_inputs_and_labels("chosen_", model.device, batch)
            rej_ids, rej_mask, rej_gather_labels, rej_loss_mask = get_inputs_and_labels("rejected_", model.device, batch)
            
            pi_chosen_output = model(input_ids=chosen_ids, attention_mask=chosen_mask)
            pi_rej_output = model(input_ids=rej_ids, attention_mask=rej_mask)
            pi_chosen_logps = get_logps(pi_chosen_output, chosen_gather_labels, chosen_loss_mask)
            pi_rej_logps = get_logps(pi_rej_output, rej_gather_labels, rej_loss_mask)

            with model.disable_adapter():
                ref_chosen_output = model(input_ids=chosen_ids, attention_mask=chosen_mask)
                ref_rej_output = model(input_ids=rej_ids, attention_mask=rej_mask)
                ref_chosen_logps = get_logps(ref_chosen_output, chosen_gather_labels, chosen_loss_mask)
                ref_rej_logps = get_logps(ref_rej_output, rej_gather_labels, rej_loss_mask)

            batch_loss, rewards = dpo_loss(pi_chosen_logps, pi_rej_logps, ref_chosen_logps, ref_rej_logps)


            total_loss += batch_loss.item()
            total_accuracy += (rewards > 0).float().mean().item()
    
    accuracy = total_accuracy / len(dataloader)
    loss = total_loss / len(dataloader)
    return accuracy, loss



def dpo_loss(pi_logps_chosen: torch.Tensor, pi_logps_rejected: torch.Tensor, ref_logps_chosen: torch.Tensor, ref_logps_rejected:torch.Tensor, beta: float = 0.1):
    # loss is logsigmoid(beta * pi_log_diffs - ref_log_diffs)
    pi_log_diffs = pi_logps_chosen - pi_logps_rejected
    ref_log_diffs = ref_logps_chosen - ref_logps_rejected
    loss = -logsigmoid(beta * (pi_log_diffs - ref_log_diffs))

    # reward is (beta * pi_log_diffs - ref_log_diffs) + (beta * Z(x)) ---> Z(x) just some constant so ignore it
    rewards = beta * (pi_log_diffs - ref_log_diffs).detach()
    return loss.mean(), rewards
    
