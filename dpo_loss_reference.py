import torch
import torch.nn.functional as F

def dpo_loss(pi_logps, ref_logps, yw_inds, yl_inds, beta) : 
    """
    pi_logps : The log probability of each pi(y|x). Shape of (B,) 
    ref_logps: The log probability of each ref(y|x). Shape of (B,)
    yw_inds : The indices for each winning y. Shape of (B/2,) where each value is in [0, B-1]
    y_l inds : The indices for each losing y. Shape of (B/2,) where each value is in [0, B-1]
    beta : The KL divergence hyperparameter
    
    """

    pi_logyw, pi_logyl = pi_logps[yw_inds], pi_logps[yl_inds]
    ref_logyw, ref_logyl = ref_logps[yw_inds], ref_logps[yl_inds]

    # DPO loss is - logsigmoid(beta * [log(pi(yw) / ref(yw)) - log(pi(yl) / ref(yl))])
    # will express this then as - logsigmoid (beta * [logpiyw - logpiyl - (logrefyw - logrefyl)])

    pi_log_diffs = pi_logyw - pi_logyl
    ref_log_diffs = ref_logyw - ref_logyl
    loss = -F.logsigmoid(beta * (pi_log_diffs - ref_log_diffs))

    # In DPO, the reward r(x,y) is solved to be :
    #   B * log(pi(y) / ref(y)) + B * log Z(x) --> where Z(x) can just be viewed as some constant across yw and yl --> we can then just define the reward without it 

    rewards = beta * (pi_logps - ref_logps).detach() # call detach so the reward is not mapped to the gradient computation graph, reward will be (B,) and defined for pi(y), ref(y) for the same y 

    return loss, rewards