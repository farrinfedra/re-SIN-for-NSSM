from einops import repeat, rearrange
import torch
import torch.nn.functional as F

def kl_normal(qm, qv, pm, pv, sequence_lengths, T_reduction='mean'):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    Args:
        qm: tensor: (batch, sequence_dim, dim): q mean
        qv: tensor: (batch, sequence_dim, dim): q variance
        pm: tensor: (batch, sequence_dim, dim): p mean
        pv: tensor: (batch, sequence_dim, dim): p variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    element_wise = element_wise.to(sequence_lengths.device)
    # mask out the padding after sequence length for each datapoint
    bs, max_sequence_length, _ = element_wise.shape

    # range_tensor = torch.arange(max_sequence_length).unsqueeze(0).expand(bs, -1) # [seq] -> [1, seq] -> [bs, seq]
    range_tensor = repeat(torch.arange(max_sequence_length), 'l -> b l', b=bs).to(sequence_lengths.device)
    mask = range_tensor < rearrange(sequence_lengths, 'b -> b ()')
    mask = mask.to(sequence_lengths.device)
    mask = rearrange(mask, 'b s -> b s ()')

    kl = element_wise * mask.float()
    kl = kl.sum(-1) #sum over latent dim
    # sum_T = sequence_lengths.float().sum(-1)
    if T_reduction == 'mean':
        kl = kl.mean(-1) #mean over sequence length
        
    elif T_reduction == 'sum':
        kl = kl.sum(-1)

    
    return kl



def log_bernoulli_with_logits(x, logits, sequence_lengths, T_reduction='mean'):
    """
    Computes the log probability of a Bernoulli given its logits

    Args:
        x: tensor: (batch, dim): Observation
        logits: tensor: (batch, dim): Bernoulli logits

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    log_prob = F.binary_cross_entropy(input=logits, target=x, reduction='none') #shape: (batch, seq_len, dim=88)
    bs, max_sequence_length, _ = x.shape
    
    range_tensor = repeat(torch.arange(max_sequence_length), 'l -> b l', b=bs).to(sequence_lengths.device) #shape: (batch, seq_len)
    mask = range_tensor < rearrange(sequence_lengths, 'b -> b ()')
    mask = mask.to(sequence_lengths.device)
    mask = rearrange(mask, 'b s -> b s ()')
    log_prob = log_prob.to(sequence_lengths.device)
    nll = log_prob * mask.float()
    #take sum over latent and mean of sequence lenghts
    nll = nll.sum(-1) #sum over latent dim
    # sum_T = sequence_lengths.float().sum(-1)
    if T_reduction == 'mean':
        nll = nll.mean(-1) #mean over sequence length
    elif T_reduction == 'sum':
        nll = nll.sum(-1)
    
    return nll