from einops import repeat, rearrange
import torch
import torch.nn.functional as F

def kl_normal(qm, qv, pm, pv, sequence_lengths):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)

    # mask out the padding after sequence length for each datapoint
    bs, max_sequence_length, _ = element_wise.shape

    # range_tensor = torch.arange(max_sequence_length).unsqueeze(0).expand(bs, -1) # [seq] -> [1, seq] -> [bs, seq]
    range_tensor = repeat(torch.arange(max_sequence_length), 'l -> b l', b=bs)
    mask = range_tensor < rearrange(sequence_lengths, 'b -> b ()')
    mask = rearrange(mask, 'b s -> b s ()')

    kl = element_wise * mask.float()
    kl = kl.sum(-1).sum(-1) #sum over latent_dim and t
    
    return kl



def log_bernoulli_with_logits(x, logits, sequence_lengths):
    """
    Computes the log probability of a Bernoulli given its logits

    Args:
        x: tensor: (batch, dim): Observation
        logits: tensor: (batch, dim): Bernoulli logits

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    
    log_prob = F.binary_cross_entropy(input=logits, target=x) 
    bs, max_sequence_length, _ = x.shape
    
    range_tensor = repeat(torch.arange(max_sequence_length), 'l -> b l', b=bs)
    mask = range_tensor < rearrange(sequence_lengths, 'b -> b ()')
    mask = rearrange(mask, 'b s -> b s ()')

    nll = log_prob * mask.float()
    
    return nll.sum(-1).sum(-1)